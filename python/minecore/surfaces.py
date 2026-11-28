"""Triangulated surfaces (DTM): topography and other horizons.

A surface is a 2.5D triangulation of points in plan (Delaunay on x, y); its
elevation at any (x, y) inside the hull is the linear interpolation on the
triangle containing it. Outside the hull the nearest point's elevation is
used and reported (``outside`` count), never silently extrapolated by slope.

Breaklines (OC06): ``build_surface`` makes every breakline segment an edge
of the triangulation (conforming Delaunay: a segment that is not yet an edge
is split at its midpoint, the new vertex taking the breakline's linearly
interpolated elevation, until every piece is an edge). The surface is then
linear along each breakline exactly as drawn. Breaklines that cross in plan
are split at the crossing when their elevations agree there and refused
otherwise. An optional closed boundary trims the triangles outside it; its
edges are made triangle edges as well, with elevations taken from the
surface.

Stored as one ``surface`` catalogue object with a Parquet table of points
(x, y, z) and, when trimmed, a second table with the boundary ring;
triangles are recomputed on load (Delaunay is deterministic for a given
point set, and the inserted breakline vertices are stored as points).
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from minecore._core import MineCoreError
from minecore.blockmodel import BlockModel
from minecore.io.dxf import read_dxf
from minecore.project import Project, bulk

KIND = "surface"
Floats = npt.NDArray[np.float64]


class SurfaceError(MineCoreError):
    """Invalid surface data."""


def _plan_origin(points: Floats) -> Floats:
    """Local origin for triangulation: qhull loses precision at map coordinates."""
    return np.floor(np.asarray(points, np.float64)[:, :2].min(axis=0) / 1000.0) * 1000.0


@dataclass(frozen=True)
class Surface:
    """Delaunay triangulation of ``points`` in plan, optionally trimmed.

    ``boundary``: closed rings (each m x 2) applied with the even-odd rule, so
    a ring inside another is a hole (island). ``max_edge``: triangles with a
    plan edge longer than this (m, user's choice) are removed. A triangle is
    kept when its centroid is inside the boundary; ``build_surface`` makes the
    boundary edges triangle edges so that this test is exact.
    """

    points: Floats  # n x 3
    boundary: tuple[Floats, ...] = ()
    max_edge: float | None = None

    def __post_init__(self) -> None:
        p = np.asarray(self.points, np.float64)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
            raise SurfaceError("a surface needs at least three x, y, z points")
        if not np.isfinite(p).all():
            raise SurfaceError("surface points must be finite")
        for ring in self.boundary:
            b = np.asarray(ring, np.float64)
            if b.ndim != 2 or b.shape[1] != 2 or len(b) < 3 or not np.isfinite(b).all():
                raise SurfaceError("a boundary ring needs at least three finite x, y points")
        if self.max_edge is not None and not (np.isfinite(self.max_edge) and self.max_edge > 0):
            raise SurfaceError("the longest-edge limit must be > 0")

    def _origin(self) -> Floats:
        return _plan_origin(self.points)

    def _delaunay(self) -> Delaunay:
        return self._triangulation

    @cached_property
    def _triangulation(self) -> Delaunay:
        # Cached: a surface is immutable and queried many times (sections, grids).
        try:
            return Delaunay(self.points[:, :2] - self._origin())
        except Exception as exc:
            raise SurfaceError(f"points cannot be triangulated in plan: {exc}") from exc

    def _kept(self, simplices: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        keep = np.ones(len(simplices), bool)
        xy = self.points[simplices, :2]
        if self.boundary:
            keep &= inside_rings(xy.mean(axis=1), self.boundary)
        if self.max_edge is not None:
            edges = np.linalg.norm(xy - np.roll(xy, -1, axis=1), axis=2)
            keep &= edges.max(axis=1) <= self.max_edge
        return keep

    @property
    def trimmed(self) -> bool:
        return bool(self.boundary) or self.max_edge is not None

    def triangles(self) -> npt.NDArray[np.int64]:
        simplices = np.asarray(self._delaunay().simplices, np.int64)
        return simplices[self._kept(simplices)] if self.trimmed else simplices

    def contains(self, xy: Floats) -> npt.NDArray[np.bool_]:
        """Whether each (x, y) lies on a kept triangle."""
        tri = self._delaunay()
        local = np.asarray(xy, np.float64) - self._origin()
        simplex = np.asarray(tri.find_simplex(local), np.int64)
        inside = simplex >= 0
        if self.trimmed and inside.any():
            kept = self._kept(np.asarray(tri.simplices, np.int64))
            inside[inside] = kept[simplex[inside]]
        return inside

    def elevation(self, xy: Floats) -> tuple[Floats, int]:
        """Elevation at each (x, y) and the number of points outside the surface.

        Outside the hull (or the trimmed area) the nearest point's elevation is used.
        """
        xy = np.asarray(xy, np.float64)
        tri = self._delaunay()
        linear = LinearNDInterpolator(tri, self.points[:, 2])
        z = np.asarray(linear(xy - self._origin()), np.float64)
        if self.trimmed:
            z[~self.contains(xy)] = np.nan
        outside = ~np.isfinite(z)
        if outside.any():
            nearest = NearestNDInterpolator(self.points[:, :2], self.points[:, 2])
            z[outside] = nearest(xy[outside])
        return z, int(outside.sum())


def read_points_csv(path: Path) -> Surface:
    """x, y, z columns (header names x/y/z or east/north/elev; else first three)."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t ")
    rows = list(csv.reader(text.splitlines(), dialect))
    header = [h.strip().lower() for h in rows[0]]
    names = {
        "x": ("x", "east", "easting", "dogu", "doğu", "y_saga"),
        "y": ("y", "north", "northing", "kuzey"),
        "z": ("z", "elev", "elevation", "rl", "kot", "h"),
    }
    cols = []
    for key in ("x", "y", "z"):
        found = next((i for i, h in enumerate(header) if h in names[key]), None)
        cols.append(found)
    start = 1
    if None in cols:
        cols, start = [0, 1, 2], (0 if _numeric(rows[0][:3]) else 1)
    data = []
    for row in rows[start:]:
        try:
            data.append([float(row[c].replace(",", ".")) for c in cols])  # type: ignore[index]
        except (ValueError, IndexError):
            continue
    if len(data) < 3:
        raise SurfaceError(f"{path.name}: fewer than three numeric x, y, z rows")
    return Surface(np.array(data))


def _numeric(cells: list[str]) -> bool:
    try:
        [float(c.replace(",", ".")) for c in cells]
    except ValueError:
        return False
    return True


def save_surface(project: Project, name: str, surface: Surface) -> str:
    table = pa.table({c: surface.points[:, i] for i, c in enumerate("xyz")})
    # Saving under an existing name replaces it (new bulk file, same object).
    existing = [r for r in project.objects(KIND) if r.name == name]
    record = (
        existing[0] if existing else project.register_object(KIND, name, metadata={"format": 1})
    )
    ref = bulk.write_table(project.path, record.object_id, "points", table)
    metadata: dict[str, object] = {"format": 1, **ref.to_json()}
    if surface.boundary:
        rings = [np.asarray(r, np.float64) for r in surface.boundary]
        ring_id = np.concatenate([np.full(len(r), k, np.int32) for k, r in enumerate(rings)])
        xy = np.vstack(rings)
        table = pa.table({"ring": ring_id, "x": xy[:, 0], "y": xy[:, 1]})
        metadata["boundary"] = bulk.write_table(
            project.path, record.object_id, "boundary", table
        ).to_json()
    if surface.max_edge is not None:
        metadata["max_edge_m"] = float(surface.max_edge)
    project.update_object(record.object_id, metadata=metadata, description=f"Store {name!r}")
    return record.object_id


def load_surface(project: Project, name: str) -> Surface:
    for record in project.objects(KIND):
        if record.name == name:
            table = bulk.read_table(project.path, bulk.FileRef.from_json(record.metadata))
            points = np.column_stack([table.column(c).to_numpy() for c in "xyz"])
            rings: list[Floats] = []
            if "boundary" in record.metadata:
                t = bulk.read_table(
                    project.path, bulk.FileRef.from_json(record.metadata["boundary"])
                )
                ids = t.column("ring").to_numpy()
                xy = np.column_stack([t.column(c).to_numpy() for c in "xy"])
                rings = [xy[ids == k] for k in np.unique(ids)]
            max_edge = record.metadata.get("max_edge_m")
            return Surface(points, tuple(rings), None if max_edge is None else float(max_edge))
    raise SurfaceError(f"no surface named {name!r}")


def surface_names(project: Project) -> list[str]:
    return [r.name for r in project.objects(KIND)]


def flag_below(model: BlockModel, surface: Surface, attribute: str = "kaya") -> tuple[int, int]:
    """Store 1 for blocks whose centroid is below the surface (rock), else 0 (air).

    Returns (rock blocks, block columns outside the surface hull).
    """
    grid = model.grid
    nx, ny, _ = grid.count
    c = grid.centroids()
    column_xy = c[: nx * ny, :2]
    top, outside = surface.elevation(column_xy)
    k_top = np.tile(top, grid.count[2])
    rock = (c[:, 2] <= k_top).astype(np.float64)
    model.set_attributes(
        {attribute: rock},
        {attribute: {"unit": None}},
        f"Flag blocks below surface as {attribute!r}",
    )
    return int(rock.sum()), outside


# -- breaklines and boundaries (OC06) -----------------------------------------


def inside_polygon(xy: Floats, ring: Floats) -> npt.NDArray[np.bool_]:
    """Even-odd test of points against a closed ring (first point not repeated)."""
    xy = np.asarray(xy, np.float64)
    ring = np.asarray(ring, np.float64)[:, :2]
    if len(ring) > 1 and np.allclose(ring[0], ring[-1], rtol=0.0, atol=1e-9):
        ring = ring[:-1]
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    x1, y1 = ring[:, 0][None, :], ring[:, 1][None, :]
    x2, y2 = np.roll(ring[:, 0], -1)[None, :], np.roll(ring[:, 1], -1)[None, :]
    straddle = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return np.asarray(((straddle & (x < xc)).sum(axis=1) % 2) == 1)


def inside_rings(xy: Floats, rings: tuple[Floats, ...] | list[Floats]) -> npt.NDArray[np.bool_]:
    """Even-odd rule over several rings: a ring inside another makes a hole."""
    count = np.zeros(len(xy), np.int64)
    for ring in rings:
        count += inside_polygon(xy, ring)
    return np.asarray(count % 2 == 1)


@dataclass(frozen=True)
class SurfaceReport:
    """What ``build_surface`` did to honour the breaklines."""

    input_points: int
    breakline_vertices: int
    duplicates_merged: int  # points within the tolerance of a kept vertex (plan)
    elevation_conflicts: int  # merged pairs whose elevations differed > z tolerance
    points_on_breaklines: int  # spot points dropped because a breakline passes over them
    crossings_split: int  # breakline crossings with agreeing elevations
    inserted: int  # vertices added on breaklines/boundary to make them triangle edges
    triangles: int
    short_pieces_left: int = 0  # tiny pieces left unconstrained (tolerate_small_angles)


def _box_pairs(box: Floats) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Pairs i < j of plan boxes (xmin, ymin, xmax, ymax) that overlap.

    A uniform grid of median box size: memory grows with the boxes, not with
    the longest one (a few long hull edges no longer pair with everything).
    """
    empty = np.zeros(0, np.int64)
    if len(box) < 2:
        return empty, empty
    ext = (box[:, 2:] - box[:, :2]).max(axis=1)
    cell = float(np.median(ext))
    if not cell > 0:
        cell = max(float(ext.max()), 1.0)
    lo = box[:, :2].min(axis=0)
    i0 = np.floor((box[:, :2] - lo) / cell).astype(np.int64)
    i1 = np.floor((box[:, 2:] - lo) / cell).astype(np.int64)
    nx, ny = i1[:, 0] - i0[:, 0] + 1, i1[:, 1] - i0[:, 1] + 1
    count = nx * ny
    owner = np.repeat(np.arange(len(box)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    key = (i0[owner, 0] + k % nx[owner]) * 4_000_003 + (i0[owner, 1] + k // nx[owner])
    order = np.lexsort((owner, key))
    key, owner = key[order], owner[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    sizes = np.diff(np.r_[starts, len(key)])
    pi, pj = [], []
    for size in np.unique(sizes[sizes > 1]):
        grp = starts[sizes == size]
        idx = owner[grp[:, None] + np.arange(size)[None, :]]
        u, v = np.triu_indices(int(size), 1)
        pi.append(idx[:, u].ravel())
        pj.append(idx[:, v].ravel())
    if not pi:
        return empty, empty
    i, j = np.concatenate(pi), np.concatenate(pj)
    pair = np.unique(np.minimum(i, j) * len(box) + np.maximum(i, j))
    i, j = pair // len(box), pair % len(box)
    keep = (
        (box[i, 0] <= box[j, 2])
        & (box[j, 0] <= box[i, 2])
        & (box[i, 1] <= box[j, 3])
        & (box[j, 1] <= box[i, 3])
    )
    return i[keep], j[keep]


def _segment_crossings(
    p: Floats, segs: npt.NDArray[np.int64], tol: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], Floats, Floats]:
    """Proper plan crossings between segments: (seg i, seg j, t on i, t on j)."""
    empty = np.zeros(0, np.int64)
    if len(segs) < 2:
        return empty, empty, np.zeros(0), np.zeros(0)
    a, b = p[segs[:, 0], :2], p[segs[:, 1], :2]
    i, j = _box_pairs(np.column_stack([np.minimum(a, b) - tol, np.maximum(a, b) + tol]))
    if len(i) == 0:
        return empty, empty, np.zeros(0), np.zeros(0)
    shared = (
        (segs[i, 0] == segs[j, 0])
        | (segs[i, 0] == segs[j, 1])
        | (segs[i, 1] == segs[j, 0])
        | (segs[i, 1] == segs[j, 1])
    )
    i, j = i[~shared], j[~shared]
    r, s = b[i] - a[i], b[j] - a[j]
    q = a[j] - a[i]
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    ok = np.abs(den) > 1e-15
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / den
        u = (q[:, 0] * r[:, 1] - q[:, 1] * r[:, 0]) / den
    eps_t = tol / np.maximum(np.linalg.norm(r, axis=1), 1e-300)
    eps_u = tol / np.maximum(np.linalg.norm(s, axis=1), 1e-300)
    hit = ok & (t > eps_t) & (t < 1 - eps_t) & (u > eps_u) & (u < 1 - eps_u)
    return i[hit], j[hit], t[hit], u[hit]


def _edges_of(simplices: npt.NDArray[np.int64], n: int) -> npt.NDArray[np.int64]:
    e = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e[:, 0] * n + e[:, 1])


def _vertices_on(
    p: Floats, segs: npt.NDArray[np.int64], fixed: npt.NDArray[np.bool_], tol: float
) -> tuple[set[int], dict[int, list[tuple[float, int]]]]:
    """Vertices within ``tol`` of a segment's interior: (spot points, fixed per segment)."""
    a, b = p[segs[:, 0], :2], p[segs[:, 1], :2]
    tree = cKDTree(p[:, :2])
    mid, half = 0.5 * (a + b), 0.5 * np.linalg.norm(b - a, axis=1)
    spots: set[int] = set()
    splits: dict[int, list[tuple[float, int]]] = {}
    for k, cand in enumerate(tree.query_ball_point(mid, half + tol)):
        idx = np.array([c for c in cand if c not in (segs[k, 0], segs[k, 1])], np.int64)
        if len(idx) == 0:
            continue
        d = b[k] - a[k]
        t = ((p[idx, :2] - a[k]) @ d) / max(float(d @ d), 1e-300)
        dist = np.linalg.norm(p[idx, :2] - (a[k] + t[:, None] * d), axis=1)
        hit = (dist <= tol) & (t > 0) & (t < 1)
        for m in np.flatnonzero(hit):
            if fixed[idx[m]]:
                splits.setdefault(k, []).append((float(t[m]), int(idx[m])))
            else:
                spots.add(int(idx[m]))
    return spots, splits


def _conform(  # noqa: PLR0915 - one loop, several blocking cases
    p: Floats,
    segs: npt.NDArray[np.int64],
    fixed: npt.NDArray[np.bool_],
    tol: float,
    unhonoured: list[int] | None = None,
) -> tuple[Floats, npt.NDArray[np.int64], int, int]:
    """Split segments until each is a Delaunay edge; drop spot points lying on them.

    Returns (points, segments, vertices inserted, spot points dropped). With
    ``unhonoured`` (a list), pieces no longer than 4 x ``tol`` that stay blocked
    by breakline vertices (breaklines meeting at a very small angle) are left
    out instead of failing; their count is appended to the list.
    """
    inserted = dropped = 0
    for _ in range(64):
        tri = Delaunay(p[:, :2] - _plan_origin(p))
        n = len(p)
        s = np.sort(segs, axis=1)
        present = np.isin(s[:, 0] * n + s[:, 1], _edges_of(np.asarray(tri.simplices), n))
        if present.all():
            return p, segs, inserted, dropped
        missing = segs[~present]
        # A vertex within ``tol`` of a missing segment blocks it: a spot point is
        # dropped (the breakline wins), a breakline vertex splits the segment.
        a, b = p[missing[:, 0], :2], p[missing[:, 1], :2]
        on, splits = _vertices_on(p, missing, fixed, tol)
        if on:
            keep = np.ones(len(p), bool)
            keep[list(on)] = False
            remap = np.cumsum(keep) - 1
            p, fixed, segs = p[keep], fixed[keep], remap[segs]
            dropped += len(on)
            continue
        if splits:
            rows = [segs[present]]
            done = set(splits)
            for k, cut in splits.items():
                chain = [int(missing[k, 0]), *[c[1] for c in sorted(cut)], int(missing[k, 1])]
                rows.append(np.array(list(itertools.pairwise(chain)), np.int64))
            rest = [k for k in range(len(missing)) if k not in done]
            rows.append(missing[rest])
            segs = np.vstack(rows)
            continue
        length = np.linalg.norm(b - a, axis=1)
        short = length <= 4.0 * tol
        if short.any():
            # Pieces this short are still blocked by a nearby vertex: spot points
            # inside their diametral circle are within ~tol of the breakline and
            # are dropped; a breakline vertex there means breaklines nearly touch.
            mid = 0.5 * (a[short] + b[short])
            near = cKDTree(p[:, :2]).query_ball_point(mid, 0.5 * length[short] + 1e-12)
            ends = missing[short]
            block = {c for k, cand in enumerate(near) for c in cand if c not in ends[k]}
            spots = {c for c in block if not fixed[c]}
            if not spots:
                if unhonoured is None:
                    break
                # Leave these few-tolerance pieces to the Delaunay triangulation.
                idx = np.flatnonzero(~present)[short]
                keep_seg = np.ones(len(segs), bool)
                keep_seg[idx] = False
                segs = segs[keep_seg]
                unhonoured.append(len(idx))
                continue
            keep = np.ones(len(p), bool)
            keep[list(spots)] = False
            remap = np.cumsum(keep) - 1
            p, fixed, segs = p[keep], fixed[keep], remap[segs]
            dropped += len(spots)
            continue
        mids = 0.5 * (p[missing[:, 0]] + p[missing[:, 1]])
        first = len(p)
        idx = np.arange(first, first + len(missing))
        p = np.vstack([p, mids])
        fixed = np.r_[fixed, np.ones(len(missing), bool)]
        segs = np.vstack(
            [
                segs[present],
                np.column_stack([missing[:, 0], idx]),
                np.column_stack([idx, missing[:, 1]]),
            ]
        )
        inserted += len(missing)
    raise SurfaceError(
        "breaklines could not be honoured; check for breaklines that touch or nearly "
        "cross each other within the merge tolerance"
    )


def _merge(
    p: Floats, segs: npt.NDArray[np.int64], tol: float, z_tol: float
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64], int]:
    """Plan clusters within ``tol``: keep the lowest index (breakline vertices first).

    Returns (kept mask, segments renumbered to kept points, elevation conflicts).
    """
    rep = np.arange(len(p))

    def root(k: int) -> int:
        while rep[k] != k:
            k = int(rep[k])
        return k

    conflicts = 0
    for i, j in sorted(map(tuple, cKDTree(p[:, :2]).query_pairs(tol))):
        ri, rj = root(int(i)), root(int(j))
        if ri != rj:
            rep[max(ri, rj)] = min(ri, rj)
            conflicts += int(abs(p[i, 2] - p[j, 2]) > z_tol)
    rep = np.array([root(k) for k in range(len(p))], np.int64)
    kept = rep == np.arange(len(p))
    remap = np.cumsum(kept) - 1
    if len(segs):
        segs = remap[rep[segs]]
        segs = segs[segs[:, 0] != segs[:, 1]]
        segs = np.unique(np.sort(segs, axis=1), axis=0)
    return kept, segs, conflicts


def _split_crossings(
    p: Floats,
    segs: npt.NDArray[np.int64],
    tol: float,
    z_tol: float,
    authority: npt.NDArray[np.bool_] | None = None,
) -> tuple[Floats, npt.NDArray[np.int64], int]:
    """Split crossing segments at the crossing.

    Without ``authority`` both elevations must agree within ``z_tol``; with it,
    a crossing of an authoritative and another segment takes the elevation of
    the authoritative one (surface edges against a trimming boundary).
    """
    i, j, t, u = _segment_crossings(p, segs, tol)
    if len(i) == 0:
        return p, segs, 0
    zi = p[segs[i, 0], 2] + t * (p[segs[i, 1], 2] - p[segs[i, 0], 2])
    zj = p[segs[j, 0], 2] + u * (p[segs[j, 1], 2] - p[segs[j, 0], 2])
    if authority is None:
        bad = np.abs(zi - zj) > z_tol
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            a0 = p[segs[i[k], 0], :2]
            xy = a0 + t[k] * (p[segs[i[k], 1], :2] - a0)
            raise SurfaceError(
                f"{int(bad.sum())} breakline crossing(s) with different elevations, "
                f"e.g. at x={xy[0]:.3f} y={xy[1]:.3f}: z {zi[k]:.3f} vs {zj[k]:.3f}"
            )
        z = 0.5 * (zi + zj)
    else:
        z = np.where(authority[j] & ~authority[i], zj, zi)
    a0 = p[segs[i, 0], :2]
    xy = a0 + t[:, None] * (p[segs[i, 1], :2] - a0)
    new_pts = np.column_stack([xy, z])
    idx = len(p) + np.arange(len(i))
    cuts: dict[int, list[tuple[float, int]]] = {}
    for k in range(len(i)):
        cuts.setdefault(int(i[k]), []).append((float(t[k]), int(idx[k])))
        cuts.setdefault(int(j[k]), []).append((float(u[k]), int(idx[k])))
    rows = [segs[k] for k in range(len(segs)) if k not in cuts]
    for k, cut in cuts.items():
        chain = [int(segs[k, 0]), *[c[1] for c in sorted(cut)], int(segs[k, 1])]
        rows += [np.array(pair) for pair in itertools.pairwise(chain)]
    return np.vstack([p, new_pts]), np.array(rows, np.int64).reshape(-1, 2), len(i)


def _tin_edges(p: Floats) -> npt.NDArray[np.int64]:
    n = len(p)
    keys = _edges_of(np.asarray(Delaunay(p[:, :2] - _plan_origin(p)).simplices, np.int64), n)
    return np.column_stack([keys // n, keys % n]).astype(np.int64)


def _insert_rings(p: Floats, rings: list[Floats], tol: float) -> tuple[Floats, int]:
    """Make ring edges triangle edges without changing the surface shape.

    Every current triangle edge is kept as a constraint; ring edges are split
    where they cross triangle edges (elevation from the triangle edge), so each
    piece lies inside one original triangle and stays exact when its new
    vertices interpolate linearly.
    """
    base = Surface(p)
    tin = _tin_edges(p)
    ring_segs = []
    extra: list[Floats] = []
    tree = cKDTree(p[:, :2])
    for ring in rings:
        z, _ = base.elevation(ring)
        near = tree.query(ring, distance_upper_bound=tol)[1]
        ids = []
        for k in range(len(ring)):
            if near[k] < len(p):
                ids.append(int(near[k]))
            else:
                ids.append(len(p) + len(extra))
                extra.append(np.array([ring[k, 0], ring[k, 1], z[k]]))
        ring_segs += [[ids[k], ids[(k + 1) % len(ids)]] for k in range(len(ids))]
    n_before = len(p)
    if extra:
        p = np.vstack([p, np.array(extra)])
    rsegs = np.array(ring_segs, np.int64)
    rsegs = rsegs[rsegs[:, 0] != rsegs[:, 1]]
    segs = np.vstack([tin, rsegs])
    authority = np.r_[np.ones(len(tin), bool), np.zeros(len(rsegs), bool)]
    p, segs, _ = _split_crossings(p, segs, tol, 0.0, authority)
    p, _, _, _ = _conform(p, segs, np.ones(len(p), bool), tol)
    return p, len(p) - n_before


def _clean_rings(rings: list[Floats] | tuple[Floats, ...]) -> list[Floats]:
    out = []
    for ring in rings:
        r = np.asarray(ring, np.float64)[:, :2]
        if len(r) > 1 and np.allclose(r[0], r[-1], rtol=0.0, atol=1e-9):
            r = r[:-1]
        if len(r) < 3:
            raise SurfaceError("a boundary ring needs at least three points")
        out.append(r)
    return out


def build_surface(  # noqa: PLR0913 - surface inputs and the user's tolerances
    points: Floats | None = None,
    breaklines: list[Floats] | tuple[Floats, ...] = (),
    boundary: list[Floats] | tuple[Floats, ...] = (),
    *,
    max_edge: float | None = None,
    tolerance: float = 0.001,
    z_tolerance: float = 0.001,
    tolerate_small_angles: bool = False,
) -> tuple[Surface, SurfaceReport]:
    """Triangulated surface honouring breaklines, a trimming boundary and holes.

    ``points``: spot heights (n x 3). ``breaklines``: polylines (k x 3); each
    segment becomes a chain of triangle edges. ``boundary``: closed rings
    (x, y[, z]) under the even-odd rule (a ring inside another is a hole);
    triangles outside are removed and the ring edges become triangle edges.
    ``max_edge``: longest plan edge kept (m). ``tolerance`` (m, plan) merges
    coincident vertices; breakline vertices win over spot points.
    ``z_tolerance`` (m) decides whether crossing breaklines agree in elevation.
    Every limit is the caller's choice (data precision, survey practice).
    ``tolerate_small_angles``: breakline pieces no longer than 4 x ``tolerance``
    that cannot be made triangle edges (breaklines meeting at a tiny angle) are
    left to the triangulation instead of failing (counted in the report).
    """
    if not (tolerance > 0 and z_tolerance >= 0):
        raise SurfaceError("tolerances must be positive")
    spot = np.zeros((0, 3)) if points is None else np.asarray(points, np.float64).reshape(-1, 3)
    lines = [np.asarray(b, np.float64).reshape(-1, 3) for b in breaklines]
    lines = [b for b in lines if len(b) >= 2]
    if not all(np.isfinite(a).all() for a in [spot, *lines]):
        raise SurfaceError("surface input must be finite")
    p = np.vstack([*lines, spot])
    n_line = sum(len(b) for b in lines)
    starts = np.cumsum([0] + [len(b) for b in lines])
    seg_list = [
        np.column_stack([np.arange(s0, s1 - 1), np.arange(s0 + 1, s1)])
        for s0, s1 in itertools.pairwise(starts)
    ]
    segs = np.vstack(seg_list).astype(np.int64) if seg_list else np.zeros((0, 2), np.int64)
    kept, segs, conflicts = (
        _merge(p, segs, tolerance, z_tolerance) if len(p) else (np.ones(0, bool), segs, 0)
    )
    fixed = (np.arange(len(p)) < n_line)[kept]
    p = p[kept]
    crossings = 0
    if len(segs):
        p, segs, crossings = _split_crossings(p, segs, tolerance, z_tolerance)
        fixed = np.r_[fixed, np.ones(len(p) - len(fixed), bool)]
    if len(p) < 3:
        raise SurfaceError("a surface needs at least three distinct x, y points")
    skipped: list[int] | None = [] if tolerate_small_angles else None
    p, segs, inserted, on_lines = _conform(p, segs, fixed, tolerance, skipped)
    rings = _clean_rings(boundary)
    if rings:
        p, more = _insert_rings(p, rings, tolerance)
        inserted += more
    surface = Surface(p, tuple(rings), max_edge)
    report = SurfaceReport(
        input_points=len(spot),
        breakline_vertices=n_line,
        duplicates_merged=int((~kept).sum()),
        elevation_conflicts=conflicts,
        points_on_breaklines=on_lines,
        crossings_split=crossings,
        inserted=inserted,
        triangles=len(surface.triangles()),
        short_pieces_left=sum(skipped or []),
    )
    return surface, report


def clip_surface(
    surface: Surface, rings: list[Floats] | tuple[Floats, ...], *, tolerance: float = 0.001
) -> Surface:
    """The part of ``surface`` inside ``rings`` (even-odd), with exact clipped edges.

    The surface shape is unchanged inside: every existing triangle edge is
    kept and new vertices on the rings take the surface elevation. A longest
    edge limit is kept. A surface that already has a boundary is refused
    (rebuild it with the combined boundary instead).
    """
    if surface.boundary:
        raise SurfaceError(
            "the surface already has a boundary; rebuild it with the combined boundary"
        )
    new = _clean_rings(rings)
    p, _ = _insert_rings(surface.points, new, tolerance)
    clipped = Surface(p, tuple(new), surface.max_edge)
    if len(clipped.triangles()) == 0:
        raise SurfaceError("the clipping polygon does not overlap the surface")
    return clipped


@dataclass(frozen=True)
class DxfSurfaceLayers:
    """Which DXF layers feed a surface (``None``: see ``read_dxf_surface``)."""

    points: list[str] | None = None
    breaklines: list[str] | None = None
    boundary: str | None = None


def read_dxf_surface(
    path: Path,
    layers: DxfSurfaceLayers | None = None,
    *,
    max_edge: float | None = None,
    tolerance: float = 0.001,
    z_tolerance: float = 0.001,
) -> tuple[Surface, SurfaceReport]:
    """Surface from an ASCII DXF.

    Spot heights: ``POINT`` entities on ``layers.points`` (``None``: every
    layer not used for breaklines or the boundary). Triangles (``3DFACE``,
    polyface meshes) on those layers keep their edges as breaklines, so an
    existing TIN is reproduced exactly. Breaklines: polylines and lines on
    ``layers.breaklines`` (``None``: every polyline/line not on the boundary
    layer; contour ``LWPOLYLINE`` entities carry their elevation). Boundary:
    every closed polyline on ``layers.boundary`` (even-odd: inner rings are
    holes).
    """
    layers = layers or DxfSurfaceLayers()
    point_layers, breakline_layers = layers.points, layers.breaklines
    boundary_layer = layers.boundary
    ents = read_dxf(path)
    special = {boundary_layer} if boundary_layer else set()

    def is_point_layer(layer: str) -> bool:
        if point_layers is not None:
            return layer in point_layers
        return layer not in special and (breakline_layers is None or layer not in breakline_layers)

    def is_line_layer(layer: str) -> bool:
        if breakline_layers is not None:
            return layer in breakline_layers
        return layer not in special

    spot = [xyz for layer, xyz in ents.points if is_point_layer(layer)]
    lines = [np.vstack([f, f[:1]]) for layer, f in ents.faces if is_point_layer(layer)]
    lines += [
        pl.points if not pl.closed else np.vstack([pl.points, pl.points[:1]])
        for pl in ents.polylines
        if is_line_layer(pl.layer)
    ]
    rings: list[Floats] = []
    if boundary_layer:
        rings = [
            pl.points
            for pl in ents.polylines
            if pl.layer == boundary_layer
            and (pl.closed or np.allclose(pl.points[0], pl.points[-1], rtol=0.0, atol=1e-9))
        ]
        if not rings:
            raise SurfaceError(f"{path.name}: no closed polyline on layer {boundary_layer!r}")
    if not spot and not lines:
        raise SurfaceError(f"{path.name}: no points, faces or polylines on the chosen layers")
    return build_surface(
        np.array(spot) if spot else None,
        lines,
        rings,
        max_edge=max_edge,
        tolerance=tolerance,
        z_tolerance=z_tolerance,
    )
