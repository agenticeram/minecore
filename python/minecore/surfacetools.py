"""Surface (DTM) analysis tools: contours, slope and aspect, profiles,
surface intersection lines and vertical offset (catalogue OC10, TP02 part;
gap list E-CD-02, E-OL-05).

Every result is exact for the triangulated surface, never resampled:

* ``contours``: on each triangle the surface is a plane, so a contour level
  crosses it in one straight segment between two edge crossings. Segments are
  chained across shared edges into polylines; a polyline that returns to its
  start is closed. Every segment is oriented with higher ground on its left,
  so a closed contour around a hill runs counter-clockwise and one around a
  pit bottom clockwise. A vertex lying exactly on a level counts as above it,
  so neighbouring triangles always agree on the crossing.
* ``slope_map``: slope (degrees from horizontal) and aspect (direction of
  steepest descent, degrees clockwise from grid north) of every triangle,
  with plan and true (sloping) area. Class limits are the user's.
* ``profile``: elevations of one or more surfaces along a plan polyline at
  every point where the line crosses a triangle edge (plus the polyline's own
  vertices); between those points each surface is linear, so the profile is
  the exact surface trace.
* ``intersection_lines``: where two surfaces have the same elevation (a pit
  design against topography gives the daylight line). The plan overlay of
  both triangulations has pieces on which the elevation difference is linear;
  its zero line crosses a piece in one segment. Segments are chained with
  ``a`` above ``b`` on the left.
* ``offset_surface``: the surface shifted vertically by ``dz``.

Plan coordinates are shifted to a local origin before any geometry is formed,
so UTM-sized coordinates lose no precision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from minecore._core import MineCoreError
from minecore.surfaces import Surface
from minecore.survey.volumes import _edges, _intersect, _tris

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]

SECTORS_8 = ("K", "KD", "D", "GD", "G", "GB", "B", "KB")  # N, NE, E, SE, S, SW, W, NW


class SurfaceToolError(MineCoreError):
    """Invalid surface tool input."""


# -- chaining ------------------------------------------------------------------


def _chain(start: Ints, end: Ints) -> list[tuple[list[int], bool]]:
    """Chain directed segments (node ``start[k]`` -> ``end[k]``) into paths.

    Returns (node list, closed) per path. Open paths start at nodes with no
    incoming segment; what remains are loops. Nodes met by more than one
    incoming and outgoing segment (a saddle exactly on a level) split paths.
    """
    keep = start != end
    start, end = start[keep], end[keep]
    out: dict[int, list[int]] = {}
    indeg: dict[int, int] = {}
    for k, (s, e) in enumerate(zip(start.tolist(), end.tolist(), strict=True)):
        out.setdefault(s, []).append(k)
        indeg[e] = indeg.get(e, 0) + 1
    used = np.zeros(len(start), bool)

    def walk(k: int) -> list[int]:
        nodes = [int(start[k])]
        while True:
            used[k] = True
            node = int(end[k])
            nodes.append(node)
            nxt = [j for j in out.get(node, ()) if not used[j]]
            if not nxt:
                return nodes
            k = nxt[0]

    paths: list[tuple[list[int], bool]] = []
    for k in range(len(start)):
        if not used[k] and indeg.get(int(start[k]), 0) == 0:
            paths.append((walk(k), False))
    for k in range(len(start)):
        if not used[k]:
            nodes = walk(k)
            closed = nodes[0] == nodes[-1]
            paths.append((nodes[:-1] if closed else nodes, closed))
    return paths


def _plane_gradient(xy: Floats, z: Floats) -> Floats:
    """Plan gradient (dz/dx, dz/dy) of triangles (m x 3 x 2 corners, m x 3 z)."""
    u, v = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dz1, dz2 = z[:, 1] - z[:, 0], z[:, 2] - z[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (dz1 * v[:, 1] - dz2 * u[:, 1]) / det
        b = (dz2 * u[:, 0] - dz1 * v[:, 0]) / det
    return np.column_stack([a, b])


# -- contours ------------------------------------------------------------------


@dataclass(frozen=True)
class Contour:
    level: float
    points: Floats  # n x 3, z = level
    closed: bool
    index: bool  # major (index) contour, labelled

    @property
    def length(self) -> float:
        p = self.points[:, :2]
        if self.closed:
            p = np.vstack([p, p[:1]])
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def contour_levels(zmin: float, zmax: float, interval: float, base: float = 0.0) -> Floats:
    """Levels ``base + k * interval`` within [zmin, zmax]."""
    if not (math.isfinite(interval) and interval > 0):
        raise SurfaceToolError("the contour interval must be > 0 m")
    k0 = math.ceil((zmin - base) / interval - 1e-9)
    k1 = math.floor((zmax - base) / interval + 1e-9)
    return base + interval * np.arange(k0, k1 + 1, dtype=np.float64)


def contours(  # noqa: PLR0915 - one vectorised pass per level
    surface: Surface,
    interval: float | None = None,
    *,
    base: float = 0.0,
    index_every: int = 5,
    levels: Sequence[float] | None = None,
) -> list[Contour]:
    """Contour lines at ``levels`` or every ``interval`` m from ``base``.

    ``index_every``: every n-th level (counted from ``base``) is an index
    contour; 0 marks none. All values are the user's choice.
    """
    tri = surface.triangles()
    if len(tri) == 0:
        raise SurfaceToolError("the surface has no triangles")
    pts = surface.points
    origin = pts[:, :2].min(axis=0)
    local = pts[:, :2] - origin
    z = pts[:, 2]
    used = np.unique(tri)
    if levels is None:
        if interval is None:
            raise SurfaceToolError("give a contour interval or a list of levels")
        lv = contour_levels(float(z[used].min()), float(z[used].max()), interval, base)
    else:
        lv = np.unique(np.asarray(levels, np.float64))
        if not np.isfinite(lv).all():
            raise SurfaceToolError("contour levels must be finite")
    if index_every < 0:
        raise SurfaceToolError("the index contour step must be >= 0")
    n = len(pts)
    grad = _plane_gradient(local[tri], z[tri])
    zt = z[tri]
    lo_t, hi_t = zt.min(axis=1), zt.max(axis=1)
    out: list[Contour] = []
    for level in lv.tolist():
        cand = np.flatnonzero((lo_t < level) & (hi_t >= level))
        if len(cand) == 0:
            continue
        t3 = tri[cand]
        above = z[t3] >= level
        mixed = above.any(axis=1) & ~above.all(axis=1)
        t3, g = t3[mixed], grad[cand[mixed]]
        above = above[mixed]
        keys = np.zeros((len(t3), 3), np.int64)
        xy = np.zeros((len(t3), 3, 2))
        cross = np.zeros((len(t3), 3), bool)
        for e in range(3):
            i0, i1 = t3[:, e], t3[:, (e + 1) % 3]
            i, j = np.minimum(i0, i1), np.maximum(i0, i1)
            cross[:, e] = above[:, e] != above[:, (e + 1) % 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(cross[:, e], (level - z[i]) / (z[j] - z[i]), 0.0)
            xy[:, e] = local[i] + t[:, None] * (local[j] - local[i])
            key = n + i * n + j
            keys[:, e] = np.where(t <= 0.0, i, np.where(t >= 1.0, j, key))
        # Exactly two crossing edges per mixed triangle.
        first = np.argmax(cross, axis=1)
        second = 2 - np.argmax(cross[:, ::-1], axis=1)
        rows = np.arange(len(t3))
        ka, kb = keys[rows, first], keys[rows, second]
        pa, pb = xy[rows, first], xy[rows, second]
        s = pb - pa
        flip = s[:, 0] * g[:, 1] - s[:, 1] * g[:, 0] < 0
        ka, kb = np.where(flip, kb, ka), np.where(flip, ka, kb)
        pa2 = np.where(flip[:, None], pb, pa)
        pb2 = np.where(flip[:, None], pa, pb)
        node_keys, inverse = np.unique(np.r_[ka, kb], return_inverse=True)
        coords = np.zeros((len(node_keys), 2))
        coords[inverse] = np.r_[pa2, pb2]
        m = len(ka)
        k_step = round((level - base) / interval) if interval else None
        is_index = bool(index_every and k_step is not None and k_step % index_every == 0)
        for nodes, closed in _chain(inverse[:m], inverse[m:]):
            p = coords[nodes] + origin
            if len(p) < 2:
                continue
            out.append(
                Contour(level, np.column_stack([p, np.full(len(p), level)]), closed, is_index)
            )
    return out


def label_points(contour: Contour, spacing: float) -> list[tuple[Floats, float]]:
    """Label positions every ``spacing`` m along a contour: (x, y, z), angle in degrees.

    The angle is the line direction in plan, turned to read left to right.
    The first label sits half a spacing from the start.
    """
    if not (math.isfinite(spacing) and spacing > 0):
        raise SurfaceToolError("the label spacing must be > 0 m")
    p = contour.points
    if contour.closed:
        p = np.vstack([p, p[:1]])
    seg = np.diff(p[:, :2], axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    total = float(cum[-1])
    labels = []
    at = 0.5 * spacing if total >= spacing else 0.5 * total
    while at <= total and total > 0:
        k = min(int(np.searchsorted(cum, at, "right")) - 1, len(seg) - 1)
        f = (at - cum[k]) / seg_len[k] if seg_len[k] > 0 else 0.0
        xyz = p[k] + f * (p[k + 1] - p[k])
        angle = math.degrees(math.atan2(seg[k, 1], seg[k, 0]))
        if angle > 90.0:
            angle -= 180.0
        elif angle <= -90.0:
            angle += 180.0
        labels.append((xyz, angle))
        at += spacing
    return labels


# -- slope and aspect ------------------------------------------------------------


@dataclass(frozen=True)
class SlopeMap:
    corners: Floats  # m x 3 x 3 triangle corners (world coordinates)
    slope: Floats  # degrees from horizontal
    aspect: Floats  # degrees clockwise from grid north, downslope; NaN where flat
    plan_area: Floats  # m2
    true_area: Floats  # m2 on the sloping surface


def _slope_aspect(gradient: Floats) -> tuple[Floats, Floats]:
    a, b = gradient[:, 0], gradient[:, 1]
    g = np.hypot(a, b)
    slope = np.degrees(np.arctan(g))
    aspect = np.degrees(np.arctan2(-a, -b)) % 360.0
    aspect = np.where(g > 1e-12, aspect, np.nan)
    return slope, aspect


def slope_map(surface: Surface) -> SlopeMap:
    tri = surface.triangles()
    p = surface.points[tri]
    origin = surface.points[:, :2].min(axis=0)
    xy = p[:, :, :2] - origin
    u, v = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    plan = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    ok = plan > 1e-12
    p, xy, plan = p[ok], xy[ok], plan[ok]
    slope, aspect = _slope_aspect(_plane_gradient(xy, p[:, :, 2]))
    true = plan / np.cos(np.radians(slope))
    return SlopeMap(p, slope, aspect, plan, true)


@dataclass(frozen=True)
class ClassRow:
    label: str
    low: float
    high: float
    plan_area: float
    true_area: float
    share: float  # of the surface plan area


def slope_classes(sm: SlopeMap, limits: Sequence[float]) -> list[ClassRow]:
    """Plan and true area per slope class [limit_k, limit_k+1); the last includes 90°.

    ``limits``: increasing class limits in degrees within [0, 90] (user's).
    Area outside the limits is reported in a row labelled "dışında".
    """
    lim = np.asarray(limits, np.float64)
    if len(lim) < 2 or not np.all(np.diff(lim) > 0) or lim[0] < 0 or lim[-1] > 90:
        raise SurfaceToolError("slope class limits must increase within 0-90 degrees")
    total = float(sm.plan_area.sum())
    rows = []
    counted = np.zeros(len(sm.slope), bool)
    for k in range(len(lim) - 1):
        lo, hi = float(lim[k]), float(lim[k + 1])
        last = k == len(lim) - 2
        m = (sm.slope >= lo) & ((sm.slope <= hi) if last else (sm.slope < hi))
        counted |= m
        rows.append(
            ClassRow(
                f"{lo:g}-{hi:g}°",
                lo,
                hi,
                float(sm.plan_area[m].sum()),
                float(sm.true_area[m].sum()),
                float(sm.plan_area[m].sum()) / total if total else 0.0,
            )
        )
    if (~counted).any():
        a = float(sm.plan_area[~counted].sum())
        rows.append(
            ClassRow(
                "dışında",
                math.nan,
                math.nan,
                a,
                float(sm.true_area[~counted].sum()),
                a / total if total else 0.0,
            )
        )
    return rows


def aspect_sectors(sm: SlopeMap, sectors: int = 8) -> list[ClassRow]:
    """Plan area by downslope direction in equal sectors centred on north.

    Flat triangles (no aspect) get their own row labelled "düz".
    """
    if sectors < 1 or 360 % sectors:
        raise SurfaceToolError("the number of aspect sectors must divide 360")
    width = 360.0 / sectors
    total = float(sm.plan_area.sum())
    names = SECTORS_8 if sectors == 8 else tuple(f"{k * width:g}°" for k in range(sectors))
    flat = np.isnan(sm.aspect)
    idx = np.floor(((np.nan_to_num(sm.aspect) + width / 2) % 360.0) / width).astype(int)
    rows = []
    for k in range(sectors):
        m = ~flat & (idx == k)
        a = float(sm.plan_area[m].sum())
        lo = (k * width - width / 2) % 360.0
        rows.append(
            ClassRow(
                names[k],
                lo,
                (lo + width) % 360.0,
                a,
                float(sm.true_area[m].sum()),
                a / total if total else 0.0,
            )
        )
    if flat.any():
        a = float(sm.plan_area[flat].sum())
        rows.append(
            ClassRow(
                "düz",
                math.nan,
                math.nan,
                a,
                float(sm.true_area[flat].sum()),
                a / total if total else 0.0,
            )
        )
    return rows


@dataclass(frozen=True)
class SurfaceGrid:
    """Values at cell centres from the triangle under each centre (NaN outside)."""

    x: Floats  # nx cell centres
    y: Floats  # ny cell centres
    cell: float
    z: Floats  # ny x nx (row 0 = lowest y)
    slope: Floats
    aspect: Floats


def surface_grid(surface: Surface, cell: float) -> SurfaceGrid:
    """Elevation, slope and aspect on a regular grid of ``cell`` m (user's)."""
    if not (math.isfinite(cell) and cell > 0):
        raise SurfaceToolError("the grid cell must be > 0 m")
    tri = surface.triangles()
    used = surface.points[np.unique(tri)]
    lo, hi = used[:, :2].min(axis=0), used[:, :2].max(axis=0)
    nx = max(1, math.ceil((hi[0] - lo[0]) / cell))
    ny = max(1, math.ceil((hi[1] - lo[1]) / cell))
    if nx * ny > 25_000_000:
        raise SurfaceToolError(f"{nx} x {ny} cells is too many; choose a larger cell")
    x = lo[0] + cell * (np.arange(nx) + 0.5)
    y = lo[1] + cell * (np.arange(ny) + 0.5)
    gx, gy = np.meshgrid(x, y)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    origin = surface.points[:, :2].min(axis=0)
    delaunay = surface._delaunay()
    simplex = np.asarray(delaunay.find_simplex(xy - surface._origin()), np.int64)
    inside = surface.contains(xy)
    z = np.full(len(xy), np.nan)
    slope = np.full(len(xy), np.nan)
    aspect = np.full(len(xy), np.nan)
    if inside.any():
        z[inside] = surface.elevation(xy[inside])[0]
        simp = np.asarray(delaunay.simplices, np.int64)[simplex[inside]]
        p = surface.points[simp]
        s, a = _slope_aspect(_plane_gradient(p[:, :, :2] - origin, p[:, :, 2]))
        slope[inside], aspect[inside] = s, a
    return SurfaceGrid(
        x, y, float(cell), z.reshape(ny, nx), slope.reshape(ny, nx), aspect.reshape(ny, nx)
    )


def write_ascii_grid(
    path: Path, grid: SurfaceGrid, values: Floats, nodata: float = -9999.0
) -> None:
    """ESRI ASCII grid (open, documented text raster read by every GIS)."""
    ny, nx = values.shape
    lines = [
        f"ncols {nx}",
        f"nrows {ny}",
        f"xllcorner {grid.x[0] - grid.cell / 2:.6f}",
        f"yllcorner {grid.y[0] - grid.cell / 2:.6f}",
        f"cellsize {grid.cell:.6f}",
        f"NODATA_value {nodata:g}",
    ]
    v = np.where(np.isfinite(values), values, nodata)
    lines += [" ".join(f"{c:.4f}" for c in row) for row in v[::-1]]  # north row first
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# -- profiles ----------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    chainage: Floats  # m along the plan polyline
    xy: Floats  # k x 2
    z: dict[str, Floats]  # per surface, NaN where the surface is absent
    vertex_chainage: Floats  # chainage of the polyline's own vertices


def profile(surfaces: Mapping[str, Surface], path: Floats) -> Profile:
    """Exact traces of ``surfaces`` along the plan polyline ``path`` (n x 2 or 3)."""
    line = np.asarray(path, np.float64)[:, :2]
    if len(line) < 2 or not np.isfinite(line).all():
        raise SurfaceToolError("a profile needs a line of at least two points")
    if not surfaces:
        raise SurfaceToolError("choose at least one surface")
    edge_sets = [_edges(s) for s in surfaces.values()]
    seg_len = np.linalg.norm(np.diff(line, axis=0), axis=1)
    if not (seg_len > 0).any():
        raise SurfaceToolError("the profile line has no length")
    cum = np.r_[0.0, np.cumsum(seg_len)]
    stations = [cum]
    for k in range(len(line) - 1):
        if seg_len[k] == 0:
            continue
        p0, d = line[k], line[k + 1] - line[k]
        for e in edge_sets:
            a, s = e[:, 0], e[:, 1] - e[:, 0]
            q = a - p0
            den = d[0] * s[:, 1] - d[1] * s[:, 0]
            ok = np.abs(den) > 1e-15
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / den
                u = (q[:, 0] * d[1] - q[:, 1] * d[0]) / den
            hit = ok & (t > 0) & (t < 1) & (u >= 0) & (u <= 1)
            stations.append(cum[k] + t[hit] * seg_len[k])
    ch = np.unique(np.round(np.concatenate(stations), 9))
    seg = np.clip(np.searchsorted(cum, ch, "right") - 1, 0, len(line) - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(seg_len[seg] > 0, (ch - cum[seg]) / seg_len[seg], 0.0)
    xy = line[seg] + f[:, None] * (line[seg + 1] - line[seg])
    z: dict[str, Floats] = {}
    for name, surf in surfaces.items():
        values = surf.elevation(xy)[0]
        values[~surf.contains(xy)] = np.nan
        z[name] = values
    return Profile(ch, xy, z, cum)


def draw_profile(ax: Any, prof: Profile, exaggeration: float = 1.0, title: str = "") -> None:
    """Draw a profile on a matplotlib Axes (vertical exaggeration is the user's)."""
    if not (math.isfinite(exaggeration) and exaggeration > 0):
        raise SurfaceToolError("the vertical exaggeration must be > 0")
    for name, z in prof.z.items():
        ax.plot(prof.chainage, z, label=name, linewidth=1.2)
    for c in prof.vertex_chainage[1:-1]:
        ax.axvline(c, color="0.7", linewidth=0.6, linestyle=":")
    ax.set_aspect(exaggeration)
    ax.set_xlabel("Mesafe (m)")
    ax.set_ylabel("Kot (m)")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(loc="best", fontsize="small")
    ax.set_title(title or f"Profil (düşey abartma {exaggeration:g}x)")


def save_profile_plot(path: Path, prof: Profile, exaggeration: float = 1.0) -> None:
    """PNG, PDF or SVG by the file extension."""
    from matplotlib.figure import Figure  # noqa: PLC0415 - plotting only on request

    fig = Figure(figsize=(11.0, 4.5))
    draw_profile(fig.add_subplot(111), prof, exaggeration)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# -- intersection and offset ----------------------------------------------------------


@dataclass(frozen=True)
class IntersectionLine:
    points: Floats  # n x 3 on both surfaces
    closed: bool

    @property
    def length(self) -> float:
        p = self.points[:, :2]
        if self.closed:
            p = np.vstack([p, p[:1]])
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _z_range(corners: Floats, plane: Floats) -> Floats:
    z = (
        plane[:, 0, None] * corners[:, :, 0]
        + plane[:, 1, None] * corners[:, :, 1]
        + plane[:, 2, None]
    )
    return np.column_stack([z.min(axis=1), z.max(axis=1)])


def intersection_lines(  # noqa: PLR0915 - overlay, zero lines, joining
    a: Surface, b: Surface, *, tolerance: float = 0.001
) -> list[IntersectionLine]:
    """Lines where ``a`` and ``b`` have the same elevation, ``a`` above ``b`` on the left.

    ``tolerance`` (m, plan) joins segment ends computed on neighbouring pieces.
    """
    if not tolerance > 0:
        raise SurfaceToolError("the tolerance must be > 0 m")
    origin = a.points[:, :2].mean(axis=0)
    ta, tb = _tris(a, origin), _tris(b, origin)
    poly = np.pad(ta.corners, ((0, 0), (0, 1), (0, 0)))
    n = np.full(len(poly), 3, np.int64)
    poly, n, ia, ib = _intersect(
        poly,
        n,
        tb.corners,
        z_ranges=(_z_range(ta.corners, ta.plane), _z_range(tb.corners, tb.plane)),
    )
    if len(poly) == 0:
        return []
    d_plane = ta.plane[ia] - tb.plane[ib]
    k_max = poly.shape[1]
    d = (
        d_plane[:, 0, None] * poly[:, :, 0]
        + d_plane[:, 1, None] * poly[:, :, 1]
        + d_plane[:, 2, None]
    )
    valid = np.arange(k_max)[None, :] < n[:, None]
    above = d >= 0
    mixed = (above & valid).any(axis=1) & (~above & valid).any(axis=1)
    rows = np.flatnonzero(mixed)
    if len(rows) == 0:
        return []
    poly, n, d, above, plane_a = poly[rows], n[rows], d[rows], above[rows], ta.plane[ia[rows]]
    grad = d_plane[rows, :2]
    m = len(rows)
    found = np.zeros((m, 2, 2))
    count = np.zeros(m, np.int64)
    r = np.arange(m)
    for i in range(k_max):
        j = np.where(i + 1 < n, i + 1, 0)
        ok = (i < n) & (above[:, i] != above[r, j])
        di, dj = d[:, i], d[r, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ok, di / (di - dj), 0.0)
        pt = poly[:, i] + t[:, None] * (poly[r, j] - poly[:, i])
        sel = ok & (count < 2)
        found[r[sel], count[sel]] = pt[sel]
        count += sel
    good = count == 2
    pa, pb, grad, plane_a = found[good, 0], found[good, 1], grad[good], plane_a[good]
    s = pb - pa
    flip = s[:, 0] * grad[:, 1] - s[:, 1] * grad[:, 0] < 0
    pa, pb = np.where(flip[:, None], pb, pa), np.where(flip[:, None], pa, pb)
    za = plane_a[:, 0] * pa[:, 0] + plane_a[:, 1] * pa[:, 1] + plane_a[:, 2]
    zb = plane_a[:, 0] * pb[:, 0] + plane_a[:, 1] * pb[:, 1] + plane_a[:, 2]
    pts = np.vstack([np.column_stack([pa, za]), np.column_stack([pb, zb])])
    # Join ends that coincide within the tolerance (union-find on plan pairs).
    rep = np.arange(len(pts))

    def root(k: int) -> int:
        while rep[k] != k:
            rep[k] = rep[rep[k]]
            k = int(rep[k])
        return k

    for i, j in cKDTree(pts[:, :2]).query_pairs(tolerance):
        ri, rj = root(int(i)), root(int(j))
        if ri != rj:
            rep[max(ri, rj)] = min(ri, rj)
    node = np.array([root(k) for k in range(len(pts))], np.int64)
    half = len(pa)
    lines = []
    for nodes, closed in _chain(node[:half], node[half:]):
        p = pts[nodes]
        if len(p) >= 2:
            lines.append(IntersectionLine(p + np.array([origin[0], origin[1], 0.0]), closed))
    return lines


def offset_surface(surface: Surface, dz: float) -> Surface:
    """The surface moved vertically by ``dz`` m (boundary and edge limit kept)."""
    if not math.isfinite(dz):
        raise SurfaceToolError("the vertical offset must be finite")
    return Surface(surface.points + np.array([0.0, 0.0, dz]), surface.boundary, surface.max_edge)
