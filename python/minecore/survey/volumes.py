"""Volumes between surfaces and by cross-sections (catalogue OC07).

Three methods, each reporting cut (upper above lower), fill (lower above
upper), net and gross volume and the plan area used:

* ``triangle_volume`` (exact): the two triangulations are overlaid in plan by
  clipping every overlapping triangle pair (convex polygon clipping); on each
  piece both surfaces are planar, so the thickness is linear and its integral
  is area x thickness at the centroid. Pieces where the thickness changes
  sign are split along the zero line, so cut and fill are exact. A limit
  polygon (with holes, even-odd) is applied as a third overlay layer. A
  constant level can replace the lower surface.
* ``grid_volume``: thickness at cell centres x cell area on a user-chosen
  cell; converges to the exact value as the cell shrinks.
* sections: cut and fill areas on vertical sections perpendicular to an axis
  at a user-chosen interval (exact areas along each section line), summed by
  the average end area rule ``V = L (A1 + A2) / 2`` and the prismoidal rule
  ``V = L / 6 (A1 + 4 Am + A2)`` with ``Am`` measured on the mid section.

Coordinates are shifted to a local origin before any area is formed, so
UTM-sized coordinates lose no precision. Tonnes need a density given by the
user; nothing is assumed.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from minecore._core import MineCoreError
from minecore.surfaces import Surface, build_surface, inside_rings

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]


class VolumeError(MineCoreError):
    """Invalid volume calculation input."""


@dataclass(frozen=True)
class VolumeReport:
    method: str  # "triangle", "grid <cell> m", "sections <interval> m"
    cut: float  # m3, upper above lower
    fill: float  # m3, lower above upper
    area: float  # m2 of plan where both surfaces (and the limit) exist
    cut_area: float  # m2 of plan with cut
    fill_area: float  # m2 of plan with fill

    @property
    def net(self) -> float:
        return self.cut - self.fill

    @property
    def gross(self) -> float:
        return self.cut + self.fill

    def tonnes(self, density: float) -> tuple[float, float]:
        """(cut, fill) tonnes for a density in t/m3 given by the user."""
        if not (math.isfinite(density) and density > 0):
            raise VolumeError("density must be > 0 t/m3")
        return self.cut * density, self.fill * density


# -- triangle overlay ---------------------------------------------------------


@dataclass(frozen=True)
class _Tris:
    corners: Floats  # m x 3 x 2 (local plan coordinates, counter-clockwise)
    plane: Floats  # m x 3: z = a x + b y + c (local coordinates)


def _tris(surface: Surface, origin: Floats) -> _Tris:
    tri = surface.triangles()
    p = surface.points[tri]  # m x 3 x 3
    xy = p[:, :, :2] - origin
    z = p[:, :, 2]
    u, v = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    ok = np.abs(det) > 1e-12 * np.maximum(1.0, (u**2).sum(1) + (v**2).sum(1))
    xy, z, u, v, det = xy[ok], z[ok], u[ok], v[ok], det[ok]
    dz1, dz2 = z[:, 1] - z[:, 0], z[:, 2] - z[:, 0]
    a = (dz1 * v[:, 1] - dz2 * u[:, 1]) / det
    b = (dz2 * u[:, 0] - dz1 * v[:, 0]) / det
    c = z[:, 0] - a * xy[:, 0, 0] - b * xy[:, 0, 1]
    flip = det < 0
    xy[flip] = xy[flip][:, ::-1]
    return _Tris(xy, np.column_stack([a, b, c]))


def _candidate_pairs(box_a: Floats, box_b: Floats) -> tuple[Ints, Ints]:
    """Index pairs whose plan bounding boxes (xmin, ymin, xmax, ymax) overlap."""
    if len(box_a) == 0 or len(box_b) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ext = np.r_[box_a[:, 2:] - box_a[:, :2], box_b[:, 2:] - box_b[:, :2]]
    cell = float(np.median(ext.max(axis=1)))
    if not cell > 0:
        cell = 1.0
    lo = np.minimum(box_a[:, :2].min(0), box_b[:, :2].min(0))

    def cells(box: Floats) -> tuple[Ints, Ints]:
        i0 = np.floor((box[:, :2] - lo) / cell).astype(np.int64)
        i1 = np.floor((box[:, 2:] - lo) / cell).astype(np.int64)
        nx, ny = i1[:, 0] - i0[:, 0] + 1, i1[:, 1] - i0[:, 1] + 1
        count = nx * ny
        owner = np.repeat(np.arange(len(box)), count)
        k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        cx = i0[owner, 0] + k % nx[owner]
        cy = i0[owner, 1] + k // nx[owner]
        return owner, cx * 4_000_003 + cy

    oa, ca = cells(box_a)
    ob, cb = cells(box_b)
    order = np.argsort(cb, kind="stable")
    cb, ob = cb[order], ob[order]
    left = np.searchsorted(cb, ca, "left")
    right = np.searchsorted(cb, ca, "right")
    n = right - left
    ia = np.repeat(oa, n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ib = ob[np.repeat(left, n) + k]
    key = np.unique(ia * len(box_b) + ib)
    ia, ib = key // len(box_b), key % len(box_b)
    keep = (
        (box_a[ia, 0] <= box_b[ib, 2])
        & (box_b[ib, 0] <= box_a[ia, 2])
        & (box_a[ia, 1] <= box_b[ib, 3])
        & (box_b[ib, 1] <= box_a[ia, 3])
    )
    return ia[keep], ib[keep]


def _clip(poly: Floats, n: Ints, line: Floats) -> tuple[Floats, Ints]:
    """Clip convex polygons (m x K x 2, n vertices each) by a x + b y + c >= 0."""
    m, k_max, _ = poly.shape
    out = np.zeros((m, k_max + 1, 2))
    count = np.zeros(m, np.int64)
    rows = np.arange(m)
    d = line[:, 0, None] * poly[:, :, 0] + line[:, 1, None] * poly[:, :, 1] + line[:, 2, None]
    for i in range(k_max):
        valid = i < n
        j = np.where(i + 1 < n, i + 1, 0)
        cur, nxt = poly[:, i], poly[rows, j]
        dc, dn = d[:, i], d[rows, j]
        keep = valid & (dc >= 0)
        out[rows[keep], count[keep]] = cur[keep]
        count += keep
        cross = valid & ((dc >= 0) != (dn >= 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(cross, dc / (dc - dn), 0.0)
        pt = cur + t[:, None] * (nxt - cur)
        out[rows[cross], count[cross]] = pt[cross]
        count += cross
    return out, count


def _area_centroid(poly: Floats, n: Ints) -> tuple[Floats, Floats]:
    """Area and centroid of convex polygons (fan from the first vertex)."""
    m, k_max, _ = poly.shape
    area = np.zeros(m)
    moment = np.zeros((m, 2))
    p0 = poly[:, 0]
    for i in range(1, k_max - 1):
        valid = i + 1 < n
        u, v = poly[:, i] - p0, poly[:, i + 1] - p0
        a = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) * valid
        area += a
        moment += a[:, None] * (p0 + poly[:, i] + poly[:, i + 1]) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = np.where(area[:, None] > 0, moment / area[:, None], p0)
    return area, centroid


def _edge_lines(tri: Floats) -> list[Floats]:
    """Inside half-planes of counter-clockwise triangles, one array per edge."""
    lines = []
    for e in range(3):
        p, q = tri[:, e], tri[:, (e + 1) % 3]
        a, b = -(q[:, 1] - p[:, 1]), q[:, 0] - p[:, 0]
        lines.append(np.column_stack([a, b, -(a * p[:, 0] + b * p[:, 1])]))
    return lines


def _boxes(poly: Floats, n: Ints) -> Floats:
    k = np.arange(poly.shape[1])[None, :] < n[:, None]
    big = np.where(k[:, :, None], poly, np.nan)
    return np.column_stack([np.nanmin(big, axis=1), np.nanmax(big, axis=1)])


def _intersect(
    poly: Floats,
    n: Ints,
    tri: Floats,
    chunk: int = 200_000,
    z_ranges: tuple[Floats, Floats] | None = None,
) -> tuple[Floats, Ints, Ints, Ints]:
    """Pieces of convex polygons inside triangles: (polygons, counts, poly id, tri id).

    ``z_ranges``: optional (min, max) elevation per polygon and per triangle
    (each k x 2); pairs whose ranges do not overlap are skipped.
    """
    tb = np.column_stack([tri.min(axis=1), tri.max(axis=1)])
    ip, it = _candidate_pairs(_boxes(poly, n), tb)
    if z_ranges is not None:
        zp, zt = z_ranges
        overlap = (zp[ip, 0] <= zt[it, 1]) & (zt[it, 0] <= zp[ip, 1])
        ip, it = ip[overlap], it[overlap]
    parts_p, parts_n, parts_ip, parts_it = [], [], [], []
    for s in range(0, len(ip), chunk):
        a, b = ip[s : s + chunk], it[s : s + chunk]
        q, c = poly[a], n[a]
        for line in _edge_lines(tri[b]):
            q, c = _clip(q, c, line)
        area, _ = _area_centroid(q, c)
        keep = (c >= 3) & (area > 1e-12)
        parts_p.append(q[keep])
        parts_n.append(c[keep])
        parts_ip.append(a[keep])
        parts_it.append(b[keep])
    if not parts_p:
        return (
            np.zeros((0, 3, 2)),
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
        )
    k_max = max(p.shape[1] for p in parts_p)
    padded = [np.pad(p, ((0, 0), (0, k_max - p.shape[1]), (0, 0))) for p in parts_p]
    return (
        np.concatenate(padded),
        np.concatenate(parts_n),
        np.concatenate(parts_ip),
        np.concatenate(parts_it),
    )


def _limit_triangles(rings: list[Floats] | tuple[Floats, ...], origin: Floats) -> Floats:
    """Triangles exactly covering the limit polygon (even-odd rings)."""
    pts = np.vstack([np.asarray(r, np.float64)[:, :2] for r in rings])
    flat, _ = build_surface(np.column_stack([pts, np.zeros(len(pts))]), boundary=list(rings))
    t = _tris(flat, origin)
    return t.corners


def triangle_volume(
    upper: Surface,
    lower: Surface | float,
    *,
    limit: list[Floats] | tuple[Floats, ...] = (),
) -> VolumeReport:
    """Exact cut and fill between ``upper`` and ``lower`` (a surface or a level).

    ``limit``: optional closed rings (x, y) restricting the plan area.
    """
    origin = upper.points[:, :2].mean(axis=0)
    ta = _tris(upper, origin)
    poly = np.pad(ta.corners, ((0, 0), (0, 1), (0, 0)))
    n = np.full(len(poly), 3, np.int64)
    if isinstance(lower, Surface):
        tb = _tris(lower, origin)
        poly, n, ia, ib = _intersect(poly, n, tb.corners)
        dplane = ta.plane[ia] - tb.plane[ib]
    else:
        if not math.isfinite(float(lower)):
            raise VolumeError("the level must be finite")
        dplane = ta.plane - np.array([0.0, 0.0, float(lower)])
    if limit:
        lt = _limit_triangles(limit, origin)
        poly, n, ip, _ = _intersect(poly, n, lt)
        dplane = dplane[ip]
    if len(poly) == 0:
        raise VolumeError("the surfaces do not overlap in plan")
    total, _ = _area_centroid(poly, n)
    volumes = []
    for sign in (1.0, -1.0):
        q, c = _clip(poly, n, sign * dplane)
        area, cen = _area_centroid(q, c)
        dz = dplane[:, 0] * cen[:, 0] + dplane[:, 1] * cen[:, 1] + dplane[:, 2]
        volumes.append((float((area * np.abs(dz)).sum()), float(area.sum())))
    (cut, cut_area), (fill, fill_area) = volumes
    return VolumeReport("triangle", cut, fill, float(total.sum()), cut_area, fill_area)


# -- grid --------------------------------------------------------------------


@dataclass(frozen=True)
class ThicknessGrid:
    """Upper minus lower at cell centres (NaN where either is missing)."""

    x: Floats  # cell centre x (nx)
    y: Floats  # cell centre y (ny)
    dz: Floats  # nx x ny
    cell: float


def thickness_grid(
    upper: Surface,
    lower: Surface | float,
    cell: float,
    *,
    limit: list[Floats] | tuple[Floats, ...] = (),
    max_cells: int = 25_000_000,
) -> ThicknessGrid:
    """Thickness at cell centres over the overlap of the two surfaces."""
    if not (math.isfinite(cell) and cell > 0):
        raise VolumeError("cell size must be > 0")
    tri_a = upper.points[upper.triangles()][:, :, :2].reshape(-1, 2)
    lo, hi = tri_a.min(axis=0), tri_a.max(axis=0)
    if isinstance(lower, Surface):
        tri_b = lower.points[lower.triangles()][:, :, :2].reshape(-1, 2)
        lo, hi = np.maximum(lo, tri_b.min(axis=0)), np.minimum(hi, tri_b.max(axis=0))
    if np.any(hi <= lo):
        raise VolumeError("the surfaces do not overlap in plan")
    n = np.ceil((hi - lo) / cell).astype(np.int64)
    if int(n[0]) * int(n[1]) > max_cells:
        raise VolumeError("too many grid cells; use a larger cell size")
    xs = lo[0] + (np.arange(n[0]) + 0.5) * cell
    ys = lo[1] + (np.arange(n[1]) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    ok = upper.contains(xy)
    za = upper.elevation(xy)[0]
    if isinstance(lower, Surface):
        ok &= lower.contains(xy)
        zb = lower.elevation(xy)[0]
    else:
        zb = np.full(len(xy), float(lower))
    if limit:
        ok &= inside_rings(xy, limit)
    dz = np.where(ok, za - zb, np.nan)
    return ThicknessGrid(xs, ys, dz.reshape(len(xs), len(ys)), float(cell))


def grid_volume(
    upper: Surface,
    lower: Surface | float,
    cell: float,
    *,
    limit: list[Floats] | tuple[Floats, ...] = (),
) -> VolumeReport:
    """Cut and fill as thickness at cell centres x cell area."""
    g = thickness_grid(upper, lower, cell, limit=limit)
    dz = g.dz[np.isfinite(g.dz)]
    if len(dz) == 0:
        raise VolumeError("the surfaces do not overlap in plan")
    a = cell * cell
    return VolumeReport(
        f"grid {cell:g} m",
        float(np.clip(dz, 0, None).sum() * a),
        float(np.clip(-dz, 0, None).sum() * a),
        float(len(dz) * a),
        float((dz > 0).sum() * a),
        float((dz < 0).sum() * a),
    )


# -- sections ----------------------------------------------------------------


def end_area_volume(length: float, a1: float, a2: float) -> float:
    """Average end area rule: ``L (A1 + A2) / 2``."""
    return float(length) * (float(a1) + float(a2)) / 2.0


def prismoidal_volume(length: float, a1: float, am: float, a2: float) -> float:
    """Prismoidal rule: ``L / 6 (A1 + 4 Am + A2)``; exact when area is quadratic in L."""
    return float(length) / 6.0 * (float(a1) + 4.0 * float(am) + float(a2))


def _edges(surface: Surface) -> Floats:
    tri = surface.triangles()
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    return surface.points[e][:, :, :2]  # k x 2 x 2


def section_area(
    upper: Surface,
    lower: Surface | float,
    start: Floats,
    end: Floats,
    *,
    edges: tuple[Floats, Floats | None] | None = None,
) -> tuple[float, float]:
    """(cut, fill) area (m2) on the vertical section from ``start`` to ``end``.

    Exact: both surfaces are linear between the points where the section line
    crosses triangle edges, and sign changes are split at the zero.
    """
    p0, p1 = np.asarray(start, np.float64)[:2], np.asarray(end, np.float64)[:2]
    d = p1 - p0
    length = float(np.hypot(*d))
    if length <= 0:
        raise VolumeError("a section needs two different end points")
    if edges is None:
        edges = (_edges(upper), _edges(lower) if isinstance(lower, Surface) else None)
    ts = [np.array([0.0, 1.0])]
    for e in edges:
        if e is None:
            continue
        a, b = e[:, 0], e[:, 1]
        s = b - a
        q = a - p0
        den = d[0] * s[:, 1] - d[1] * s[:, 0]
        ok = np.abs(den) > 1e-15
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / den
            u = (q[:, 0] * d[1] - q[:, 1] * d[0]) / den
        hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        ts.append(t[hit])
    t = np.unique(np.concatenate(ts))
    xy = p0 + t[:, None] * d
    za = upper.elevation(xy)[0]
    zb = lower.elevation(xy)[0] if isinstance(lower, Surface) else np.full(len(t), float(lower))
    mid = p0 + (0.5 * (t[:-1] + t[1:]))[:, None] * d
    ok = upper.contains(mid)
    if isinstance(lower, Surface):
        ok &= lower.contains(mid)
    h0, h1 = (za - zb)[:-1], (za - zb)[1:]
    w = np.diff(t) * length
    cut = fill = 0.0
    same = (h0 >= 0) == (h1 >= 0)
    trap = 0.5 * (h0 + h1) * w
    cut += float(np.where(ok & same & (h0 >= 0), trap, 0.0).sum())
    fill -= float(np.where(ok & same & (h0 < 0), trap, 0.0).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(same, 0.0, h0 / (h0 - h1))
    tri0 = 0.5 * h0 * f * w
    tri1 = 0.5 * h1 * (1 - f) * w
    split = ok & ~same
    cut += float(np.where(split, np.clip(tri0, 0, None) + np.clip(tri1, 0, None), 0).sum())
    fill += float(np.where(split, np.clip(-tri0, 0, None) + np.clip(-tri1, 0, None), 0).sum())
    return cut, fill


@dataclass(frozen=True)
class SectionRow:
    station: float  # m along the axis
    cut_area: float
    fill_area: float


@dataclass(frozen=True)
class SectionInterval:
    start: float
    end: float
    mid_cut_area: float
    mid_fill_area: float
    cut_end_area: float  # m3
    fill_end_area: float
    cut_prismoidal: float
    fill_prismoidal: float


@dataclass(frozen=True)
class SectionReport:
    sections: list[SectionRow]
    intervals: list[SectionInterval]
    interval: float
    half_width: float

    def total(self, rule: str) -> VolumeReport:
        """Totals by ``"end_area"`` or ``"prismoidal"``."""
        if rule == "end_area":
            cut = sum(i.cut_end_area for i in self.intervals)
            fill = sum(i.fill_end_area for i in self.intervals)
        elif rule == "prismoidal":
            cut = sum(i.cut_prismoidal for i in self.intervals)
            fill = sum(i.fill_prismoidal for i in self.intervals)
        else:
            raise VolumeError(f"unknown section rule {rule!r}")
        # Plan areas are not measured by sections (NaN).
        return VolumeReport(
            f"sections {self.interval:g} m {rule}", cut, fill, math.nan, math.nan, math.nan
        )


def _station(axis: Floats, cum: Floats, s: float) -> tuple[Floats, Floats]:
    k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(axis) - 2))
    seg = axis[k + 1] - axis[k]
    seg_len = float(np.hypot(*seg))
    direction = seg / seg_len
    return axis[k] + direction * (s - cum[k]), direction


def section_volumes(
    upper: Surface,
    lower: Surface | float,
    axis: Floats,
    interval: float,
    half_width: float,
) -> SectionReport:
    """Sections perpendicular to ``axis`` (plan polyline) every ``interval`` m.

    Each section reaches ``half_width`` m to both sides of the axis; the last
    section is at the axis end. Both the interval and the width are the
    user's choice (survey practice), never assumed.
    """
    axis = np.asarray(axis, np.float64)[:, :2]
    if len(axis) < 2:
        raise VolumeError("the axis needs at least two points")
    if not (
        math.isfinite(interval) and interval > 0 and math.isfinite(half_width) and half_width > 0
    ):
        raise VolumeError("interval and half width must be > 0")
    seg = np.linalg.norm(np.diff(axis, axis=0), axis=1)
    if (seg <= 0).any():
        raise VolumeError("the axis has repeated points")
    cum = np.r_[0.0, np.cumsum(seg)]
    stations = np.arange(0.0, cum[-1], interval)
    stations = np.r_[stations, cum[-1]] if cum[-1] - stations[-1] > 1e-9 else stations
    edges = (_edges(upper), _edges(lower) if isinstance(lower, Surface) else None)

    def areas(s: float) -> tuple[float, float]:
        centre, direction = _station(axis, cum, s)
        normal = np.array([-direction[1], direction[0]])
        return section_area(
            upper, lower, centre - half_width * normal, centre + half_width * normal, edges=edges
        )

    rows = [SectionRow(float(s), *areas(float(s))) for s in stations]
    intervals = []
    for r0, r1 in itertools.pairwise(rows):
        length = r1.station - r0.station
        mc, mf = areas(0.5 * (r0.station + r1.station))
        intervals.append(
            SectionInterval(
                r0.station,
                r1.station,
                mc,
                mf,
                end_area_volume(length, r0.cut_area, r1.cut_area),
                end_area_volume(length, r0.fill_area, r1.fill_area),
                prismoidal_volume(length, r0.cut_area, mc, r1.cut_area),
                prismoidal_volume(length, r0.fill_area, mf, r1.fill_area),
            )
        )
    return SectionReport(rows, intervals, float(interval), float(half_width))
