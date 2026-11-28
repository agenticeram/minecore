"""Surface (DTM) editing: merge/patch, normal offset, smoothing, decimation and
grid <-> TIN conversion (gap list E-CD-02, catalogue TP02 part).

* ``merge_surfaces``: upper or lower envelope of two surfaces, or ``a``
  patched into ``b`` (``a`` wins where it exists). The plan arrangement of
  both triangulations is formed exactly: every edge of each surface is split
  where it crosses an edge of the other and where the two surfaces swap order
  along it, and the intersection lines are added. On every face of that
  arrangement both surfaces are planes, so the envelope is one plane; the
  result is triangulated honouring all those segments and is therefore the
  exact envelope (no resampling). Where one surface ends above the other the
  step is bridged by the triangles next to the edge (a TIN has one elevation
  per plan point).
* ``normal_offset``: every triangle plane moved ``distance`` along its upward
  unit normal; each vertex goes to the least-squares meeting point of the
  moved planes around it (exact on planes, ridges and symmetric apexes),
  limited to ``max_factor`` x the distance. Folded triangles are refused.
* ``smooth_surface``: elevations only (plan positions and triangles kept).
  Each pass moves a vertex ``factor`` of the way to the least-squares plane
  through its neighbours, so a plane is left exactly as it is; a user limit
  caps the total change per vertex and boundary vertices can be held.
* ``decimate_surface``: greedy insertion. Starting from the boundary, the
  vertex with the largest vertical error in each triangle is added until
  every original vertex is within the user's vertical tolerance.
* ``read_ascii_grid`` / ``grid_surface``: ESRI ASCII grid to a TIN of the cell
  centres (no-data cells become holes), optionally decimated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay

from minecore._core import MineCoreError
from minecore.surfaces import Surface, build_surface
from minecore.survey.volumes import _candidate_pairs

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]

MODES = ("upper", "lower", "patch")


class SurfaceEditError(MineCoreError):
    """Invalid surface editing input."""


# -- mesh helpers --------------------------------------------------------------------


def _ccw(points: Floats, tri: Ints) -> Ints:
    """Triangles ordered counter-clockwise in plan."""
    p = points[tri, :2]
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    flip = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0] < 0
    out = tri.copy()
    out[flip] = out[flip][:, ::-1]
    return out


def _unique_edges(tri: Ints) -> Ints:
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def _is_rim(edges: Ints, tri: Ints) -> npt.NDArray[np.bool_]:
    """Which of the sorted unique ``edges`` lie on the footprint boundary."""
    rim = np.sort(_boundary_edges(tri), axis=1)
    n = int(max(edges.max(), tri.max())) + 1
    return np.isin(edges[:, 0] * n + edges[:, 1], rim[:, 0] * n + rim[:, 1])


def _boundary_edges(tri: Ints) -> Ints:
    """Directed edges used by one triangle only (counter-clockwise triangles)."""
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, count = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    out: Ints = e[count[inv.ravel()] == 1]
    return out


def boundary_rings(points: Floats, tri: Ints) -> list[Floats]:
    """Closed plan rings around a triangle set (outer rings and holes).

    Under the even-odd rule the rings reproduce the triangle set's footprint
    exactly, whatever order the boundary edges are chained in.
    """
    edges = _boundary_edges(_ccw(points, tri))
    nxt: dict[int, list[int]] = {}
    for s, t in edges:
        nxt.setdefault(int(s), []).append(int(t))
    rings = []
    while nxt:
        start = next(iter(nxt))
        ring = [start]
        k = start
        while True:
            ends = nxt[k]
            t = ends.pop()
            if not ends:
                del nxt[k]
            if t == start:
                break
            ring.append(t)
            k = t
            if k not in nxt:  # open chain (should not happen for a valid mesh)
                break
        if len(ring) >= 3:
            rings.append(points[ring, :2].copy())
    return rings


def _planes(points: Floats, tri: Ints, origin: Floats) -> Floats:
    """z = a x + b y + c per triangle (plan coordinates relative to ``origin``)."""
    p = points[tri]
    xy = p[:, :, :2] - origin
    u, v = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    det = np.where(np.abs(det) > 1e-300, det, 1e-300)
    dz1, dz2 = p[:, 1, 2] - p[:, 0, 2], p[:, 2, 2] - p[:, 0, 2]
    a = (dz1 * v[:, 1] - dz2 * u[:, 1]) / det
    b = (dz2 * u[:, 0] - dz1 * v[:, 0]) / det
    c = p[:, 0, 2] - a * xy[:, 0, 0] - b * xy[:, 0, 1]
    return np.column_stack([a, b, c])


def _z_at(surface: Surface, xy: Floats) -> Floats:
    """Elevation on the kept triangle under each point, NaN off the surface."""
    origin = surface._origin()
    d = surface._delaunay()
    simplex = np.asarray(d.find_simplex(xy - origin), np.int64)
    inside = surface.contains(xy)
    z = np.full(len(xy), np.nan)
    if inside.any():
        tri = np.asarray(d.simplices, np.int64)[simplex[inside]]
        pl = _planes(surface.points, tri, origin)
        q = xy[inside] - origin
        z[inside] = pl[:, 0] * q[:, 0] + pl[:, 1] * q[:, 1] + pl[:, 2]
    return z


# -- merge / patch -----------------------------------------------------------------------


@dataclass(frozen=True)
class MergeReport:
    mode: str
    triangles: int
    area_both: float  # m2 where both surfaces exist
    area_a_only: float
    area_b_only: float
    area_from_a: float  # m2 of the result taken from a (where both exist)
    area_from_b: float
    crossings: int  # edge crossings between the two triangulations


def _segment_hits(
    pa: Floats, pb: Floats, qa: Floats, qb: Floats
) -> tuple[Ints, Ints, Floats, Floats]:
    """Crossings of segments p (pa->pb) with segments q (qa->qb), ends included."""
    box_p = np.column_stack([np.minimum(pa, pb), np.maximum(pa, pb)])
    box_q = np.column_stack([np.minimum(qa, qb), np.maximum(qa, qb)])
    i, j = _candidate_pairs(box_p, box_q)
    r, s = pb[i] - pa[i], qb[j] - qa[j]
    w = qa[j] - pa[i]
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    scale = np.linalg.norm(r, axis=1) * np.linalg.norm(s, axis=1)
    ok = np.abs(den) > 1e-12 * np.maximum(scale, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * s[:, 1] - w[:, 1] * s[:, 0]) / den
        u = (w[:, 0] * r[:, 1] - w[:, 1] * r[:, 0]) / den
    eps = 1e-12
    hit = ok & (t >= -eps) & (t <= 1 + eps) & (u >= -eps) & (u <= 1 + eps)
    return i[hit], j[hit], np.clip(t[hit], 0.0, 1.0), np.clip(u[hit], 0.0, 1.0)


def _pieces(params: list[list[float]]) -> tuple[Ints, Floats, Floats]:
    """(segment id, t0, t1) of each piece after splitting every segment at its params."""
    seg, t0, t1 = [], [], []
    for k, ts in enumerate(params):
        cuts = np.unique(np.r_[0.0, np.asarray(ts, np.float64), 1.0])
        cuts = cuts[np.r_[True, np.diff(cuts) > 1e-12]]
        cuts[-1] = max(cuts[-1], 1.0)
        seg += [k] * (len(cuts) - 1)
        t0 += list(cuts[:-1])
        t1 += list(cuts[1:])
    return np.asarray(seg, np.int64), np.asarray(t0), np.asarray(t1)


def _envelope(za: Floats, zb: Floats, mode: str) -> Floats:
    if mode == "patch":
        return np.where(np.isfinite(za), za, zb)
    both = np.isfinite(za) & np.isfinite(zb)
    pick = np.fmax if mode == "upper" else np.fmin
    z = pick(za, zb)  # fmax/fmin ignore NaN: one surface only
    return np.where(both, np.maximum(za, zb) if mode == "upper" else np.minimum(za, zb), z)


def merge_surfaces(  # noqa: PLR0912, PLR0915 - arrangement, envelope, trimming
    a: Surface, b: Surface, mode: str = "upper", *, tolerance: float = 0.001
) -> tuple[Surface, MergeReport]:
    """Upper or lower envelope of ``a`` and ``b``, or ``a`` patched into ``b``.

    ``mode``: ``upper`` (higher of the two where both exist), ``lower`` (lower,
    e.g. a pit design cut into topography), ``patch`` (``a`` where it exists,
    ``b`` elsewhere). Outside the other surface each surface is kept as it is.
    ``tolerance`` (m, plan) merges coincident vertices.
    """
    if mode not in MODES:
        raise SurfaceEditError(f"merge mode must be one of {', '.join(MODES)}")
    if not tolerance > 0:
        raise SurfaceEditError("the tolerance must be > 0 m")
    from minecore.surfacetools import intersection_lines  # noqa: PLC0415 - avoid a cycle

    origin = np.minimum(a.points[:, :2].min(0), b.points[:, :2].min(0))
    ta, tb = a.triangles(), b.triangles()
    if len(ta) == 0 or len(tb) == 0:
        raise SurfaceEditError("both surfaces need triangles")
    ea, eb = _unique_edges(ta), _unique_edges(tb)
    rim_a, rim_b = _is_rim(ea, ta), _is_rim(eb, tb)
    pa0, pa1 = a.points[ea[:, 0]], a.points[ea[:, 1]]
    pb0, pb1 = b.points[eb[:, 0]], b.points[eb[:, 1]]
    i, j, t, u = _segment_hits(
        pa0[:, :2] - origin, pa1[:, :2] - origin, pb0[:, :2] - origin, pb1[:, :2] - origin
    )
    params_a: list[list[float]] = [[] for _ in range(len(ea))]
    params_b: list[list[float]] = [[] for _ in range(len(eb))]
    for k in range(len(i)):
        params_a[i[k]].append(float(t[k]))
        params_b[j[k]].append(float(u[k]))

    # Elevation of the other surface at each crossing, from its own edge (exact
    # even where the crossing is on that surface's outer boundary).
    z_on_b = pb0[j, 2] + u * (pb1[j, 2] - pb0[j, 2])
    z_on_a = pa0[i, 2] + t * (pa1[i, 2] - pa0[i, 2])
    segs: list[Floats] = []
    for other, p0, p1, params, rim, own_is_a, cross in (
        (b, pa0, pa1, params_a, rim_a, True, (i, t, z_on_b)),
        (a, pb0, pb1, params_b, rim_b, False, (j, u, z_on_a)),
    ):
        sid, s0, s1 = _pieces(params)
        q0 = p0[sid] + s0[:, None] * (p1[sid] - p0[sid])  # exact on the own edge
        q1 = p0[sid] + s1[:, None] * (p1[sid] - p0[sid])
        mid = 0.5 * (q0 + q1)
        # The other surface is one plane along a piece (pieces end at its edges).
        on_other = np.isfinite(_z_at(other, mid[:, :2]))
        zo0 = np.full(len(sid), np.nan)
        zo1 = np.full(len(sid), np.nan)
        if on_other.any():
            o_origin = other._origin()
            dl = other._delaunay()
            simplex = np.asarray(dl.find_simplex(mid[on_other, :2] - o_origin), np.int64)
            pl = _planes(other.points, np.asarray(dl.simplices, np.int64)[simplex], o_origin)
            for zo, q in ((zo0, q0), (zo1, q1)):
                r = q[on_other, :2] - o_origin
                zo[on_other] = pl[:, 0] * r[:, 0] + pl[:, 1] * r[:, 1] + pl[:, 2]
        # A piece end at a crossing lies on the other surface's edge.
        c_sid, c_t, c_z = cross
        c_order = np.lexsort((c_t, c_sid))
        c_sid, c_t, c_z = c_sid[c_order], c_t[c_order], c_z[c_order]
        for zo, q, sv in ((zo0, q0, s0), (zo1, q1, s1)):
            if len(c_sid):
                key = c_sid.astype(np.float64) * 4.0 + c_t
                at = np.clip(np.searchsorted(key, sid * 4.0 + sv), 0, len(key) - 1)
                for kk in (at, np.maximum(at - 1, 0)):
                    hit = (c_sid[kk] == sid) & (np.abs(c_t[kk] - sv) <= 1e-9)
                    zo[hit] = c_z[kk[hit]]
            gap = ~np.isfinite(zo)
            if gap.any():
                zo[gap] = _z_at(other, q[gap, :2])
        za0, zb0 = (q0[:, 2], zo0) if own_is_a else (zo0, q0[:, 2])
        za1, zb1 = (q1[:, 2], zo1) if own_is_a else (zo1, q1[:, 2])
        e0, e1 = _envelope(za0, zb0, mode), _envelope(za1, zb1, mode)
        d0, d1 = za0 - zb0, za1 - zb1
        swap = on_other & np.isfinite(d0) & np.isfinite(d1) & (d0 * d1 < 0) & (mode != "patch")
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(swap, d0 / (d0 - d1), 0.5)
        xm = q0[:, :2] + f[:, None] * (q1[:, :2] - q0[:, :2])
        zm = q0[:, 2] + f * (q1[:, 2] - q0[:, 2])  # on the own surface = on both
        a0 = np.column_stack([q0[:, :2], e0])
        a1 = np.column_stack([q1[:, :2], e1])
        m = np.column_stack([xm, zm])
        # Only pieces of the surface that forms the result are kinks of it; the
        # other surface's edges there are not constraints (fewer tiny angles).
        zm_a = np.where(swap, zm, np.nan)
        halves = (
            (
                a0,
                np.where(swap[:, None], m, a1),
                za0,
                np.where(swap, zm_a, za1),
                zb0,
                np.where(swap, zm_a, zb1),
            ),
            (m, a1, zm_a, za1, zm_a, zb1),
        )
        runs: list[tuple[Ints, Floats, Floats, Floats, Floats]] = []
        for h, (s_from, s_to, ha0, ha1, hb0, hb1) in enumerate(halves):
            rows = np.ones(len(sid), bool) if h == 0 else swap
            zam, zbm = 0.5 * (ha0 + ha1), 0.5 * (hb0 + hb1)
            env = _envelope(zam, zbm, mode)
            own = zam if own_is_a else zbm
            top = rows & np.isfinite(own) & (np.abs(own - env) <= 1e-9 * (1.0 + np.abs(env)))
            # Footprint edges always: a step where one surface ends above the
            # other is then bridged only outside it.
            rim_only = rows & rim[sid] & ~top
            segs += [np.stack([s_from[r], s_to[r]]) for r in np.flatnonzero(rim_only)]
            t_from = s0 if h == 0 else s0 + f * (s1 - s0)
            t_to = np.where(swap, s0 + f * (s1 - s0), s1) if h == 0 else s1
            kt = np.flatnonzero(top)
            runs.append((sid[kt], t_from[kt], t_to[kt], s_from[kt], s_to[kt]))
        # Consecutive pieces of one edge on which the own surface forms the
        # result are one straight, linear segment: join them (the other
        # surface's edges crossing there are no kinks of the result).
        sid_k, tf, tt, pf, pt = (np.concatenate(x) for x in zip(*runs, strict=True))
        order = np.lexsort((tf, sid_k))
        sid_k, tf, tt, pf, pt = sid_k[order], tf[order], tt[order], pf[order], pt[order]
        new_run = np.r_[True, (sid_k[1:] != sid_k[:-1]) | (np.abs(tf[1:] - tt[:-1]) > 1e-12)]
        first = np.flatnonzero(new_run)
        last = np.r_[first[1:] - 1, len(sid_k) - 1]
        segs += [np.stack([pf[i0], pt[i1]]) for i0, i1 in zip(first, last, strict=True)]
    lines: list[Floats] = []
    if mode != "patch":
        lines = [
            np.vstack([x.points, x.points[:1]]) if x.closed else x.points
            for x in intersection_lines(a, b, tolerance=tolerance)
        ]
    breaklines = [s for s in segs if np.linalg.norm(s[1, :2] - s[0, :2]) > tolerance] + lines
    built, _ = build_surface(
        None,
        breaklines,
        tolerance=tolerance,
        z_tolerance=max(tolerance, 1e-6),
        tolerate_small_angles=True,
    )
    tri = built.triangles()
    c = built.points[tri, :2].mean(axis=1)
    za, zb = _z_at(a, c), _z_at(b, c)
    keep = np.isfinite(za) | np.isfinite(zb)
    tri = tri[keep]
    za, zb = za[keep], zb[keep]
    rings = boundary_rings(built.points, tri) if not keep.all() else []
    result = Surface(built.points, tuple(rings)) if rings else built
    p = built.points[tri, :2]
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    both = np.isfinite(za) & np.isfinite(zb)
    if mode == "patch":
        from_a = both
    elif mode == "upper":
        from_a = both & (za >= zb)
    else:
        from_a = both & (za <= zb)
    report = MergeReport(
        mode=mode,
        triangles=len(tri),
        area_both=float(area[both].sum()),
        area_a_only=float(area[np.isfinite(za) & ~both].sum()),
        area_b_only=float(area[np.isfinite(zb) & ~both].sum()),
        area_from_a=float(area[from_a].sum()),
        area_from_b=float(area[both & ~from_a].sum()),
        crossings=len(i),
    )
    return result, report


# -- normal offset ------------------------------------------------------------------------


@dataclass(frozen=True)
class OffsetReport:
    distance: float
    vertices: int
    limited: int  # vertices whose move was capped at max_factor x distance
    max_move: float  # m, largest vertex move
    triangles: int


def _vertex_angles(points: Floats, tri: Ints) -> Floats:
    """Interior angle of each triangle at each corner (3D), m x 3."""
    p = points[tri]
    out = np.zeros(tri.shape)
    for k in range(3):
        u = p[:, (k + 1) % 3] - p[:, k]
        v = p[:, (k + 2) % 3] - p[:, k]
        cos = (u * v).sum(1) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300
        )
        out[:, k] = np.arccos(np.clip(cos, -1.0, 1.0))
    return out


def normal_offset(
    surface: Surface, distance: float, *, max_factor: float = 3.0, tolerance: float = 0.001
) -> tuple[Surface, OffsetReport]:
    """The surface moved ``distance`` m along its upward normal (- moves down).

    Each triangle's plane is moved by ``distance``; a vertex goes to the
    least-squares point on the moved planes around it (angle weighted), never
    further than ``max_factor`` x ``distance``. If any triangle folds over in
    plan the offset is refused: reduce the distance or smooth first.
    """
    if not (math.isfinite(distance) and distance != 0):
        raise SurfaceEditError("the offset distance must be a non-zero number of m")
    if not (math.isfinite(max_factor) and max_factor >= 1):
        raise SurfaceEditError("the move limit factor must be >= 1")
    tri = _ccw(surface.points, surface.triangles())
    used = np.unique(tri)
    remap = np.full(len(surface.points), -1, np.int64)
    remap[used] = np.arange(len(used))
    pts = surface.points[used] - np.r_[surface._origin(), 0.0]
    tri = remap[tri]
    p = pts[tri]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    w = _vertex_angles(pts, tri)
    m = np.zeros((len(pts), 3, 3))
    r = np.zeros((len(pts), 3))
    outer = n[:, :, None] * n[:, None, :]
    for k in range(3):
        np.add.at(m, tri[:, k], w[:, k, None, None] * outer)
        np.add.at(r, tri[:, k], w[:, k, None] * distance * n)
    val, vec = np.linalg.eigh(m)
    cut = 1e-6 * val[:, -1:]
    inv = np.where(val > cut, 1.0 / np.where(val > cut, val, 1.0), 0.0)
    coef = np.einsum("nji,nj->ni", vec, r) * inv
    move = np.einsum("nij,nj->ni", vec, coef)
    length = np.linalg.norm(move, axis=1)
    cap = max_factor * abs(distance)
    limited = length > cap
    move[limited] *= (cap / length[limited])[:, None]
    new = pts + move
    q = new[tri, :2]
    area = (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (
        q[:, 1, 1] - q[:, 0, 1]
    )
    folded = int((area <= 0).sum())
    if folded:
        raise SurfaceEditError(
            f"{folded} triangles fold over at {distance:g} m (concave areas tighter than the "
            "offset); use a smaller distance or smooth the surface first"
        )
    new = new + np.r_[surface._origin(), 0.0]
    edges = _unique_edges(tri)
    rings = boundary_rings(new, tri)
    out, _ = build_surface(
        None,
        [new[e] for e in edges],
        rings,
        tolerance=tolerance,
        z_tolerance=1e9,
        tolerate_small_angles=True,
    )
    return out, OffsetReport(
        distance=float(distance),
        vertices=len(pts),
        limited=int(limited.sum()),
        max_move=float(np.linalg.norm(move, axis=1).max()),
        triangles=len(out.triangles()),
    )


# -- smoothing -------------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothReport:
    iterations: int
    max_change: float  # m, largest |dz| of a vertex
    mean_change: float  # m, mean |dz|
    volume_change: float  # m3, smoothed minus original (exact, same triangles)
    limited: int  # vertices held at the user's change limit


def smooth_surface(
    surface: Surface,
    iterations: int = 1,
    factor: float = 0.5,
    *,
    max_change: float | None = None,
    hold_boundary: bool = True,
) -> tuple[Surface, SmoothReport]:
    """Elevations smoothed towards the local plane of each vertex's neighbours.

    ``factor`` (0-1) is the share of the way moved per pass; ``max_change`` (m,
    user's, e.g. survey accuracy) caps the total change of any vertex;
    ``hold_boundary`` keeps boundary vertices fixed. Plan positions and
    triangles are unchanged, so a planar surface stays exactly planar.
    """
    if not (isinstance(iterations, int) and iterations >= 1):
        raise SurfaceEditError("smoothing needs at least one pass")
    if not (0 < factor <= 1):
        raise SurfaceEditError("the smoothing factor must be in (0, 1]")
    if max_change is not None and not (math.isfinite(max_change) and max_change > 0):
        raise SurfaceEditError("the change limit must be > 0 m")
    tri = surface.triangles()
    edges = _unique_edges(tri)
    origin = surface._origin()
    xy = surface.points[:, :2] - origin
    z0 = surface.points[:, 2].copy()
    z = z0.copy()
    hold = np.zeros(len(z), bool)
    if hold_boundary:
        hold[np.unique(_boundary_edges(_ccw(surface.points, tri)))] = True
    used = np.zeros(len(z), bool)
    used[tri.ravel()] = True
    src = np.r_[edges[:, 0], edges[:, 1]]
    dst = np.r_[edges[:, 1], edges[:, 0]]
    dx = xy[dst] - xy[src]
    # Normal equations of the neighbour plane z = c + gx dx + gy dy around each vertex.
    a = np.zeros((len(z), 3, 3))
    basis = np.column_stack([np.ones(len(src)), dx])
    np.add.at(a, src, basis[:, :, None] * basis[:, None, :])
    det_ok = np.abs(np.linalg.det(a)) > 1e-9 * np.maximum(a[:, 1, 1] * a[:, 2, 2], 1e-300)
    inv = np.zeros_like(a)
    inv[det_ok] = np.linalg.inv(a[det_ok])
    count = np.bincount(src, minlength=len(z)).astype(float)
    limited = np.zeros(len(z), bool)
    for _ in range(iterations):
        rhs = np.zeros((len(z), 3))
        np.add.at(rhs, src, basis * z[dst, None])
        target = np.einsum("nij,nj->ni", inv, rhs)[:, 0]
        mean = np.bincount(src, weights=z[dst], minlength=len(z)) / np.maximum(count, 1)
        target = np.where(det_ok, target, mean)
        move = used & ~hold & (count > 0)
        z = np.where(move, z + factor * (target - z), z)
        if max_change is not None:
            over = np.abs(z - z0) > max_change
            limited |= over
            z = np.where(over, z0 + np.sign(z - z0) * max_change, z)
    dz = z - z0
    p = xy[tri]
    plan = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    volume = float((plan * dz[tri].mean(axis=1)).sum())
    pts = surface.points.copy()
    pts[:, 2] = z
    moved = np.abs(dz[used])
    return Surface(pts, surface.boundary, surface.max_edge), SmoothReport(
        iterations=iterations,
        max_change=float(moved.max()) if len(moved) else 0.0,
        mean_change=float(moved.mean()) if len(moved) else 0.0,
        volume_change=volume,
        limited=int(limited.sum()),
    )


# -- decimation ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DecimateReport:
    tolerance: float
    points_before: int
    points_after: int
    triangles: int
    max_error: float  # m, largest vertical error at an original vertex
    rounds: int


def _errors(sel_pts: Floats, all_pts: Floats, origin: Floats) -> tuple[Floats, Ints]:
    d = Delaunay(sel_pts[:, :2] - origin)
    q = all_pts[:, :2] - origin
    simplex = np.asarray(d.find_simplex(q), np.int64)
    err = np.full(len(q), np.inf)
    ok = simplex >= 0
    tri = np.asarray(d.simplices, np.int64)[simplex[ok]]
    pl = _planes(sel_pts, tri, origin)
    zi = pl[:, 0] * q[ok, 0] + pl[:, 1] * q[ok, 1] + pl[:, 2]
    err[ok] = np.abs(all_pts[ok, 2] - zi)
    return err, simplex


def decimate_surface(
    surface: Surface, tolerance: float, *, max_rounds: int = 200
) -> tuple[Surface, DecimateReport]:
    """Fewer points, every original vertex within ``tolerance`` m vertically.

    ``tolerance`` is the user's (e.g. survey accuracy). Boundary vertices are
    always kept, so the footprint does not change.
    """
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise SurfaceEditError("the vertical tolerance must be >= 0 m")
    tri = surface.triangles()
    used = np.unique(tri)
    pts = surface.points[used]
    remap = np.full(len(surface.points), -1, np.int64)
    remap[used] = np.arange(len(used))
    tri = remap[tri]
    origin = surface._origin()
    bverts = np.unique(_boundary_edges(_ccw(pts, tri)))
    hull = Delaunay(pts[:, :2] - origin).convex_hull
    selected = np.zeros(len(pts), bool)
    selected[bverts] = True
    selected[np.unique(hull)] = True
    rings = boundary_rings(pts, tri) if surface.trimmed else []
    rounds = 0
    err = np.zeros(len(pts))
    while rounds < max_rounds:
        rounds += 1
        sel = np.flatnonzero(selected)
        err, simplex = _errors(pts[sel], pts, origin)
        err[selected] = 0.0
        bad = np.flatnonzero(err > tolerance)
        if len(bad) == 0:
            break
        # The worst point of every triangle that still has an error over the limit.
        order = bad[np.lexsort((-err[bad], simplex[bad]))]
        first = np.r_[True, np.diff(simplex[order]) != 0]
        selected[order[first]] = True
    final = Surface(pts[selected], tuple(rings)) if rings else Surface(pts[selected])
    if rings:
        final, _ = build_surface(pts[selected], (), rings, tolerance=1e-6)
    z_final = _z_at(final, pts[:, :2])
    max_error = float(np.nanmax(np.abs(z_final - pts[:, 2]))) if len(pts) else 0.0
    return final, DecimateReport(
        tolerance=float(tolerance),
        points_before=len(pts),
        points_after=int(selected.sum()),
        triangles=len(final.triangles()),
        max_error=max_error,
        rounds=rounds,
    )


# -- grid <-> TIN -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AsciiGrid:
    x: Floats  # nx cell centres
    y: Floats  # ny cell centres, ascending
    cell: float
    z: Floats  # ny x nx, row 0 = lowest y, NaN = no data


def read_ascii_grid(path: Path) -> AsciiGrid:
    """ESRI ASCII grid (``.asc``): corner or centre registration, NODATA cells."""
    header: dict[str, float] = {}
    with Path(path).open(encoding="ascii", errors="replace") as f:
        lines = f.read().split("\n")
    k = 0
    while k < len(lines):
        parts = lines[k].split()
        if len(parts) == 2 and parts[0][0].isalpha():
            header[parts[0].lower()] = float(parts[1])
            k += 1
            continue
        if not parts:
            k += 1
            continue
        break
    try:
        nx, ny, cell = int(header["ncols"]), int(header["nrows"]), header["cellsize"]
    except KeyError as exc:
        raise SurfaceEditError(f"{Path(path).name}: missing grid header {exc}") from exc
    if not (nx > 0 and ny > 0 and cell > 0):
        raise SurfaceEditError(f"{Path(path).name}: invalid grid size")
    if "xllcenter" in header:
        x0, y0 = header["xllcenter"], header["yllcenter"]
    else:
        x0, y0 = header["xllcorner"] + cell / 2, header["yllcorner"] + cell / 2
    values = np.array(" ".join(lines[k:]).split(), np.float64)
    if len(values) != nx * ny:
        raise SurfaceEditError(f"{Path(path).name}: {len(values)} values for a {nx} x {ny} grid")
    z = values.reshape(ny, nx)[::-1].copy()  # file is north row first
    nodata = header.get("nodata_value")
    if nodata is not None:
        z[z == nodata] = np.nan
    xs = np.asarray(x0 + cell * np.arange(nx), np.float64)
    ys = np.asarray(y0 + cell * np.arange(ny), np.float64)
    return AsciiGrid(xs, ys, float(cell), z)


def grid_surface(grid: AsciiGrid, tolerance: float | None = None) -> tuple[Surface, int]:
    """TIN through the cell centres with data (holes at no-data cells).

    ``tolerance`` (m, optional): decimate to that vertical accuracy. Returns
    the surface and the number of grid points with data.
    """
    gx, gy = np.meshgrid(grid.x, grid.y)
    ok = np.isfinite(grid.z)
    pts = np.column_stack([gx[ok], gy[ok], grid.z[ok]])
    if len(pts) < 3:
        raise SurfaceEditError("the grid has fewer than three cells with data")
    holes = not ok.all()
    # Cell triangles only: a diagonal is cell * sqrt(2); longer edges span no-data.
    s = Surface(pts, (), grid.cell * math.sqrt(2.0) * 1.0001 if holes else None)
    if tolerance is not None:
        s, _ = decimate_surface(s, tolerance)
    return s, len(pts)
