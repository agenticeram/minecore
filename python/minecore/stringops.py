"""String (polyline) editing operations: the CAD toolbox for strings (E-CD-01).

All operations take and return :class:`minecore.strings.String` values and
never change their input. Plan geometry (x, y) drives every operation; z is
carried along by linear interpolation unless an operation says otherwise.

Operations
----------
* ``reverse``, ``split``, ``join``, ``set_closed``: topology edits.
* ``measure`` / ``segment_table``: plan and slope length, plan area of closed
  strings, bearing (degrees clockwise from grid north), gradient and dip of
  every segment.
* ``simplify``: Douglas-Peucker in 3-D with a distance tolerance; end points
  (and, for closed strings, the two vertices farthest apart) are kept.
* ``resample``: points at a fixed spacing along the slope length.
* ``smooth``: Chaikin corner cutting; open strings keep their end points.
* ``intersections``: plan crossings of two strings with z on each string.
* ``trim_extend``: cut an end back to, or extend it forward to, a boundary
  string.
* ``boolean``: plan union, intersection, difference and xor of sets of closed
  strings (even-odd rule; results have holes clockwise).
* ``fillet`` / ``chamfer``: round or cut every corner.
* ``offset``: parallel string at a plan distance with mitre or round joins and
  an optional vertical shift (constant ``dz`` or a slope angle), as used to
  move a toe to a crest or a berm. Self-intersection loops created by
  concave corners are removed: a loop is dropped when it runs the wrong way
  round or when any of its points lies closer to the source than the offset
  distance. A closed string offset inwards may split into several strings or
  vanish.

Sign convention for ``offset``: a positive distance moves a closed string
outwards and an open string to its left (looking along the digitising
direction); a negative distance moves it inwards / to the right.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from minecore.strings import String, StringError

Floats = npt.NDArray[np.float64]
Join = Literal["mitre", "round"]
_EPS = 1e-9
SLIVER_AREA = 1e-6  # m2
MITRE_LIMIT = 10.0  # mitre length / offset distance above which a corner is rounded


# -- helpers ----------------------------------------------------------------


def _clean(points: npt.ArrayLike, closed: bool) -> Floats:
    """n x 3 float array without repeated points (and without a closing duplicate)."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    if len(p) == 0:
        return p
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > _EPS]
    p = p[keep]
    if closed and len(p) > 1 and np.linalg.norm(p[0] - p[-1]) <= _EPS:
        p = p[:-1]
    return p


def _ring(p: Floats, closed: bool) -> Floats:
    """Vertices with the first repeated at the end when closed (segment walk)."""
    return np.vstack([p, p[:1]]) if closed and len(p) > 1 else p


def _new(points: Floats, closed: bool, layer: str) -> String:
    return String(np.ascontiguousarray(points, np.float64), closed, layer)


def _signed_area(xy: Floats) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _need(s: String, n: int = 2) -> Floats:
    p = _clean(s.points, s.closed)
    if len(p) < n:
        raise StringError(f"the string needs at least {n} distinct points")
    return p


# -- topology -----------------------------------------------------------------


def reverse(s: String) -> String:
    """Same string digitised the other way."""
    p = _clean(s.points, s.closed)
    if s.closed and len(p) > 1:
        return _new(np.vstack([p[:1], p[:0:-1]]), True, s.layer)
    return _new(p[::-1], s.closed, s.layer)


def set_closed(s: String, closed: bool) -> String:
    p = _clean(s.points, s.closed)
    if closed and len(p) < 3:
        raise StringError("a closed string needs at least three distinct points")
    return _new(p, closed, s.layer)


def split(s: String, vertex: int) -> list[String]:
    """Cut at a vertex. An open string gives two strings sharing that vertex;
    a closed string is opened there (one string starting and ending at it)."""
    p = _clean(s.points, s.closed)
    n = len(p)
    if not -n <= vertex < n:
        raise StringError(f"vertex {vertex} is outside the string (0..{n - 1})")
    k = vertex % n
    if s.closed:
        q = np.roll(p, -k, axis=0)
        return [_new(np.vstack([q, q[:1]]), False, s.layer)]
    if k in (0, n - 1):
        raise StringError("an open string cannot be split at its end point")
    return [_new(p[: k + 1], False, s.layer), _new(p[k:], False, s.layer)]


def join(a: String, b: String, tolerance: float = math.inf) -> String:
    """Join two open strings at their nearest end points (plan distance).

    Coincident ends are merged; otherwise a connecting segment is added. A
    gap larger than ``tolerance`` is refused."""
    if a.closed or b.closed:
        raise StringError("only open strings can be joined")
    p, q = _need(a, 1), _need(b, 1)
    ends = {
        ("end", "start"): (p, q),
        ("end", "end"): (p, q[::-1]),
        ("start", "start"): (p[::-1], q),
        ("start", "end"): (q, p),
    }
    best = min(ends.values(), key=lambda pq: float(np.linalg.norm(pq[0][-1, :2] - pq[1][0, :2])))
    first, second = best
    gap = float(np.linalg.norm(first[-1, :2] - second[0, :2]))
    if gap > tolerance:
        raise StringError(f"nearest ends are {gap:.3f} m apart, more than {tolerance:g} m")
    joined = np.vstack([first, second])
    return _new(_clean(joined, False), False, a.layer)


# -- measurement --------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    start: int
    plan_length: float
    slope_length: float
    bearing: float  # degrees clockwise from grid north, 0..360
    gradient: float  # rise / run (dimensionless); nan for a vertical segment
    dip: float  # degrees from horizontal, positive upwards


@dataclass(frozen=True)
class Measure:
    points: int
    closed: bool
    plan_length: float
    slope_length: float
    area: float  # plan area, closed strings only (else 0)
    z_min: float
    z_max: float


def segment_table(s: String) -> list[Segment]:
    p = _ring(_need(s), s.closed)
    out = []
    for k in range(len(p) - 1):
        d = p[k + 1] - p[k]
        run = math.hypot(d[0], d[1])
        out.append(
            Segment(
                start=k,
                plan_length=run,
                slope_length=float(np.linalg.norm(d)),
                bearing=math.degrees(math.atan2(d[0], d[1])) % 360.0 if run > _EPS else math.nan,
                gradient=d[2] / run if run > _EPS else math.nan,
                dip=math.degrees(math.atan2(d[2], run)),
            )
        )
    return out


def measure(s: String) -> Measure:
    p = _clean(s.points, s.closed)
    if len(p) == 0:
        raise StringError("the string is empty")
    w = _ring(p, s.closed)
    d = np.diff(w, axis=0)
    return Measure(
        points=len(p),
        closed=s.closed,
        plan_length=float(np.hypot(d[:, 0], d[:, 1]).sum()),
        slope_length=float(np.linalg.norm(d, axis=1).sum()),
        area=abs(_signed_area(p[:, :2])) if s.closed and len(p) >= 3 else 0.0,
        z_min=float(p[:, 2].min()),
        z_max=float(p[:, 2].max()),
    )


# -- shape --------------------------------------------------------------------


def _dp_keep(p: Floats, tol: float) -> npt.NDArray[np.bool_]:
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        len2 = float(ab @ ab)
        q = p[i + 1 : j] - a
        if len2 <= _EPS * _EPS:
            dist = np.linalg.norm(q, axis=1)
        else:
            t = np.clip(q @ ab / len2, 0.0, 1.0)
            dist = np.linalg.norm(q - t[:, None] * ab, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return keep


def simplify(s: String, tolerance: float) -> String:
    """Douglas-Peucker: drop vertices closer than ``tolerance`` (3-D distance)
    to the simplified string."""
    if tolerance < 0:
        raise StringError("tolerance must not be negative")
    p = _need(s)
    if not s.closed:
        return _new(p[_dp_keep(p, tolerance)], False, s.layer)
    # Anchor a closed string at the two vertices farthest apart in plan.
    j = int(np.argmax(np.linalg.norm(p[:, :2] - p[0, :2], axis=1)))
    i = int(np.argmax(np.linalg.norm(p[:, :2] - p[j, :2], axis=1)))
    q = np.roll(p, -i, axis=0)
    m = (j - i) % len(p)
    w = np.vstack([q, q[:1]])
    keep = np.zeros(len(w), bool)
    keep[: m + 1] |= _dp_keep(w[: m + 1], tolerance)
    keep[m:] |= _dp_keep(w[m:], tolerance)
    out = w[keep][:-1]
    if len(out) < 3:
        raise StringError("the tolerance removes the whole closed string")
    return _new(out, True, s.layer)


def resample(s: String, spacing: float) -> String:
    """Points every ``spacing`` metres of slope length; the last point is kept."""
    if spacing <= 0:
        raise StringError("spacing must be positive")
    p = _need(s)
    w = _ring(p, s.closed)
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = float(cum[-1])
    stations = np.arange(0.0, total, spacing)
    if s.closed:
        if total - stations[-1] < 1e-6 * spacing and len(stations) > 1:
            stations = stations[:-1]
    else:
        stations = np.r_[stations, total] if total - stations[-1] > 1e-9 else stations
    out = np.column_stack([np.interp(stations, cum, w[:, c]) for c in range(3)])
    if s.closed and len(out) < 3:
        raise StringError("the spacing is too long for this closed string")
    return _new(out, s.closed, s.layer)


def smooth(s: String, iterations: int = 1) -> String:
    """Chaikin corner cutting (each pass replaces a corner by two points at a
    quarter and three quarters of its segments). Open strings keep their ends."""
    if iterations < 0:
        raise StringError("iterations must not be negative")
    p = _need(s, 3 if s.closed else 2)
    for _ in range(iterations):
        w = _ring(p, s.closed)
        a, b = w[:-1], w[1:]
        q = np.empty((2 * len(a), 3))
        q[0::2] = 0.75 * a + 0.25 * b
        q[1::2] = 0.25 * a + 0.75 * b
        p = q if s.closed else np.vstack([w[:1], q[1:-1], w[-1:]])
    return _new(p, s.closed, s.layer)


# -- intersections and trim/extend -----------------------------------------------


@dataclass(frozen=True)
class Crossing:
    x: float
    y: float
    z_a: float
    z_b: float
    segment_a: int
    t_a: float  # position along segment_a, 0..1
    segment_b: int
    t_b: float


def _segments(p: Floats, closed: bool) -> tuple[Floats, Floats]:
    w = _ring(p, closed)
    return w[:-1], w[1:]


def _cross2(u: Floats, v: Floats) -> Floats:
    return np.asarray(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], np.float64)


def _candidate_pairs(
    a0: Floats, a1: Floats, b0: Floats, b1: Floats
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Index pairs of segments whose plan bounding circles overlap."""
    ma, mb = (a0[:, :2] + a1[:, :2]) / 2, (b0[:, :2] + b1[:, :2]) / 2
    ra = np.linalg.norm(a1[:, :2] - a0[:, :2], axis=1) / 2
    rb = np.linalg.norm(b1[:, :2] - b0[:, :2], axis=1) / 2
    if len(a0) * len(b0) <= 250_000:
        i, j = np.indices((len(a0), len(b0))).reshape(2, -1)
    else:
        found = cKDTree(mb).query_ball_point(ma, ra + float(rb.max()) + 1e-9)
        i = np.repeat(np.arange(len(a0)), [len(f) for f in found])
        j = np.fromiter((k for f in found for k in f), np.int64, count=len(i))
    near = np.hypot(*(ma[i] - mb[j]).T) <= ra[i] + rb[j] + 1e-9
    return i[near].astype(np.int64), j[near].astype(np.int64)


def _seg_hits(
    a0: Floats, a1: Floats, b0: Floats, b1: Floats
) -> list[tuple[int, int, float, float]]:
    """All proper plan crossings between segment lists (parallel overlaps skipped)."""
    i, j = _candidate_pairs(a0, a1, b0, b1)
    r = (a1 - a0)[i, :2]
    s_ = (b1 - b0)[j, :2]
    qp = (b0[j] - a0[i])[:, :2]
    den = _cross2(r, s_)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross2(qp, s_) / den
        u = _cross2(qp, r) / den
    tol = 1e-12
    ok = (np.abs(den) > tol) & (t >= -tol) & (t <= 1 + tol) & (u >= -tol) & (u <= 1 + tol)
    t, u = np.clip(t[ok], 0, 1), np.clip(u[ok], 0, 1)
    return sorted(zip(i[ok].tolist(), j[ok].tolist(), t.tolist(), u.tolist(), strict=True))


def intersections(a: String, b: String) -> list[Crossing]:
    """Plan crossings of two strings, ordered along ``a``."""
    p, q = _need(a), _need(b)
    a0, a1 = _segments(p, a.closed)
    b0, b1 = _segments(q, b.closed)
    out: list[Crossing] = []
    seen: set[tuple[float, float]] = set()
    for i, j, t, u in sorted(_seg_hits(a0, a1, b0, b1), key=lambda h: (h[0], h[2])):
        pt = a0[i] + t * (a1[i] - a0[i])
        key = (round(float(pt[0]), 6), round(float(pt[1]), 6))
        if key in seen:  # a crossing exactly at a shared vertex is found twice
            continue
        seen.add(key)
        zb = b0[j, 2] + u * (b1[j, 2] - b0[j, 2])
        out.append(Crossing(float(pt[0]), float(pt[1]), float(pt[2]), float(zb), i, t, j, u))
    return out


def trim_extend(
    s: String,
    boundary: String,
    end: Literal["start", "end", "both"] = "end",
    max_extension: float = math.inf,
) -> String:
    """Trim or extend an open string's end(s) to a boundary string.

    If the string crosses the boundary, the end is cut back to the crossing
    nearest that end. Otherwise the end segment is extended along its plan
    direction to the first boundary crossing within ``max_extension``; z
    continues at the end segment's gradient."""
    if s.closed:
        raise StringError("only open strings can be trimmed or extended")
    p = _need(s)
    if end == "both":
        return trim_extend(
            trim_extend(s, boundary, "start", max_extension), boundary, "end", max_extension
        )
    if end == "start":
        return reverse(trim_extend(reverse(s), boundary, "end", max_extension))
    q = _need(boundary)
    hits = intersections(_new(p, False, ""), _new(q, boundary.closed, ""))
    if hits:
        h = max(hits, key=lambda c: (c.segment_a, c.t_a))
        cut = np.array([h.x, h.y, h.z_a])
        out = np.vstack([p[: h.segment_a + 1], cut])
        return _new(_clean(out, False), False, s.layer)
    a, b = p[-2], p[-1]
    d = b - a
    run = math.hypot(d[0], d[1])
    if run <= _EPS:
        raise StringError("the end segment has no plan length to extend along")
    span = np.vstack([p[:, :2], q[:, :2]])
    reach = 2.0 * float(np.linalg.norm(span.max(axis=0) - span.min(axis=0))) + 1.0
    far = min(max_extension, reach)
    ray_end = b + d / run * far
    b0, b1 = _segments(q, boundary.closed)
    cands = _seg_hits(np.array([b]), np.array([ray_end]), b0, b1)
    cands = [c for c in cands if c[2] * far > 1e-9]
    if not cands:
        raise StringError("the extended end does not reach the boundary")
    t = min(c[2] for c in cands)
    return _new(np.vstack([p, b + t * (ray_end - b)]), False, s.layer)


# -- offset -------------------------------------------------------------------


def _point_to_polyline(pts: Floats, poly: Floats, closed: bool) -> Floats:
    """Plan distance from each point to a polyline."""
    a0, a1 = _segments(poly, closed)
    a, ab = a0[None, :, :2], (a1 - a0)[None, :, :2]
    q = pts[:, None, :2] - a
    len2 = np.maximum((ab * ab).sum(axis=2), 1e-300)
    t = np.clip((q * ab).sum(axis=2) / len2, 0.0, 1.0)
    d = q - t[..., None] * ab
    return np.asarray(np.sqrt((d * d).sum(axis=2)).min(axis=1), np.float64)


def _raw_offset(p: Floats, closed: bool, d: float, join: Join, arc_step: float) -> Floats:
    """Offset each segment to the left by ``d`` and connect the pieces.

    Round joins are drawn as polylines circumscribed about the arc (every
    chord touches the circle), so no point of a join is closer than ``|d|``
    to its corner. Across the inside of a bend the two offset ends are simply
    linked; :func:`_clip_band` removes that link and every other part that
    comes closer than ``|d|`` to the source."""
    w = _ring(p, closed)
    a, b = w[:-1], w[1:]
    t = (b - a)[:, :2]
    t /= np.linalg.norm(t, axis=1)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])  # left normal
    pa = a[:, :2] + d * n
    pb = b[:, :2] + d * n
    m = len(a)
    pts: list[list[float]] = []

    def add(xy: Floats, z: float) -> None:
        pts.append([float(xy[0]), float(xy[1]), z])

    if not closed:
        add(pa[0], float(a[0, 2]))
    for k in range(m) if closed else range(1, m):
        i, j = (k - 1) % m, k  # segment i ends where segment j starts
        corner = a[j]
        z = float(corner[2])
        turn = float(_cross2(t[i], t[j]))
        reversal = abs(turn) < 1e-12 and float(t[i] @ t[j]) < 0
        if not (turn * d < 0 or reversal):
            # inside of the bend: link the two offset ends through the corner
            # (the winding fill of a closed string and the distance clip of an
            # open one both remove this link)
            add(pb[i], z)
            if abs(turn) > 1e-12:
                add(corner[:2], z)
            add(pa[j], z)
            continue
        if join == "mitre" and not reversal:
            s_ = float(_cross2(pa[j] - pb[i], t[j])) / turn
            mitre = pb[i] + s_ * t[i]
            if np.linalg.norm(mitre - corner[:2]) <= MITRE_LIMIT * abs(d):
                add(mitre, z)
                continue  # else: very sharp corner, round it instead
        a0 = math.atan2(*(pb[i] - corner[:2])[::-1])
        # the normals turn with the tangents: sweep through the turn angle
        sweep = -math.pi if d > 0 else math.pi
        if not reversal:
            sweep = math.atan2(turn, float(t[i] @ t[j]))
        steps = max(1, math.ceil(abs(sweep) / arc_step))
        step = sweep / steps
        radius = abs(d) / math.cos(step / 2)
        add(pb[i], z)
        for k2 in range(steps):
            ang = a0 + step * (k2 + 0.5)
            add(corner[:2] + radius * np.array([math.cos(ang), math.sin(ang)]), z)
        add(pa[j], z)
    if not closed:
        add(pb[-1], float(b[-1, 2]))
    return _clean(np.array(pts, np.float64).reshape(-1, 3), closed)


def _first_self_hit(p: Floats, closed: bool) -> tuple[int, int, float, float] | None:
    """First crossing between two non-neighbouring segments (None if simple)."""
    a0, a1 = _segments(p, closed)
    m = len(a0)
    if m < 3:
        return None
    for i, j, t, u in sorted(_seg_hits(a0, a1, a0, a1)):
        if j > i + 1 and not (closed and i == 0 and j == m - 1):
            return i, j, t, u
    return None


def _slab(
    s0: Floats, ds: Floats, low: Floats | float, high: Floats | float
) -> tuple[Floats, Floats]:
    """Parameter range where ``low < s0 + u * ds < high`` (empty as lo > hi)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        e0 = (low - s0) / ds
        e1 = (high - s0) / ds
    flat = np.abs(ds) < 1e-300
    inside = (s0 > low) & (s0 < high)
    lo = np.where(flat, np.where(inside, -np.inf, np.inf), np.minimum(e0, e1))
    hi = np.where(flat, np.where(inside, np.inf, -np.inf), np.maximum(e0, e1))
    return lo, hi


def _band_intervals(
    p0: Floats, p1: Floats, src0: Floats, src1: Floats, r: float
) -> list[tuple[float, float]]:
    """Parameter intervals of segment p0-p1 (plan) lying strictly within ``r``
    of any source segment: the segment clipped by each source capsule (two
    end discs and the rectangle between them; their union is convex, so the
    clipped part is one interval per capsule)."""
    dv = p1 - p0
    dd = float(dv @ dv)
    if dd <= 0:
        return []
    lo = np.full(len(src0), np.inf)
    hi = np.full(len(src0), -np.inf)
    for c in (src0, src1):  # end discs
        f = p0 - c
        bq = f @ dv
        disc = bq * bq - dd * ((f * f).sum(axis=1) - r * r)
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        lo = np.where(ok, np.minimum(lo, (-bq - sq) / dd), lo)
        hi = np.where(ok, np.maximum(hi, (-bq + sq) / dd), hi)
    ab = src1 - src0
    length = np.hypot(ab[:, 0], ab[:, 1])
    tt = ab / np.maximum(length, 1e-300)[:, None]
    nn = np.column_stack([-tt[:, 1], tt[:, 0]])
    rel = p0 - src0
    l1, h1 = _slab((rel * tt).sum(axis=1), tt @ dv, 0.0, length)
    l2, h2 = _slab((rel * nn).sum(axis=1), nn @ dv, -r, r)
    u0 = np.maximum(np.maximum(l1, l2), 0.0)
    u1 = np.minimum(np.minimum(h1, h2), 1.0)
    rect = (u0 < u1) & (length > 0)
    lo = np.where(rect, np.minimum(lo, u0), lo)
    hi = np.where(rect, np.maximum(hi, u1), hi)
    lo, hi = np.maximum(lo, 0.0), np.minimum(hi, 1.0)
    keep = lo < hi
    return list(zip(lo[keep].tolist(), hi[keep].tolist(), strict=True))


def _clip_band(
    raw: Floats, src: Floats, closed_src: bool, closed: bool, dist: float
) -> list[tuple[Floats, bool]]:
    """Remove every part of the raw offset that comes closer than ``dist`` to
    the source, then chain what is left into strings (loop flag per chain)."""
    r = dist * (1 - 1e-9)
    s0, s1 = _segments(src, closed_src)
    a0, a1 = _segments(raw, closed)
    pieces: list[list[Floats]] = []
    current: list[Floats] | None = None
    for k in range(len(a0)):
        p0, p1 = a0[k], a1[k]
        cut = sorted(_band_intervals(p0[:2], p1[:2], s0[:, :2], s1[:, :2], r))
        free: list[tuple[float, float]] = []
        u = 0.0
        for lo, hi in cut:
            if lo > u:
                free.append((u, lo))
            u = max(u, hi)
        if u < 1.0:
            free.append((u, 1.0))
        for f0, f1 in free:
            q0, q1 = p0 + f0 * (p1 - p0), p0 + f1 * (p1 - p0)
            if current is not None and f0 == 0.0:
                current.append(q1)
            else:
                current = [q0, q1]
                pieces.append(current)
            if f1 < 1.0:
                current = None
        if not free:
            current = None
    if closed and len(pieces) > 1 and current is pieces[-1] and np.allclose(pieces[0][0], a0[0]):
        pieces[0] = pieces.pop()[:-1] + pieces[0]  # the walk wrapped round the start
    scale = max(float(np.abs(raw[:, :2]).max()), dist, 1.0)
    tol = 1e-7 * scale
    left = [np.array(pc) for pc in pieces if len(pc) >= 2]
    left = [pc for pc in left if np.linalg.norm(np.diff(pc[:, :2], axis=0), axis=1).sum() > tol]
    return _chain(left, tol)


def _winding(q: Floats, ring: Floats) -> npt.NDArray[np.int64]:
    """Winding number of each plan point about a closed ring.

    Edges are binned into horizontal bands so each point is tested only
    against the edges spanning its band."""
    a = ring[:, :2]
    b = np.roll(a, -1, axis=0)
    out = np.zeros(len(q), np.int64)
    lo_y, hi_y = float(min(a[:, 1].min(), q[:, 1].min())), float(max(a[:, 1].max(), q[:, 1].max()))
    bands = max(1, min(512, len(a) // 8))
    width = (hi_y - lo_y) / bands or 1.0
    qb = np.clip(((q[:, 1] - lo_y) / width).astype(np.int64), 0, bands - 1)
    e_lo = np.clip(((np.minimum(a[:, 1], b[:, 1]) - lo_y) / width).astype(np.int64), 0, bands - 1)
    e_hi = np.clip(((np.maximum(a[:, 1], b[:, 1]) - lo_y) / width).astype(np.int64), 0, bands - 1)
    for band in np.unique(qb):
        rows = np.nonzero(qb == band)[0]
        cols = np.nonzero((e_lo <= band) & (e_hi >= band))[0]
        if not len(cols):
            continue
        ea, eb = a[cols], b[cols]
        for c in range(0, len(rows), 512):
            r = rows[c : c + 512]
            qx, qy = q[r, 0][:, None], q[r, 1][:, None]
            left = (eb[:, 0] - ea[:, 0]) * (qy - ea[:, 1]) - (qx - ea[:, 0]) * (eb[:, 1] - ea[:, 1])
            up = (ea[:, 1] <= qy) & (eb[:, 1] > qy) & (left > 0)
            down = (ea[:, 1] > qy) & (eb[:, 1] <= qy) & (left < 0)
            out[r] = up.sum(axis=1) - down.sum(axis=1)
    return out


def _winding_fill(raw: Floats) -> list[Floats]:
    """Outline of the region the counter-clockwise raw ring winds round a
    positive number of times (the "positive" fill rule used for polygon
    offsetting): split the ring at its self-crossings and keep each piece
    that has winding > 0 on its left and <= 0 on its right."""
    a0, a1 = _segments(raw, True)
    m = len(a0)
    cuts: list[list[tuple[float, Floats]]] = [[] for _ in range(m)]
    for i, j, t, u in _seg_hits(a0, a1, a0, a1):
        if j <= i or j == i + 1 or (i == 0 and j == m - 1):
            continue
        x = a0[i] + t * (a1[i] - a0[i])
        cuts[i].append((t, x))
        cuts[j].append((u, x))
    subs: list[tuple[Floats, Floats]] = []
    parent: list[int] = []
    for k in range(m):
        stops = [a0[k]] + [x for _t, x in sorted(cuts[k], key=lambda c: c[0])] + [a1[k]]
        for p0, p1 in itertools.pairwise(stops):
            if np.linalg.norm(p1[:2] - p0[:2]) > 1e-12:
                subs.append((p0, p1))
                parent.append(k)
    if not subs:
        return []
    scale = max(float(np.abs(raw[:, :2]).max()), 1.0)
    p0s = np.array([s_[0][:2] for s_ in subs])
    p1s = np.array([s_[1][:2] for s_ in subs])
    mid = (p0s + p1s) / 2
    tan = p1s - p0s
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    # Sample the winding just left and right of each piece, closer than any
    # other part of the ring (thin loops at shallow bends are very narrow).
    cap = 1e-7 * scale
    box = np.column_stack([mid, np.zeros(len(mid))])
    reach = np.array([cap * 4, 0.0, 0.0])
    ci, cj = _candidate_pairs(box - reach, box + reach, a0, a1)
    keep = cj != np.asarray(parent)[ci]
    ci, cj = ci[keep], cj[keep]
    ab = (a1 - a0)[cj, :2]
    rel = mid[ci] - a0[cj, :2]
    tt = np.clip((rel * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
    gap = np.full(len(subs), np.inf)
    np.minimum.at(gap, ci, np.linalg.norm(rel - tt[:, None] * ab, axis=1))
    eps = np.minimum(cap, 0.25 * gap)[:, None]
    eps = np.maximum(eps, 1e-13 * scale)
    w_left = _winding(mid + eps * nrm, raw)
    w_right = _winding(mid - eps * nrm, raw)
    kept = [subs[k] for k in np.nonzero((w_left > 0) & (w_right <= 0))[0]]
    chains = _chain([np.array(e) for e in kept], 1e-6 * scale)
    return [c for c, loop in chains if loop]


def _chain(pieces: list[Floats], tol: float) -> list[tuple[Floats, bool]]:
    """Join pieces end to start into strings; flag the ones that close.

    Chains start at pieces nobody leads into, so open strings come out whole."""
    if not pieces:
        return []
    starts = cKDTree(np.array([pc[0, :2] for pc in pieces]))
    ends = cKDTree(np.array([pc[-1, :2] for pc in pieces]))
    heads = [i for i, pc in enumerate(pieces) if not ends.query_ball_point(pc[0, :2], tol)]
    head_set = set(heads)
    used = np.zeros(len(pieces), bool)
    out: list[tuple[Floats, bool]] = []
    for first in heads + list(range(len(pieces))):
        if used[first]:
            continue
        used[first] = True
        parts = [pieces[first]]
        head = pieces[first][0, :2]
        count = len(pieces[first])
        while True:
            tail = parts[-1][-1, :2]
            if count > 2 and np.hypot(*(tail - head)) <= tol and first not in head_set:
                out.append((np.vstack([parts[0], *(pc[1:] for pc in parts[1:])])[:-1], True))
                break
            near = [k for k in starts.query_ball_point(tail, tol) if not used[k]]
            if not near:
                out.append((np.vstack([parts[0], *(pc[1:] for pc in parts[1:])]), False))
                break
            k = min(near, key=lambda c: float(np.hypot(*(pieces[c][0, :2] - tail))))
            used[k] = True
            parts.append(pieces[k])
            count += len(pieces[k]) - 1
    return out


def _untangle_open(p: Floats) -> Floats:
    """Cut out the loops of a self-crossing open string (mitre spikes)."""
    for _ in range(10_000):
        hit = _first_self_hit(p, False)
        if hit is None:
            return p
        i, j, t, _u = hit
        x = p[i] + t * (p[i + 1] - p[i])
        p = _clean(np.vstack([p[: i + 1], x, p[j + 1 :]]), False)
    return p


def offset(  # noqa: PLR0913 - keyword-only options
    s: String,
    distance: float,
    *,
    join: Join = "mitre",
    dz: float = 0.0,
    slope_deg: float | None = None,
    arc_step_deg: float = 10.0,
) -> list[String]:
    """Plan offset by ``distance`` (see the module sign convention).

    z of the result is the source z plus ``dz``; with ``slope_deg`` the shift
    is ``|distance| * tan(slope_deg)`` instead (positive angle goes up).
    Returns the offset strings, longest first for an open input; a closed
    input gives zero or more closed strings. No point of the result is closer
    than ``|distance|`` to the source."""
    if distance == 0:
        raise StringError("the offset distance must not be zero")
    if not 0.1 <= arc_step_deg <= 45:
        raise StringError("the arc step must be between 0.1 and 45 degrees")
    if join not in ("mitre", "round"):
        raise StringError("join must be 'mitre' or 'round'")
    p = _need(s, 3 if s.closed else 2)
    if slope_deg is not None:
        if not -89.0 <= slope_deg <= 89.0:
            raise StringError("the slope angle must be between -89 and 89 degrees")
        dz = abs(distance) * math.tan(math.radians(slope_deg))
    origin = np.array([*p[:, :2].mean(axis=0), 0.0])  # work in local coordinates
    q = p - origin
    left = distance
    sign = 1.0
    if s.closed:
        area = _signed_area(q[:, :2])
        if abs(area) <= _EPS:
            raise StringError("the closed string has no plan area")
        sign = 1.0 if area > 0 else -1.0
        left = -distance * sign  # outward is to the right of a counter-clockwise ring
    raw = _raw_offset(q, s.closed, left, join, math.radians(arc_step_deg))
    shift = origin + np.array([0.0, 0.0, dz])
    if s.closed:
        ccw = raw if sign > 0 else raw[::-1]
        rings = [_clean(r, True) for r in _winding_fill(ccw)]
        # slivers left by round-off at near-collinear crossings have no area
        rings = [r for r in rings if len(r) >= 3 and _signed_area(r[:, :2]) > SLIVER_AREA]
        rings = [r if sign > 0 else r[::-1] for r in rings]
        rings.sort(key=lambda r: -abs(_signed_area(r[:, :2])))
        return [_new(r + shift, True, s.layer) for r in rings]
    chains = _clip_band(raw, q, False, False, abs(distance))
    lines = [_untangle_open(_clean(c, False)) for c, loop in chains if not loop]
    lines = [ln for ln in lines if len(ln) >= 2]
    if not lines:
        raise StringError("the offset leaves nothing of the string")
    lines.sort(key=lambda ln: -float(np.linalg.norm(np.diff(ln[:, :2], axis=0), axis=1).sum()))
    return [_new(ln + shift, False, s.layer) for ln in lines]


# -- polygon booleans ----------------------------------------------------------

BoolOp = Literal["union", "intersection", "difference", "xor"]
Mask = npt.NDArray[np.bool_]
_BOOL_RULES: dict[str, Callable[[Mask, Mask], Mask]] = {
    "union": np.logical_or,
    "intersection": np.logical_and,
    "difference": lambda a, b: np.logical_and(a, np.logical_not(b)),
    "xor": np.logical_xor,
}


def _drop_collinear(r: Floats, tol: float) -> Floats:
    """Remove vertices lying on the straight 3-D line between their neighbours."""
    while len(r) > 3:
        prev, nxt = np.roll(r, 1, axis=0), np.roll(r, -1, axis=0)
        u, v = r - prev, nxt - prev
        lv = np.linalg.norm(v, axis=1)
        dist = np.linalg.norm(np.cross(u, v), axis=1) / np.maximum(lv, 1e-300)
        between = (u * v).sum(axis=1) > 0
        drop = np.nonzero((dist <= tol) & between & (lv > 0))[0]
        if not len(drop):
            break
        r = np.delete(r, drop[0], axis=0)
    return r


def _cut_at_contacts(seg0: Floats, seg1: Floats, tol: float) -> tuple[Floats, Floats]:
    """Split segments at every crossing and at every vertex lying on them, so
    that boundaries meet only at piece ends."""
    m = len(seg0)
    cuts: list[list[float]] = [[] for _ in range(m)]
    for i, j, t, _u in _seg_hits(seg0, seg1, seg0, seg1):
        if i == j:
            continue
        li = float(np.linalg.norm(seg1[i, :2] - seg0[i, :2]))
        if t * li > tol and (1 - t) * li > tol:
            cuts[i].append(t)
    # vertices lying on other segments (touching and overlapping boundaries)
    ci, cj = _candidate_pairs(seg0, seg0, seg0, seg1)
    ab = (seg1 - seg0)[cj, :2]
    length = np.sqrt((ab * ab).sum(axis=1))
    tt = ((seg0[ci, :2] - seg0[cj, :2]) * ab).sum(axis=1) / np.maximum(length**2, 1e-300)
    foot = seg0[cj, :2] + tt[:, None] * ab
    near = np.hypot(*(seg0[ci, :2] - foot).T) <= tol
    inner = (tt * length > tol) & ((1 - tt) * length > tol)
    for k in np.nonzero(near & inner)[0]:
        cuts[cj[k]].append(float(tt[k]))
    p0s: list[Floats] = []
    p1s: list[Floats] = []
    for k in range(m):
        ts = np.unique(np.r_[0.0, np.clip(cuts[k], 0, 1), 1.0])
        pts = seg0[k] + ts[:, None] * (seg1[k] - seg0[k])
        for s0, s1 in itertools.pairwise(pts):
            if np.linalg.norm(s1[:2] - s0[:2]) > tol:
                p0s.append(s0)
                p1s.append(s1)
    return np.array(p0s), np.array(p1s)


def _side_offset(mid: Floats, seg0: Floats, seg1: Floats, scale: float, tol: float) -> Floats:
    """Distance to sample each side of a piece: closer than any other boundary
    that does not run through the piece's midpoint."""
    cap = 1e-7 * scale
    box = np.column_stack([mid, np.zeros(len(mid))])
    reach = np.array([cap * 4, 0.0, 0.0])
    ci, cj = _candidate_pairs(box - reach, box + reach, seg0, seg1)
    ab = (seg1 - seg0)[cj, :2]
    rel = mid[ci] - seg0[cj, :2]
    tt = np.clip((rel * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
    dist = np.linalg.norm(rel - tt[:, None] * ab, axis=1)
    gap = np.full(len(mid), np.inf)
    far = dist > tol
    np.minimum.at(gap, ci[far], dist[far])
    return np.asarray(np.maximum(np.minimum(cap, 0.25 * gap), 1e-13 * scale), np.float64)


def is_outer(s: String) -> bool:
    """True for a counter-clockwise closed string (an outer boundary in
    :func:`boolean` results); False for a clockwise one (a hole)."""
    return _signed_area(_need(s, 3)[:, :2]) > 0


def boolean(a: list[String], b: list[String], op: BoolOp) -> list[String]:
    """Plan boolean of two sets of closed strings.

    Each set is a region under the even-odd rule: a string nested inside
    another of the same set is a hole. The result is a list of closed strings,
    outer boundaries counter-clockwise and holes clockwise, largest first.
    Points on the result carry z interpolated from the source string they come
    from; where boundaries coincide, z is taken from one of them."""
    if op not in _BOOL_RULES:
        raise StringError("op must be union, intersection, difference or xor")
    rings: list[Floats] = []
    owner: list[int] = []
    for side, group in enumerate((a, b)):
        for s in group:
            if not s.closed:
                raise StringError("boolean operations need closed strings")
            p = _need(s, 3)
            if abs(_signed_area(p[:, :2] - p[0, :2])) <= _EPS:
                raise StringError("a closed string has no plan area")
            rings.append(p)
            owner.append(side)
    if not any(o == 0 for o in owner):
        raise StringError("the first set has no closed strings")
    layer = a[0].layer
    origin = np.array([*np.vstack(rings)[:, :2].mean(axis=0), 0.0])
    rings = [r - origin for r in rings]
    scale = max(float(np.abs(np.vstack(rings)[:, :2]).max()), 1.0)
    tol = 1e-9 * scale

    seg0 = np.vstack([_segments(r, True)[0] for r in rings])
    seg1 = np.vstack([_segments(r, True)[1] for r in rings])
    q0, q1 = _cut_at_contacts(seg0, seg1, tol)
    mid = (q0[:, :2] + q1[:, :2]) / 2
    tan = q1[:, :2] - q0[:, :2]
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    eps = _side_offset(mid, seg0, seg1, scale, tol)[:, None]

    def inside(q: Floats, side: int) -> npt.NDArray[np.bool_]:
        w = np.zeros(len(q), np.int64)
        for r, o in zip(rings, owner, strict=True):
            if o == side:
                w += _winding(q, r)
        return np.asarray(w % 2 == 1)

    rule = _BOOL_RULES[op]
    left_q, right_q = mid + eps * nrm, mid - eps * nrm
    in_left = rule(inside(left_q, 0), inside(left_q, 1))
    in_right = rule(inside(right_q, 0), inside(right_q, 1))
    pieces: dict[tuple[tuple[float, float], tuple[float, float]], Floats] = {}
    for k in np.nonzero(in_left != in_right)[0]:
        e = np.array([q0[k], q1[k]]) if in_left[k] else np.array([q1[k], q0[k]])
        key = (
            (round(float(e[0, 0]) / tol), round(float(e[0, 1]) / tol)),
            (round(float(e[1, 0]) / tol), round(float(e[1, 1]) / tol)),
        )
        pieces.setdefault(key, e)  # coincident boundaries give the same piece twice
    chains = _chain(list(pieces.values()), 1e3 * tol)
    out = [_drop_collinear(_clean(c, True), tol) for c, loop in chains if loop]
    out = [r for r in out if len(r) >= 3 and abs(_signed_area(r[:, :2])) > SLIVER_AREA]
    out.sort(key=lambda r: -abs(_signed_area(r[:, :2])))
    return [_new(r + origin, True, layer) for r in out]


# -- corners ---------------------------------------------------------------------


def _corner_cut(
    s: String, size: float, mode: Literal["fillet", "chamfer"], arc_step_deg: float
) -> String:
    if size <= 0:
        raise StringError("the radius or distance must be positive")
    if not 0.1 <= arc_step_deg <= 45:
        raise StringError("the arc step must be between 0.1 and 45 degrees")
    p = _need(s, 3)
    n = len(p)
    corners = range(n) if s.closed else range(1, n - 1)
    out: list[Floats] = [] if s.closed else [p[:1]]
    for k in corners:
        prev, here, nxt = p[k - 1], p[k], p[(k + 1) % n]
        u = here[:2] - prev[:2]
        v = nxt[:2] - here[:2]
        lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        u, v = u / lu, v / lv
        turn = math.atan2(float(_cross2(u, v)), float(np.dot(u, v)))
        if abs(turn) < 1e-9 or abs(abs(turn) - math.pi) < 1e-9:
            out.append(here[None, :])
            continue
        # each corner may use at most half of each neighbouring segment
        room = min(lu, lv) / 2
        back = size * math.tan(abs(turn) / 2) if mode == "fillet" else size
        back = min(back, room)
        a = here + (prev - here) * (back / lu)
        b = here + (nxt - here) * (back / lv)
        if mode == "chamfer":
            out.append(np.array([a, b]))
            continue
        radius = back / math.tan(abs(turn) / 2)
        side = 1.0 if turn > 0 else -1.0
        centre = a[:2] + side * radius * np.array([-u[1], u[0]])
        steps = max(1, math.ceil(abs(turn) / math.radians(arc_step_deg)))
        start = math.atan2(*(a[:2] - centre)[::-1])
        ang = start + np.linspace(0.0, turn, steps + 1)
        frac = np.linspace(0.0, 1.0, steps + 1)
        arc = np.column_stack(
            [
                centre[0] + radius * np.cos(ang),
                centre[1] + radius * np.sin(ang),
                a[2] + frac * (b[2] - a[2]),
            ]
        )
        out.append(arc)
    if not s.closed:
        out.append(p[-1:])
    return _new(_clean(np.vstack(out), s.closed), s.closed, s.layer)


def fillet(s: String, radius: float, arc_step_deg: float = 10.0) -> String:
    """Round every corner with a circular arc of ``radius`` (plan).

    Where the arc would need more than half of a neighbouring segment the
    radius at that corner is reduced to fit. z runs linearly along each arc
    between its tangent points."""
    return _corner_cut(s, radius, "fillet", arc_step_deg)


def chamfer(s: String, distance: float) -> String:
    """Cut every corner with a straight line from ``distance`` back along the
    incoming segment to ``distance`` along the outgoing one (reduced to half
    a segment where the segment is shorter)."""
    return _corner_cut(s, distance, "chamfer", 10.0)
