"""Solid (wireframe) tools: verification, repair, exact booleans, plane and surface cuts.

Verification reports open edges (used by one triangle), non-manifold edges
(used by more than two), degenerate triangles (zero area), duplicate
triangles, edges walked the same way by both of their triangles
(inconsistent orientation) and whether the normals point inwards (negative
signed volume). Repair merges vertices closer than a user tolerance, drops
degenerate and duplicate triangles, makes the orientation consistent across
each connected shell (flood fill over shared edges) and orients every shell
by its nesting: outer shells outwards, cavities (shells inside an odd number
of other shells) inwards, so hollow solids keep their voids.

Booleans (union, intersection, difference) and cuts are exact on the
triangles: they are computed by the Manifold library (Apache-2.0,
``manifold3d``), which intersects the triangle meshes in double precision
with symbolic perturbation, so coplanar faces, shared edges and touching
solids give a closed, manifold result instead of slivers or holes. The
inputs must be closed and consistently oriented; problems are reported by
name (open edges, non-manifold edges, flipped triangles, inward normals)
and can be fixed first with ``repair``. Work happens in a local frame (the
lower corner of the inputs) so mine-grid coordinates keep full precision.
The user tolerance (m) merges vertices during repair and is Manifold's
simplification tolerance: features finer than it may be merged in results.

Cut by a plane: both closed halves and the section outline (closed rings in
the plane, outer rings counter-clockwise seen from the normal side, holes
clockwise) with its area. Cut by a surface: the surface TIN is extruded to a
closed prism below and above it and intersected with the solid, so the
cut follows every triangle of the surface exactly; the part of the solid
outside the surface footprint is reported.

Properties: volume, surface area and centroid from the divergence theorem
(exact for the triangles), bounding box and per-shell volumes.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from minecore.solids import Solid, SolidError
from minecore.surfaces import Surface

if TYPE_CHECKING:
    import manifold3d

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SolidReport:
    triangles: int
    vertices: int
    open_edges: int
    non_manifold_edges: int
    degenerate: int
    duplicates: int
    flipped_edges: int  # interior edges both triangles walk the same way
    shells: int
    volume: float  # signed; negative = normals inwards
    area: float

    @property
    def valid(self) -> bool:
        return (
            self.open_edges == 0
            and self.non_manifold_edges == 0
            and self.degenerate == 0
            and self.duplicates == 0
            and self.flipped_edges == 0
            and self.volume > 0
        )


def _areas(s: Solid) -> Floats:
    a, b, c = (s.vertices[s.triangles[:, i]] for i in range(3))
    return np.asarray(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1), np.float64)


def _signed_volume(v: Floats, t: Ints) -> float:
    # In a local frame: with mine-grid coordinates the products would cancel badly.
    if len(t) == 0:
        return 0.0
    local = v - v[t[0, 0]]
    a, b, c = (local[t[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _directed_edges(t: Ints) -> Ints:
    return np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])


def _edge_keys(edges: Ints, n_vertices: int) -> Ints:
    """One integer per (start, end) pair; sort the pair first for undirected edges."""
    return np.asarray(edges[:, 0] * np.int64(max(n_vertices, 1)) + edges[:, 1], np.int64)


def _undirected_ids(t: Ints) -> tuple[Ints, Ints]:
    """Edge id of each of the 3m triangle edges and the use count of each edge id."""
    n = int(t.max()) + 1 if len(t) else 1
    _, inv, counts = np.unique(
        _edge_keys(np.sort(_directed_edges(t), axis=1), n), return_inverse=True, return_counts=True
    )
    return np.asarray(inv.ravel(), np.int64), np.asarray(counts, np.int64)


def _shells(t: Ints) -> Ints:
    """Connected component (over shared edges) of each triangle."""
    n = len(t)
    if n == 0:
        return np.zeros(0, np.int64)
    inv, _ = _undirected_ids(t)
    owner = np.tile(np.arange(n), 3)
    order = np.argsort(inv, kind="stable")
    same = inv[order][1:] == inv[order][:-1]
    i, j = owner[order][:-1][same], owner[order][1:][same]
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    return np.asarray(label, np.int64)


def _duplicate_triangles(t: Ints) -> int:
    st = np.sort(t, axis=1)
    order = np.lexsort((st[:, 2], st[:, 1], st[:, 0]))
    st = st[order]
    return int((st[1:] == st[:-1]).all(axis=1).sum())


def verify(s: Solid) -> SolidReport:
    t = s.triangles
    n = len(s.vertices)
    _, counts = _undirected_ids(t)
    _, dcounts = np.unique(_edge_keys(_directed_edges(t), n), return_counts=True)
    areas = _areas(s)
    return SolidReport(
        len(t),
        n,
        int((counts == 1).sum()),
        int((counts > 2).sum()),
        int((areas <= 1e-12).sum()),
        _duplicate_triangles(t),
        int((dcounts > 1).sum()),
        int(_shells(t).max() + 1) if len(t) else 0,
        _signed_volume(s.vertices, t),
        float(areas.sum()),
    )


def merge_vertices(vertices: Floats, tolerance: float) -> tuple[Floats, Ints]:
    """Vertices closer than ``tolerance`` become one (chains merge transitively).

    Returns the merged vertices and, for each input vertex, its new index.
    """
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise SolidError("the vertex merge tolerance must be >= 0")
    v = np.asarray(vertices, np.float64)
    n = len(v)
    if tolerance == 0 or n == 0:
        _, first, inv = np.unique(v, axis=0, return_index=True, return_inverse=True)
    else:
        pairs = cKDTree(v - v.min(axis=0)).query_pairs(tolerance, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        _, first, inv = np.unique(label, return_index=True, return_inverse=True)
    return v[first], np.asarray(inv.ravel(), np.int64)


def repair(s: Solid, tolerance: float = 1e-6) -> Solid:
    """Merge vertices, drop bad triangles, orient every shell consistently.

    Outer shells face outwards and cavities (inside an odd number of other
    shells) face inwards, so a hollow solid keeps its void.
    """
    v, inv = merge_vertices(s.vertices, tolerance)
    t = inv[s.triangles]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 2] != t[:, 0])]
    a, b, c = (v[t[:, i]] for i in range(3))
    t = t[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-12]
    _, keep = np.unique(np.sort(t, axis=1), axis=0, return_index=True)
    t = t[np.sort(keep)]
    if len(t) == 0:
        raise SolidError("nothing is left after removing degenerate triangles")
    t = _orient(t)
    shells = _shells(t)
    count = int(shells.max()) + 1
    depth = _nesting_depth(v, t, shells, count)
    for k in range(count):
        sel = shells == k
        outward = depth[k] % 2 == 0
        if (_signed_volume(v, t[sel]) > 0) != outward:
            t[sel] = t[sel][:, ::-1]
    return Solid(v, t.astype(np.int64))


def _nesting_depth(v: Floats, t: Ints, shells: Ints, count: int) -> list[int]:
    """How many other shells enclose each shell (tested at one of its vertices)."""
    if count == 1:
        return [0]
    local = v - v.min(axis=0)
    probes = np.array([local[t[shells == k][0, 0]] for k in range(count)])
    depth = [0] * count
    for k in range(count):
        shell = Solid(local, t[shells == k])
        inside = shell.inside(probes)
        for j in range(count):
            if j != k and inside[j]:
                depth[j] += 1
    return depth


def _orient(t: Ints) -> Ints:
    """Flip triangles so neighbours walk shared edges in opposite directions."""
    t = t.copy()
    n = len(t)
    by_edge: dict[tuple[int, int], list[int]] = {}
    for i, (a, b, c) in enumerate(t.tolist()):
        for e in ((a, b), (b, c), (c, a)):
            by_edge.setdefault((min(e), max(e)), []).append(i)
    seen = np.zeros(n, bool)
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        while queue:
            i = queue.popleft()
            a, b, c = t[i].tolist()
            for u, w in ((a, b), (b, c), (c, a)):
                for j in by_edge[(min(u, w), max(u, w))]:
                    if j == i or seen[j]:
                        continue
                    x, y, z = t[j].tolist()
                    same = (x, y) == (u, w) or (y, z) == (u, w) or (z, x) == (u, w)
                    if same:
                        t[j] = t[j][::-1]
                    seen[j] = True
                    queue.append(j)
    return t


# -- exact operations --------------------------------------------------------


class Boolean(StrEnum):
    UNION = "union"
    INTERSECTION = "intersection"
    DIFFERENCE = "difference"  # a minus b


DEFAULT_TOLERANCE = 0.001  # m; shown to and changed by the user in the CLI and GUI


def input_problems(s: Solid) -> list[str]:
    """Why ``s`` cannot take part in an exact boolean (empty list: it can)."""
    r = verify(s)
    problems = []
    if r.open_edges:
        problems.append(f"{r.open_edges:,} open edges (holes in the wireframe)")
    if r.non_manifold_edges:
        problems.append(f"{r.non_manifold_edges:,} edges shared by more than two triangles")
    if r.duplicates:
        problems.append(f"{r.duplicates:,} duplicate triangles")
    if r.flipped_edges:
        problems.append(
            f"{r.flipped_edges:,} edges where neighbouring triangles face opposite ways"
        )
    if not problems and r.volume <= 0:
        problems.append("the normals point inwards (negative volume)")
    return problems


def _checked(s: Solid, label: str, tolerance: float, repair_first: bool) -> Solid:
    if repair_first:
        s = repair(s, tolerance)
    problems = input_problems(s)
    if problems:
        hint = (
            ""
            if repair_first
            else " Repair it first (solid verify --repair, or the repair option)."
        )
        raise SolidError(
            f"solid {label!r} is not closed and consistently oriented: "
            + "; ".join(problems)
            + "."
            + hint
        )
    return s


def _check_tolerance(tolerance: float) -> float:
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise SolidError("the tolerance must be >= 0 m")
    return float(tolerance)


def _manifold(s: Solid, origin: Floats, tolerance: float, label: str) -> manifold3d.Manifold:
    import manifold3d  # noqa: PLC0415 - compiled library, loaded on first use

    mesh = manifold3d.Mesh64(
        np.ascontiguousarray(s.vertices - origin, np.float64),
        np.ascontiguousarray(s.triangles, np.uint64),
        tolerance=tolerance,
    )
    m = manifold3d.Manifold(mesh)
    if m.status() != manifold3d.Error.NoError:
        raise SolidError(f"solid {label!r} was rejected by the boolean kernel: {m.status().name}")
    return m


def _solid(m: manifold3d.Manifold, origin: Floats, tolerance: float) -> Solid | None:
    """Result mesh in world coordinates, without zero-thickness shells.

    Solids that only touch (shared faces) can leave a flat, zero-volume
    double sheet; shells thinner on average than the tolerance are dropped.
    """
    import manifold3d  # noqa: PLC0415

    if m.is_empty():
        return None
    parts = m.decompose()
    keep = [
        p
        for p in parts
        if abs(p.volume()) > max(tolerance * p.surface_area() / 2, 1e-12 * p.surface_area() ** 1.5)
    ]
    if not keep:
        return None
    if len(keep) < len(parts):
        m = manifold3d.Manifold.compose(keep)
    mesh = m.to_mesh64()
    v = np.asarray(mesh.vert_properties, np.float64)[:, :3] + origin
    return Solid(v, np.asarray(mesh.tri_verts, np.int64))


def boolean(
    a: Solid,
    b: Solid,
    op: Boolean,
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    repair_inputs: bool = False,
) -> Solid:
    """Exact ``a op b`` of two closed solids (see the module text)."""
    tol = _check_tolerance(tolerance)
    a = _checked(a, "A", tol, repair_inputs)
    b = _checked(b, "B", tol, repair_inputs)
    origin = np.minimum(a.vertices.min(axis=0), b.vertices.min(axis=0))
    ma, mb = _manifold(a, origin, tol, "A"), _manifold(b, origin, tol, "B")
    op = Boolean(op)
    m = {Boolean.UNION: ma + mb, Boolean.INTERSECTION: ma ^ mb, Boolean.DIFFERENCE: ma - mb}[op]
    result = _solid(m, origin, tol)
    if result is None:
        raise SolidError(f"the {op.value} is empty (the solids do not overlap)")
    return result


def union_all(
    solids: list[Solid], tolerance: float = DEFAULT_TOLERANCE, *, repair_inputs: bool = False
) -> Solid:
    """Exact union of many closed solids at once."""
    import manifold3d  # noqa: PLC0415

    if not solids:
        raise SolidError("no solids to join")
    tol = _check_tolerance(tolerance)
    checked = [_checked(s, f"#{i + 1}", tol, repair_inputs) for i, s in enumerate(solids)]
    origin = np.asarray(np.min([s.vertices.min(axis=0) for s in checked], axis=0), np.float64)
    parts = [_manifold(s, origin, tol, f"#{i + 1}") for i, s in enumerate(checked)]
    result = _solid(manifold3d.Manifold.batch_boolean(parts, manifold3d.OpType.Add), origin, tol)
    if result is None:
        raise SolidError("the union is empty")
    return result


# -- plane cut ---------------------------------------------------------------


def plane_normal(dip: float, dip_direction: float) -> Floats:
    """Upward unit normal of a plane given by dip and dip direction (degrees).

    A horizontal plane (dip 0) has normal +z; a vertical plane (dip 90) has a
    horizontal normal pointing towards the dip direction (azimuth from north).
    """
    if not (0 <= dip <= 90):
        raise SolidError("dip must be between 0 and 90 degrees")
    d, a = math.radians(dip), math.radians(dip_direction)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


@dataclass(frozen=True)
class PlaneCut:
    upper: Solid | None  # on the side the normal points to
    lower: Solid | None
    section: list[Floats]  # closed rings (k x 3) in the plane
    section_area: float  # m2, holes subtracted
    volume_upper: float
    volume_lower: float


def _frame(normal: Floats) -> Floats:
    """Rows u, w, n: a right-handed orthonormal frame with n along ``normal``."""
    n = np.asarray(normal, np.float64)
    length = float(np.linalg.norm(n))
    if not (math.isfinite(length) and length > 0):
        raise SolidError("the plane normal must be a non-zero vector")
    n = n / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    return np.vstack([u, np.cross(n, u), n])


def cut_by_plane(
    s: Solid,
    point: Floats,
    normal: Floats,
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    repair_input: bool = False,
) -> PlaneCut:
    """Both closed halves of ``s`` on either side of a plane, and the section."""
    tol = _check_tolerance(tolerance)
    s = _checked(s, "solid", tol, repair_input)
    frame = _frame(normal)
    origin = s.vertices.min(axis=0)
    m = _manifold(s, origin, tol, "solid")
    offset = float(frame[2] @ (np.asarray(point, np.float64) - origin))
    up, low = m.split_by_plane(tuple(frame[2]), offset)
    upper, lower = _solid(up, origin, tol), _solid(low, origin, tol)
    if upper is None and lower is None:
        raise SolidError("the cut is empty")
    rings, area = _section(lower, upper, frame[2], np.asarray(point, np.float64))
    return PlaneCut(upper, lower, rings, area, float(up.volume()), float(low.volume()))


def _section(
    lower: Solid | None, upper: Solid | None, n: Floats, point: Floats
) -> tuple[list[Floats], float]:
    """Section rings and area from the cap faces the cut adds in the plane.

    The cap of the lower half faces along ``n``; its boundary edges, chained,
    are the section outline (outer rings counter-clockwise seen from ``n``).
    """
    piece, sign = (lower, 1.0) if lower is not None else (upper, -1.0)
    assert piece is not None
    v, t = piece.vertices, piece.triangles
    scale = max(1.0, float((v.max(axis=0) - v.min(axis=0)).max()))
    on = np.abs((v - point) @ n) <= 1e-9 * scale
    tri = t[on[t].all(axis=1)]
    a, b, c = (v[tri[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    tri = tri[sign * (cross @ n) > 0]
    if sign < 0:
        tri = tri[:, ::-1]
    if len(tri) == 0:
        return [], 0.0
    a, b, c = (v[tri[:, i]] for i in range(3))
    area = float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    e = _directed_edges(tri)
    _, inv, count = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True, return_counts=True)
    nxt: dict[int, list[int]] = {}
    for s0, s1 in e[count[inv.ravel()] == 1].tolist():
        nxt.setdefault(s0, []).append(s1)
    rings = []
    while nxt:
        start = next(iter(nxt))
        ring, k = [start], start
        while True:
            ends = nxt.get(k)
            if not ends:
                break  # open chain: cannot happen for a closed cap
            step = ends.pop()
            if not ends:
                del nxt[k]
            if step == start:
                break
            ring.append(step)
            k = step
        pts = _drop_collinear(v[ring], 1e-9 * scale)
        if len(pts) >= 3:
            rings.append(pts)
    return rings, area


def _drop_collinear(ring: Floats, eps: float) -> Floats:
    """Remove ring vertices lying on the line through their neighbours."""
    pts = ring
    while len(pts) > 3:
        prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
        span = nxt - prev
        dist = np.linalg.norm(np.cross(pts - prev, span), axis=1) / np.maximum(
            np.linalg.norm(span, axis=1), 1e-300
        )
        flat = np.flatnonzero(dist <= eps)
        if len(flat) == 0:
            break
        pts = np.delete(pts, flat[0], axis=0)
    return np.asarray(pts, np.float64)


# -- surface cut -------------------------------------------------------------


def _surface_prism(surface: Surface, z_end: float, upwards: bool) -> Solid:
    """Closed prism between the surface TIN and the level ``z_end`` over its footprint."""
    p = np.asarray(surface.points, np.float64)
    tri = surface.triangles()
    xy = p[tri, :2] - p[:, :2].min(axis=0)
    u, w = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    cross = u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]
    tri = tri[np.abs(cross) > 1e-12]
    flip = cross[np.abs(cross) > 1e-12] < 0
    tri[flip] = tri[flip][:, ::-1]  # counter-clockwise in plan: normals up
    used, tri = np.unique(tri, return_inverse=True)
    tri = tri.reshape(-1, 3)
    top = p[used]
    n = len(top)
    flat = top.copy()
    flat[:, 2] = z_end
    e = _directed_edges(tri)
    _, inv, count = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True, return_counts=True)
    rim = e[count[inv.ravel()] == 1]  # boundary edges, interior on the left
    a, b = rim[:, 0], rim[:, 1]
    if upwards:  # surface is the floor (faces down), the cap at z_end faces up
        faces = [
            tri[:, ::-1],
            tri + n,
            np.column_stack([a, b, b + n]),
            np.column_stack([a, b + n, a + n]),
        ]
    else:  # surface is the roof (faces up), the cap at z_end faces down
        faces = [
            tri,
            tri[:, ::-1] + n,
            np.column_stack([b, a, a + n]),
            np.column_stack([b, a + n, b + n]),
        ]
    return Solid(np.vstack([top, flat]), np.vstack(faces).astype(np.int64))


@dataclass(frozen=True)
class SurfaceCut:
    below: Solid | None
    above: Solid | None
    volume_below: float
    volume_above: float
    volume_outside: float  # part of the solid outside the surface footprint


def cut_by_surface(
    s: Solid,
    surface: Surface,
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    repair_input: bool = False,
) -> SurfaceCut:
    """Exact parts of ``s`` below and above a surface (see the module text)."""
    tol = _check_tolerance(tolerance)
    s = _checked(s, "solid", tol, repair_input)
    zs, zp = s.vertices[:, 2], surface.points[:, 2]
    margin = max(1.0, 0.01 * float(max(zs.max(), zp.max()) - min(zs.min(), zp.min())))
    down = _surface_prism(surface, float(min(zs.min(), zp.min())) - margin, upwards=False)
    up = _surface_prism(surface, float(max(zs.max(), zp.max())) + margin, upwards=True)
    for prism, side in ((down, "below"), (up, "above")):
        problems = input_problems(prism)
        if problems:
            raise SolidError(
                f"the surface cannot be closed into a prism {side} it: "
                + "; ".join(problems)
                + " (check for pinched corners in its outline)"
            )
    origin = np.minimum(s.vertices.min(axis=0), down.vertices.min(axis=0))
    m = _manifold(s, origin, tol, "solid")
    below = m ^ _manifold(down, origin, tol, "surface (below)")
    above = m ^ _manifold(up, origin, tol, "surface (above)")
    vb, va = float(below.volume()), float(above.volume())
    return SurfaceCut(
        _solid(below, origin, tol),
        _solid(above, origin, tol),
        vb,
        va,
        max(0.0, float(m.volume()) - vb - va),
    )


# -- properties --------------------------------------------------------------


@dataclass(frozen=True)
class SolidProperties:
    volume: float  # m3 (signed sum over shells; cavities count negative)
    area: float  # m2, all triangles
    centroid: Floats  # x, y, z of the volume
    lower: Floats  # bounding box corners
    upper: Floats
    shell_volumes: tuple[float, ...]  # signed, per connected shell
    closed: bool


def properties(s: Solid) -> SolidProperties:
    """Volume, area and centroid (divergence theorem, local frame), box and shells."""
    origin = s.vertices.min(axis=0)
    v = s.vertices - origin
    a, b, c = (v[s.triangles[:, i]] for i in range(3))
    six_v = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = float(six_v.sum() / 6.0)
    area = float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    if abs(volume) > 0:
        centroid = origin + (six_v[:, None] * (a + b + c)).sum(axis=0) / (24.0 * volume)
    else:
        centroid = np.full(3, np.nan)
    shells = _shells(s.triangles)
    shell_volumes = tuple(
        float(np.bincount(shells, weights=six_v, minlength=int(shells.max()) + 1)[k] / 6.0)
        for k in range(int(shells.max()) + 1)
    )
    return SolidProperties(
        volume,
        area,
        np.asarray(centroid, np.float64),
        s.vertices.min(axis=0),
        s.vertices.max(axis=0),
        shell_volumes,
        not input_problems(s),
    )
