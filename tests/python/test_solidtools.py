"""Solid verification, repair, exact booleans and cuts against analytic volumes."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minecore.solids import Solid, SolidError
from minecore.solidtools import (
    Boolean,
    boolean,
    cut_by_plane,
    cut_by_surface,
    input_problems,
    plane_normal,
    properties,
    repair,
    union_all,
    verify,
)
from minecore.surfaces import Surface

FACES = np.array(
    [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
     [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
)  # fmt: skip
CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    float,
)
MINE = np.array([500_000.0, 4_500_000.0, 1_000.0])  # mine-grid coordinates


def at(dx: float, dy: float, dz: float) -> np.ndarray:
    return np.asarray(MINE + np.array([dx, dy, dz], float))


def box(origin: object, size: object) -> Solid:
    return Solid(np.asarray(origin, float) + np.asarray(size, float) * CORNERS, FACES.copy())


def icosphere(centre: np.ndarray, radius: float, levels: int) -> Solid:
    t = (1 + 5**0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]  # fmt: skip
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]  # fmt: skip
    pts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(levels):
        mid: dict[tuple[int, int], int] = {}

        def m(i: int, j: int, mid: dict[tuple[int, int], int] = mid) -> int:
            key = (min(i, j), max(i, j))
            if key not in mid:
                p = pts[i] + pts[j]
                pts.append(p / np.linalg.norm(p))
                mid[key] = len(pts) - 1
            return mid[key]

        f = [
            tri
            for a, b, c in f
            for tri in (
                [a, m(a, b), m(c, a)],
                [b, m(b, c), m(a, b)],
                [c, m(c, a), m(b, c)],
                [m(a, b), m(b, c), m(c, a)],
            )
        ]
    return Solid(centre + radius * np.array(pts), np.array(f, np.int64))


def vol(s: Solid) -> float:
    return properties(s).volume


def test_verify_and_repair_a_flipped_cube() -> None:
    good = box(MINE, 10)
    assert verify(good).valid
    assert verify(good).volume == pytest.approx(1000.0, abs=1e-9)
    bad = Solid(good.vertices, np.vstack([good.triangles[:5], good.triangles[5:][:, ::-1]]))
    report = verify(bad)
    assert not report.valid
    assert report.flipped_edges > 0
    fixed = repair(bad)
    assert verify(fixed).valid
    assert verify(fixed).volume == pytest.approx(1000.0, abs=1e-9)


def test_overlapping_cubes_give_the_analytic_volumes() -> None:
    a, b = box(MINE, 10), box(at(5, 5, 5), 10)  # overlap 5^3 = 125
    expected = {Boolean.UNION: 1875.0, Boolean.INTERSECTION: 125.0, Boolean.DIFFERENCE: 875.0}
    for op, v in expected.items():
        r = boolean(a, b, op)
        assert not input_problems(r), op
        assert vol(r) == pytest.approx(v, abs=1e-6), op
    inter = boolean(a, b, Boolean.INTERSECTION)
    assert properties(inter).centroid == pytest.approx(MINE + 7.5, abs=1e-9)
    assert properties(inter).area == pytest.approx(150.0, abs=1e-6)


def test_coplanar_and_touching_cases() -> None:
    a = box(MINE, 10)
    assert vol(boolean(a, a, Boolean.UNION)) == pytest.approx(1000.0, abs=1e-6)
    assert vol(boolean(a, a, Boolean.INTERSECTION)) == pytest.approx(1000.0, abs=1e-6)
    with pytest.raises(SolidError, match="empty"):
        boolean(a, a, Boolean.DIFFERENCE)
    side = box(at(10, 0, 0), 10)  # shares a whole face
    joined = boolean(a, side, Boolean.UNION)
    assert vol(joined) == pytest.approx(2000.0, abs=1e-6)
    assert properties(joined).shell_volumes == pytest.approx((2000.0,), abs=1e-6)
    with pytest.raises(SolidError, match="empty"):
        boolean(a, side, Boolean.INTERSECTION)
    flush = box(at(0, 0, 5), [10, 10, 5])  # top half, faces flush on five sides
    assert vol(boolean(a, flush, Boolean.DIFFERENCE)) == pytest.approx(500.0, abs=1e-6)
    assert vol(boolean(a, flush, Boolean.INTERSECTION)) == pytest.approx(500.0, abs=1e-6)


def test_nested_solids_make_a_cavity_that_repair_keeps() -> None:
    outer, inner = box(MINE, 10), box(MINE + 3, 4)
    hollow = boolean(outer, inner, Boolean.DIFFERENCE)
    p = properties(hollow)
    assert p.volume == pytest.approx(936.0, abs=1e-6)
    assert sorted(p.shell_volumes) == pytest.approx([-64.0, 1000.0], abs=1e-6)
    assert vol(repair(hollow)) == pytest.approx(936.0, abs=1e-6)
    assert vol(boolean(outer, inner, Boolean.UNION)) == pytest.approx(1000.0, abs=1e-6)
    assert vol(boolean(outer, inner, Boolean.INTERSECTION)) == pytest.approx(64.0, abs=1e-6)


def test_spheres_match_the_lens_formula() -> None:
    r, d = 50.0, 60.0
    a = icosphere(MINE, r, 5)
    b = icosphere(at(d, 0, 0), r, 5)
    va = vol(a)
    assert va == pytest.approx(4 / 3 * math.pi * r**3, rel=2e-3)  # faceting error
    lens = math.pi * (4 * r + d) * (2 * r - d) ** 2 / 12
    inter = vol(boolean(a, b, Boolean.INTERSECTION))
    union = vol(boolean(a, b, Boolean.UNION))
    assert inter == pytest.approx(lens, rel=5e-3)
    assert union + inter == pytest.approx(va + vol(b), rel=1e-12)
    assert vol(boolean(a, b, Boolean.DIFFERENCE)) == pytest.approx(va - inter, rel=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.floats(-8, 8, allow_nan=False), min_size=3, max_size=3),
    st.lists(st.floats(1, 12, allow_nan=False), min_size=3, max_size=3),
)
def test_random_boxes_match_the_overlap(shift: list[float], size: list[float]) -> None:
    a = box(MINE, 10)
    b = box(MINE + np.array(shift), size)
    lo = np.maximum(0, shift)
    hi = np.minimum(10, np.array(shift) + size)
    overlap = float(np.prod(np.clip(hi - lo, 0, None)))
    if overlap < 1e-3:
        return
    union = 1000 + float(np.prod(size)) - overlap
    # Tolerance 0: no simplification, the volumes are exact to rounding.
    assert vol(boolean(a, b, Boolean.INTERSECTION, 0.0)) == pytest.approx(overlap, abs=1e-6)
    assert vol(boolean(a, b, Boolean.UNION, 0.0)) == pytest.approx(union, abs=1e-6)
    # Default 1 mm: faces closer than that may merge, so the error is bounded by
    # tolerance x area of the second box.
    bound = 1e-3 * 2 * (size[0] * size[1] + size[1] * size[2] + size[0] * size[2])
    assert vol(boolean(a, b, Boolean.UNION)) == pytest.approx(union, abs=bound)


def test_union_of_many() -> None:
    parts = [box(at(10 * i, 0, 0), 10) for i in range(4)] + [box(at(5, 5, 5), 10)]
    # 4 cubes in a row (4000) plus a cube overlapping two of them by 125 each.
    assert vol(union_all(parts)) == pytest.approx(4000 + 1000 - 250, abs=1e-6)


def test_open_and_inverted_inputs_are_reported() -> None:
    a = box(MINE, 10)
    open_box = Solid(a.vertices, a.triangles[:-1])
    with pytest.raises(SolidError, match="open edges"):
        boolean(open_box, a, Boolean.UNION)
    inverted = Solid(a.vertices, a.triangles[:, ::-1])
    with pytest.raises(SolidError, match="inwards"):
        boolean(inverted, a, Boolean.UNION)
    b = box(MINE + 5, 10)
    assert vol(boolean(inverted, b, Boolean.UNION, repair_inputs=True)) == pytest.approx(1875.0)
    with pytest.raises(SolidError, match="tolerance"):
        boolean(a, b, Boolean.UNION, -1.0)


def test_repair_merges_vertices_within_the_tolerance() -> None:
    a = box(MINE, 10)
    # Split vertex 6 into a twin 0.4 mm away: two triangles use the twin.
    v = np.vstack([a.vertices, a.vertices[6] + [0.0004, 0, 0]])
    t = a.triangles.copy()
    t[2] = [4, 5, 8]
    t[3] = [4, 8, 7]
    cracked = Solid(v, t)
    assert input_problems(cracked)
    assert not input_problems(repair(cracked, 0.001))
    assert input_problems(repair(cracked, 0.0001))


def test_plane_cuts_a_cube_on_a_diagonal() -> None:
    a = box(MINE, 10)
    # Vertical plane through the centre, normal along +x+y (dip 90, dip direction 45).
    cut = cut_by_plane(a, MINE + 5, plane_normal(90, 45))
    assert cut.volume_upper == pytest.approx(500.0, abs=1e-6)
    assert cut.volume_lower == pytest.approx(500.0, abs=1e-6)
    assert cut.section_area == pytest.approx(100 * math.sqrt(2), abs=1e-9)
    assert len(cut.section) == 1
    assert len(cut.section[0]) == 4
    assert cut.upper is not None
    assert properties(cut.upper).centroid[0] > MINE[0] + 5
    # Horizontal cut at z = 3 m: section is the 10 x 10 square, outer ring CCW from above.
    flat = cut_by_plane(a, at(0, 0, 3), plane_normal(0, 0))
    assert (flat.volume_upper, flat.volume_lower) == pytest.approx((700.0, 300.0), abs=1e-6)
    ring = flat.section[0]
    assert np.allclose(ring[:, 2], MINE[2] + 3)
    xy = ring[:, :2] - MINE[:2]
    assert 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]) == (
        pytest.approx(100.0)
    )


def test_plane_section_of_a_hollow_solid_has_a_hole() -> None:
    hollow = boolean(box(MINE, 10), box(MINE + 3, 4), Boolean.DIFFERENCE)
    cut = cut_by_plane(hollow, MINE + 5, plane_normal(0, 0))
    assert cut.section_area == pytest.approx(100.0 - 16.0, abs=1e-6)
    assert len(cut.section) == 2


def test_cut_by_a_tilted_surface_is_exact() -> None:
    a = box(MINE, 10)
    # Plane z = 1000 + 2 + 0.4 x over a footprint wider than the cube.
    xy = np.array([[-5, -5], [20, -5], [-5, 20], [20, 20]], float)
    pts = np.column_stack([xy + MINE[:2], MINE[2] + 2 + 0.4 * xy[:, 0]])
    cut = cut_by_surface(a, Surface(pts))
    below = 10 * (2 * 10 + 0.4 * 10**2 / 2)  # integral of (2 + 0.4 x) dx dy
    assert cut.volume_below == pytest.approx(below, abs=1e-6)
    assert cut.volume_above == pytest.approx(1000 - below, abs=1e-6)
    assert cut.volume_outside == pytest.approx(0.0, abs=1e-6)
    assert cut.below is not None
    assert not input_problems(cut.below)


def test_cut_by_a_surface_reports_the_part_outside_its_footprint() -> None:
    a = box(MINE, 10)
    xy = np.array([[2, -5], [20, -5], [2, 20], [20, 20]], float)
    pts = np.column_stack([xy + MINE[:2], np.full(4, MINE[2] + 4)])
    cut = cut_by_surface(a, Surface(pts))
    assert cut.volume_below == pytest.approx(8 * 10 * 4, abs=1e-6)
    assert cut.volume_above == pytest.approx(8 * 10 * 6, abs=1e-6)
    assert cut.volume_outside == pytest.approx(200.0, abs=1e-6)


def test_properties_of_a_box() -> None:
    p = properties(box(MINE, [10, 20, 30]))
    assert p.volume == pytest.approx(6000.0, abs=1e-9)
    assert p.area == pytest.approx(2 * (200 + 300 + 600), abs=1e-9)
    assert p.centroid == pytest.approx(at(5, 10, 15), abs=1e-9)
    assert p.closed
    assert len(p.shell_volumes) == 1


def test_solid_cli_boolean_cut_and_properties(tmp_path: Path) -> None:
    from minecore.cli import main  # noqa: PLC0415
    from minecore.project import Project  # noqa: PLC0415
    from minecore.solids import load_solid, save_solid  # noqa: PLC0415
    from minecore.strings import load_strings  # noqa: PLC0415
    from minecore.surfaces import save_surface  # noqa: PLC0415

    p = str(tmp_path / "k.mcproj")
    with Project.create(p, "k") as project:
        save_solid(project, "a", box(MINE, 10))
        save_solid(project, "b", box(MINE + 5, 10))
        xy = np.array([[-5, -5], [20, -5], [-5, 20], [20, 20]], float)
        save_surface(
            project, "topo", Surface(np.column_stack([xy + MINE[:2], np.full(4, MINE[2] + 4)]))
        )
    out = io.StringIO()
    assert main(["solid", "boolean", p, "a", "union", "b", "u", "--tolerance", "0"], out) == 0
    assert main(["solid", "join", p, "j", "a", "b"], out) == 0
    assert main(["solid", "cut", p, "a", "topo", "lo", "--other", "hi"], out) == 0
    args = ["solid", "plane-cut", p, "a", "--point", *map(str, MINE + 5), "--dip", "90", "45"]
    assert main([*args, "--upper", "e", "--lower", "w", "--section", "sec"], out) == 0
    text = out.getvalue()
    assert "u: 36 triangles, 1 shells, volume 1,875.000 m3" in text
    assert "below 400.000 m3, above 600.000 m3" in text
    assert "section 141.421 m2 in 1 rings" in text
    out = io.StringIO()
    assert main(["solid", "properties", p, "a", "u", "--density", "2.7"], out) == 0
    rows = out.getvalue().splitlines()
    assert rows[1].startswith("a,12,1,yes,1000.000,600.000")
    assert rows[1].endswith(",2700.000")
    assert rows[2].split(",")[4] == "1875.000"
    with Project.open(p) as project:
        assert vol(load_solid(project, "hi")) == pytest.approx(600.0, abs=1e-6)
        assert vol(load_solid(project, "e")) == pytest.approx(500.0, abs=1e-6)
        assert len(load_strings(project, "sec")) == 1
