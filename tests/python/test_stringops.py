"""String editing operations (E-CD-01) against analytic answers."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minecore import cli
from minecore import stringops as so
from minecore.project import Project
from minecore.strings import String, StringError, load_strings, save_strings


def _s(xy: list[list[float]], closed: bool = False, z: float = 0.0) -> String:
    a = np.asarray(xy, float)
    if a.shape[1] == 2:
        a = np.column_stack([a, np.full(len(a), z)])
    return String(a, closed)


SQUARE = _s([[0, 0], [10, 0], [10, 10], [0, 10]], closed=True)
SQUARE_CW = so.reverse(SQUARE)
L_SHAPE = _s([[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]], closed=True)


# -- topology -------------------------------------------------------------------


def test_reverse_open_and_closed() -> None:
    s = _s([[0, 0], [1, 0], [2, 1]])
    assert np.array_equal(so.reverse(s).points[:, :2], [[2, 1], [1, 0], [0, 0]])
    r = so.reverse(SQUARE)
    assert np.array_equal(r.points[0], SQUARE.points[0])
    assert so.measure(r).area == pytest.approx(100.0)
    assert so._signed_area(r.points[:, :2]) == pytest.approx(-100.0)


def test_split_and_join_round_trip() -> None:
    s = _s([[0, 0], [1, 0], [2, 0], [3, 1]])
    a, b = so.split(s, 2)
    assert len(a.points) == 3
    assert len(b.points) == 2
    back = so.join(b, a)  # order and direction are found from the nearest ends
    assert so.measure(back).plan_length == pytest.approx(so.measure(s).plan_length)
    with pytest.raises(StringError):
        so.split(s, 0)
    opened = so.split(SQUARE, 2)[0]
    assert not opened.closed
    assert np.array_equal(opened.points[0], opened.points[-1])
    assert so.measure(opened).plan_length == pytest.approx(40.0)


def test_join_refuses_large_gap() -> None:
    with pytest.raises(StringError):
        so.join(_s([[0, 0], [1, 0]]), _s([[5, 0], [6, 0]]), tolerance=1.0)
    j = so.join(_s([[0, 0], [1, 0]]), _s([[6, 0], [5, 0]]))
    assert np.array_equal(j.points[:, 0], [0, 1, 5, 6])


# -- measurement ----------------------------------------------------------------


def test_measure_and_segment_bearings() -> None:
    s = String(np.array([[0, 0, 0], [0, 10, 0], [10, 10, 10], [10, 0, 10]], float), False)
    m = so.measure(s)
    assert m.plan_length == pytest.approx(30.0)
    assert m.slope_length == pytest.approx(20 + math.sqrt(200))
    segs = so.segment_table(s)
    assert [round(g.bearing) for g in segs] == [0, 90, 180]
    assert segs[1].gradient == pytest.approx(1.0)
    assert segs[1].dip == pytest.approx(45.0)
    assert so.measure(L_SHAPE).area == pytest.approx(300.0)
    assert so.measure(L_SHAPE).plan_length == pytest.approx(80.0)


# -- shape ----------------------------------------------------------------------


def test_simplify_removes_collinear_and_small_bumps() -> None:
    s = _s([[0, 0], [1, 0], [2, 0.01], [3, 0], [4, 0], [4, 5]])
    out = so.simplify(s, 0.05)
    assert np.array_equal(out.points[:, :2], [[0, 0], [4, 0], [4, 5]])
    dense = so.resample(SQUARE, 0.5)
    back = so.simplify(dense, 1e-6)
    assert len(back.points) == 4
    assert so.measure(back).area == pytest.approx(100.0)


@settings(max_examples=30, deadline=None)
@given(st.floats(0.05, 3.0), st.integers(5, 60))
def test_simplify_stays_within_tolerance(tol: float, n: int) -> None:
    t = np.linspace(0, 3 * np.pi, n)
    s = String(np.column_stack([t * 5, 4 * np.sin(t), t]), False)
    out = so.simplify(s, tol)
    # every original vertex is within tol of the simplified string (3-D)
    q = out.points
    for p in s.points:
        a, b = q[:-1], q[1:]
        ab = b - a
        tt = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-300), 0, 1)
        assert np.min(np.linalg.norm(a + tt[:, None] * ab - p, axis=1)) <= tol + 1e-9


def test_resample_spacing_and_length() -> None:
    s = String(np.array([[0, 0, 0], [30, 40, 0]], float), False)  # 50 m
    r = so.resample(s, 10.0)
    assert len(r.points) == 6
    assert np.allclose(np.linalg.norm(np.diff(r.points, axis=0), axis=1), 10.0)
    c = so.resample(SQUARE, 2.5)
    assert len(c.points) == 16
    assert c.closed
    assert so.measure(c).plan_length == pytest.approx(40.0)


def test_smooth_keeps_open_ends_and_shrinks_square() -> None:
    s = _s([[0, 0], [5, 5], [10, 0]])
    out = so.smooth(s, 2)
    assert np.array_equal(out.points[0], s.points[0])
    assert np.array_equal(out.points[-1], s.points[-1])
    sq = so.smooth(SQUARE, 1)
    # one Chaikin pass cuts each corner triangle of legs 2.5 m: 4 * 2.5^2 / 2
    assert so.measure(sq).area == pytest.approx(100 - 4 * 3.125)


# -- intersections, trim, extend ---------------------------------------------------


def test_intersections_with_z() -> None:
    a = String(np.array([[0, 5, 0], [10, 5, 10]], float), False)
    b = String(np.array([[4, 0, 100], [4, 10, 100]], float), False)
    (c,) = so.intersections(a, b)
    assert (c.x, c.y, c.z_a, c.z_b) == pytest.approx((4, 5, 4, 100))
    assert len(so.intersections(_s([[-5, 5], [15, 5]]), SQUARE)) == 2


def test_extend_and_trim_to_boundary() -> None:
    wall = _s([[10, -10], [10, 10]])
    s = String(np.array([[0, 0, 0], [5, 0, 1]], float), False)
    ext = so.trim_extend(s, wall)
    assert ext.points[-1] == pytest.approx([10, 0, 2])  # gradient carried on
    long = _s([[0, 0], [15, 0]])
    assert so.trim_extend(long, wall).points[-1, :2] == pytest.approx([10, 0])
    both = so.trim_extend(_s([[2, 0], [5, 0]]), _s([[0, -1], [0, 1]]), end="start")
    assert both.points[0, :2] == pytest.approx([0, 0])
    with pytest.raises(StringError):
        so.trim_extend(s, wall, max_extension=2.0)


# -- offset ---------------------------------------------------------------------


@pytest.mark.parametrize("ring", [SQUARE, SQUARE_CW])
def test_offset_square_outward_mitre_and_round(ring: String) -> None:
    (m,) = so.offset(ring, 2.0, join="mitre")
    assert so.measure(m).area == pytest.approx(14.0**2)
    (r,) = so.offset(ring, 2.0, join="round", arc_step_deg=0.5)
    exact = 100 + 4 * 10 * 2 + math.pi * 4
    assert so.measure(r).area == pytest.approx(exact, rel=1e-4)


@pytest.mark.parametrize("ring", [SQUARE, SQUARE_CW])
def test_offset_square_inward_and_collapse(ring: String) -> None:
    (m,) = so.offset(ring, -2.0)
    assert so.measure(m).area == pytest.approx(36.0)
    assert so.offset(ring, -6.0) == []


def test_offset_concave_l_shape() -> None:
    (out,) = so.offset(L_SHAPE, 1.0)
    # mitre offset of an orthogonal polygon: area + perimeter*d + (convex - concave)*d^2
    assert so.measure(out).area == pytest.approx(300 + 80 + (5 - 1))
    (inner,) = so.offset(L_SHAPE, -1.0)
    assert so.measure(inner).area == pytest.approx(300 - 80 + (5 - 1))


def test_offset_inward_splits_dumbbell() -> None:
    db = _s(
        [
            [0, 0],
            [10, 0],
            [10, 4],
            [20, 4],
            [20, 0],
            [30, 0],
            [30, 10],
            [20, 10],
            [20, 6],
            [10, 6],
            [10, 10],
            [0, 10],
        ],
        closed=True,
    )
    parts = so.offset(db, -1.5)
    assert len(parts) == 2
    assert sorted(so.measure(p).area for p in parts) == pytest.approx([49.0, 49.0])


def test_offset_open_left_right_and_slope() -> None:
    s = _s([[0, 0], [10, 0], [10, 10]], z=100.0)
    (left,) = so.offset(s, 1.0)
    assert left.points[0, :2] == pytest.approx([0, 1])
    assert left.points[-1, :2] == pytest.approx([9, 10])
    (right,) = so.offset(s, -1.0, slope_deg=45.0)
    assert right.points[1, :2] == pytest.approx([11, -1])  # mitre corner
    assert np.allclose(right.points[:, 2], 101.0)
    (down,) = so.offset(s, 2.0, dz=-5.0)
    assert np.allclose(down.points[:, 2], 95.0)


@settings(max_examples=25, deadline=None)
@given(st.floats(1.0, 50.0), st.floats(0.1, 5.0))
def test_offset_circle_radius(radius: float, d: float) -> None:
    t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    c = String(np.column_stack([radius * np.cos(t), radius * np.sin(t), np.zeros(180)]), True)
    (out,) = so.offset(c, d, join="round")
    r_out = np.hypot(out.points[:, 0], out.points[:, 1])
    assert r_out.min() >= radius * math.cos(math.pi / 180) + d - 1e-6
    # round joins are circumscribed about the arc: corners lie at d / cos(half step)
    assert r_out.max() <= radius + d / math.cos(math.radians(1.0)) + 1e-6


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(3.0, 20.0), min_size=5, max_size=24),
    st.floats(-4.0, 4.0).filter(lambda d: abs(d) > 0.05),
    st.sampled_from(["mitre", "round"]),
)
def test_offset_star_polygons_are_clean(radii: list[float], d: float, join: str) -> None:
    n = len(radii)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.asarray(radii)
    ring = String(np.column_stack([r * np.cos(t), r * np.sin(t), np.zeros(n)]), True)
    parts = so.offset(ring, d, join=join)  # type: ignore[arg-type]
    if d > 0:
        assert len(parts) == 1
    for part in parts:
        assert so._first_self_hit(part.points, True) is None
        dist = so._point_to_polyline(part.points, ring.points, True)
        assert dist.min() >= abs(d) * (1 - 1e-6) - 1e-9
        inside = _inside(part.points[:, :2], ring.points[:, :2])
        assert inside.all() if d < 0 else not inside.any()


def _inside(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path  # noqa: PLC0415

    return np.asarray(Path(poly).contains_points(pts))


# -- command line ---------------------------------------------------------------


def test_cli_offset_measure_trim(tmp_path: Path) -> None:
    path = tmp_path / "c.mcproj"
    with Project.create(path, "c") as project:
        save_strings(
            project,
            "toe",
            [String(np.column_stack([SQUARE.points[:, :2], np.full(4, 100.0)]), True)],
        )
        save_strings(project, "wall", [_s([[20, -50], [20, 50]])])
        save_strings(project, "road", [_s([[5, 5], [8, 5]])])
    out = io.StringIO()
    assert (
        cli.main(
            [
                "string",
                "edit",
                str(path),
                "toe",
                "offset",
                "--distance",
                "2",
                "--slope",
                "45",
                "--out",
                "crest",
            ],
            out,
        )
        == 0
    )
    out = io.StringIO()
    assert cli.main(["string", "list", str(path), "crest"], out) == 0
    row = out.getvalue().splitlines()[1].split()
    assert float(row[5]) == pytest.approx(196.0)  # (10 + 2 * 2)^2
    assert float(row[6]) == pytest.approx(102.0)  # z raised by 2 m at 45 degrees
    assert (
        cli.main(
            ["string", "edit", str(path), "road", "trim", "--boundary", "wall", "--out", "road2"],
            io.StringIO(),
        )
        == 0
    )
    csv_path = tmp_path / "seg.csv"
    out = io.StringIO()
    assert (
        cli.main(
            ["string", "measure", str(path), "road2", "--segments", "--csv", str(csv_path)], out
        )
        == 0
    )
    assert "plan 15.000 m" in out.getvalue()
    assert "bearing 90.00" in out.getvalue()
    assert csv_path.read_text(encoding="utf-8").startswith("string,segment")
    with Project.open(path) as project:
        assert len(load_strings(project, "crest")) == 1
    err = io.StringIO()
    assert cli.main(["string", "edit", str(path), "road", "simplify", "--out", "x"], err) == 1


def test_cli_split_join_intersect(tmp_path: Path) -> None:
    path = tmp_path / "j.mcproj"
    with Project.create(path, "j") as project:
        save_strings(project, "a", [_s([[0, 0], [5, 0], [10, 0]]), _s([[5, -5], [5, 5]])])
    assert (
        cli.main(
            ["string", "split", str(path), "a", "--index", "0", "--vertex", "1", "--out", "b"],
            io.StringIO(),
        )
        == 0
    )
    assert (
        cli.main(
            ["string", "join", str(path), "b", "--index", "0", "--index", "1", "--out", "c"],
            io.StringIO(),
        )
        == 0
    )
    with Project.open(path) as project:
        assert len(load_strings(project, "b")) == 3
        c = load_strings(project, "c")
        assert len(c) == 2
        assert so.measure(c[-1]).plan_length == pytest.approx(10.0)
    out = io.StringIO()
    assert cli.main(["string", "intersect", str(path), "a", "a"], out) == 0
    assert "5.000 0.000" in out.getvalue()
