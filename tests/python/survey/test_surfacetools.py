"""Contours, slope/aspect, profiles, intersection lines and offset (OC10, E-CD-02).

Known answers: a square pyramid (apex H over a 2R square) has contour squares
of perimeter 8R(1 - L/H), one face slope atan(H/R), true area
plan / cos(slope), and meets a level plane in the same square. Exactness on
random triangulations: every contour vertex lies on the surface at its level,
every intersection vertex lies on both surfaces, and the profile interpolated
linearly between its points equals the surface everywhere.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest

from minecore import strings
from minecore import surfacetools as st
from minecore.cli import main
from minecore.project import Project
from minecore.surfaces import Surface, save_surface

E0, N0 = 512_000.0, 4_400_000.0  # UTM-sized coordinates
H, R = 20.0, 50.0
OFF = np.array([E0, N0])


def _pyramid(sign: float = 1.0, base: float = 1000.0) -> Surface:
    pts = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [50, 50, sign * H]], float)
    return Surface(pts + np.array([E0, N0, base]))


def _flat(z: float) -> Surface:
    xy = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float) + OFF
    return Surface(np.column_stack([xy, np.full(4, z)]))


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean()
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def test_pyramid_contours_are_squares_counter_clockwise() -> None:
    cs = st.contours(_pyramid(), 2.0, base=0.0, index_every=5)
    assert [c.level for c in cs] == [1002.0 + 2 * k for k in range(9)]
    for c in cs:
        level = c.level - 1000.0
        assert c.closed
        assert len(c.points) == 4
        assert c.length == pytest.approx(8 * R * (1 - level / H), abs=1e-9)
        assert np.all(c.points[:, 2] == c.level)
        assert _signed_area(c.points) > 0  # hill: higher ground on the left
        assert c.index == (round(c.level / 2.0) % 5 == 0)
    # A pit bottom runs clockwise.
    pit = st.contours(_pyramid(-1.0), 5.0, base=0.0)
    assert pit
    assert all(c.closed and _signed_area(c.points) < 0 for c in pit)


def test_contour_through_vertices_is_one_unbroken_line() -> None:
    gx, gy = np.meshgrid(np.arange(11) * 10.0, np.arange(11) * 10.0)
    pts = np.column_stack([gx.ravel() + E0, gy.ravel() + N0, gx.ravel() / 10.0])
    cs = st.contours(Surface(pts), 1.0)  # every level passes through a vertex row
    # Levels 1..10: a vertex on a level counts as above it, so the top level 10
    # is traced along the east edge and the bottom level 0 is not.
    assert [c.level for c in cs] == [float(k) for k in range(1, 11)]
    for c in cs:
        assert not c.closed
        assert c.length == pytest.approx(100.0, abs=1e-9)
        assert np.allclose(c.points[:, 0] - E0, c.level * 10.0)
        # Higher ground (east) on the left: the line runs north to south.
        assert c.points[0, 1] > c.points[-1, 1]


def test_contours_lie_on_random_surface() -> None:
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 500, (400, 2))
    z = 300 + 20 * np.sin(xy[:, 0] / 60) + 15 * np.cos(xy[:, 1] / 45) + rng.normal(0, 1, 400)
    s = Surface(np.column_stack([xy + OFF, z]))
    cs = st.contours(s, 2.5)
    assert len(cs) > 10
    for c in cs:
        # Ends of open lines sit on the hull edge, where the point lookup may
        # fall just outside; every other vertex is checked strictly.
        inner = c.points if c.closed else c.points[1:-1]
        zc, outside = s.elevation(inner[:, :2])
        assert outside == 0
        assert np.allclose(zc, c.level, atol=1e-6)
    explicit = st.contours(s, levels=[301.3])
    assert explicit
    assert {c.level for c in explicit} == {301.3}


def test_label_points_spacing_and_reading_angle() -> None:
    c = st.contours(_pyramid(), levels=[1010.0], index_every=0)[0]  # 200 m square
    labels = st.label_points(c, 50.0)
    assert len(labels) == 4
    for xyz, angle in labels:
        assert -90.0 < angle <= 90.0
        assert xyz[2] == 1010.0


def test_slope_and_aspect_of_pyramid_and_planes() -> None:
    sm = st.slope_map(_pyramid())
    assert np.allclose(sm.slope, math.degrees(math.atan(H / R)))
    assert sorted(sm.aspect.tolist()) == [0.0, 90.0, 180.0, 270.0]
    assert sm.plan_area.sum() == pytest.approx(10_000.0)
    assert sm.true_area.sum() == pytest.approx(4 * 0.5 * 100 * math.hypot(R, H))
    rows = st.slope_classes(sm, [0, 10, 20, 30, 90])
    assert [r.share for r in rows] == [0.0, 0.0, 1.0, 0.0]
    sectors = {r.label: r.plan_area for r in st.aspect_sectors(sm)}
    assert sectors == pytest.approx(
        {
            "K": 2500.0,
            "KD": 0.0,
            "D": 2500.0,
            "GD": 0.0,
            "G": 2500.0,
            "GB": 0.0,
            "B": 2500.0,
            "KB": 0.0,
        }
    )
    # z rising to the east at 30 degrees: slope 30, facing west (270).
    xy = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    east = Surface(np.column_stack([xy + OFF, xy[:, 0] * math.tan(math.radians(30))]))
    se = st.slope_map(east)
    assert np.allclose(se.slope, 30.0)
    assert np.allclose(se.aspect, 270.0)
    # Flat ground has no aspect and its own row.
    sf = st.slope_map(_flat(5.0))
    assert np.isnan(sf.aspect).all()
    assert st.aspect_sectors(sf)[-1].label == "düz"
    with pytest.raises(st.SurfaceToolError):
        st.slope_classes(sm, [10, 5])


def test_surface_grid_and_ascii_grid(tmp_path: Path) -> None:
    g = st.surface_grid(_pyramid(), 10.0)
    assert g.z.shape == (10, 10)
    assert np.allclose(g.slope, math.degrees(math.atan(H / R)))
    # Cell (row 0, col 0) centre (5, 5) is on the south face: z = 1000 + H * 5 / R.
    assert g.z[0, 0] == pytest.approx(1000.0 + H * 5.0 / R)
    assert g.aspect[0, 5] == pytest.approx(180.0)
    path = tmp_path / "z.asc"
    st.write_ascii_grid(path, g, g.z)
    lines = path.read_text().splitlines()
    assert lines[:2] == ["ncols 10", "nrows 10"]
    assert lines[2] == f"xllcorner {E0:.6f}"
    # First data row is the northern row.
    assert float(lines[6].split()[0]) == pytest.approx(g.z[-1, 0], abs=1e-4)


def test_profile_is_exact_trace() -> None:
    pyr, flat = _pyramid(), _flat(1010.0)
    line = np.array([[E0, N0 + 50], [E0 + 100, N0 + 50]])
    p = st.profile({"pyr": pyr, "flat": flat}, line)
    assert (p.chainage[0], p.chainage[-1]) == (0.0, 100.0)
    k = int(np.argmax(p.z["pyr"]))
    assert p.chainage[k] == pytest.approx(50.0)
    assert p.z["pyr"][k] == pytest.approx(1020.0)
    assert np.allclose(p.z["flat"], 1010.0)
    # Bent line over a random surface: linear interpolation of the profile
    # between its points reproduces the surface at every chainage.
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 300, (200, 2))
    s = Surface(np.column_stack([xy + OFF, rng.uniform(100, 150, 200)]))
    bent = np.array([[60, 40], [200, 120], [240, 260]], float) + OFF
    pr = st.profile({"s": s}, bent)
    seg = np.linalg.norm(np.diff(bent, axis=0), axis=1)
    for c in np.linspace(0, seg.sum(), 97):
        k = min(int(np.searchsorted(np.r_[0, np.cumsum(seg)], c, "right")) - 1, 1)
        f = (c - np.r_[0, np.cumsum(seg)][k]) / seg[k]
        q = bent[k] + f * (bent[k + 1] - bent[k])
        assert np.interp(c, pr.chainage, pr.z["s"]) == pytest.approx(
            s.elevation(q[None])[0][0], abs=1e-6
        )


def test_intersection_lines_known_answers() -> None:
    lines = st.intersection_lines(_pyramid(), _flat(1010.0))
    assert len(lines) == 1
    sq = lines[0]
    assert sq.closed
    assert sq.length == pytest.approx(8 * R * (1 - 10 / H), abs=1e-6)
    assert np.allclose(sq.points[:, 2], 1010.0)
    assert _signed_area(sq.points) > 0  # pyramid (a) above the plane on the left
    # A 0.5 gradient plane meets z = 10 along x = 20.
    xy = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    ramp = Surface(np.column_stack([xy + OFF, 0.5 * xy[:, 0]]))
    flat = Surface(np.column_stack([xy + OFF, np.full(4, 10.0)]))
    (line,) = st.intersection_lines(ramp, flat)
    assert not line.closed
    assert line.length == pytest.approx(100.0, abs=1e-6)
    assert np.allclose(line.points[:, 0] - E0, 20.0)
    # The ramp is above the plane east of x = 20; east is on the left of a
    # southward line, so the line starts at the north end.
    assert line.points[0, 1] > line.points[-1, 1]


def test_intersection_points_lie_on_both_random_surfaces() -> None:
    rng = np.random.default_rng(11)
    xy1, xy2 = rng.uniform(0, 400, (300, 2)), rng.uniform(0, 400, (250, 2))
    a = Surface(np.column_stack([xy1 + OFF, 200 + 0.05 * xy1[:, 0] + rng.normal(0, 2, 300)]))
    b = Surface(np.column_stack([xy2 + OFF, 210 + rng.normal(0, 2, 250)]))
    lines = st.intersection_lines(a, b)
    assert lines
    for line in lines:
        inner = line.points if line.closed else line.points[1:-1]  # ends on a hull edge
        za, out_a = a.elevation(inner[:, :2])
        zb, out_b = b.elevation(inner[:, :2])
        assert out_a == out_b == 0
        assert np.allclose(za, inner[:, 2], atol=1e-6)
        assert np.allclose(zb, inner[:, 2], atol=1e-6)


def test_offset_surface() -> None:
    s = st.offset_surface(_pyramid(), -0.5)
    assert s.points[:, 2].max() == pytest.approx(1019.5)
    assert np.allclose(s.points[:, :2], _pyramid().points[:, :2])


def _run(*argv: str) -> str:
    out = io.StringIO()
    assert main(list(argv), out) == 0, out.getvalue()
    return out.getvalue()


def test_surface_tool_commands(tmp_path: Path) -> None:
    p = str(tmp_path / "t.mcproj")
    _run("project", "create", p, "--name", "t")
    with Project.open(p) as project:
        save_surface(project, "topo", _flat(1010.0))
        save_surface(project, "pit", _pyramid())
        line = np.array([[E0, N0 + 50, 0], [E0 + 100, N0 + 50, 0]])
        strings.save_strings(project, "hat", [strings.String(line, False, "H")])
    text = _run(
        "surface", "contours", p, "pit", "--interval", "2", "--save", "kontur",
        "--dxf", str(tmp_path / "k.dxf"), "--label-spacing", "50",
    )  # fmt: skip
    assert "9 contour lines on 9 levels" in text
    dxf = (tmp_path / "k.dxf").read_text()
    assert dxf.count("POLYLINE") == 9
    assert "TEXT" in dxf
    assert "KONTUR_ANA" in dxf
    text = _run(
        "surface", "slope", p, "pit", "--classes", "0", "20", "30", "90",
        "--csv", str(tmp_path / "s.csv"), "--cell", "10", "--asc-slope", str(tmp_path / "s.asc"),
    )  # fmt: skip
    assert "20-30°  10,000.0" in text
    assert "K  2,500.0  25.0%" in text
    assert (tmp_path / "s.asc").read_text().startswith("ncols 10")
    text = _run(
        "surface", "profile", p, "hat", "pit", "topo",
        "--csv", str(tmp_path / "p.csv"), "--plot", str(tmp_path / "p.png"),
    )  # fmt: skip
    assert "pit: z 1000.00 to 1020.00 m" in text
    assert (tmp_path / "p.png").stat().st_size > 1000
    assert (tmp_path / "p.csv").read_text().splitlines()[0] == "chainage_m,x,y,z_pit,z_topo"
    text = _run("surface", "intersect", p, "pit", "topo", "--save", "tepe")
    assert "1 intersection lines, 1 closed, total length 200.0 m" in text
    _run("surface", "offset", p, "topo", "--dz", "2.5", "--save", "topo2")
    with Project.open(p) as project:
        assert len(strings.load_strings(project, "kontur")) == 9
        assert strings.load_strings(project, "tepe")[0].closed
