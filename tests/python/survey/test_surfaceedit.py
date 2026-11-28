"""Surface merge/patch, normal offset, smoothing, decimation, grid <-> TIN (E-CD-02).

Known answers: an inverted square pyramid pit (z = |x| + |y| - 5) cut into a
level topography holds 1/3 x 50 m2 x 5 m = 83.333 m3 below it; an inclined
plane offset along its normal rises d / cos(slope); a V-trough offset keeps
every vertex exactly d from both planes; smoothing leaves a plane unchanged.
Exactness on random triangulations: the merged surface equals the upper or
lower envelope (or the patch) at random points, and decimation keeps every
original vertex within the tolerance.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from minecore import surfaceedit as se
from minecore import surfacetools as st
from minecore.cli import main
from minecore.project import Project
from minecore.surfaces import Surface, load_surface, save_surface
from minecore.survey.volumes import triangle_volume

E0, N0 = 512_000.0, 4_400_000.0
OFF = np.array([E0, N0, 0.0])


def _grid(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lo: float = -10.0,
    hi: float = 10.0,
    n: int = 21,
) -> Surface:
    g = np.linspace(lo, hi, n)
    x, y = (v.ravel() for v in np.meshgrid(g, g))
    return Surface(np.column_stack([x, y, fn(x, y)]) + OFF)


def _random(seed: int, lo: float, hi: float, n: int = 300) -> Surface:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(lo, hi, (n, 2))
    return Surface(np.column_stack([xy, rng.normal(0, 3, n)]) + OFF)


def test_pit_into_topography_volume() -> None:
    topo = _grid(lambda x, y: np.zeros_like(x))
    pit = _grid(lambda x, y: np.abs(x) + np.abs(y) - 5.0)
    final, rep = se.merge_surfaces(topo, pit, "lower")
    vol = triangle_volume(topo, final)
    assert vol.cut == pytest.approx(250.0 / 3.0, rel=1e-9)
    assert vol.fill == pytest.approx(0.0, abs=1e-9)
    assert rep.area_from_b == pytest.approx(50.0, rel=1e-9)
    assert rep.area_both == pytest.approx(400.0, rel=1e-9)
    upper, rep = se.merge_surfaces(topo, pit, "upper")
    assert triangle_volume(upper, topo).cut == pytest.approx(
        triangle_volume(pit, topo).cut, rel=1e-9
    )


@pytest.mark.parametrize("mode", se.MODES)
def test_merge_is_exact_envelope(mode: str) -> None:
    a, b = _random(1, 0, 100), _random(2, 30, 130)
    m, rep = se.merge_surfaces(a, b, mode, tolerance=1e-5)
    rng = np.random.default_rng(5)
    q = rng.uniform(0, 130, (4000, 2)) + OFF[:2]
    za, zb, zm = se._z_at(a, q), se._z_at(b, q), se._z_at(m, q)
    ok = np.isfinite(za) & np.isfinite(zb) if mode != "patch" else np.isfinite(za)
    assert ok.sum() > 1000
    assert np.max(np.abs(zm[ok] - se._envelope(za, zb, mode)[ok])) < 1e-6
    # Away from the other surface each one is kept as it is (a step at the
    # other surface's edge is bridged by the triangles next to it).
    only_a = np.isfinite(za) & ~np.isfinite(zb)
    inner = only_a & (q[:, 0] - E0 < 15)
    assert np.max(np.abs(zm[inner] - za[inner])) < 1e-6
    assert rep.area_a_only > 0
    assert rep.area_b_only > 0


def test_normal_offset_plane_and_trough() -> None:
    plane = _grid(lambda x, y: 0.5 * x, 0, 100, 11)
    o, rep = se.normal_offset(plane, 2.0)
    q = np.random.default_rng(0).uniform(20, 80, (100, 2))
    z = se._z_at(o, q + OFF[:2])
    assert np.allclose(z, 0.5 * q[:, 0] + 2.0 * math.sqrt(1.25), atol=1e-9)
    assert rep.limited == 0
    trough = _grid(lambda x, y: 0.5 * np.abs(x - 50), 0, 100, 11)
    o, rep = se.normal_offset(trough, 2.0)
    p = o.points - OFF - np.array([50.0, 0, 0])
    inner = (np.abs(p[:, 0]) < 45) & (p[:, 1] > 5) & (p[:, 1] < 95)
    n1 = np.array([-0.5, 0, 1]) / math.sqrt(1.25)
    n2 = np.array([0.5, 0, 1]) / math.sqrt(1.25)
    d = np.minimum(np.abs(p @ n1), np.abs(p @ n2))
    assert np.allclose(d[inner], 2.0, atol=1e-9)
    assert rep.max_move == pytest.approx(2.0 * math.sqrt(1.25), rel=1e-9)  # mitre at the floor
    with pytest.raises(se.SurfaceEditError):
        se.normal_offset(trough, 0.0)


def test_smoothing_keeps_planes_and_respects_limit() -> None:
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 100, (2000, 2))
    plane = Surface(np.column_stack([xy, 0.3 * xy[:, 0] + 0.1 * xy[:, 1]]) + OFF)
    _, rep = se.smooth_surface(plane, 5)
    assert rep.max_change < 1e-9
    noisy = Surface(np.column_stack([xy, 0.3 * xy[:, 0] + rng.normal(0, 0.5, 2000)]) + OFF)
    s, rep = se.smooth_surface(noisy, 5)
    assert np.std(s.points[:, 2] - 0.3 * xy[:, 0]) < 0.25
    s, rep = se.smooth_surface(noisy, 5, max_change=0.2)
    assert rep.max_change <= 0.2 + 1e-12
    assert rep.limited > 0
    np.testing.assert_array_equal(s.points[:, :2], noisy.points[:, :2])


def test_decimation_within_tolerance() -> None:
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 100, (5000, 2))
    z = 10 * np.sin(xy[:, 0] / 20) + 5 * np.cos(xy[:, 1] / 15)
    s = Surface(np.column_stack([xy, z]) + OFF)
    d, rep = se.decimate_surface(s, 0.1)
    assert rep.points_after < rep.points_before / 4
    zd = se._z_at(d, s.points[:, :2])
    assert np.nanmax(np.abs(zd - s.points[:, 2])) <= 0.1 + 1e-9
    assert rep.max_error <= 0.1 + 1e-9


def test_grid_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (500, 2))
    s = Surface(np.column_stack([xy, 2 * xy[:, 0] + xy[:, 1]]) + OFF)
    g = st.surface_grid(s, 5.0)
    st.write_ascii_grid(tmp_path / "g.asc", g, g.z)
    back = se.read_ascii_grid(tmp_path / "g.asc")
    assert np.allclose(back.x, g.x)
    assert np.allclose(back.y, g.y)
    assert np.isnan(back.z).sum() == np.isnan(g.z).sum() > 0  # corners outside the hull
    tin, n = se.grid_surface(back)
    assert n == np.isfinite(g.z).sum()
    q = rng.uniform(15, 85, (200, 2))
    z = se._z_at(tin, q + OFF[:2])
    ok = np.isfinite(z)
    assert ok.sum() > 150
    assert np.max(np.abs(z[ok] - (2 * q[ok, 0] + q[ok, 1]))) < 1e-3  # 4 decimals in the file
    # A plane decimates to a handful of points.
    thin, _ = se.grid_surface(back, 0.01)
    assert len(thin.points) < n / 3


def test_cli(tmp_path: Path) -> None:
    path = tmp_path / "p.mcproj"
    with Project.create(path, "p") as project:
        save_surface(project, "topo", _grid(lambda x, y: np.zeros_like(x)))
        save_surface(project, "pit", _grid(lambda x, y: np.abs(x) + np.abs(y) - 5.0))
    p = str(path)
    asc = str(tmp_path / "t.asc")
    cmds = [
        ["surface", "merge", p, "pit", "topo", "--mode", "lower", "--save", "final"],
        ["surface", "normal-offset", p, "pit", "--distance", "0.5", "--save", "pit_n"],
        ["surface", "smooth", p, "final", "--passes", "2", "--max-change", "0.1", "--save", "sm"],
        ["surface", "decimate", p, "final", "--tolerance", "0.001", "--save", "thin"],
        ["surface", "to-grid", p, "final", "--cell", "1", "--asc", asc],
        ["surface", "from-grid", p, asc, "--save", "from_asc"],
    ]
    for cmd in cmds:
        out = io.StringIO()
        assert main(cmd, out=out) == 0, cmd
        assert "surface" in out.getvalue() or ".asc" in out.getvalue()
    with Project.open(path) as project:
        final = load_surface(project, "final")
        topo = load_surface(project, "topo")
        assert triangle_volume(topo, final).cut == pytest.approx(250.0 / 3.0, rel=1e-9)
        thin = load_surface(project, "thin")
        # Within tolerance x area: 0.001 m x 400 m2.
        assert triangle_volume(topo, thin).cut == pytest.approx(250.0 / 3.0, abs=0.4)
        assert len(load_surface(project, "from_asc").points) == 400
