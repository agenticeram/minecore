"""OC07: volumes between surfaces and by sections, against analytic answers."""

from __future__ import annotations

import numpy as np
import pytest

from minecore.strings import link_sections
from minecore.surfaces import Surface
from minecore.survey.volumes import (
    end_area_volume,
    grid_volume,
    prismoidal_volume,
    section_volumes,
    triangle_volume,
)


def _plane(n: int, seed: int, f: object, origin: tuple[float, float] = (0.0, 0.0)) -> Surface:
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.uniform(0, 100, (n, 2)), [[0, 0], [100, 0], [0, 100], [100, 100]]])
    z = f(xy[:, 0], xy[:, 1])  # type: ignore[operator]
    return Surface(np.column_stack([xy + np.array(origin), z]))


def test_catalogue_flat_surfaces_100000_m3_every_method() -> None:
    # Catalogue OC07: two flat surfaces 10 m apart over 100 x 100 m -> 100000 m3.
    top = _plane(60, 1, lambda x, y: 10.0 + 0 * x)
    base = _plane(90, 2, lambda x, y: 0 * x)
    tri = triangle_volume(top, base)
    assert tri.cut == pytest.approx(100_000.0, rel=1e-12)
    assert tri.fill == pytest.approx(0.0, abs=1e-6)
    assert tri.area == pytest.approx(10_000.0, rel=1e-12)
    assert grid_volume(top, base, 1.0).cut == pytest.approx(100_000.0, rel=1e-12)
    sec = section_volumes(top, base, np.array([[0.0, 50.0], [100.0, 50.0]]), 10.0, 60.0)
    assert sec.total("end_area").cut == pytest.approx(100_000.0, rel=1e-12)
    assert sec.total("prismoidal").cut == pytest.approx(100_000.0, rel=1e-12)


def test_catalogue_section_rules_2333_and_2500() -> None:
    # Catalogue OC07: A1 = 100, Am = 225, A2 = 400 m2, L = 10 m (square frustum).
    assert prismoidal_volume(10.0, 100.0, 225.0, 400.0) == pytest.approx(7000.0 / 3.0)
    assert round(prismoidal_volume(10.0, 100.0, 225.0, 400.0), 1) == 2333.3
    assert end_area_volume(10.0, 100.0, 400.0) == pytest.approx(2500.0)
    # Independent check: the frustum solid linked from its two square outlines.
    small = np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0.0]])
    large = np.array([[-10, -10, 10], [10, -10, 10], [10, 10, 10], [-10, 10, 10.0]])
    solid = link_sections([small, large])
    a, b, c = (solid.vertices[solid.triangles[:, i]] for i in range(3))
    volume = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    assert volume == pytest.approx(7000.0 / 3.0, rel=1e-12)


def test_cut_and_fill_split_on_the_zero_line() -> None:
    # z = x - 50 against z = 0 over 100 x 100: cut = fill = 50 x 50 / 2 x 100.
    a = _plane(70, 3, lambda x, y: x - 50.0)
    b = _plane(80, 4, lambda x, y: 0 * x)
    v = triangle_volume(a, b)
    assert v.cut == pytest.approx(125_000.0, rel=1e-12)
    assert v.fill == pytest.approx(125_000.0, rel=1e-12)
    assert v.cut_area == pytest.approx(5_000.0, rel=1e-12)
    assert v.net == pytest.approx(0.0, abs=1e-6)
    lv = triangle_volume(a, 0.0)
    assert lv.cut == pytest.approx(125_000.0, rel=1e-12)
    assert v.tonnes(2.5)[0] == pytest.approx(312_500.0, rel=1e-12)


def test_limit_polygon_with_hole_and_utm_coordinates() -> None:
    e0, n0 = 500_000.0, 4_400_000.0
    top = _plane(60, 1, lambda x, y: 10.0 + 0 * x, (e0, n0))
    base = _plane(90, 2, lambda x, y: 0 * x, (e0, n0))
    outer = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float) + np.array([e0, n0])
    hole = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], float) + np.array([e0, n0])
    v = triangle_volume(top, base, limit=[outer, hole])
    assert v.cut == pytest.approx(10.0 * (80 * 80 - 20 * 20), rel=1e-10)


def test_pyramid_prismoidal_exact_end_area_high_and_grid_converges() -> None:
    # Square pyramid, base 2R x 2R, height h: V = 4 R^2 h / 3. Section areas
    # across it are quadratic, so the prismoidal rule is exact.
    r, h = 50.0, 20.0
    base = np.array([[-r, -r, 0], [r, -r, 0], [r, r, 0], [-r, r, 0.0]])
    pyramid = Surface(np.vstack([base, [[0.0, 0.0, h]]]))
    flat = Surface(base)
    exact = 4 * r * r * h / 3
    assert triangle_volume(pyramid, flat).cut == pytest.approx(exact, rel=1e-12)
    sec = section_volumes(pyramid, flat, np.array([[-r, 0.0], [r, 0.0]]), 10.0, r)
    assert sec.total("prismoidal").cut == pytest.approx(exact, rel=1e-12)
    assert sec.total("end_area").cut > exact * 0.98  # end area differs from exact
    assert sec.total("end_area").cut != pytest.approx(exact, rel=1e-6)
    coarse = abs(grid_volume(pyramid, flat, 10.0).cut - exact)
    fine = abs(grid_volume(pyramid, flat, 1.0).cut - exact)
    assert fine < coarse or fine < 1e-6
