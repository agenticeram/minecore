# mypy: disable-error-code="union-attr,index"
"""Search passes, simple kriging far from data, exact kriging at samples and category passes."""

from __future__ import annotations

import numpy as np
import pytest

from minecore.blockmodel.classify import Criteria, categorise
from minecore.estimation import (
    Ellipsoid,
    Method,
    ModelType,
    Search,
    Structure,
    VariogramModel,
    estimate,
)


def sph(nugget: float, sill: float, a: float) -> VariogramModel:
    return VariogramModel(nugget, (Structure(ModelType.SPHERICAL, sill, Ellipsoid.isotropic(a)),))


def line_samples() -> tuple[np.ndarray, np.ndarray]:
    xyz = np.column_stack([np.arange(11.0), np.zeros(11), np.zeros(11)])
    return xyz, 2.0 * xyz[:, 0]


def test_second_pass_estimates_far_target_only() -> None:
    xyz, values = line_samples()
    targets = np.array([[5.0, 0, 0], [100.0, 0, 0]])
    pass1 = Search(Ellipsoid.isotropic(20.0), min_samples=1, max_samples=24)
    pass2 = pass1.expanded(10.0)

    two = estimate(xyz, values, targets, [pass1, pass2], Method.NEAREST)
    assert two.pass_number.tolist() == [1, 2]
    assert two.unestimated == 0
    assert np.isfinite(two.value).all()
    # Nearest sample to x=100 is x=10, value 20.
    assert two.value[1] == pytest.approx(20.0)

    one = estimate(xyz, values, targets, pass1, Method.NEAREST)
    assert one.pass_number.tolist() == [1, 0]
    assert one.unestimated == 1
    assert np.isnan(one.value[1])


def test_simple_kriging_beyond_range_returns_mean() -> None:
    xyz, values = line_samples()
    # Variogram range 10; every sample is >= 990 away from the target, yet inside the search.
    v = sph(0.0, 1.0, 10.0)
    search = Search(Ellipsoid.isotropic(2000.0), min_samples=1, max_samples=24)
    r = estimate(
        xyz,
        values,
        np.array([[1000.0, 0, 0]]),
        search,
        Method.SIMPLE_KRIGING,
        variogram=v,
        mean=7.5,
    )
    assert r.unestimated == 0
    assert r.value[0] == pytest.approx(7.5, abs=1e-12)
    assert r.efficiency[0] == pytest.approx(0.0, abs=1e-9)


def test_ordinary_kriging_at_sample_is_exact() -> None:
    xyz, values = line_samples()
    v = sph(0.0, 1.0, 20.0)
    search = Search(Ellipsoid.isotropic(50.0), min_samples=1, max_samples=24)
    r = estimate(xyz, values, xyz[[5]], search, Method.ORDINARY_KRIGING, variogram=v)
    assert r.value[0] == pytest.approx(values[5], abs=1e-8)
    assert r.slope[0] == pytest.approx(1.0, abs=1e-6)
    assert r.efficiency[0] == pytest.approx(1.0, abs=1e-6)


def test_categorise_by_pass_and_by_slope() -> None:
    dist = np.array([10.0, 10.0, 10.0])
    samples = np.array([5.0, 5.0, 5.0])

    # Category 1 needs estimation in pass 1; block 2 was estimated in pass 2.
    by_pass = [Criteria(max_distance=50.0, min_samples=1, max_pass=1), Criteria(100.0, 1)]
    cat = categorise(dist, samples, None, by_pass, pass_number=np.array([1.0, 2.0, 1.0]))
    assert cat.tolist() == [1.0, 2.0, 1.0]

    # Category 1 needs slope >= 0.5; block 3 (slope 0.2) falls to category 2.
    by_slope = [Criteria(max_distance=50.0, min_samples=1, min_slope=0.5), Criteria(100.0, 1)]
    cat = categorise(dist, samples, None, by_slope, slope=np.array([0.8, 0.8, 0.2]))
    assert cat.tolist() == [1.0, 1.0, 2.0]
