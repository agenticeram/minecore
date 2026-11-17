"""Grade estimation at points or blocks: NN, inverse distance, ordinary and simple kriging.

All methods share one anisotropic search (:class:`Search`): candidate samples
are those inside the search ellipsoid, nearest first (by reduced distance),
at most ``max_samples``; a target with fewer than ``min_samples`` is left
unestimated (NaN) and counted.

Ordinary kriging solves, per target, ``[C 1; 1' 0] [w; mu] = [c; 1]`` where
``C`` holds sample-sample covariances and ``c`` sample-target covariances.
For blocks, ``c`` and the block variance average point covariances over a
regular discretisation of the block (``discretisation`` points per axis).
Kriging variance is ``C_bb - w'c - mu``. Simple kriging (known mean ``m``)
drops the unbiasedness row: ``C w = c``, estimate ``m + w'(z - m)``.

Kriging also reports, per target, the slope of regression of the true value
on the estimate, ``w'c / (w'c - mu)`` (1 for simple kriging), and the kriging
efficiency ``(BV - KV) / BV`` with ``BV = C_bb`` the block variance.

Several search passes (:func:`estimate` with a sequence of searches) estimate
first with the tightest search; targets left unestimated go to the next,
usually larger, search. The pass number of every estimate is returned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from minecore._core import estimate_neighbourhood
from minecore.estimation.variogram import Ellipsoid, EstimationError, VariogramModel
from minecore.jobs import Progress

Floats = npt.NDArray[np.float64]


class Method(StrEnum):
    NEAREST = "nearest"
    INVERSE_DISTANCE = "inverse_distance"
    ORDINARY_KRIGING = "ordinary_kriging"
    SIMPLE_KRIGING = "simple_kriging"

    @property
    def is_kriging(self) -> bool:
        return self in (Method.ORDINARY_KRIGING, Method.SIMPLE_KRIGING)


@dataclass(frozen=True)
class Search:
    ellipsoid: Ellipsoid
    min_samples: int = 4
    max_samples: int = 24
    max_per_hole: int = 0
    """0 = no limit; otherwise at most this many samples from one hole."""

    def __post_init__(self) -> None:
        if not (1 <= self.min_samples <= self.max_samples):
            raise EstimationError("search needs 1 <= min_samples <= max_samples")
        if self.max_per_hole < 0:
            raise EstimationError("max_per_hole must be >= 0")

    def expanded(
        self, factor: float, min_samples: int | None = None, max_samples: int | None = None
    ) -> Search:
        """This search with ranges times ``factor`` (a later search pass)."""
        if not factor > 0:
            raise EstimationError("search expansion factor must be > 0")
        e = self.ellipsoid
        return Search(
            replace(e, major=e.major * factor, semi=e.semi * factor, minor=e.minor * factor),
            self.min_samples if min_samples is None else min_samples,
            self.max_samples if max_samples is None else max_samples,
            self.max_per_hole,
        )


@dataclass(frozen=True)
class EstimationResult:
    value: Floats
    variance: Floats
    """Kriging variance (NaN for other methods)."""
    samples_used: npt.NDArray[np.int64]
    mean_distance: Floats
    """Mean anisotropic (reduced) distance of the samples used."""
    unestimated: int
    slope: Floats | None = None
    """Slope of regression (kriging only, else NaN)."""
    efficiency: Floats | None = None
    """Kriging efficiency (kriging only, else NaN)."""
    pass_number: npt.NDArray[np.int64] | None = None
    """Search pass that estimated each target (0 = not estimated)."""


def _discretise(size: tuple[float, float, float], n: tuple[int, int, int]) -> Floats:
    axes = [(np.arange(k) + 0.5) / k * s - s / 2 for s, k in zip(size, n, strict=True)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def estimate(  # noqa: PLR0913 - options are keyword-only
    samples: Floats,
    values: Floats,
    targets: Floats,
    search: Search | Sequence[Search],
    method: Method,
    *,
    variogram: VariogramModel | None = None,
    power: float = 2.0,
    mean: float | None = None,
    block_size: tuple[float, float, float] | None = None,
    discretisation: tuple[int, int, int] = (1, 1, 1),
    block_axes: Floats | None = None,
    hole_ids: npt.NDArray[np.int64] | None = None,
    exclude_hole: npt.NDArray[np.int64] | None = None,
    progress: Progress | None = None,
) -> EstimationResult:
    """Estimate ``values`` known at ``samples`` (n, 3) at ``targets`` (m, 3).

    ``search`` is one search or a sequence of passes: pass ``i + 1`` only
    estimates the targets that passes ``1..i`` left unestimated.
    ``mean`` is the known mean for simple kriging (default: sample mean).
    ``exclude_hole[t]`` (with ``hole_ids``) drops every sample of that hole
    from target ``t``'s neighbourhood: leave-one-hole-out cross-validation.
    ``block_axes`` (3 x 3, columns = world directions of the block axes)
    orients the block discretisation of a rotated block model.
    """
    passes = [search] if isinstance(search, Search) else list(search)
    if not passes:
        raise EstimationError("at least one search pass is needed")
    targets = np.asarray(targets, np.float64)
    if method is Method.SIMPLE_KRIGING and mean is None:
        v = np.asarray(values, np.float64)
        mean = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
    m = len(targets)
    value = np.full(m, np.nan)
    variance = np.full(m, np.nan)
    used = np.zeros(m, np.int64)
    mean_dist = np.full(m, np.nan)
    slope = np.full(m, np.nan)
    efficiency = np.full(m, np.nan)
    pass_no = np.zeros(m, np.int64)
    todo = np.arange(m)
    for number, s in enumerate(passes, start=1):
        if len(todo) == 0:
            break
        n = len(passes)
        part = None if progress is None else progress.span((number - 1) / n, number / n)
        r = _estimate_pass(
            samples,
            values,
            targets[todo],
            s,
            method,
            variogram=variogram,
            power=power,
            mean=0.0 if mean is None else mean,
            block_size=block_size,
            discretisation=discretisation,
            block_axes=block_axes,
            hole_ids=hole_ids,
            exclude_hole=None if exclude_hole is None else np.asarray(exclude_hole)[todo],
            progress=part,
        )
        done = np.isfinite(r.value)
        idx = todo[done]
        value[idx], variance[idx], used[idx] = r.value[done], r.variance[done], r.samples_used[done]
        mean_dist[idx] = r.mean_distance[done]
        assert r.slope is not None
        assert r.efficiency is not None
        slope[idx], efficiency[idx] = r.slope[done], r.efficiency[done]
        pass_no[idx] = number
        todo = todo[~done]
    return EstimationResult(value, variance, used, mean_dist, len(todo), slope, efficiency, pass_no)


def _estimate_pass(  # noqa: PLR0913 - options are keyword-only
    samples: Floats,
    values: Floats,
    targets: Floats,
    search: Search,
    method: Method,
    *,
    variogram: VariogramModel | None,
    power: float,
    mean: float,
    block_size: tuple[float, float, float] | None = None,
    discretisation: tuple[int, int, int] = (1, 1, 1),
    block_axes: Floats | None = None,
    hole_ids: npt.NDArray[np.int64] | None = None,
    exclude_hole: npt.NDArray[np.int64] | None = None,
    progress: Progress | None = None,
) -> EstimationResult:
    samples = np.asarray(samples, np.float64)
    values = np.asarray(values, np.float64)
    targets = np.asarray(targets, np.float64)
    ok = np.isfinite(values) & np.isfinite(samples).all(axis=1)
    samples, values = samples[ok], values[ok]
    holes = None if hole_ids is None else np.asarray(hole_ids)[ok]
    if method.is_kriging and variogram is None:
        raise EstimationError("kriging needs a variogram model")
    m = len(targets)
    value = np.full(m, np.nan)
    variance = np.full(m, np.nan)
    used = np.zeros(m, np.int64)
    mean_dist = np.full(m, np.nan)
    slope = np.full(m, np.nan)
    efficiency = np.full(m, np.nan)
    if len(values) == 0 or m == 0:
        return EstimationResult(value, variance, used, mean_dist, m, slope, efficiency)

    ell = search.ellipsoid
    transform = (ell.axes().T / np.array([ell.major, ell.semi, ell.minor])).astype(np.float64)
    tree = cKDTree(samples @ transform)
    k = min(len(values), search.max_samples * (4 if search.max_per_hole else 1))
    if exclude_hole is not None:
        if holes is None:
            raise EstimationError("exclude_hole needs hole_ids")
        k = min(len(values), search.max_samples * 4 + 200)
    if block_size is not None and method.is_kriging:
        offsets = _discretise(block_size, discretisation)
        if block_axes is not None:
            offsets = offsets @ np.asarray(block_axes, np.float64).T
    else:
        offsets = np.zeros((1, 3))
    structures = []
    if method.is_kriging:
        assert variogram is not None
        for s in variogram.structures:
            e = s.ranges
            rows = e.axes() / np.array([[e.major], [e.semi], [e.minor]])
            structures.append((s.type.value, float(s.sill), rows.tolist()))
    scaled_targets = targets @ transform
    chunk = 100_000
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        dist, idx = tree.query(scaled_targets[lo:hi], k=k, distance_upper_bound=1.0)
        dist = np.asarray(dist, np.float64).reshape(hi - lo, -1)
        idx = np.asarray(idx).reshape(hi - lo, -1)
        idx = np.where(idx >= len(values), np.iinfo(np.uint32).max, idx).astype(np.uint32)
        v, var, n_used, md, sl, ke = estimate_neighbourhood(
            method.value,
            samples,
            values,
            targets[lo:hi],
            idx,
            dist,
            min_samples=search.min_samples,
            max_samples=search.max_samples,
            max_per_hole=search.max_per_hole if holes is not None else 0,
            holes=None if holes is None else holes.astype(np.int64),
            exclude_hole=None
            if exclude_hole is None
            else np.asarray(exclude_hole[lo:hi], np.int64),
            power=power,
            nugget=0.0 if variogram is None else variogram.nugget,
            structures=structures,
            offsets=offsets,
            mean=mean,
        )
        value[lo:hi], variance[lo:hi], used[lo:hi], mean_dist[lo:hi] = v, var, n_used, md
        slope[lo:hi], efficiency[lo:hi] = sl, ke
        if progress is not None:
            progress.update(hi / m, f"{hi}/{m}")
    unestimated = int(np.count_nonzero(np.isnan(value)))
    return EstimationResult(value, variance, used, mean_dist, unestimated, slope, efficiency)


def inverse_distance_weights(distances: Floats, power: float) -> Floats:
    """Normalised IDW weights (used in tests and reports)."""
    w = 1.0 / np.asarray(distances, np.float64) ** power
    return w / w.sum()
