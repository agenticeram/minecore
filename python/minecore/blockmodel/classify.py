"""Confidence categories for estimated blocks from user criteria.

Each category (1 = highest confidence) has thresholds on the estimation
statistics stored with an estimate: mean distance to the samples used
(``<attr>_dist``), number of samples (``<attr>_n``) and optionally the kriging
variance relative to the variogram's total sill (``<attr>_kv``), the highest
search pass allowed (``<attr>_pass``) and the lowest slope of regression
(``<attr>_sor``). A block gets
the first category whose every threshold it meets; estimated blocks that meet
none get 0; unestimated blocks stay empty.

These categories are a geometric/statistical aid only. They are not a
resource classification under JORC, NI 43-101, PERC or similar codes; that
requires a Competent/Qualified Person (see CLAUDE.md rule 8).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from minecore.blockmodel.model import BlockModel, BlockModelError

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Criteria:
    max_distance: float  # mean sample distance, m
    min_samples: int
    max_kv_ratio: float | None = None  # kriging variance / total sill
    max_pass: int | None = None  # estimated in this search pass or earlier
    min_slope: float | None = None  # slope of regression at least this

    def __post_init__(self) -> None:
        if not self.max_distance > 0 or self.min_samples < 1:
            raise BlockModelError("distance must be > 0 and samples >= 1")
        if self.max_kv_ratio is not None and not self.max_kv_ratio > 0:
            raise BlockModelError("kriging variance ratio must be > 0")
        if self.max_pass is not None and self.max_pass < 1:
            raise BlockModelError("search pass must be >= 1")
        if self.min_slope is not None and not math.isfinite(self.min_slope):
            raise BlockModelError("slope of regression must be finite")


def total_sill(model: BlockModel, attribute: str) -> float | None:
    vg = model.attributes[attribute].get("variogram")
    if not vg:
        return None
    return float(vg["nugget"]) + sum(float(s["sill"]) for s in vg["structures"])


def categorise(  # noqa: PLR0913, PLR0917 - optional statistics
    distance: Floats,
    samples: Floats,
    kv_ratio: Floats | None,
    criteria: Sequence[Criteria],
    pass_number: Floats | None = None,
    slope: Floats | None = None,
) -> Floats:
    """Category per block (1-based, 0 = none, NaN = not estimated)."""
    out = np.full(len(distance), np.nan)
    estimated = np.isfinite(distance) & np.isfinite(samples) & (samples > 0)
    out[estimated] = 0.0
    for n in range(len(criteria), 0, -1):  # best category last so it wins
        c = criteria[n - 1]
        ok = estimated & (distance <= c.max_distance) & (samples >= c.min_samples)
        if c.max_kv_ratio is not None:
            if kv_ratio is None:
                raise BlockModelError("kriging variance criteria need an ordinary kriging estimate")
            ok &= np.isfinite(kv_ratio) & (kv_ratio <= c.max_kv_ratio)
        if c.max_pass is not None:
            if pass_number is None:
                raise BlockModelError("search pass criteria need an estimate with search passes")
            ok &= np.isfinite(pass_number) & (pass_number <= c.max_pass)
        if c.min_slope is not None:
            if slope is None:
                raise BlockModelError("slope of regression criteria need a kriging estimate")
            ok &= np.isfinite(slope) & (slope >= c.min_slope)
        out[ok] = float(n)
    return out


def classify(
    model: BlockModel, attribute: str, criteria: Sequence[Criteria], target: str | None = None
) -> dict[int, int]:
    """Write the category attribute (default ``kat_<attribute>``); block counts per category."""
    if not criteria:
        raise BlockModelError("at least one category is needed")
    attrs = model.attributes
    for suffix in ("_dist", "_n"):
        if f"{attribute}{suffix}" not in attrs:
            raise BlockModelError(f"{attribute!r} has no estimation statistics ({suffix})")
    kv_ratio = None
    if f"{attribute}_kv" in attrs:
        sill = total_sill(model, attribute)
        if sill is not None and math.isfinite(sill) and sill > 0:
            kv_ratio = model.values(f"{attribute}_kv") / sill

    def optional(suffix: str) -> Floats | None:
        name = f"{attribute}{suffix}"
        return model.values(name) if name in attrs else None

    cat = categorise(
        model.values(f"{attribute}_dist"),
        model.values(f"{attribute}_n"),
        kv_ratio,
        criteria,
        optional("_pass"),
        optional("_sor"),
    )
    name = target or f"kat_{attribute}"
    model.set_attributes(
        {name: cat},
        {
            name: {
                "unit": None,
                "source": attribute,
                "criteria": [c.__dict__ for c in criteria],
                "note": "confidence categories from user criteria; not a code-compliant "
                "resource classification",
            }
        },
        f"Categorise {attribute!r}",
    )
    finite = cat[np.isfinite(cat)].astype(int)
    return {n: int((finite == n).sum()) for n in range(len(criteria) + 1)}
