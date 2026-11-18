"""Month-end progress survey and excavation reconciliation (catalogue OC09).

Between the surface surveyed at the start of a period and the one surveyed
at its end:

* the exact excavated (start above end) and filled (end above start, e.g.
  dumps) volumes come from the triangle overlay of
  ``minecore.survey.volumes``;
* each block's share of its volume inside the excavated solid is measured
  on ``samples x samples`` vertical lines spread evenly over the block (the
  user's choice). On each line the overlap of the block's height with the
  interval between the two surfaces is exact, so a level cut is exact for
  any ``samples``; sloping surfaces converge as ``samples`` grows. Shares
  below 1e-9 (round-off where a surface lies on a block face) are zero. Where
  either surface is missing the thickness is zero;
* tonnes and grades are prorated by that share and tabulated by phase,
  bench and material. Material comes either from a category attribute or
  from grade limits given by the user (no default cut-off);
* the surveyed tonnes are compared with the truck count: the ratio gives
  the suggested correction of the truck load factor.

The block-model volume is reported next to the exact survey volume; the
difference shows the excavation outside the model or the sampling error.
This is a production reconciliation, not a resource statement (rule 8).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from minecore._core import MineCoreError
from minecore.blockmodel import BlockModel
from minecore.surfaces import Surface
from minecore.survey.volumes import triangle_volume

Floats = npt.NDArray[np.float64]
UNESTIMATED = "(tenör yok)"
_TINY = 1e-9


class MonthEndError(MineCoreError):
    """Invalid month-end input."""


def mined_fraction(model: BlockModel, start: Surface, end: Surface, samples: int = 5) -> Floats:
    """Share (0..1) of each block's volume below ``start`` and above ``end``."""
    if not (isinstance(samples, int) and 1 <= samples <= 50):
        raise MonthEndError("samples per block side must be 1..50")
    grid = model.grid
    nx, ny, nz = grid.count
    sx, sy, sz = grid.size
    u = (np.arange(samples) + 0.5) / samples
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")  # ny x nx
    col = (jj * nx + ii).ravel()  # i fastest, then j
    ua, ub = np.meshgrid(u, u, indexing="ij")
    lx = (ii.ravel()[:, None] + ua.ravel()[None, :]) * sx
    ly = (jj.ravel()[:, None] + ub.ravel()[None, :]) * sy
    local = np.column_stack([lx.ravel(), ly.ravel(), np.zeros(lx.size)])
    xy = grid.to_world(local)[:, :2]
    # Only columns inside both surfaces' plan boxes can hold any excavation.
    lo = np.maximum(start.points[:, :2].min(axis=0), end.points[:, :2].min(axis=0))
    hi = np.minimum(start.points[:, :2].max(axis=0), end.points[:, :2].max(axis=0))
    box = ((xy >= lo) & (xy <= hi)).all(axis=1)
    top = np.full(len(xy), np.nan)
    bot = np.full(len(xy), np.nan)
    if box.any():
        sel = xy[box]
        ok = start.contains(sel) & end.contains(sel)
        t = np.where(ok, start.elevation(sel)[0], np.nan)
        b = np.where(ok, end.elevation(sel)[0], np.nan)
        top[box], bot[box] = t, b
    thick = np.nan_to_num(top - bot, nan=0.0)
    s2 = samples * samples
    top2, bot2 = top.reshape(-1, s2), bot.reshape(-1, s2)
    active = (thick.reshape(-1, s2) > 0).any(axis=1)
    frac = np.zeros(grid.blocks)
    if not active.any():
        return frac
    top2, bot2, cols = top2[active], bot2[active], col[active]
    base = grid.origin[2]
    for k in range(nz):
        zlo = base + k * sz
        zhi = zlo + sz
        ov = np.minimum(top2, zhi) - np.maximum(bot2, zlo)
        share = np.nan_to_num(np.clip(ov, 0.0, sz), nan=0.0).mean(axis=1) / sz
        share[share < _TINY] = 0.0  # interpolation round-off at a block face
        frac[k * nx * ny + cols] = share
    return frac


@dataclass(frozen=True)
class GradeClass:
    """Blocks with grade >= ``min_grade`` (and below the next higher class)."""

    name: str
    min_grade: float


def classify_material(
    grades: Floats, classes: list[GradeClass], below: str
) -> npt.NDArray[np.object_]:
    """Material name per block from user grade limits; ``below`` under the lowest."""
    if not classes:
        raise MonthEndError("give at least one material grade limit")
    if any(not math.isfinite(c.min_grade) for c in classes):
        raise MonthEndError("material grade limits must be finite")
    out = np.full(len(grades), below, dtype=object)
    for c in sorted(classes, key=lambda c: c.min_grade):
        out[np.isfinite(grades) & (grades >= c.min_grade)] = c.name
    out[~np.isfinite(grades)] = UNESTIMATED
    return out


@dataclass(frozen=True)
class MonthEndRow:
    phase: str  # "" when not split
    bench: float  # bench floor elevation, m
    material: str
    volume: float  # m3
    tonnes: float
    grade: float  # tonnage-weighted, NaN without a grade
    metal: float  # tonnes x grade


@dataclass(frozen=True)
class MonthEndReport:
    rows: list[MonthEndRow]
    excavated: float  # m3, exact between the surfaces
    filled: float  # m3, exact (end above start: dumps, backfill)
    model_volume: float  # m3 of excavation found inside the block model
    fraction: Floats  # per block

    @property
    def tonnes(self) -> float:
        return sum(r.tonnes for r in self.rows)

    @property
    def model_coverage(self) -> float:
        return self.model_volume / self.excavated if self.excavated > 0 else math.nan

    def by_material(self) -> dict[str, tuple[float, float, float]]:
        """material -> (volume m3, tonnes, tonnage-weighted grade)."""
        acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for r in self.rows:
            a = acc[r.material]
            a[0] += r.volume
            a[1] += r.tonnes
            if math.isfinite(r.grade):
                a[2] += r.metal
                a[3] += r.tonnes
        return {
            m: (a[0], a[1], a[2] / a[3] if a[3] > 0 else math.nan) for m, a in sorted(acc.items())
        }


def _labels(values: Floats) -> npt.NDArray[np.object_]:
    out = []
    for v in values:
        if not math.isfinite(v):
            out.append("-")
        else:
            out.append(str(int(v)) if float(v).is_integer() else f"{v:g}")
    return np.array(out, dtype=object)


def month_end(  # noqa: PLR0913 - one report, keyword options
    model: BlockModel,
    start: Surface,
    end: Surface,
    *,
    density: float | str,
    grade: str | None = None,
    classes: list[GradeClass] | None = None,
    below: str = "pasa",
    material: str | None = None,
    phase: str | None = None,
    bench_height: float | None = None,
    samples: int = 5,
) -> MonthEndReport:
    """Excavation between ``start`` and ``end`` intersected with the block model.

    ``density``: t/m3 or a density attribute. Material: ``material`` (a
    category attribute) or ``grade`` with ``classes`` (user limits) and the
    name ``below`` for blocks under the lowest limit. ``phase``: optional
    attribute (pushback, stage) to split by. ``bench_height`` groups block
    layers from the model base (default one bench per layer).
    """
    grid = model.grid
    if isinstance(density, str):
        rho = model.values(density)
    else:
        if not (math.isfinite(density) and density > 0):
            raise MonthEndError("density must be > 0 t/m3")
        rho = np.full(grid.blocks, float(density))
    g = model.values(grade) if grade else np.full(grid.blocks, np.nan)
    if material:
        mat = _labels(model.values(material))
    elif grade and classes:
        mat = classify_material(g, classes, below)
    else:
        raise MonthEndError("give a material attribute or a grade with grade limits")
    ph = _labels(model.values(phase)) if phase else np.full(grid.blocks, "", dtype=object)
    height = bench_height or grid.size[2]
    if not (math.isfinite(height) and height > 0):
        raise MonthEndError("bench height must be > 0")
    v = triangle_volume(start, end)
    frac = mined_fraction(model, start, end, samples)
    vol = frac * grid.block_volume
    tonnes = vol * np.nan_to_num(rho, nan=0.0)
    k = np.arange(grid.blocks) // (grid.count[0] * grid.count[1])
    bench = np.floor(k * grid.size[2] / height + 1e-9) * height + grid.origin[2]
    rows = []
    hit = np.flatnonzero(frac > 0)
    keys = sorted({(ph[i], bench[i], mat[i]) for i in hit}, key=lambda t: (t[0], -t[1], t[2]))
    for p, b, m in keys:
        sel = hit[(ph[hit] == p) & (bench[hit] == b) & (mat[hit] == m)]
        t = float(tonnes[sel].sum())
        known = sel[np.isfinite(g[sel])]
        tk = float(tonnes[known].sum())
        metal = float((tonnes[known] * g[known]).sum())
        rows.append(
            MonthEndRow(
                str(p),
                float(b),
                str(m),
                float(vol[sel].sum()),
                t,
                metal / tk if tk > 0 else math.nan,
                metal,
            )
        )
    return MonthEndReport(rows, v.cut, v.fill, float(vol.sum()), frac)


@dataclass(frozen=True)
class TruckReconciliation:
    survey_tonnes: float
    truck_tonnes: float  # loads x load factor
    loads: int
    load_factor: float  # t per load used for the truck count
    ratio: float  # survey / truck
    suggested_load_factor: float  # t per load that would match the survey


def truck_reconciliation(
    survey_tonnes: float, loads: int, load_factor: float
) -> TruckReconciliation:
    """Compare surveyed tonnes with the truck count (loads x t per load, user's values)."""
    if loads <= 0 or not (math.isfinite(load_factor) and load_factor > 0):
        raise MonthEndError("truck loads and load factor must be > 0")
    if not (math.isfinite(survey_tonnes) and survey_tonnes >= 0):
        raise MonthEndError("surveyed tonnes must be >= 0")
    truck = loads * load_factor
    return TruckReconciliation(
        survey_tonnes, truck, loads, load_factor, survey_tonnes / truck, survey_tonnes / loads
    )
