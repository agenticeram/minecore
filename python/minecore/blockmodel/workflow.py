"""Estimate block grades from drillhole intervals (usually composites)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from minecore.blockmodel.model import BlockModel
from minecore.drillholes import DrillholeDatabase, DrillholeError, FieldType, Flag, validate
from minecore.drillholes.workflow import hole_geometry
from minecore.estimation import EstimationResult, Method, Search, VariogramModel, estimate
from minecore.jobs import Progress
from minecore.project import Project

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SampleSet:
    xyz: Floats
    values: Floats
    hole_index: npt.NDArray[np.int64]
    unit: str | None


def interval_samples(
    db: DrillholeDatabase, table: str, field: str, *, below_detection_factor: float = 0.5
) -> SampleSet:
    """Interval midpoints and values of a numeric field, usable rows only.

    Uses stored ``mid_x/mid_y/mid_z`` (composites) when present, else desurveys.
    """
    report = validate(db)
    info = db.table_info(table)
    spec = next((f for f in info.fields if f.name == field), None)
    if spec is None or spec.type is not FieldType.NUMERIC:
        raise DrillholeError(f"table {table!r} has no numeric field {field!r}")
    data = db.table(table)
    usable = report.usable[table]
    values = np.asarray(data.column(field).to_numpy(zero_copy_only=False), np.float64).copy()
    flags = np.asarray(data.column(spec.flag_column).to_numpy(zero_copy_only=False))
    values[flags == Flag.BELOW_DETECTION] *= below_detection_factor
    values[flags >= Flag.MISSING] = np.nan
    ids = np.array(data.column("hole_id").to_pylist(), dtype=object)
    if {"mid_x", "mid_y", "mid_z"} <= set(data.column_names):
        xyz = np.column_stack(
            [
                np.asarray(data.column(c).to_numpy(zero_copy_only=False), float)
                for c in ("mid_x", "mid_y", "mid_z")
            ]
        )
    else:
        geometry = hole_geometry(db, report)
        mid = (
            np.asarray(data.column("from").to_numpy(), float)
            + np.asarray(data.column("to").to_numpy(), float)
        ) / 2
        xyz = np.full((data.num_rows, 3), np.nan)
        for hole, g in geometry.items():
            sel = usable & (ids == hole)
            if sel.any():
                xyz[sel] = g.at(mid[sel])
    keep = usable & np.isfinite(values) & np.isfinite(xyz).all(axis=1)
    names = {h: i for i, h in enumerate(sorted(set(ids[keep].tolist())))}
    holes = np.array([names[h] for h in ids[keep]], np.int64)
    return SampleSet(xyz[keep], values[keep], holes, spec.unit)


def estimate_blocks(  # noqa: PLR0913 - options are keyword-only
    model: BlockModel,
    samples: SampleSet,
    attribute: str,
    *,
    method: Method,
    search: Search | Sequence[Search],
    variogram: VariogramModel | None = None,
    power: float = 2.0,
    mean: float | None = None,
    discretisation: tuple[int, int, int] = (2, 2, 2),
    mask: npt.NDArray[np.bool_] | None = None,
    top_cut: float | None = None,
    progress: Progress | None = None,
) -> int:
    """Estimate ``attribute`` into ``model``; returns the number of unestimated blocks.

    ``mask`` limits estimation to selected blocks (e.g. below topography or
    inside a domain); the others stay empty and are not counted as unestimated.
    ``top_cut`` caps sample values before estimation (outlier treatment).
    With a ``mask``, existing values of the attribute outside the mask are
    kept, so domains can be estimated one after another into one attribute.
    ``search`` may be a sequence of passes (see :func:`estimation.estimate`);
    the pass of each block is stored as ``<attr>_pass``. Kriging also stores
    the kriging variance (``_kv``), slope of regression (``_sor``) and kriging
    efficiency (``_ke``). ``mean`` is the simple-kriging mean (default: the
    mean of the (top-cut) samples).
    """
    passes = [search] if isinstance(search, Search) else list(search)
    grid = model.grid
    targets = grid.centroids()
    selected = np.ones(grid.blocks, bool) if mask is None else np.asarray(mask, bool)
    sample_values = samples.values if top_cut is None else np.minimum(samples.values, top_cut)
    if method is Method.SIMPLE_KRIGING and mean is None:
        mean = float(np.mean(sample_values)) if len(sample_values) else 0.0
    sub = estimate(
        samples.xyz,
        sample_values,
        targets[selected],
        passes,
        method,
        variogram=variogram,
        power=power,
        mean=mean,
        block_size=grid.size,
        discretisation=discretisation if method.is_kriging else (1, 1, 1),
        block_axes=grid.axes if grid.rotated else None,
        hole_ids=samples.hole_index,
        progress=progress,
    )
    result = _expand(sub, selected)
    info: dict[str, dict[str, Any]] = {
        attribute: {
            "unit": samples.unit,
            "method": method.value,
            "variogram": variogram.to_json() if variogram else None,
            "search": _search_info(passes[0]),
            "passes": [_search_info(p) for p in passes],
            "power": power if method is Method.INVERSE_DISTANCE else None,
            "mean": mean if method is Method.SIMPLE_KRIGING else None,
            "top_cut": top_cut,
        },
        f"{attribute}_n": {"unit": None},
        f"{attribute}_dist": {"unit": None},
        f"{attribute}_pass": {"unit": None},
    }
    assert result.pass_number is not None
    values = {
        attribute: result.value,
        f"{attribute}_n": result.samples_used.astype(np.float64),
        f"{attribute}_dist": result.mean_distance,
        f"{attribute}_pass": np.where(result.pass_number > 0, result.pass_number, np.nan),
    }
    if method.is_kriging:
        assert result.slope is not None
        assert result.efficiency is not None
        values[f"{attribute}_kv"] = result.variance
        values[f"{attribute}_sor"] = result.slope
        values[f"{attribute}_ke"] = result.efficiency
        for suffix in ("_kv", "_sor", "_ke"):
            info[f"{attribute}{suffix}"] = {"unit": None}
    if mask is not None:
        # Estimating domain by domain into one attribute: keep the other domains.
        existing = model.attributes
        for name in list(values):
            if name in existing:
                old = model.values(name)
                values[name] = np.where(selected, values[name], old)
        if attribute in existing:
            previous = existing[attribute].get("domains") or [
                {k: v for k, v in existing[attribute].items() if k != "domains"}
            ]
            info[attribute]["domains"] = [*previous, dict(info[attribute])]
    model.set_attributes(values, info, f"Estimate {attribute!r} by {method.value}")
    return sub.unestimated


def _search_info(search: Search) -> dict[str, Any]:
    e = search.ellipsoid
    return {
        "ranges": [e.major, e.semi, e.minor],
        "angles": [e.azimuth, e.dip, e.rake],
        "min_samples": search.min_samples,
        "max_samples": search.max_samples,
        "max_per_hole": search.max_per_hole,
    }


def _expand(sub: EstimationResult, selected: npt.NDArray[np.bool_]) -> EstimationResult:
    def full(values: Any, fill: Any) -> Any:
        out = np.full(len(selected), fill, dtype=np.asarray(values).dtype)
        out[selected] = values
        return out

    return EstimationResult(
        full(sub.value, np.nan),
        full(sub.variance, np.nan),
        full(sub.samples_used, 0),
        full(sub.mean_distance, np.nan),
        sub.unestimated,
        None if sub.slope is None else full(sub.slope, np.nan),
        None if sub.efficiency is None else full(sub.efficiency, np.nan),
        None if sub.pass_number is None else full(sub.pass_number, 0),
    )


VARIOGRAM_KIND = "variogram_model"


def save_variogram(
    project: Project, name: str, model: VariogramModel, source: dict[str, Any]
) -> None:
    """Store a variogram model in the project (replaces one with the same name)."""
    meta = {"model": model.to_json(), "source": source}
    for record in project.objects(VARIOGRAM_KIND):
        if record.name == name:
            project.update_object(
                record.object_id, metadata=meta, description=f"Update variogram {name!r}"
            )
            return
    project.register_object(VARIOGRAM_KIND, name, metadata=meta)


def load_variograms(project: Project) -> dict[str, VariogramModel]:
    """All stored variogram models by name."""
    return {
        r.name: VariogramModel.from_json(r.metadata["model"])
        for r in project.objects(VARIOGRAM_KIND)
    }
