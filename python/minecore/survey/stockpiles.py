"""Stockpile volume and tonnes from a surveyed pile surface (catalogue OC08).

The pile is limited by a closed toe outline (drawn by the surveyor). Its
volume is the volume of the pile surface above a base inside the outline;
only positive thickness counts, and the volume below the base (negative
thickness) is reported separately as a check. Four bases:

* ``surface``: the surveyed ground before the pile was built (most accurate);
* ``boundary``: a triangulation of the outline vertices (sloping ground),
  each vertex taking its elevation from the outline itself when it is 3D,
  otherwise from the pile surface;
* ``plane``: the least squares plane through those vertices; the RMS of the
  vertex residuals is reported so a poor fit can be seen;
* ``level``: a constant elevation.

Volumes use the exact triangle overlay or the grid method of
``minecore.survey.volumes``. Tonnes need a density given by the user, kept
with the date and source of its measurement; nothing is assumed. Survey
results can be stored in the project, so the change since the previous
survey of the same pile is reported.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from minecore._core import MineCoreError
from minecore.project import Project
from minecore.surfaces import Surface, build_surface
from minecore.survey.volumes import VolumeReport, grid_volume, triangle_volume

Floats = npt.NDArray[np.float64]
KIND = "stockpile_survey"
BASE_METHODS = ("surface", "boundary", "plane", "level")


class StockpileError(MineCoreError):
    """Invalid stockpile input."""


@dataclass(frozen=True)
class Density:
    """Pile density in t/m3, with when and how it was measured (user's record)."""

    value: float
    measured: str = ""  # date of the measurement, e.g. 2026-11-02
    source: str = ""  # e.g. "sand replacement test", "truck weighbridge"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.value) and self.value > 0):
            raise StockpileError("density must be > 0 t/m3")


@dataclass(frozen=True)
class StockpileResult:
    name: str
    material: str
    survey_date: str
    base_method: str
    volume_method: str  # "triangle" or "grid <cell> m"
    volume: float  # m3 of pile above the base inside the outline
    below_base: float  # m3 where the pile surface is under the base (check)
    area: float  # m2 of the outline covered by both surfaces
    outline_area: float  # m2 inside the outline
    base_rms: float  # m, plane fit residual (NaN for other bases)
    density: float | None  # t/m3
    density_measured: str
    density_source: str

    @property
    def tonnes(self) -> float | None:
        return None if self.density is None else self.volume * self.density

    @property
    def coverage(self) -> float:
        """Share of the outline covered by the surfaces (1.0 when complete)."""
        return self.area / self.outline_area if self.outline_area > 0 else math.nan


def _ring(outline: Floats) -> Floats:
    r = np.asarray(outline, np.float64)
    if r.ndim != 2 or r.shape[1] not in (2, 3) or not np.isfinite(r).all():
        raise StockpileError("the pile outline must be finite x, y[, z] points")
    if len(r) > 1 and np.allclose(r[0, :2], r[-1, :2], rtol=0.0, atol=1e-9):
        r = r[:-1]
    if len(r) < 3:
        raise StockpileError("the pile outline needs at least three points")
    return r


def polygon_area(ring: Floats) -> float:
    x, y = ring[:, 0] - ring[:, 0].mean(), ring[:, 1] - ring[:, 1].mean()
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0)


def _densify(ring: Floats, spacing: float | None) -> Floats:
    if spacing is None:
        return ring
    if not (math.isfinite(spacing) and spacing > 0):
        raise StockpileError("outline point spacing must be > 0")
    out = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0), strict=True):
        n = max(1, math.ceil(float(np.linalg.norm(b[:2] - a[:2])) / spacing))
        t = np.arange(n)[:, None] / n
        out.append(a + t * (b - a))
    return np.vstack(out)


def outline_elevations(pile: Surface, outline: Floats, spacing: float | None = None) -> Floats:
    """Outline vertices (n x 3): z from a 3D outline, else from the pile surface.

    ``spacing`` (m, user's choice) adds vertices along the outline edges.
    """
    ring = _densify(_ring(outline), spacing)
    if ring.shape[1] == 3:
        return ring
    inside = pile.contains(ring)
    if not inside.all():
        raise StockpileError(
            f"{int((~inside).sum())} outline points lie outside the pile surface; "
            "give a 3D outline or a larger pile surface"
        )
    z, _ = pile.elevation(ring)
    return np.column_stack([ring, z])


def fit_plane(points: Floats) -> tuple[Floats, float]:
    """Least squares plane ``z = a x + b y + c`` through points and its RMS residual."""
    p = np.asarray(points, np.float64)
    o = p[:, :2].mean(axis=0)
    a = np.column_stack([p[:, :2] - o, np.ones(len(p))])
    coef, *_ = np.linalg.lstsq(a, p[:, 2], rcond=None)
    if np.linalg.matrix_rank(a) < 3:
        raise StockpileError("the outline points are collinear; no plane can be fitted")
    rms = float(np.sqrt(np.mean((a @ coef - p[:, 2]) ** 2)))
    c = coef[2] - coef[0] * o[0] - coef[1] * o[1]
    return np.array([coef[0], coef[1], c]), rms


def stockpile_base(  # noqa: PLR0913 - base inputs, keyword options
    pile: Surface,
    outline: Floats,
    method: str,
    *,
    base: Surface | None = None,
    level: float | None = None,
    spacing: float | None = None,
) -> tuple[Surface | float, float]:
    """The base under the pile and the plane fit RMS (NaN unless ``plane``)."""
    if method not in BASE_METHODS:
        raise StockpileError(f"base method must be one of {', '.join(BASE_METHODS)}")
    if method == "surface":
        if base is None:
            raise StockpileError("the surface base needs the surveyed ground surface")
        return base, math.nan
    if method == "level":
        if level is None or not math.isfinite(level):
            raise StockpileError("the level base needs a finite elevation")
        return float(level), math.nan
    pts = outline_elevations(pile, outline, spacing)
    if method == "boundary":
        surface, _ = build_surface(pts, boundary=[pts[:, :2]])
        return surface, math.nan
    plane, rms = fit_plane(pts)
    lo, hi = pts[:, :2].min(axis=0) - 1.0, pts[:, :2].max(axis=0) + 1.0
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    z = corners @ plane[:2] + plane[2]
    return Surface(np.column_stack([corners, z])), rms


def stockpile_volume(  # noqa: PLR0913 - one survey, keyword options
    pile: Surface,
    outline: Floats,
    *,
    base_method: str,
    base: Surface | None = None,
    level: float | None = None,
    spacing: float | None = None,
    cell: float | None = None,
    density: Density | None = None,
    name: str = "",
    material: str = "",
    survey_date: str = "",
) -> StockpileResult:
    """Pile volume above the chosen base inside ``outline``.

    ``cell`` (m) selects the grid method; without it the exact triangle
    overlay is used.
    """
    ring = _ring(outline)
    floor, rms = stockpile_base(pile, ring, base_method, base=base, level=level, spacing=spacing)
    limit = [ring[:, :2]]
    v: VolumeReport = (
        triangle_volume(pile, floor, limit=limit)
        if cell is None
        else grid_volume(pile, floor, cell, limit=limit)
    )
    return StockpileResult(
        name=name,
        material=material,
        survey_date=survey_date,
        base_method=base_method,
        volume_method=v.method,
        volume=v.cut,
        below_base=v.fill,
        area=v.area,
        outline_area=polygon_area(ring),
        base_rms=rms,
        density=None if density is None else density.value,
        density_measured="" if density is None else density.measured,
        density_source="" if density is None else density.source,
    )


# -- survey records -----------------------------------------------------------


def save_survey(project: Project, result: StockpileResult) -> str:
    """Store a pile survey result (pile name + survey date must be given)."""
    if not result.name.strip() or not result.survey_date.strip():
        raise StockpileError("a stored survey needs a pile name and a survey date")
    meta = {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in asdict(result).items()
    }
    meta["format"] = 1
    label = f"{result.name.strip()} {result.survey_date.strip()}"
    existing = [r for r in project.objects(KIND) if r.name == label]
    if existing:
        project.update_object(existing[0].object_id, metadata=meta)
        return existing[0].object_id
    return project.register_object(KIND, label, metadata=meta).object_id


def _from_meta(meta: dict[str, object]) -> StockpileResult:
    fields = {k: meta[k] for k in StockpileResult.__dataclass_fields__}
    if fields["base_rms"] is None:
        fields["base_rms"] = math.nan
    return StockpileResult(**fields)  # type: ignore[arg-type]


def surveys(project: Project, name: str | None = None) -> list[StockpileResult]:
    """Stored surveys, by pile name then survey date."""
    rows = [_from_meta(r.metadata) for r in project.objects(KIND)]
    if name is not None:
        rows = [r for r in rows if r.name == name]
    return sorted(rows, key=lambda r: (r.name, r.survey_date))


@dataclass(frozen=True)
class Change:
    previous_date: str
    volume: float  # m3, current minus previous
    tonnes: float | None


def change_since_previous(project: Project, result: StockpileResult) -> Change | None:
    """Change against the latest stored survey of the same pile before this one."""
    earlier = [r for r in surveys(project, result.name) if r.survey_date < result.survey_date]
    if not earlier:
        return None
    prev = earlier[-1]
    t_now, t_prev = result.tonnes, prev.tonnes
    dt = None if t_now is None or t_prev is None else t_now - t_prev
    return Change(prev.survey_date, result.volume - prev.volume, dt)
