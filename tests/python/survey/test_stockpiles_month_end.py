"""OC08 stockpile volumes and OC09 month-end reconciliation against known answers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from minecore.blockmodel import BlockModel
from minecore.blockmodel.model import Grid
from minecore.project import Project
from minecore.surfaces import Surface
from minecore.survey.month_end import (
    GradeClass,
    mined_fraction,
    month_end,
    truck_reconciliation,
)
from minecore.survey.stockpiles import (
    Density,
    change_since_previous,
    save_survey,
    stockpile_volume,
    surveys,
)

E0, N0 = 500_000.0, 4_400_000.0
EN = np.array([E0, N0], np.float64)


def _cone(r: float, h: float, n: int, base: object = None) -> Surface:
    """Cone of radius r, height h over a ground z = base(x, y) (0 when None)."""
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rings = [np.zeros((1, 2))]
    for rr in np.linspace(0.0, 3 * r, 61)[1:]:
        rings.append(np.column_stack([rr * np.cos(ang), rr * np.sin(ang)]))
    xy = np.vstack(rings)
    rad = np.hypot(xy[:, 0], xy[:, 1])
    z = np.clip(h * (1.0 - rad / r), 0.0, None)
    if base is not None:
        z = z + base(xy[:, 0], xy[:, 1])  # type: ignore[operator]
    return Surface(np.column_stack([xy + np.array([E0, N0]), z]))


def _square(half: float) -> np.ndarray:
    sq = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return np.asarray(sq + EN, np.float64)


def test_catalogue_cone_523_6_m3_and_grid_converges() -> None:
    # Catalogue OC08: cone r = 10 m, h = 5 m on flat ground -> pi r^2 h / 3 = 523.6 m3.
    exact = math.pi * 100.0 * 5.0 / 3.0
    assert round(exact, 1) == 523.6
    n = 720
    pile = _cone(10.0, 5.0, n)
    # The TIN is an inscribed pyramid with n faces: V = h/3 * n/2 r^2 sin(2 pi / n).
    tin = 5.0 / 3.0 * n / 2.0 * 100.0 * math.sin(2 * math.pi / n)
    out = _square(15.0)
    v = stockpile_volume(pile, out, base_method="level", level=0.0)
    assert v.volume == pytest.approx(tin, rel=1e-9)
    assert v.volume == pytest.approx(exact, rel=1e-4)
    assert v.below_base == pytest.approx(0.0, abs=1e-6)
    assert v.coverage == pytest.approx(1.0, rel=1e-12)
    errors = [
        abs(stockpile_volume(pile, out, base_method="level", level=0.0, cell=c).volume - tin)
        for c in (2.0, 1.0, 0.1)
    ]
    # Converges (not strictly monotonic: cell centres alias with the cone apex).
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] / tin < 1e-5


def test_bases_on_sloping_ground_recover_the_cone() -> None:
    def ground(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 100.0 + 0.05 * x - 0.02 * y

    n = 720
    pile = _cone(10.0, 5.0, n, ground)
    tin = 5.0 / 3.0 * n / 2.0 * 100.0 * math.sin(2 * math.pi / n)
    out = _square(15.0)
    for method in ("boundary", "plane"):
        v = stockpile_volume(pile, out, base_method=method, spacing=2.0)
        assert v.volume == pytest.approx(tin, rel=1e-9), method
        assert v.below_base == pytest.approx(0.0, abs=1e-6)
    assert stockpile_volume(pile, out, base_method="plane").base_rms < 1e-9
    xy = np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]])
    ground_tin = Surface(np.column_stack([xy + EN, ground(xy[:, 0], xy[:, 1])]))
    v = stockpile_volume(pile, out, base_method="surface", base=ground_tin)
    assert v.volume == pytest.approx(tin, rel=1e-9)
    # A flat level at the lowest outline corner over-counts the sloping wedge.
    low = float(ground(np.array([-15.0]), np.array([15.0]))[0])
    assert stockpile_volume(pile, out, base_method="level", level=low).volume > tin + 100


def test_density_tonnes_and_change_since_previous(tmp_path: Path) -> None:
    pile = _cone(10.0, 5.0, 360)
    out = _square(15.0)
    rho = Density(1.8, "2026-10-28", "kum konisi")
    with Project.create(tmp_path / "s.mcproj", "s") as project:
        before = stockpile_volume(
            _cone(10.0, 4.0, 360),
            out,
            base_method="level",
            level=0.0,
            density=rho,
            name="ROM-1",
            material="cevher",
            survey_date="2026-10-31",
        )
        now = stockpile_volume(
            pile,
            out,
            base_method="level",
            level=0.0,
            density=rho,
            name="ROM-1",
            material="cevher",
            survey_date="2026-11-30",
        )
        assert now.tonnes == pytest.approx(now.volume * 1.8)
        save_survey(project, before)
        assert change_since_previous(project, before) is None
        save_survey(project, now)
        change = change_since_previous(project, now)
        assert change is not None
        assert change.previous_date == "2026-10-31"
        assert change.volume == pytest.approx(now.volume * (1 - 4 / 5), rel=1e-9)
        assert change.tonnes == pytest.approx(change.volume * 1.8)
        stored = surveys(project, "ROM-1")
        assert [s.survey_date for s in stored] == ["2026-10-31", "2026-11-30"]
        assert stored[1].density_source == "kum konisi"
        assert math.isnan(stored[1].base_rms)


def _flat(z: float, extent: float = 200.0) -> Surface:
    xy = np.array([[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]])
    return Surface(np.column_stack([xy + EN, np.full(4, z)]))


def _model(project: Project, azimuth: float = 0.0) -> BlockModel:
    grid = Grid((E0 - 50.0, N0 - 50.0, 80.0), (10.0, 10.0, 10.0), (10, 10, 4), azimuth)
    model = BlockModel.create(project, "BM", grid)
    k = np.arange(grid.blocks) // 100
    i = np.arange(grid.blocks) % 10
    grade = np.where(i < 5, 1.2, 0.2)  # ore in the west half
    grade[0] = np.nan
    model.set_attributes(
        {
            "au": grade,
            "rho": np.full(grid.blocks, 2.5),
            "pb": (i >= 5).astype(float) + 1,
            "k": k.astype(float),
        },
        {},
        "test model",
    )
    return model


def test_catalogue_flat_cut_one_block_high_equals_block_tonnes(tmp_path: Path) -> None:
    # Catalogue OC09: a flat cut of one block height gives the sum of the block tonnes.
    with Project.create(tmp_path / "m.mcproj", "m") as project:
        model = _model(project)
        classes = [GradeClass("cevher", 0.5)]
        r = month_end(model, _flat(110.0), _flat(100.0), density="rho", grade="au", classes=classes)
        layer = model.values("k") == 2
        expected_t = float(layer.sum()) * 1000.0 * 2.5
        assert r.tonnes == pytest.approx(expected_t, rel=1e-12)
        assert r.excavated == pytest.approx(400.0 * 400.0 * 10.0, rel=1e-12)
        assert r.model_volume == pytest.approx(100 * 1000.0, rel=1e-12)
        assert np.allclose(r.fraction[layer], 1.0)
        assert np.allclose(r.fraction[~layer], 0.0)
        mat = r.by_material()
        assert mat["cevher"][1] == pytest.approx(50 * 2500.0)
        assert mat["cevher"][2] == pytest.approx(1.2)
        assert mat["pasa"][1] == pytest.approx(50 * 2500.0)
        assert {row.bench for row in r.rows} == {100.0}
        # Half a block: tonnes prorate exactly for any sampling density.
        half = month_end(model, _flat(105.0), _flat(100.0), density=2.5, material="pb", samples=1)
        assert half.tonnes == pytest.approx(expected_t / 2, rel=1e-12)
        assert set(half.by_material()) == {"1", "2"}


def test_sloping_cut_converges_to_exact_volume_and_rotated_model(tmp_path: Path) -> None:
    # Cut between z = 110 + 0.1 x and z = 95: entirely inside the model in plan.
    xy = np.array([[-40.0, -40.0], [40.0, -40.0], [40.0, 40.0], [-40.0, 40.0]])
    start = Surface(np.column_stack([xy + EN, 110.0 + 0.1 * xy[:, 0]]))
    end = Surface(np.column_stack([xy + EN, np.full(4, 95.0)]))
    exact = 80.0 * 80.0 * 15.0
    with Project.create(tmp_path / "r.mcproj", "r") as project:
        model = _model(project)
        r = month_end(model, start, end, density=2.5, material="pb", samples=8)
        assert r.excavated == pytest.approx(exact, rel=1e-12)
        assert r.model_coverage == pytest.approx(1.0, rel=1e-12)
        rot = Grid((E0 - 65.0 * math.sqrt(2.0), N0, 80.0), (10.0, 10.0, 10.0), (13, 13, 4), 45.0)
        rm = BlockModel.create(project, "ROT", rot)
        f = mined_fraction(rm, start, end, samples=20)
        assert float(f.sum()) * 1000.0 == pytest.approx(exact, rel=2e-2)


def test_truck_reconciliation_suggests_load_factor() -> None:
    t = truck_reconciliation(95_000.0, 1000, 100.0)
    assert t.ratio == pytest.approx(0.95)
    assert t.suggested_load_factor == pytest.approx(95.0)
