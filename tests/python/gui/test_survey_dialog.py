"""Stockpile and month-end tabs of the survey dialog (offscreen Qt)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from minecore import strings
from minecore.blockmodel import BlockModel
from minecore.blockmodel.model import Grid
from minecore.gui.survey import SurveyDialog
from minecore.project import Project
from minecore.surfaces import Surface, save_surface
from minecore.survey.stockpiles import surveys


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


def _cell(d: SurveyDialog, i: int, j: int) -> str:
    item = d.month_table.item(i, j)
    assert item is not None
    return item.text()


def _flat(z: float) -> Surface:
    xy = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    return Surface(np.column_stack([xy, np.full(4, z)]))


def test_survey_dialog_pile_and_month(qapp: QApplication, tmp_path: Path) -> None:
    with Project.create(tmp_path / "g.mcproj", "g") as project:
        save_surface(project, "a_ekim", _flat(110.0))
        save_surface(project, "b_kasim", _flat(100.0))
        ring = np.array([[10, 10, 0], [90, 10, 0], [90, 90, 0], [10, 90, 0.0]])
        strings.save_strings(project, "sinir", [strings.String(ring, True, "S")])
        grid = Grid((0.0, 0.0, 80.0), (10.0, 10.0, 10.0), (10, 10, 4))
        BlockModel.create(project, "BM", grid).set_attributes(
            {"au": np.where(np.arange(400) % 10 < 5, 1.0, 0.1)}, {}, "test"
        )
        parent = QWidget()
        d = SurveyDialog(project, parent)
        # Pile: a_ekim above a constant level of 100 m inside the 80 x 80 m outline.
        d.pile.setCurrentText("a_ekim")
        d.base.setCurrentIndex(3)
        d.level.setValue(100.0)
        d.density.setValue(2.0)
        d.pile_name.setText("ROM")
        d.pile_volume(save=True)
        assert "Hacim 64,000.0 m³" in d.pile_label.text()
        assert "Ton: 128,000" in d.pile_label.text()
        assert [s.volume for s in surveys(project, "ROM")] == [pytest.approx(64_000.0)]
        assert d.pile_table.rowCount() == 1
        d.start.setCurrentText("a_ekim")
        d.end.setCurrentText("b_kasim")
        d.grade.setCurrentText("au")
        d.classes.setText("cevher:0.5")
        d.loads.setValue(100)
        d.factor.setValue(2500.0)
        d.month()
        assert "Kazı 100,000.0 m³" in d.month_label.text()
        assert "toplam 270,000 t" in d.month_label.text()
        totals = {
            _cell(d, i, 2): _cell(d, i, 4)
            for i in range(d.month_table.rowCount())
            if _cell(d, i, 0) == "TOPLAM"
        }
        assert totals == {"cevher": "135,000", "pasa": "135,000"}
