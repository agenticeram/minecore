"""Surface tools dialog (offscreen Qt): every tab runs on a known pyramid."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from minecore import strings
from minecore.gui.surfacetools import SurfaceToolsDialog
from minecore.project import Project
from minecore.surfaces import Surface, load_surface, save_surface


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


def test_surface_tools_dialog(qapp: QApplication, tmp_path: Path) -> None:  # noqa: PLR0915
    pyramid = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [50, 50, 20]], float)
    flat = np.array([[0, 0, 10], [100, 0, 10], [100, 100, 10], [0, 100, 10]], float)
    with Project.create(tmp_path / "y.mcproj", "y") as project:
        save_surface(project, "pit", Surface(pyramid))
        save_surface(project, "topo", Surface(flat))
        line = np.array([[0, 50, 0], [100, 50, 0]], float)
        strings.save_strings(project, "hat", [strings.String(line, False, "H")])
        parent = QWidget()
        d = SurfaceToolsDialog(project, parent, "slope")
        assert d.tabs.currentIndex() == 1
        d.c_surface.setCurrentText("pit")
        d.c_interval.setValue(2.0)
        d.make_contours()
        assert len(strings.load_strings(project, "kontur")) == 9
        d.export_contours(str(tmp_path / "k.dxf"))
        assert "TEXT" in (tmp_path / "k.dxf").read_text()
        d.s_surface.setCurrentText("pit")
        d.s_classes.setText("0 20 30 90")
        d.compute_slope()
        item = d.slope_table.item(1, 3)
        assert item is not None
        assert item.text() == "100.0"
        d.export_slope(str(tmp_path))
        assert (tmp_path / "pit_slope.asc").exists()
        d.p_line.setCurrentText("hat")
        for i in range(d.p_surfaces.count()):
            d.p_surfaces.item(i).setSelected(True)
        d.draw()
        assert d.profile_result is not None
        assert np.nanmax(d.profile_result.z["pit"]) == pytest.approx(20.0)
        d.i_a.setCurrentText("pit")
        d.i_b.setCurrentText("topo")
        d.intersect()
        (sq,) = strings.load_strings(project, "kesisim")
        assert sq.closed
        assert np.allclose(sq.points[:, 2], 10.0)
        d.o_surface.setCurrentText("topo")
        d.o_dz.setValue(-1.5)
        d.o_name.setText("topo_alt")
        d.offset()
        assert load_surface(project, "topo_alt").points[:, 2].max() == pytest.approx(8.5)
        assert d.changed
        # Merge, normal offset, smoothing, decimation, grid.
        d.m_a.setCurrentText("pit")
        d.m_b.setCurrentText("topo")
        d.m_mode.setCurrentIndex(1)  # upper
        d.m_name.setText("ust")
        d.merge()
        assert load_surface(project, "ust").points[:, 2].min() == pytest.approx(10.0)
        d.e_surface.setCurrentText("ust")
        for slot, name in ((d.normal_offset, "n"), (d.smooth, "s"), (d.decimate, "t")):
            d.e_name.setText(name)
            slot()
            assert name in d.e_result.text()
        d.to_grid(str(tmp_path / "u.asc"))
        d.e_name.setText("g")
        d.from_grid(str(tmp_path / "u.asc"))
        assert len(load_surface(project, "g").points) > 3
        d2 = SurfaceToolsDialog(project, parent, "edit")
        assert d2.tabs.currentIndex() == 4
