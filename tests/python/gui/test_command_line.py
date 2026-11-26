"""Workbench command line: command words and headless commands (offscreen Qt)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from minecore.blockmodel import BlockModel
from minecore.gui.app import MainWindow
from minecore.gui.command_line import AliasTable, normalise, workbench_aliases
from minecore.project import Project

SAMPLE = Path(__file__).resolve().parents[3] / "python" / "minecore" / "gui" / "sample"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


def test_words_match_without_case_accents_or_full_spelling() -> None:
    assert normalise(" kompozİt ") == "KOMPOZIT"
    assert normalise("ölçme") == "OLCME"
    calls: list[str] = []
    table = AliasTable()
    table.add("KESTIRIM", "", lambda: calls.append("k"))
    table.add("KDOGRULA", "", lambda: calls.append("d"))
    table.add("KOMPOZIT", "", lambda: calls.append("c"))
    assert table.match("kest")[0] is not None
    assert table.match("K") == (None, ["KDOGRULA", "KESTIRIM", "KOMPOZIT"])
    assert table.match("xyz") == (None, [])
    with pytest.raises(ValueError, match="duplicate"):
        table.add("kompozit", "", print)


def test_command_line_runs_words_and_commands_on_the_open_project(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    log = window.command.log
    window.run_command("drillholes validate")
    assert "önce bir proje" in log.toPlainText()

    window._set_project(Project.create(tmp_path / "p.mcproj", "p"))
    window.run_command(f'drillholes import "{SAMPLE}"')
    window.run_command("drillholes composite assay AU_PPM --length 2")
    window.run_command("bm create BM --source assay_komp_2m/AU_PPM --size 20 20 10")
    window.run_command("bm estimate BM assay_komp_2m/AU_PPM au --method idw --radius 150")
    window.run_command("minecore bm report BM au --cutoffs 0 --density 2.5")
    text = log.toPlainText()
    assert "Komut başarısız" not in text, text
    assert "tonnes" in text
    assert window.project is not None  # reopened after each command
    assert window.db is not None
    assert "au" in BlockModel.open(window.project, "BM").attributes

    window.run_command("bm report BM nothing --cutoffs 0")
    assert "Komut başarısız (kod 1)" in log.toPlainText()
    window.run_command("bm report --nonsense")
    assert "Komut başarısız (kod 2)" in log.toPlainText()

    opened: list[bool] = []
    monkeypatch.setattr(window, "composite", lambda: opened.append(True))
    window.aliases = workbench_aliases(window)
    window.run_command("komp")
    assert opened == [True]
    window.run_command("K")
    assert "Birden çok komut" in log.toPlainText()
    window.run_command("YARDIM")
    assert "KOMPOZIT" in log.toPlainText()
    window.run_command("TEMIZLE")
    assert log.toPlainText() == ""
    window.close()
