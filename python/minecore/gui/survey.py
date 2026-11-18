"""Stockpile volume (OC08) and month-end progress survey (OC09)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from minecore import strings, surfaces
from minecore.blockmodel import BlockModel
from minecore.gui.layout import fit_to_screen
from minecore.gui.resource import _row, _spin, block_models
from minecore.project import Project
from minecore.survey import month_end as me
from minecore.survey import stockpiles as sp

BASES = [
    ("surface", "Önceki ölçülmüş taban yüzeyi (en doğru)"),
    ("boundary", "Sınır noktalarından üçgenleme (eğimli taban)"),
    ("plane", "Sınır noktalarına uydurulan düzlem"),
    ("level", "Sabit kot"),
]
PILE_HEADERS = ["Yığın", "Tarih", "Malzeme", "Hacim m³", "Yoğunluk", "Ton", "Taban", "Yöntem"]
MONTH_HEADERS = ["Aşama", "Basamak", "Malzeme", "Hacim m³", "Ton", "Tenör"]


def _table(headers: list[str], rows: list[list[str]]) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            t.setItem(i, j, QTableWidgetItem(v))
    return t


def _fill_table(t: QTableWidget, rows: list[list[str]]) -> None:
    t.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            t.setItem(i, j, QTableWidgetItem(v))


def _pick(box: QComboBox) -> str | None:
    """The chosen attribute, or None for the first entry ("none")."""
    return box.currentText() if box.currentIndex() > 0 else None


class SurveyDialog(QDialog):
    def __init__(self, project: Project, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ölçme: stok hacmi ve ay sonu ilerleme")
        fit_to_screen(self, 1050, 700)
        self.project = project
        tabs = QTabWidget()
        QVBoxLayout(self).addWidget(tabs)
        tabs.addTab(self._pile_tab(), "Stok hacmi")
        tabs.addTab(self._month_tab(), "Ay sonu ilerleme")

    def _guard(self, slot: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                slot()
            except Exception as exc:
                QMessageBox.critical(self, "Ölçme", str(exc))

        return run

    # -- stockpile -------------------------------------------------------------

    def _pile_tab(self) -> QWidget:  # noqa: PLR0915 - form
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        lay.addLayout(form)
        names = surfaces.surface_names(self.project)
        self.pile = QComboBox()
        self.pile.addItems(names)
        self.outline = QComboBox()
        self.outline.addItems(strings.string_names(self.project))
        self.base = QComboBox()
        for _, label in BASES:
            self.base.addItem(label)
        self.base_surface = QComboBox()
        self.base_surface.addItems(names)
        self.level = _spin(0.0, -1e5, 1e5, 1.0, 3)
        self.spacing = _spin(0.0, 0.0, 1000.0, 1.0, 2)
        self.spacing.setSpecialValueText("yalnız köşeler")
        self.cell = _spin(0.0, 0.0, 100.0, 0.25, 2)
        self.cell.setSpecialValueText("üçgen kesiştirme (tam)")
        self.density = _spin(0.0, 0.0, 10.0, 0.05, 3)
        self.density.setSpecialValueText("girilmedi")
        self.density_date = QLineEdit()
        self.density_date.setPlaceholderText("ölçüm tarihi, ör. 2026-11-02")
        self.density_source = QLineEdit()
        self.density_source.setPlaceholderText("yöntem, ör. kum konisi")
        self.pile_name = QLineEdit()
        self.material = QLineEdit()
        self.date = QDateEdit(QDate.currentDate())
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.date.setCalendarPopup(True)
        form.addRow("Yığın yüzeyi (drone / tarama)", self.pile)
        form.addRow("Yığın sınırı (kapalı çizgi seti)", self.outline)
        form.addRow("Taban yöntemi", self.base)
        form.addRow("Taban yüzeyi / sabit kot", _row(self.base_surface, self.level))
        form.addRow("Sınır nokta aralığı (m)", self.spacing)
        form.addRow("Izgara hücresi (m)", self.cell)
        form.addRow("Yoğunluk (t/m³)", _row(self.density, self.density_date, self.density_source))
        form.addRow("Yığın adı / malzeme", _row(self.pile_name, self.material))
        form.addRow("Ölçüm tarihi", self.date)
        run = QPushButton("Hacmi hesapla")
        run.clicked.connect(self._guard(self.pile_volume))
        save = QPushButton("Hesapla ve projeye kaydet")
        save.clicked.connect(self._guard(lambda: self.pile_volume(save=True)))
        form.addRow(_row(run, save))
        self.pile_label = QLabel()
        self.pile_label.setWordWrap(True)
        form.addRow(self.pile_label)
        self.pile_table = _table(PILE_HEADERS, [])
        lay.addWidget(QLabel("<b>Kayıtlı stok ölçümleri</b>"))
        lay.addWidget(self.pile_table, 1)
        self.base.currentIndexChanged.connect(self._base_changed)
        self._base_changed(0)
        self._list_piles()
        return w

    def _base_changed(self, i: int) -> None:
        key = BASES[i][0]
        self.base_surface.setEnabled(key == "surface")
        self.level.setEnabled(key == "level")
        self.spacing.setEnabled(key in ("boundary", "plane"))

    def _list_piles(self) -> None:
        rows = [
            [
                r.name,
                r.survey_date,
                r.material,
                f"{r.volume:,.1f}",
                "" if r.density is None else f"{r.density:g}",
                "" if r.tonnes is None else f"{r.tonnes:,.0f}",
                r.base_method,
                r.volume_method,
            ]
            for r in sp.surveys(self.project)
        ]
        _fill_table(self.pile_table, rows)

    def pile_volume(self, save: bool = False) -> None:
        if not self.pile.currentText() or not self.outline.currentText():
            raise RuntimeError("Yığın yüzeyi ve sınır çizgisi gerekli.")
        key = BASES[self.base.currentIndex()][0]
        closed = [
            s.points
            for s in strings.load_strings(self.project, self.outline.currentText())
            if s.closed
        ]
        if not closed:
            raise RuntimeError("Sınır çizgi setinde kapalı çizgi yok.")
        outline: Any = closed[0]
        if not (outline[:, 2] != 0).any():
            outline = outline[:, :2]
        rho = self.density.value()
        r = sp.stockpile_volume(
            surfaces.load_surface(self.project, self.pile.currentText()),
            outline,
            base_method=key,
            base=(
                surfaces.load_surface(self.project, self.base_surface.currentText())
                if key == "surface"
                else None
            ),
            level=self.level.value() if key == "level" else None,
            spacing=self.spacing.value() or None,
            cell=self.cell.value() or None,
            density=(
                sp.Density(rho, self.density_date.text(), self.density_source.text())
                if rho > 0
                else None
            ),
            name=self.pile_name.text().strip(),
            material=self.material.text().strip(),
            survey_date=self.date.date().toString("yyyy-MM-dd"),
        )
        text = (
            f"Hacim {r.volume:,.1f} m³ (taban altında kalan {r.below_base:,.1f} m³); "
            f"sınır alanı {r.outline_area:,.1f} m², kapsanan %{100 * r.coverage:.2f}."
        )
        if r.tonnes is not None:
            text += f" Ton: {r.tonnes:,.0f} ({r.density:g} t/m³)."
        if math.isfinite(r.base_rms):
            text += f" Düzlem uyumu KOK hatası {r.base_rms:.3f} m."
        if r.coverage < 0.999:
            text += " Uyarı: yüzeyler sınırın tamamını kapsamıyor."
        change = sp.change_since_previous(self.project, r) if r.name else None
        if change is not None:
            text += f" Önceki ölçüme ({change.previous_date}) göre {change.volume:+,.1f} m³"
            text += "." if change.tonnes is None else f", {change.tonnes:+,.0f} t."
        if save:
            sp.save_survey(self.project, r)
            self._list_piles()
            text += " Kaydedildi."
        self.pile_label.setText(text)

    # -- month end -------------------------------------------------------------

    def _month_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        lay.addLayout(form)
        names = surfaces.surface_names(self.project)
        self.start = QComboBox()
        self.start.addItems(names)
        self.end = QComboBox()
        self.end.addItems(names)
        if len(names) > 1:
            self.end.setCurrentIndex(1)
        self.model = QComboBox()
        self.model.addItems(block_models(self.project))
        self.density_attr = QComboBox()
        self.rho = _spin(2.7, 0.5, 10.0, 0.05, 3)
        self.grade = QComboBox()
        self.classes = QLineEdit()
        self.classes.setPlaceholderText("ad:alt tenör; ör. cevher:0.5; düşük tenör:0.3")
        self.below = QLineEdit("pasa")
        self.material_attr = QComboBox()
        self.phase = QComboBox()
        self.bench = _spin(0.0, 0.0, 100.0, 1.0, 1)
        self.bench.setSpecialValueText("blok yüksekliği")
        self.samples = QSpinBox(minimum=1, maximum=50, value=5)
        self.loads = QSpinBox(minimum=0, maximum=10_000_000, value=0)
        self.loads.setSpecialValueText("yok")
        self.factor = _spin(0.0, 0.0, 1000.0, 1.0, 2)
        form.addRow("Dönem başı yüzeyi / dönem sonu yüzeyi", _row(self.start, self.end))
        form.addRow("Blok modeli", self.model)
        form.addRow("Yoğunluk (öznitelik / sabit t/m³)", _row(self.density_attr, self.rho))
        form.addRow("Tenör ve malzeme sınırları", _row(self.grade, self.classes))
        form.addRow("En alt sınırın altı", self.below)
        form.addRow("veya malzeme özniteliği", self.material_attr)
        form.addRow("Aşamaya göre ayır", self.phase)
        form.addRow("Basamak yüksekliği (m)", self.bench)
        form.addRow("Blok kenarı başına düşey çizgi", self.samples)
        form.addRow("Kamyon sefer sayısı / sefer başı ton", _row(self.loads, self.factor))
        run = QPushButton("Ay sonu kazısını hesapla")
        run.clicked.connect(self._guard(self.month))
        form.addRow(run)
        self.month_label = QLabel()
        self.month_label.setWordWrap(True)
        form.addRow(self.month_label)
        self.month_table = _table(MONTH_HEADERS, [])
        lay.addWidget(self.month_table, 1)
        self.model.currentTextChanged.connect(self._model_changed)
        self._model_changed(self.model.currentText())
        return w

    def _model_changed(self, name: str) -> None:
        for box, first in (
            (self.density_attr, "(sabit)"),
            (self.grade, "(yok)"),
            (self.material_attr, "(tenör sınırları)"),
            (self.phase, "(ayırma)"),
        ):
            box.clear()
            box.addItem(first)
            if name:
                box.addItems(list(BlockModel.open(self.project, name).attributes))

    def _classes(self) -> list[me.GradeClass]:
        out = []
        for part in self.classes.text().split(";"):
            if not part.strip():
                continue
            name, _, value = part.rpartition(":")
            if not name.strip():
                raise RuntimeError(f"Malzeme sınırı 'ad:tenör' biçiminde olmalı: {part!r}")
            out.append(me.GradeClass(name.strip(), float(value.replace(",", "."))))
        return out

    def month(self) -> None:
        if not self.model.currentText():
            raise RuntimeError("Blok modeli gerekli.")
        r = me.month_end(
            BlockModel.open(self.project, self.model.currentText()),
            surfaces.load_surface(self.project, self.start.currentText()),
            surfaces.load_surface(self.project, self.end.currentText()),
            density=_pick(self.density_attr) or self.rho.value(),
            grade=_pick(self.grade),
            classes=self._classes(),
            below=self.below.text().strip() or "pasa",
            material=_pick(self.material_attr),
            phase=_pick(self.phase),
            bench_height=self.bench.value() or None,
            samples=self.samples.value(),
        )
        rows = [
            [
                x.phase or "-",
                f"{x.bench:g}",
                x.material,
                f"{x.volume:,.0f}",
                f"{x.tonnes:,.0f}",
                "" if math.isnan(x.grade) else f"{x.grade:.3f}",
            ]
            for x in r.rows
        ]
        for m, (v, t, g) in r.by_material().items():
            rows.append(
                ["TOPLAM", "", m, f"{v:,.0f}", f"{t:,.0f}", "" if math.isnan(g) else f"{g:.3f}"]
            )
        _fill_table(self.month_table, rows)
        text = (
            f"Kazı {r.excavated:,.1f} m³ (tam); blok modelinde {r.model_volume:,.1f} m³ "
            f"(%{100 * r.model_coverage:.1f}); dolgu/döküm {r.filled:,.1f} m³; "
            f"toplam {r.tonnes:,.0f} t."
        )
        if self.loads.value() > 0 and self.factor.value() > 0:
            tr = me.truck_reconciliation(r.tonnes, self.loads.value(), self.factor.value())
            text += (
                f" Kamyon sayımı {tr.truck_tonnes:,.0f} t; ölçme/kamyon {tr.ratio:.3f}; "
                f"ölçmeye uyan sefer başı ton {tr.suggested_load_factor:.2f}."
            )
        self.month_label.setText(text)
