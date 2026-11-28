"""Solid tools dialog: verify, repair and properties; exact booleans; surface and plane cuts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from minecore import surfaces
from minecore.gui.layout import fit_to_screen
from minecore.gui.resource import _row, _spin
from minecore.project import Project
from minecore.solids import Solid, load_solid, save_solid, solid_names
from minecore.solidtools import (
    DEFAULT_TOLERANCE,
    Boolean,
    boolean,
    cut_by_plane,
    cut_by_surface,
    plane_normal,
    properties,
    repair,
    union_all,
    verify,
)
from minecore.strings import String, save_strings

OPS = {
    "Birleşim (A + B)": Boolean.UNION,
    "Kesişim (A ve B ortak)": Boolean.INTERSECTION,
    "Fark (A eksi B)": Boolean.DIFFERENCE,
}
TABS = ("check", "boolean", "surface", "plane")


class SolidToolsDialog(QDialog):
    def __init__(self, project: Project, parent: QWidget, tab: str = "check") -> None:
        super().__init__(parent)
        self.setWindowTitle("Katı model araçları")
        fit_to_screen(self, 720, 620)
        self.project = project
        self.changed = False
        layout = QVBoxLayout(self)
        head = QFormLayout()
        names = solid_names(project)
        self.a = QComboBox()
        self.a.addItems(names)
        self.b = QComboBox()
        self.b.addItems(names)
        if len(names) > 1:
            self.b.setCurrentIndex(1)
        head.addRow("Katı A / katı B", _row(self.a, self.b))
        self.tolerance = _spin(DEFAULT_TOLERANCE, 0.0, 10.0, 0.001, 4)
        self.tolerance.setToolTip(
            "Bu mesafeden yakın köşeler onarımda birleşir; sonuçta bundan küçük "
            "ayrıntılar sadeleşebilir."
        )
        self.repair_inputs = QCheckBox("Girdileri önce onar")
        head.addRow("Tolerans (m)", _row(self.tolerance, self.repair_inputs))
        self.name = QLineEdit("sonuc_kati")
        head.addRow("Sonuç adı", self.name)
        layout.addLayout(head)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._check_tab(), "Doğrula / özellikler")
        self.tabs.addTab(self._boolean_tab(), "Boolean")
        self.tabs.addTab(self._surface_tab(), "Yüzeyle kes")
        self.tabs.addTab(self._plane_tab(), "Düzlemle kes / kesit")
        self.tabs.setCurrentIndex(TABS.index(tab) if tab in TABS else 0)
        layout.addWidget(self.tabs)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    # -- tabs ---------------------------------------------------------------

    def _check_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        check = QPushButton("A'yı doğrula")
        check.clicked.connect(self._guard(self.check))
        fix = QPushButton("A'yı onar ve sonuç adıyla kaydet")
        fix.clicked.connect(self._guard(self.fix))
        form.addRow(_row(check, fix))
        self.density = _spin(0.0, 0.0, 25.0, 0.1, 3)
        self.density.setToolTip("0 bırakılırsa tonaj hesaplanmaz.")
        props = QPushButton("A'nın hacim, alan, ağırlık merkezi")
        props.clicked.connect(self._guard(self.show_properties))
        form.addRow("Yoğunluk (t/m³)", _row(self.density, props))
        return page

    def _boolean_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.op = QComboBox()
        self.op.addItems(list(OPS))
        run = QPushButton("Boolean işlemini yap")
        run.clicked.connect(self._guard(self.run_boolean))
        form.addRow("İşlem (A ile B)", _row(self.op, run))
        self.many = QListWidget()
        self.many.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.many.addItems(solid_names(self.project))
        join = QPushButton("Seçilenlerin hepsini birleştir")
        join.clicked.connect(self._guard(self.run_join))
        form.addRow("Çoklu birleşim", self.many)
        form.addRow(join)
        note = QLabel(
            "Kesin işlem: üçgenler birbirine göre kesilir, sonuç kapalı ve köşeleri "
            "keskin kalır (voksel yuvarlaması yok). Girdiler kapalı olmalıdır."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _surface_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.surface = QComboBox()
        self.surface.addItems(surfaces.surface_names(self.project))
        self.side = QComboBox()
        self.side.addItems(["Yüzeyin altı", "Yüzeyin üstü"])
        form.addRow("A'yı yüzeyle kes", _row(self.surface, self.side))
        self.other = QCheckBox("Diğer tarafı da '<ad>_diger' olarak kaydet")
        form.addRow(self.other)
        cut = QPushButton("Kes")
        cut.clicked.connect(self._guard(self.run_cut))
        form.addRow(cut)
        return page

    def _plane_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.px = _spin(0.0, -1e8, 1e8, 1.0, 3)
        self.py = _spin(0.0, -1e8, 1e8, 1.0, 3)
        self.pz = _spin(0.0, -1e5, 1e5, 1.0, 3)
        centre = QPushButton("A'nın merkezi")
        centre.clicked.connect(self._guard(self.use_centre))
        form.addRow("Düzlem noktası X / Y / Z", _row(self.px, self.py, self.pz, centre))
        self.dip = _spin(0.0, 0.0, 90.0, 5.0, 2)
        self.dip_dir = _spin(0.0, 0.0, 360.0, 5.0, 2)
        form.addRow("Eğim / eğim yönü (°)", _row(self.dip, self.dip_dir))
        hint = QLabel(
            "Eğim 0 = yatay düzlem (kot kesiti). Üst parça '<ad>_ust' (düzlemin normali "
            "yönündeki taraf; dikey düzlemde eğim yönü tarafı), alt parça '<ad>_alt', "
            "kesit çizgisi '<ad>_kesit' çizgi seti olarak kaydedilir."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        run = QPushButton("Düzlemle kes")
        run.clicked.connect(self._guard(self.run_plane))
        form.addRow(run)
        return page

    # -- helpers ------------------------------------------------------------

    def _guard(self, slot: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                slot()
            except Exception as exc:
                QMessageBox.critical(self, "Katı model araçları", str(exc))

        return run

    def _need(self, box: QComboBox) -> str:
        if not box.currentText():
            raise RuntimeError("Önce katı model içe aktarın veya oluşturun.")
        return box.currentText()

    def _out(self) -> str:
        name = self.name.text().strip()
        if not name:
            raise RuntimeError("Sonuç adı yazın.")
        return name

    def _store(self, name: str, solid: Solid) -> str:
        save_solid(self.project, name, solid)
        self.changed = True
        for box in (self.a, self.b):
            if box.findText(name) < 0:
                box.addItem(name)
        if not self.many.findItems(name, Qt.MatchFlag.MatchExactly):
            self.many.addItem(name)
        p = properties(solid)
        return f"'{name}': hacim {p.volume:,.3f} m³, alan {p.area:,.1f} m²"

    # -- actions ------------------------------------------------------------

    def check(self) -> None:
        r = verify(load_solid(self.project, self._need(self.a)))
        self.summary.setText(
            f"{r.triangles:,} üçgen, {r.shells} kabuk, hacim {r.volume:,.3f} m³, alan "
            f"{r.area:,.1f} m². Açık kenar {r.open_edges}, çok-yüzlü kenar "
            f"{r.non_manifold_edges}, sıfır alanlı üçgen {r.degenerate}, tekrar {r.duplicates}, "
            f"ters yönlü kenar {r.flipped_edges}"
            + (", normaller içe dönük" if r.volume < 0 else "")
            + (". Geçerli." if r.valid else ". Onarım gerekli.")
        )

    def fix(self) -> None:
        out = self._out()
        fixed = repair(load_solid(self.project, self._need(self.a)), self.tolerance.value())
        text = self._store(out, fixed)
        r = verify(fixed)
        self.summary.setText(
            f"{text} kaydedildi, açık kenar {r.open_edges}"
            + (" (kapalı değil: eksik yüzler onarımla eklenmez)." if r.open_edges else ".")
        )

    def show_properties(self) -> None:
        name = self._need(self.a)
        p = properties(load_solid(self.project, name))
        cx, cy, cz = p.centroid
        lines = [
            f"'{name}': hacim {p.volume:,.3f} m³, yüzey alanı {p.area:,.3f} m²",
            f"Ağırlık merkezi X {cx:,.3f}  Y {cy:,.3f}  Z {cz:,.3f}",
            "Sınır kutusu: "
            + "  ".join(
                f"{a} {lo:,.2f}…{hi:,.2f}"
                for a, lo, hi in zip("XYZ", p.lower, p.upper, strict=True)
            )
            + f"; {len(p.shell_volumes)} kabuk"
            + (" (boşluklar eksi hacimli)" if min(p.shell_volumes) < 0 else ""),
            "Kapalı ve tutarlı." if p.closed else "Kapalı değil: önce onarın, hacim güvenilmez.",
        ]
        if self.density.value() > 0:
            lines.append(f"Tonaj {p.volume * self.density.value():,.1f} t")
        self.summary.setText("\n".join(lines))

    def run_boolean(self) -> None:
        out = self._out()
        result = boolean(
            load_solid(self.project, self._need(self.a)),
            load_solid(self.project, self._need(self.b)),
            OPS[self.op.currentText()],
            self.tolerance.value(),
            repair_inputs=self.repair_inputs.isChecked(),
        )
        self.summary.setText(self._store(out, result) + " kaydedildi.")

    def run_join(self) -> None:
        out = self._out()
        names = [i.text() for i in self.many.selectedItems()]
        if len(names) < 2:
            raise RuntimeError("Listeden en az iki katı seçin.")
        result = union_all(
            [load_solid(self.project, n) for n in names],
            self.tolerance.value(),
            repair_inputs=self.repair_inputs.isChecked(),
        )
        self.summary.setText(f"{len(names)} katı birleşti: " + self._store(out, result) + ".")

    def run_cut(self) -> None:
        out = self._out()
        if not self.surface.currentText():
            raise RuntimeError("Önce bir yüzey içe aktarın.")
        cut = cut_by_surface(
            load_solid(self.project, self._need(self.a)),
            surfaces.load_surface(self.project, self.surface.currentText()),
            self.tolerance.value(),
            repair_input=self.repair_inputs.isChecked(),
        )
        below = self.side.currentIndex() == 0
        keep, other = (cut.below, cut.above) if below else (cut.above, cut.below)
        if keep is None:
            raise RuntimeError("Katının bu tarafta hiç parçası yok.")
        parts = [self._store(out, keep)]
        if self.other.isChecked() and other is not None:
            parts.append(self._store(f"{out}_diger", other))
        self.summary.setText(
            " ve ".join(parts)
            + f" kaydedildi. Altta {cut.volume_below:,.3f} m³, üstte {cut.volume_above:,.3f} m³"
            + (
                f", yüzeyin kapsamı dışında {cut.volume_outside:,.3f} m³ (bu kısım atıldı)."
                if cut.volume_outside > 1e-6 * max(cut.volume_below + cut.volume_above, 1.0)
                else "."
            )
        )

    def use_centre(self) -> None:
        c = properties(load_solid(self.project, self._need(self.a))).centroid
        for box, v in zip((self.px, self.py, self.pz), c, strict=True):
            box.setValue(float(v))

    def run_plane(self) -> None:
        out = self._out()
        cut = cut_by_plane(
            load_solid(self.project, self._need(self.a)),
            np.array([self.px.value(), self.py.value(), self.pz.value()]),
            plane_normal(self.dip.value(), self.dip_dir.value()),
            self.tolerance.value(),
            repair_input=self.repair_inputs.isChecked(),
        )
        parts = []
        if cut.upper is not None:
            parts.append(self._store(f"{out}_ust", cut.upper))
        if cut.lower is not None:
            parts.append(self._store(f"{out}_alt", cut.lower))
        if cut.section:
            save_strings(self.project, f"{out}_kesit", [String(r, True) for r in cut.section])
            self.changed = True
        self.summary.setText(
            "; ".join(parts)
            + f" kaydedildi. Kesit alanı {cut.section_area:,.3f} m² ({len(cut.section)} halka"
            + (f", '{out}_kesit' çizgi seti)." if cut.section else ").")
        )
