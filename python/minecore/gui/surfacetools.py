"""Surface tools dialog: contours, slope/aspect, profile, intersection, offset,
merge/patch, normal offset, smoothing, decimation, grid <-> TIN."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from minecore import cli_surfacetools, strings, surfaces
from minecore import surfaceedit as se
from minecore import surfacetools as st
from minecore.gui.layout import fit_to_screen
from minecore.gui.resource import _row, _spin
from minecore.project import Project

TABS = ("contours", "slope", "profile", "intersect", "edit")
MERGE_MODES = {
    "Alt zarf (düşük olan; ör. ocak tasarımı + topoğrafya)": "lower",
    "Üst zarf (yüksek olan; ör. döküm sahası + topoğrafya)": "upper",
    "Yama (A olduğu yerde A, başka yerde B)": "patch",
}
CLASS_HEADERS = ["Sınıf", "Plan alanı m²", "Gerçek alan m²", "Pay %"]


def _fill(table: QTableWidget, rows: list[list[str]]) -> None:
    table.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            table.setItem(i, j, QTableWidgetItem(v))


def _class_rows(rows: list[st.ClassRow]) -> list[list[str]]:
    return [
        [r.label, f"{r.plan_area:,.1f}", f"{r.true_area:,.1f}", f"{100 * r.share:.1f}"]
        for r in rows
    ]


class SurfaceToolsDialog(QDialog):
    def __init__(self, project: Project, parent: QWidget, tab: str = "contours") -> None:
        super().__init__(parent)
        self.setWindowTitle("Yüzey araçları: kontur, eğim/bakı, profil, kesişim, birleştirme")
        fit_to_screen(self, 1050, 700)
        self.project = project
        self.changed = False
        self.tabs = QTabWidget()
        QVBoxLayout(self).addWidget(self.tabs)
        self.tabs.addTab(self._contour_tab(), "Kontur")
        self.tabs.addTab(self._slope_tab(), "Eğim / bakı")
        self.tabs.addTab(self._profile_tab(), "Profil")
        self.tabs.addTab(self._intersect_tab(), "Kesişim ve öteleme")
        self.tabs.addTab(self._edit_tab(), "Birleştir / düzenle")
        self.tabs.setCurrentIndex(TABS.index(tab))

    def _guard(self, slot: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                slot()
            except Exception as exc:
                QMessageBox.critical(self, "Yüzey araçları", str(exc))

        return run

    def _surface_box(self) -> QComboBox:
        box = QComboBox()
        box.addItems(surfaces.surface_names(self.project))
        return box

    def _save_path(self, title: str, filters: str) -> str:
        path, _ = QFileDialog.getSaveFileName(self, title, "", filters)
        return path

    # -- contours ----------------------------------------------------------------

    def _contour_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        lay.addLayout(form)
        self.c_surface = self._surface_box()
        self.c_interval = _spin(5.0, 0.01, 1000.0, 1.0, 2)
        self.c_base = _spin(0.0, -1e5, 1e5, 1.0, 2)
        self.c_index = QSpinBox(minimum=0, maximum=100, value=5)
        self.c_index.setSpecialValueText("ana kontur yok")
        self.c_name = QLineEdit("kontur")
        self.c_label = _spin(200.0, 0.0, 1e5, 10.0, 1)
        self.c_label.setSpecialValueText("etiket yok")
        self.c_height = _spin(2.5, 0.1, 100.0, 0.5, 2)
        form.addRow("Yüzey", self.c_surface)
        form.addRow("Kontur aralığı (m)", self.c_interval)
        form.addRow("Başlangıç kotu (m)", self.c_base)
        form.addRow("Her kaçıncı ana kontur", self.c_index)
        form.addRow("Çizgi seti adı", self.c_name)
        form.addRow("DXF etiket aralığı / yazı boyu (m)", _row(self.c_label, self.c_height))
        run = QPushButton("Konturları üret ve çizgi seti olarak kaydet")
        run.clicked.connect(self._guard(self.make_contours))
        dxf = QPushButton("Konturları etiketli DXF olarak dışa aktar...")
        dxf.clicked.connect(self._guard(self.export_contours))
        lay.addWidget(run)
        lay.addWidget(dxf)
        self.c_result = QLabel("")
        self.c_result.setWordWrap(True)
        lay.addWidget(self.c_result)
        lay.addStretch(1)
        return w

    def _contours(self) -> list[st.Contour]:
        s = surfaces.load_surface(self.project, self.c_surface.currentText())
        cs = st.contours(
            s, self.c_interval.value(), base=self.c_base.value(), index_every=self.c_index.value()
        )
        if not cs:
            raise ValueError("Bu aralıkta yüzeyi kesen kontur kotu yok.")
        return cs

    def make_contours(self) -> None:
        cs = self._contours()
        name = self.c_name.text().strip() or "kontur"
        strings.save_strings(self.project, name, cli_surfacetools.contour_strings(cs))
        self.changed = True
        levels = sorted({c.level for c in cs})
        self.c_result.setText(
            f"{len(cs)} kontur çizgisi, {len(levels)} kot ({levels[0]:g} - {levels[-1]:g} m), "
            f"{sum(c.closed for c in cs)} kapalı, {sum(c.index for c in cs)} ana kontur; "
            f"toplam uzunluk {sum(c.length for c in cs):,.0f} m. '{name}' çizgi seti "
            "olarak kaydedildi (3B görünümde ve çizgi düzenleyicide)."
        )

    def export_contours(self, path: str | None = None) -> None:
        path = path or self._save_path("Kontur DXF", "DXF (*.dxf)")
        if not path:
            return
        cs = self._contours()
        n = cli_surfacetools.contour_dxf(
            Path(path), cs, self.c_label.value() or None, self.c_height.value()
        )
        self.c_result.setText(f"{path}: {n} öğe (kontur ve etiket) yazıldı.")

    # -- slope ---------------------------------------------------------------------

    def _slope_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        lay.addLayout(form)
        self.s_surface = self._surface_box()
        self.s_classes = QLineEdit("0 10 20 30 45 90")
        self.s_classes.setToolTip("Eğim sınıf sınırları, derece, boşlukla ayrılmış")
        self.s_cell = _spin(5.0, 0.01, 1000.0, 1.0, 2)
        form.addRow("Yüzey", self.s_surface)
        form.addRow("Eğim sınıf sınırları (°)", self.s_classes)
        form.addRow("Raster hücresi (m)", self.s_cell)
        run = QPushButton("Eğim ve bakı haritasını hesapla")
        run.clicked.connect(self._guard(self.compute_slope))
        asc = QPushButton("Eğim ve bakı rasterini (ESRI ASCII .asc) kaydet...")
        asc.clicked.connect(self._guard(self.export_slope))
        lay.addWidget(run)
        lay.addWidget(asc)
        self.s_summary = QLabel("")
        lay.addWidget(self.s_summary)
        self.slope_table = QTableWidget(0, len(CLASS_HEADERS))
        self.slope_table.setHorizontalHeaderLabels(CLASS_HEADERS)
        self.aspect_table = QTableWidget(0, len(CLASS_HEADERS))
        self.aspect_table.setHorizontalHeaderLabels(["Bakı (eğim yönü)", *CLASS_HEADERS[1:]])
        lay.addWidget(self.slope_table)
        lay.addWidget(self.aspect_table)
        return w

    def compute_slope(self) -> None:
        limits = [float(v) for v in self.s_classes.text().replace(",", " ").split()]
        sm = st.slope_map(surfaces.load_surface(self.project, self.s_surface.currentText()))
        a = sm.plan_area
        self.s_summary.setText(
            f"{len(a)} üçgen; plan alanı {a.sum():,.0f} m², gerçek alan "
            f"{sm.true_area.sum():,.0f} m²; ortalama eğim "
            f"{(sm.slope * a).sum() / a.sum():.1f}°, en büyük {sm.slope.max():.1f}°"
        )
        _fill(self.slope_table, _class_rows(st.slope_classes(sm, limits)))
        _fill(self.aspect_table, _class_rows(st.aspect_sectors(sm)))

    def export_slope(self, folder: str | None = None) -> None:
        folder = folder or QFileDialog.getExistingDirectory(self, "Raster klasörü")
        if not folder:
            return
        name = self.s_surface.currentText()
        g = st.surface_grid(surfaces.load_surface(self.project, name), self.s_cell.value())
        for field in ("slope", "aspect", "z"):
            st.write_ascii_grid(Path(folder) / f"{name}_{field}.asc", g, getattr(g, field))
        self.s_summary.setText(
            f"{folder}: {name}_slope.asc, {name}_aspect.asc, {name}_z.asc "
            f"({g.z.shape[1]} x {g.z.shape[0]} hücre)"
        )

    # -- profile ---------------------------------------------------------------------

    def _profile_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        lay.addLayout(form)
        self.p_line = QComboBox()
        self.p_line.addItems(strings.string_names(self.project))
        self.p_surfaces = QListWidget()
        self.p_surfaces.addItems(surfaces.surface_names(self.project))
        self.p_surfaces.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.p_surfaces.setMaximumHeight(110)
        self.p_exag = _spin(1.0, 0.1, 50.0, 0.5, 1)
        form.addRow("Profil hattı (çizgi setinin ilk çizgisi)", self.p_line)
        form.addRow("Yüzeyler (birden çok seçilebilir)", self.p_surfaces)
        form.addRow("Düşey abartma", self.p_exag)
        run = QPushButton("Profili çiz")
        run.clicked.connect(self._guard(self.draw))
        save = QPushButton("Profili kaydet (CSV ve PNG/PDF)...")
        save.clicked.connect(self._guard(self.save_profile))
        lay.addWidget(run)
        lay.addWidget(save)
        self.figure = Figure(figsize=(9, 3.5))
        self.canvas = FigureCanvasQTAgg(self.figure)  # type: ignore[no-untyped-call]
        lay.addWidget(self.canvas, 1)
        self.profile_result: st.Profile | None = None
        return w

    def _profile(self) -> st.Profile:
        chosen = [i.text() for i in self.p_surfaces.selectedItems()]
        if not chosen:
            raise ValueError("En az bir yüzey seçin.")
        lines = strings.load_strings(self.project, self.p_line.currentText())
        if not lines:
            raise ValueError("Çizgi seti boş.")
        path = lines[0].points
        if lines[0].closed:
            path = np.vstack([path, path[:1]])
        named = {n: surfaces.load_surface(self.project, n) for n in chosen}
        return st.profile(named, path)

    def draw(self) -> None:
        self.profile_result = self._profile()
        self.figure.clear()
        st.draw_profile(self.figure.add_subplot(111), self.profile_result, self.p_exag.value())
        self.figure.tight_layout()
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def save_profile(self, path: str | None = None) -> None:
        path = path or self._save_path("Profil", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        p = self.profile_result or self._profile()
        st.save_profile_plot(Path(path), p, self.p_exag.value())
        csv_path = Path(path).with_suffix(".csv")
        rows = ["chainage_m,x,y," + ",".join(f"z_{n}" for n in p.z)]
        for k in range(len(p.chainage)):
            zs = ["" if np.isnan(z[k]) else f"{z[k]:.3f}" for z in p.z.values()]
            rows.append(f"{p.chainage[k]:.3f},{p.xy[k, 0]:.3f},{p.xy[k, 1]:.3f}," + ",".join(zs))
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        QMessageBox.information(self, "Profil", f"{path}\n{csv_path}")

    # -- intersection and offset --------------------------------------------------------

    def _intersect_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        lay.addLayout(form)
        self.i_a = self._surface_box()
        self.i_b = self._surface_box()
        if self.i_b.count() > 1:
            self.i_b.setCurrentIndex(1)
        self.i_name = QLineEdit("kesisim")
        form.addRow("Yüzey A (ör. ocak tasarımı)", self.i_a)
        form.addRow("Yüzey B (ör. topoğrafya)", self.i_b)
        form.addRow("Çizgi seti adı", self.i_name)
        run = QPushButton("Kesişim çizgisini bul ve kaydet (ör. ocak tepe / günışığı çizgisi)")
        run.clicked.connect(self._guard(self.intersect))
        lay.addWidget(run)
        self.i_result = QLabel("")
        self.i_result.setWordWrap(True)
        lay.addWidget(self.i_result)
        off = QFormLayout()
        lay.addLayout(off)
        self.o_surface = self._surface_box()
        self.o_dz = _spin(0.0, -1e4, 1e4, 0.5, 3)
        self.o_name = QLineEdit("")
        self.o_name.setPlaceholderText("yeni yüzey adı")
        off.addRow("Ötelenecek yüzey", self.o_surface)
        off.addRow("Düşey öteleme (m, + yukarı)", self.o_dz)
        off.addRow("Yeni yüzey adı", self.o_name)
        go = QPushButton("Yüzeyi ötele ve kaydet")
        go.clicked.connect(self._guard(self.offset))
        lay.addWidget(go)
        lay.addStretch(1)
        return w

    def intersect(self) -> None:
        a = surfaces.load_surface(self.project, self.i_a.currentText())
        b = surfaces.load_surface(self.project, self.i_b.currentText())
        lines = st.intersection_lines(a, b)
        if not lines:
            self.i_result.setText("Yüzeyler kesişmiyor.")
            return
        name = self.i_name.text().strip() or "kesisim"
        strings.save_strings(
            self.project, name, [strings.String(x.points, x.closed, "KESISIM") for x in lines]
        )
        self.changed = True
        self.i_result.setText(
            f"{len(lines)} kesişim çizgisi ({sum(x.closed for x in lines)} kapalı), toplam "
            f"{sum(x.length for x in lines):,.1f} m; '{name}' çizgi seti olarak kaydedildi."
        )

    def offset(self) -> None:
        name = self.o_name.text().strip()
        if not name:
            raise ValueError("Yeni yüzey adını yazın.")
        src = self.o_surface.currentText()
        s = st.offset_surface(surfaces.load_surface(self.project, src), self.o_dz.value())
        surfaces.save_surface(self.project, name, s)
        self.changed = True
        self.i_result.setText(f"'{name}': {src} {self.o_dz.value():+g} m ötelendi.")

    # -- merge, normal offset, smoothing, decimation, grid ------------------------------

    def _edit_tab(self) -> QWidget:  # noqa: PLR0915 - one form per tool
        w = QWidget()
        lay = QVBoxLayout(w)
        merge = QFormLayout()
        lay.addLayout(merge)
        self.m_a = self._surface_box()
        self.m_b = self._surface_box()
        if self.m_b.count() > 1:
            self.m_b.setCurrentIndex(1)
        self.m_mode = QComboBox()
        self.m_mode.addItems(list(MERGE_MODES))
        self.m_name = QLineEdit("")
        self.m_name.setPlaceholderText("yeni yüzey adı")
        merge.addRow("Yüzey A", self.m_a)
        merge.addRow("Yüzey B", self.m_b)
        merge.addRow("Birleştirme", self.m_mode)
        merge.addRow("Yeni yüzey adı", self.m_name)
        go = QPushButton("Yüzeyleri birleştir ve kaydet")
        go.clicked.connect(self._guard(self.merge))
        lay.addWidget(go)
        edit = QFormLayout()
        lay.addLayout(edit)
        self.e_surface = self._surface_box()
        self.e_name = QLineEdit("")
        self.e_name.setPlaceholderText("yeni yüzey adı")
        self.e_distance = _spin(1.0, -1e4, 1e4, 0.5, 3)
        self.e_passes = QSpinBox(minimum=1, maximum=100, value=3)
        self.e_factor = _spin(0.5, 0.05, 1.0, 0.05, 2)
        self.e_limit = _spin(0.0, 0.0, 1e4, 0.05, 3)
        self.e_limit.setSpecialValueText("sınır yok")
        self.e_hold = QCheckBox("Sınır noktaları sabit kalsın")
        self.e_hold.setChecked(True)
        self.e_tol = _spin(0.05, 0.0, 1e3, 0.01, 3)
        edit.addRow("Yüzey", self.e_surface)
        edit.addRow("Yeni yüzey adı", self.e_name)
        edit.addRow("Normal öteleme mesafesi (m, + yukarı)", self.e_distance)
        edit.addRow("Yumuşatma geçişi / oranı", _row(self.e_passes, self.e_factor))
        edit.addRow("Bir noktada en çok değişim (m)", self.e_limit)
        edit.addRow("", self.e_hold)
        edit.addRow("İnceltme düşey toleransı (m)", self.e_tol)
        for text, slot in (
            ("Normal yönünde ötele", self.normal_offset),
            ("Yumuşat", self.smooth),
            ("İncelt (nokta azalt)", self.decimate),
        ):
            b = QPushButton(text)
            b.clicked.connect(self._guard(slot))
            lay.addWidget(b)
        grid = QFormLayout()
        lay.addLayout(grid)
        self.g_cell = _spin(5.0, 0.01, 1e4, 1.0, 2)
        grid.addRow("Izgara hücresi (m), yüzey yukarıdaki", self.g_cell)
        imp = QPushButton("ESRI ASCII ızgaradan (.asc) yüzey oluştur (tolerans: inceltme)...")
        imp.clicked.connect(self._guard(self.from_grid))
        exp = QPushButton("Yüzeyi ESRI ASCII ızgara (.asc) olarak dışa aktar...")
        exp.clicked.connect(self._guard(self.to_grid))
        lay.addWidget(imp)
        lay.addWidget(exp)
        self.e_result = QLabel("")
        self.e_result.setWordWrap(True)
        lay.addWidget(self.e_result)
        lay.addStretch(1)
        return w

    def _new_name(self, box: QLineEdit) -> str:
        name = box.text().strip()
        if not name:
            raise ValueError("Yeni yüzey adını yazın.")
        return name

    def _store(self, name: str, surface: surfaces.Surface, text: str) -> None:
        surfaces.save_surface(self.project, name, surface)
        self.changed = True
        for box in (self.m_a, self.m_b, self.e_surface, self.o_surface, self.i_a, self.i_b):
            if box.findText(name) < 0:
                box.addItem(name)
        self.e_result.setText(text)

    def merge(self) -> None:
        name = self._new_name(self.m_name)
        a_name, b_name = self.m_a.currentText(), self.m_b.currentText()
        mode = MERGE_MODES[self.m_mode.currentText()]
        s, r = se.merge_surfaces(
            surfaces.load_surface(self.project, a_name),
            surfaces.load_surface(self.project, b_name),
            mode,
        )
        self._store(
            name,
            s,
            f"'{name}': {r.triangles} üçgen. İkisinin olduğu alan {r.area_both:,.1f} m² "
            f"({a_name}: {r.area_from_a:,.1f}, {b_name}: {r.area_from_b:,.1f}); yalnız {a_name} "
            f"{r.area_a_only:,.1f} m², yalnız {b_name} {r.area_b_only:,.1f} m².",
        )

    def normal_offset(self) -> None:
        name = self._new_name(self.e_name)
        src = self.e_surface.currentText()
        s, r = se.normal_offset(surfaces.load_surface(self.project, src), self.e_distance.value())
        self._store(
            name,
            s,
            f"'{name}': {src} normal yönünde {r.distance:+g} m ötelendi; en büyük nokta "
            f"hareketi {r.max_move:.3f} m, sınıra takılan {r.limited} nokta.",
        )

    def smooth(self) -> None:
        name = self._new_name(self.e_name)
        src = self.e_surface.currentText()
        limit = self.e_limit.value() or None
        s, r = se.smooth_surface(
            surfaces.load_surface(self.project, src),
            self.e_passes.value(),
            self.e_factor.value(),
            max_change=limit,
            hold_boundary=self.e_hold.isChecked(),
        )
        self._store(
            name,
            s,
            f"'{name}': {r.iterations} geçiş; değişim en çok {r.max_change:.3f} m, ortalama "
            f"{r.mean_change:.3f} m; hacim farkı {r.volume_change:+,.1f} m³; sınıra takılan "
            f"{r.limited} nokta.",
        )

    def decimate(self) -> None:
        name = self._new_name(self.e_name)
        src = self.e_surface.currentText()
        s, r = se.decimate_surface(surfaces.load_surface(self.project, src), self.e_tol.value())
        self._store(
            name,
            s,
            f"'{name}': {r.points_before} noktadan {r.points_after} noktaya, {r.triangles} "
            f"üçgen; en büyük düşey hata {r.max_error:.4f} m (tolerans {r.tolerance:g} m).",
        )

    def from_grid(self, path: str = "") -> None:
        name = self._new_name(self.e_name)
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "ASCII ızgara", "", "ESRI ASCII (*.asc)")
        if not path:
            return
        g = se.read_ascii_grid(Path(path))
        tol = self.e_tol.value() or None
        s, n = se.grid_surface(g, tol)
        self._store(
            name,
            s,
            f"'{name}': {g.z.shape[1]} x {g.z.shape[0]} ızgara ({g.cell:g} m), veri olan {n} "
            f"hücre; {len(s.points)} nokta, {len(s.triangles())} üçgen.",
        )

    def to_grid(self, path: str = "") -> None:
        path = path or self._save_path("ASCII ızgara", "ESRI ASCII (*.asc)")
        if not path:
            return
        src = self.e_surface.currentText()
        g = st.surface_grid(surfaces.load_surface(self.project, src), self.g_cell.value())
        st.write_ascii_grid(Path(path), g, g.z)
        self.e_result.setText(f"{path}: {g.z.shape[1]} x {g.z.shape[0]} hücre, {g.cell:g} m.")
