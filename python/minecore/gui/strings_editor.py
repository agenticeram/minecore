"""Plan-view string editor: digitise, move, insert and delete points of polylines.

Strings are edited in plan (x, y); each new or moved point takes its z from
the "Kot" field or, when a surface is chosen, from that surface (draped).
The editor works on a copy and writes the whole set back on save, so a set
can be loaded, changed and saved under the same or a new name.

The "İşlem" box applies the string toolbox (:mod:`minecore.stringops`) to the
selected string: reverse, simplify, resample, smooth, offset (with a
vertical shift), trim/extend to the background set and join with the nearest
string. "Noktadan böl" mode cuts a string at the clicked vertex. The selected
string's length, area and z range are shown under the buttons.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minecore import stringops as so
from minecore import surfaces
from minecore.gui.layout import fit_to_screen, scroll_panel
from minecore.gui.resource import _row, _spin
from minecore.project import Project
from minecore.strings import String, StringError, load_strings, save_strings, string_names

MODES = [
    "Nokta ekle",
    "Nokta taşı",
    "Araya nokta ekle",
    "Nokta sil",
    "Çizgi seç",
    "Noktadan böl",
    "İki nokta ölç",
]
SNAPS = ["Yakalama yok", "Noktaya yakala", "Nokta veya çizgiye yakala"]
# operation label, value label, default value, minimum, maximum
OPERATIONS = [
    ("Yönü ters çevir", "", 0.0, 0.0, 0.0),
    ("Sadeleştir (tolerans, m)", "Tolerans (m)", 0.1, 0.0, 1e4),
    ("Yeniden örnekle (aralık, m)", "Aralık (m)", 5.0, 0.01, 1e5),
    ("Yumuşat (tekrar sayısı)", "Tekrar", 2.0, 1.0, 10.0),
    ("Ötele: dışa / sola (m)", "Mesafe (m)", 5.0, 0.001, 1e5),
    ("Ötele: içe / sağa (m)", "Mesafe (m)", 5.0, 0.001, 1e5),
    ("Ucu arka plan setine uzat / kırp", "", 0.0, 0.0, 0.0),
    ("Başı arka plan setine uzat / kırp", "", 0.0, 0.0, 0.0),
    ("En yakın açık çizgiyle birleştir", "En büyük boşluk (m)", 1.0, 0.0, 1e5),
    ("Köşeleri yuvarlat (yarıçap, m)", "Yarıçap (m)", 5.0, 0.001, 1e5),
    ("Köşeleri pahla (mesafe, m)", "Mesafe (m)", 2.0, 0.001, 1e5),
    ("Arka plan setiyle birleşim (poligon)", "", 0.0, 0.0, 0.0),
    ("Arka plan setini çıkar (poligon)", "", 0.0, 0.0, 0.0),
    ("Arka plan setiyle kesişim (poligon)", "", 0.0, 0.0, 0.0),
]
BOOLEAN_OPS = {11: "union", 12: "difference", 13: "intersection"}
PICK_PX = 10.0


class _Line:
    def __init__(self, points: Any, closed: bool, layer: str = "") -> None:
        self.points = np.asarray(points, np.float64).reshape(-1, 3)
        self.closed = closed
        self.layer = layer


class StringEditor(QDialog):
    def __init__(self, project: Project, parent: QWidget) -> None:  # noqa: PLR0915 - form
        super().__init__(parent)
        self.setWindowTitle("Çizgi (string) düzenleyici")
        fit_to_screen(self, 1200, 800)
        self.project = project
        layout = QHBoxLayout(self)
        form = QFormLayout()
        scroll_panel(layout, form)
        self.set_pick = QComboBox()
        self.set_pick.addItem("(yeni set)")
        self.set_pick.addItems(string_names(project))
        load = QPushButton("Yükle")
        load.clicked.connect(self._guard(self.load))
        form.addRow("Çizgi seti", _row(self.set_pick, load))
        self.background = QComboBox()
        self.background.addItem("(yok)")
        self.background.addItems(string_names(project))
        self.background.currentTextChanged.connect(lambda _t: self.redraw())
        form.addRow("Arka plan seti", self.background)
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        form.addRow("Fare ile", self.mode)
        self.snap = QComboBox()
        self.snap.addItems(SNAPS)
        self.snap.setCurrentIndex(1)
        form.addRow("Yakalama", self.snap)
        self.z = _spin(0.0, -1e5, 1e5, 1.0, 2)
        self.drape = QComboBox()
        self.drape.addItem("(sabit kot)")
        self.drape.addItems(surfaces.surface_names(project))
        form.addRow("Kot (m) / yüzeye oturt", _row(self.z, self.drape))
        self.coord_x = _spin(0.0, -1e8, 1e8, 1.0, 3)
        self.coord_y = _spin(0.0, -1e8, 1e8, 1.0, 3)
        add_xy = QPushButton("Ekle")
        add_xy.clicked.connect(self._guard(self.add_typed_point))
        form.addRow("Koordinatla nokta (X, Y)", _row(self.coord_x, self.coord_y, add_xy))
        self.bearing = _spin(0.0, 0.0, 360.0, 1.0, 4)
        self.distance = _spin(10.0, 0.0, 1e6, 1.0, 3)
        self.grade = _spin(0.0, -100.0, 100.0, 1.0, 3)
        add_bd = QPushButton("Ekle")
        add_bd.clicked.connect(self._guard(self.add_bearing_point))
        form.addRow(
            "Son noktadan azimut (°), mesafe (m), eğim (%)",
            _row(self.bearing, self.distance, self.grade, add_bd),
        )
        buttons = [
            ("Yeni çizgi başlat", self.new_line),
            ("Seçili çizgiyi kapat / aç", self.toggle_closed),
            ("Seçili çizgiye kotu uygula", self.apply_z),
            ("Seçili çizgiyi yüzeye oturt", self.drape_current),
            ("Seçili çizgiyi sil", self.delete_line),
            ("Geri al", self.undo),
            ("Görünümü sığdır", self.fit_view),
        ]
        for text, slot in buttons:
            b = QPushButton(text)
            b.clicked.connect(self._guard(slot))
            form.addRow(b)
        self.operation = QComboBox()
        self.operation.addItems([op[0] for op in OPERATIONS])
        self.operation.currentIndexChanged.connect(self._operation_changed)
        form.addRow("İşlem", self.operation)
        self.op_value = _spin(0.0, 0.0, 1e5, 0.5, 3)
        self.op_value_label = QLabel()
        form.addRow(self.op_value_label, self.op_value)
        self.op_dz = _spin(0.0, -1e4, 1e4, 0.5, 3)
        self.op_round = QComboBox()
        self.op_round.addItems(["Keskin köşe", "Yuvarlak köşe"])
        form.addRow("Öteleme: kot farkı (m), köşe", _row(self.op_dz, self.op_round))
        run_op = QPushButton("Seçili çizgiye uygula")
        run_op.clicked.connect(self._guard(self.apply_operation))
        form.addRow(run_op)
        self.measure_label = QLabel()
        self.measure_label.setWordWrap(True)
        form.addRow(self.measure_label)
        self._operation_changed(0)
        self.name = QLineEdit()
        self.name.setPlaceholderText("örn. tasarim_taban")
        save = QPushButton("Kaydet")
        save.clicked.connect(self._guard(self.save))
        form.addRow("Kayıt adı", _row(self.name, save))
        self.status = QLabel(
            "Sol tık: moda göre işlem. Sağ tık: çizgiyi bitir. Yakınlaştırma için araç "
            "çubuğunu kullanın (araç seçiliyken düzenleme yapılmaz)."
        )
        self.status.setWordWrap(True)
        form.addRow(self.status)
        right = QVBoxLayout()
        self.figure = Figure(figsize=(8, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)  # type: ignore[no-untyped-call]
        self.toolbar = NavigationToolbar2QT(self.canvas, self)  # type: ignore[no-untyped-call]
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas, 1)
        layout.addLayout(right, 3)
        self.ax = self.figure.add_subplot(111)
        self.lines: list[_Line] = []
        self.current = -1
        self.history: list[tuple[list[_Line], int]] = []
        self.drag: tuple[int, int] | None = None
        self.measure_from: np.ndarray | None = None
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("button_release_event", self._release)
        self.redraw(reset=True)

    def _guard(self, slot: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                slot()
            except Exception as exc:
                QMessageBox.critical(self, "Çizgi düzenleyici", str(exc))

        return run

    # ----- data -----------------------------------------------------------------

    def _remember(self) -> None:
        self.history.append((copy.deepcopy(self.lines), self.current))
        del self.history[:-50]

    def undo(self) -> None:
        if not self.history:
            return
        self.lines, self.current = self.history.pop()
        self.redraw()

    def load(self) -> None:
        if self.set_pick.currentIndex() == 0:
            self.lines, self.current = [], -1
            self.name.clear()
        else:
            name = self.set_pick.currentText()
            self.lines = [
                _Line(s.points, s.closed, s.layer) for s in load_strings(self.project, name)
            ]
            self.current = -1
            self.name.setText(name)
        self.history.clear()
        self.redraw(reset=True)

    def save(self) -> None:
        name = self.name.text().strip()
        if not name:
            raise RuntimeError("Bir kayıt adı yazın.")
        keep = [ln for ln in self.lines if len(ln.points) >= 2]
        if not keep:
            raise RuntimeError("Kaydedilecek çizgi yok (en az iki noktalı çizgi gerekir).")
        save_strings(
            self.project,
            name,
            [String(ln.points, ln.closed and len(ln.points) >= 3, ln.layer) for ln in keep],
        )
        for box in (self.set_pick, self.background):
            if box.findText(name) < 0:
                box.addItem(name)
        self.status.setText(f"'{name}' kaydedildi: {len(keep)} çizgi.")

    def _z_at(self, xy: np.ndarray) -> np.ndarray:
        if self.drape.currentIndex() <= 0:
            return np.full(len(xy), self.z.value())
        surface = surfaces.load_surface(self.project, self.drape.currentText())
        return surface.elevation(np.asarray(xy, np.float64).reshape(-1, 2))[0]

    def new_line(self) -> None:
        self.current = -1
        self.redraw()

    def _need_current(self) -> _Line:
        if not 0 <= self.current < len(self.lines):
            raise RuntimeError("Önce 'Çizgi seç' moduyla bir çizgi seçin.")
        return self.lines[self.current]

    def toggle_closed(self) -> None:
        line = self._need_current()
        self._remember()
        line.closed = not line.closed
        self.redraw()

    def apply_z(self) -> None:
        line = self._need_current()
        self._remember()
        line.points[:, 2] = self.z.value()
        self.redraw()

    def drape_current(self) -> None:
        line = self._need_current()
        if self.drape.currentIndex() <= 0:
            raise RuntimeError("Önce bir yüzey seçin.")
        self._remember()
        line.points[:, 2] = self._z_at(line.points[:, :2])
        self.redraw()

    def delete_line(self) -> None:
        self._need_current()
        self._remember()
        del self.lines[self.current]
        self.current = -1
        self.redraw()

    # ----- editing operations (data coordinates; also used by tests) ---------------

    def add_point(self, x: float, y: float, z: float | None = None) -> None:
        self._remember()
        if z is None or self.drape.currentIndex() > 0:
            z = float(self._z_at(np.array([[x, y]]))[0])
        if not 0 <= self.current < len(self.lines):
            self.lines.append(_Line([[x, y, z]], closed=False))
            self.current = len(self.lines) - 1
        else:
            line = self.lines[self.current]
            line.points = np.vstack([line.points, [x, y, z]])
        self.redraw()

    def add_typed_point(self) -> None:
        """Add the point typed in the X, Y boxes (z from the kot box or the surface)."""
        self.add_point(self.coord_x.value(), self.coord_y.value())

    def add_bearing_point(self) -> None:
        """Add a point at a bearing and plan distance from the last point of the
        selected string; z changes by the gradient (%) unless draped."""
        line = self._need_current()
        if not len(line.points):
            raise RuntimeError("Seçili çizginin noktası yok.")
        last = line.points[-1]
        b = np.radians(self.bearing.value())
        d = self.distance.value()
        if d <= 0:
            raise RuntimeError("Mesafe sıfırdan büyük olmalı.")
        x, y = last[0] + d * np.sin(b), last[1] + d * np.cos(b)
        self.add_point(float(x), float(y), float(last[2] + d * self.grade.value() / 100.0))

    def measure_between(self, a: np.ndarray, b: np.ndarray) -> str:
        """Plan and slope distance, bearing and gradient from a to b."""
        d = b - a
        plan = float(np.hypot(d[0], d[1]))
        bearing = float(np.degrees(np.arctan2(d[0], d[1])) % 360.0)
        text = (
            f"Ölçüm: plan {plan:,.3f} m, eğik {float(np.linalg.norm(d)):,.3f} m, "
            f"azimut {bearing:.2f}°, kot farkı {d[2]:+.3f} m"
        )
        if plan > 0:
            text += f", eğim {100 * d[2] / plan:+.2f}%"
        return text

    def move_point(self, line: int, vertex: int, x: float, y: float) -> None:
        pts = self.lines[line].points
        pts[vertex, :2] = (x, y)
        if self.drape.currentIndex() > 0:
            pts[vertex, 2] = float(self._z_at(np.array([[x, y]]))[0])

    def delete_point(self, line: int, vertex: int) -> None:
        self._remember()
        ln = self.lines[line]
        ln.points = np.delete(ln.points, vertex, axis=0)
        if len(ln.points) == 0:
            del self.lines[line]
            self.current = -1
        self.redraw()

    def insert_point(self, line: int, segment: int, x: float, y: float) -> None:
        """Insert after vertex ``segment`` (z interpolated along the segment unless draped)."""
        self._remember()
        ln = self.lines[line]
        a = ln.points[segment]
        b = ln.points[(segment + 1) % len(ln.points)]
        ab = b[:2] - a[:2]
        t = float(np.clip(np.dot([x, y] - a[:2], ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
        z = a[2] + t * (b[2] - a[2])
        if self.drape.currentIndex() > 0:
            z = float(self._z_at(np.array([[x, y]]))[0])
        ln.points = np.insert(ln.points, segment + 1, [x, y, z], axis=0)
        self.current = line
        self.redraw()

    # ----- string toolbox ---------------------------------------------------------

    def _operation_changed(self, index: int) -> None:
        _label, value_label, default, lo, hi = OPERATIONS[index]
        self.op_value_label.setText(value_label or "(değer yok)")
        self.op_value.setEnabled(bool(value_label))
        self.op_value.setRange(lo, hi)
        self.op_value.setValue(default)
        offset = index in (4, 5)
        self.op_dz.setEnabled(offset)
        self.op_round.setEnabled(offset)

    def _as_string(self, line: _Line) -> String:
        return String(line.points.copy(), line.closed and len(line.points) >= 3, line.layer)

    def _replace_current(self, made: list[String]) -> None:
        lines = [_Line(m.points, m.closed, m.layer) for m in made]
        self.lines[self.current : self.current + 1] = lines
        self.current = self.current if lines else -1

    def _background_strings(self) -> list[String]:
        if self.background.currentIndex() <= 0:
            raise RuntimeError("Önce bir arka plan seti seçin (sınır çizgisi).")
        return load_strings(self.project, self.background.currentText())

    def apply_operation(self) -> None:  # noqa: PLR0912, PLR0915 - one branch per operation
        line = self._need_current()
        s = self._as_string(line)
        k, v = self.operation.currentIndex(), self.op_value.value()
        if k == 0:
            made = [so.reverse(s)]
        elif k == 1:
            made = [so.simplify(s, v)]
        elif k == 2:
            made = [so.resample(s, v)]
        elif k == 3:
            made = [so.smooth(s, int(v))]
        elif k in (4, 5):
            join = "round" if self.op_round.currentIndex() == 1 else "mitre"
            made = so.offset(s, v if k == 4 else -v, join=join, dz=self.op_dz.value())  # type: ignore[arg-type]
            if not made:
                raise RuntimeError("Öteleme sonucunda çizgi kalmadı (mesafe çok büyük).")
            self._remember()
            self.lines += [_Line(m.points, m.closed, m.layer) for m in made]
            self.current = len(self.lines) - 1
            self.status.setText(f"{len(made)} ötelenmiş çizgi eklendi; kaynak çizgi korundu.")
            self.redraw()
            return
        elif k == 9:
            made = [so.fillet(s, v)]
        elif k == 10:
            made = [so.chamfer(s, v)]
        elif k in BOOLEAN_OPS:
            if not s.closed:
                raise RuntimeError("Poligon işlemleri için seçili çizgi kapalı olmalı.")
            closed_bg = [b for b in self._background_strings() if b.closed]
            if not closed_bg:
                raise RuntimeError("Arka plan setinde kapalı çizgi yok.")
            made = so.boolean([s], closed_bg, BOOLEAN_OPS[k])  # type: ignore[arg-type]
            if not made:
                raise RuntimeError("İşlem sonucunda alan kalmadı.")
        elif k in (6, 7):
            end = "end" if k == 6 else "start"
            made = []
            for b in self._background_strings():
                try:
                    made = [so.trim_extend(s, b, end)]  # type: ignore[arg-type]
                    break
                except StringError:
                    continue  # this boundary string is not reached; try the next
            if not made:
                raise RuntimeError("Çizginin ucu arka plan setindeki hiçbir çizgiye ulaşmıyor.")
        else:
            others = [i for i, ln in enumerate(self.lines) if i != self.current and not ln.closed]
            if line.closed or not others:
                raise RuntimeError("Birleştirmek için iki açık çizgi gerekir.")

            def gap(i: int) -> float:
                q = self.lines[i].points[[0, -1], :2]
                p = line.points[[0, -1], :2]
                return float(np.min(np.linalg.norm(p[:, None] - q[None], axis=2)))

            other = min(others, key=gap)
            joined = so.join(s, self._as_string(self.lines[other]), v if v > 0 else float("inf"))
            self._remember()
            keep = [ln for i, ln in enumerate(self.lines) if i not in (self.current, other)]
            self.lines = [*keep, _Line(joined.points, False, line.layer)]
            self.current = len(self.lines) - 1
            self.redraw()
            return
        self._remember()
        self._replace_current(made)
        self.redraw()

    def split_at(self, line: int, vertex: int) -> None:
        """Cut a string at a vertex (an open string becomes two; a closed one opens)."""
        parts = so.split(self._as_string(self.lines[line]), vertex)
        self._remember()
        self.current = line
        self._replace_current(parts)
        self.redraw()

    def _measure_text(self) -> str:
        if not 0 <= self.current < len(self.lines) or len(self.lines[self.current].points) < 2:
            return ""
        m = so.measure(self._as_string(self.lines[self.current]))
        text = f"Seçili çizgi: {m.points} nokta, plan uzunluğu {m.plan_length:,.2f} m"
        if m.closed:
            text += f", alan {m.area:,.1f} m²"
        return text + f", kot {m.z_min:.2f} / {m.z_max:.2f} m"

    # ----- picking ---------------------------------------------------------------

    def _nearest_vertex(self, event: MouseEvent) -> tuple[int, int] | None:
        best, where = PICK_PX, None
        for i, ln in enumerate(self.lines):
            if not len(ln.points):
                continue
            px = self.ax.transData.transform(ln.points[:, :2])
            d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
            k = int(np.argmin(d))
            if d[k] < best:
                best, where = float(d[k]), (i, k)
        return where

    def _nearest_segment(self, event: MouseEvent) -> tuple[int, int] | None:
        best, where = PICK_PX, None
        p = np.array([event.x, event.y], np.float64)
        for i, ln in enumerate(self.lines):
            n = len(ln.points)
            if n < 2:
                continue
            px = self.ax.transData.transform(ln.points[:, :2])
            count = n if ln.closed else n - 1
            for k in range(count):
                a, b = px[k], px[(k + 1) % n]
                ab = b - a
                t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
                d = float(np.hypot(*(a + t * ab - p)))
                if d < best:
                    best, where = d, (i, k)
        return where

    def _snap(  # noqa: PLR0912 - vertex pass, then segment pass
        self, event: MouseEvent, skip: tuple[int, int] | None = None
    ) -> tuple[float, float, float | None]:
        """Mouse position snapped to the nearest vertex (or, if asked, segment) of
        the edited strings and the background set within PICK_PX; a vertex also
        gives its z."""
        x, y = float(event.xdata), float(event.ydata)  # type: ignore[arg-type]
        mode = self.snap.currentIndex()
        if mode == 0:
            return x, y, None
        sets = [(i, ln.points, ln.closed) for i, ln in enumerate(self.lines)]
        if self.background.currentIndex() > 0:
            for s in load_strings(self.project, self.background.currentText()):
                sets.append((-1, s.points, s.closed))
        p = np.array([event.x, event.y], np.float64)
        best, hit = PICK_PX, None
        for i, pts, _closed in sets:
            if not len(pts):
                continue
            px = self.ax.transData.transform(pts[:, :2])
            d = np.hypot(px[:, 0] - p[0], px[:, 1] - p[1])
            if skip is not None and skip[0] == i:
                d[skip[1]] = np.inf
            k = int(np.argmin(d))
            if d[k] < best:
                best, hit = float(d[k]), pts[k]
        if hit is not None:
            return float(hit[0]), float(hit[1]), float(hit[2])
        if mode == 2:
            best, where = PICK_PX, None
            for i, pts, closed in sets:
                n = len(pts)
                if n < 2 or (skip is not None and skip[0] == i):
                    continue
                px = self.ax.transData.transform(pts[:, :2])
                for k in range(n if closed else n - 1):
                    a, b = px[k], px[(k + 1) % n]
                    ab = b - a
                    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
                    dist = float(np.hypot(*(a + t * ab - p)))
                    if dist < best:
                        best = dist
                        where = pts[k] + t * (pts[(k + 1) % n] - pts[k])
            if where is not None:
                return float(where[0]), float(where[1]), float(where[2])
        return x, y, None

    def _press(self, event: MouseEvent) -> None:  # noqa: PLR0912 - one branch per mode
        if event.inaxes is not self.ax or self.toolbar.mode or event.xdata is None:
            return
        if event.button == MouseButton.RIGHT:
            self.new_line()
            return
        mode = self.mode.currentIndex()
        try:
            x, y, z = self._snap(event)
            if mode == 0:
                self.add_point(x, y, z)
            elif mode == 1:
                hit = self._nearest_vertex(event)
                if hit is not None:
                    self._remember()
                    self.drag, self.current = hit, hit[0]
            elif mode == 2:
                seg = self._nearest_segment(event)
                if seg is not None:
                    self.insert_point(seg[0], seg[1], x, y)
            elif mode == 3:
                hit = self._nearest_vertex(event)
                if hit is not None:
                    self.delete_point(*hit)
            elif mode == 5:
                hit = self._nearest_vertex(event)
                if hit is not None:
                    self.split_at(*hit)
            elif mode == 6:
                here = np.array([x, y, z if z is not None else self.z.value()])
                if self.measure_from is None:
                    self.measure_from = here
                    self.status.setText("Ölçüm: ikinci noktayı tıklayın.")
                else:
                    self.status.setText(self.measure_between(self.measure_from, here))
                    self.measure_from = None
            else:
                hit = self._nearest_vertex(event) or self._nearest_segment(event)
                self.current = hit[0] if hit is not None else -1
                self.redraw()
        except Exception as exc:
            QMessageBox.critical(self, "Çizgi düzenleyici", str(exc))

    def _motion(self, event: MouseEvent) -> None:
        if self.drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        x, y, _z = self._snap(event, skip=self.drag)
        self.move_point(*self.drag, x, y)
        self.redraw()

    def _release(self, _event: MouseEvent) -> None:
        self.drag = None

    # ----- drawing ---------------------------------------------------------------

    def _extent(self) -> tuple[float, float, float, float]:
        """Plan box of the strings, else of the project's surfaces, else 0..100."""
        pts = [ln.points[:, :2] for ln in self.lines if len(ln.points)]
        if self.background.currentIndex() > 0:
            pts += [
                s.points[:, :2] for s in load_strings(self.project, self.background.currentText())
            ]
        if not pts:
            pts = [
                surfaces.load_surface(self.project, n).points[:, :2]
                for n in surfaces.surface_names(self.project)
            ]
        if not pts:
            return (0.0, 0.0, 100.0, 100.0)
        xy = np.vstack(pts)
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = max(float((hi - lo).max()) * 0.05, 10.0)
        return (lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad)

    def fit_view(self) -> None:
        box = self._extent()
        self.ax.set_xlim(box[0], box[2])
        self.ax.set_ylim(box[1], box[3])
        self.redraw()

    def redraw(self, reset: bool = False) -> None:
        ax = self.ax
        limits = None if reset else (ax.get_xlim(), ax.get_ylim())
        ax.clear()
        bg = self.background.currentText() if self.background.currentIndex() > 0 else ""
        drawn = False
        if bg:
            for s in load_strings(self.project, bg):
                pts = np.vstack([s.points, s.points[:1]]) if s.closed else s.points
                ax.plot(pts[:, 0], pts[:, 1], color="#9e9e9e", lw=0.8)
                drawn = True
        for i, ln in enumerate(self.lines):
            pts = ln.points
            if not len(pts):
                continue
            show = np.vstack([pts, pts[:1]]) if ln.closed and len(pts) >= 3 else pts
            selected = i == self.current
            ax.plot(
                show[:, 0],
                show[:, 1],
                "-o",
                color="#d84315" if selected else "#1565c0",
                lw=1.8 if selected else 1.0,
                ms=4 if selected else 2.5,
            )
            drawn = True
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Doğu (m)")
        ax.set_ylabel("Kuzey (m)")
        if limits is not None and not reset:
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
        elif not drawn:
            box = self._extent()
            ax.set_xlim(box[0], box[2])
            ax.set_ylim(box[1], box[3])
        n = len(self.lines[self.current].points) if 0 <= self.current < len(self.lines) else 0
        ax.set_title(f"{len(self.lines)} çizgi" + (f", seçili çizgi {n} nokta" if n else ""))
        self.measure_label.setText(self._measure_text())
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]
