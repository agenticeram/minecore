"""Workbench command line: type commands as in a CAD program.

Two kinds of input share one line at the bottom of the main window:

* a command word (``KOMPOZIT``, ``KESTIRIM``, ``RAPOR``...) opens the same
  dialog as its menu entry; a unique abbreviation is enough (``KOMP``);
* a full headless command (``bm report model au --cutoffs 0.3 0.5``) runs the
  CLI against the open project, whose path is filled in automatically, and
  its output appears above the line.

Up/Down walk the history, Tab completes a command word, Enter on an empty
line repeats the last command, Escape clears the line.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QCompleter,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

_TR = str.maketrans("ÇĞİIÖŞÜçğıiöşü", "CGIIOSUCGIIOSU")


def normalise(word: str) -> str:
    """Upper-case ASCII form of a command word: 'kompozıt' and 'KOMPOZİT' match."""
    text = word.strip().translate(_TR).upper()
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


@dataclass(frozen=True)
class Alias:
    """A command word that runs a workbench action."""

    name: str
    description: str
    slot: Callable[[], Any]


class AliasTable:
    """Command words, matched case- and accent-insensitively, by unique prefix."""

    def __init__(self) -> None:
        self._items: dict[str, Alias] = {}

    def add(self, name: str, description: str, slot: Callable[[], Any]) -> None:
        key = normalise(name)
        if key in self._items:
            raise ValueError(f"duplicate command word {name}")
        self._items[key] = Alias(key, description, slot)

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[Alias]:
        return [self._items[k] for k in self.names()]

    def match(self, word: str) -> tuple[Alias | None, list[str]]:
        """(alias, candidates): exact or unique-prefix hit, else the candidates."""
        key = normalise(word)
        if not key:
            return None, []
        if key in self._items:
            return self._items[key], [key]
        hits = [k for k in self._items if k.startswith(key)]
        if len(hits) == 1:
            return self._items[hits[0]], hits
        return None, sorted(hits)


class _Input(QLineEdit):
    """Line edit with command history on Up/Down."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.history: list[str] = []
        self._pos = 0

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self._pos = len(self.history)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        key = event.key()
        popup = self.completer().popup() if self.completer() else None
        if popup is not None and popup.isVisible():
            super().keyPressEvent(event)
            return
        if key == Qt.Key.Key_Up and self.history:
            self._pos = max(0, self._pos - 1)
            self.setText(self.history[self._pos])
            return
        if key == Qt.Key.Key_Down and self.history:
            self._pos = min(len(self.history), self._pos + 1)
            self.setText(self.history[self._pos] if self._pos < len(self.history) else "")
            return
        if key == Qt.Key.Key_Escape:
            self.clear()
            return
        super().keyPressEvent(event)


class CommandLine(QWidget):
    """Output log above a single command input, docked under the 3D view."""

    submitted = Signal(str)

    def __init__(self, words: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(fixed)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.input = _Input(self)
        self.input.setFont(fixed)
        self.input.setPlaceholderText(
            "Komut yazın (ör. KOMPOZIT veya bm report model au --cutoffs 0.5). Liste için YARDIM."
        )
        completer = QCompleter(words, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
        self.input.setCompleter(completer)
        self.input.returnPressed.connect(self._submit)
        row = QHBoxLayout()
        prompt = QLabel("Komut:")
        bold = QFont(prompt.font())
        bold.setBold(True)
        prompt.setFont(bold)
        row.addWidget(prompt)
        row.addWidget(self.input, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.addWidget(self.log, 1)
        layout.addLayout(row)

    def _submit(self) -> None:
        text = self.input.text().strip()
        if not text and self.input.history:
            text = self.input.history[-1]  # Enter on an empty line repeats, as in CAD
        self.input.clear()
        if not text:
            return
        self.input.remember(text)
        self.write(f"Komut: {text}")
        self.submitted.emit(text)

    def write(self, text: str) -> None:
        for line in text.rstrip("\n").splitlines() or [""]:
            self.log.appendPlainText(line)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())


# Command word, description, MainWindow method name.
WORKBENCH_COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("YENI", "Yeni proje", "new_project"),
    ("AC", "Proje aç", "open_project"),
    ("GERIAL", "Son işlemi geri al", "undo"),
    ("YINELE", "Geri alınanı yinele", "redo"),
    ("OMFAL", "OMF içe aktar", "import_omf"),
    ("OMFVER", "OMF olarak dışa aktar", "export_omf"),
    ("SONDAJAL", "Sondaj klasörü içe aktar", "import_folder"),
    ("DOSYAAL", "Tek dosya içe aktar", "import_file"),
    ("DOGRULA", "Sondaj doğrulama", "revalidate"),
    ("KOMPOZIT", "Kompozit oluştur", "composite"),
    ("TABLOVER", "Tabloyu CSV olarak dışa aktar", "export_table"),
    ("BLOKMODEL", "Blok modeli oluştur", "new_block_model"),
    ("BLOKAL", "Blok modeli içe aktar (CSV)", "import_block_model"),
    ("BLOKVER", "Blok modelini CSV olarak dışa aktar", "export_block_model"),
    ("ISTATISTIK", "Örnek istatistikleri, top-cut", "sample_stats"),
    ("VARIOGRAM", "Variogram analizi", "variogram"),
    ("KESTIRIM", "Tenör kestirimi (NN/IDW/Kriging)", "estimate"),
    ("KDOGRULA", "Kestirim doğrulama (swath, çapraz)", "validate_estimate"),
    ("SINIFLA", "Güven kategorileri", "categorise"),
    ("HESAP", "Blok modeli hesaplayıcı", "calculator"),
    ("YENIDENBLOK", "Yeniden bloklama", "reblock_model"),
    ("RAPOR", "Tenör-tonaj raporu", "grade_tonnage"),
    ("TOPOAL", "Topoğrafya noktaları içe aktar", "import_surface"),
    ("TOPOKUYU", "Kuyu ağızlarından topoğrafya", "surface_from_collars"),
    ("KAYAHAVA", "Blok modelini topoğrafyayla kes", "flag_blocks"),
    ("KONTUR", "Kontur üret (etiketli DXF)", "surface_contours"),
    ("EGIM", "Eğim ve bakı haritası", "surface_slope"),
    ("PROFIL", "Profil çıkar (hat boyunca yüzeyler)", "surface_profile"),
    ("KESISIM", "Yüzey kesişim çizgisi, yüzey öteleme", "surface_intersect"),
    ("YBIRLESTIR", "Yüzey birleştir/yama, normal öteleme, yumuşat, incelt, ızgara", "surface_edit"),
    ("KATIAL", "Katı model içe aktar", "import_solid"),
    ("KATIVER", "Katı modeli DXF olarak dışa aktar", "export_solid"),
    ("KATIARAC", "Katı doğrula, onar; hacim, alan, ağırlık merkezi", "solid_tools"),
    ("KATIBOOL", "Katı Boolean: kesin birleşim, kesişim, fark", "solid_boolean"),
    ("KATIYKES", "Katıyı yüzeyle kes (kesin)", "solid_surface_cut"),
    ("KATIDKES", "Katıyı düzlemle kes, kesit çıkar", "solid_plane_cut"),
    ("CIZGIAL", "Çizgi (DXF polyline) içe aktar", "import_strings"),
    ("CIZGI", "Çizgi düzenleyici", "edit_strings"),
    ("KESITBAGLA", "Kesit çizgilerini bağlayarak katı", "link_strings"),
    ("DOMEN", "Blok modelini katı ile işaretle", "flag_solid"),
    ("TENORKABUK", "Tenör kabuğu katısı", "make_grade_shell"),
    ("IMPLICIT", "Örtük (implicit) domen modeli", "implicit_domain"),
    ("NAKLIYOL", "Taşıma yolu mesafesi", "haul_route"),
    ("OPTIMIZE", "Ocak optimizasyonu, pushback, çizelge", "optimise_pit"),
    ("BASAMAK", "Basamak tasarımı", "bench_design"),
    ("CUTOFF", "Cut-off optimizasyonu (Lane)", "lane_cutoff"),
    ("OCAKVER", "Ocağı DXF olarak dışa aktar", "export_pit"),
    ("DONEM", "Dönem sonu yüzeyleri", "export_periods"),
    ("PATLATMA", "Delme-patlatma tasarımı", "blast"),
    ("DOKUM", "Döküm sahası tasarımı", "dump"),
    ("TABANTASARIM", "Taban çizgisinden pit tasarımı", "toe_design"),
    ("TENORKONTROL", "Tenör kontrolü (patlatma delikleri)", "grade_control"),
    ("REZERV", "Tasarım rezervi, yüzeyler arası hacim", "reserves"),
    ("OLCME", "Stok hacmi ve ay sonu ilerleme", "survey"),
    ("PANO", "Pano zarfı ve pano seçimi", "stopes"),
    ("DEKLINE", "Dekline ve kat galerileri", "decline"),
    ("KESITYAZDIR", "Kesiti PDF/PNG yazdır", "print_section"),
    ("YENILE", "Ekranı yenile", "refresh"),
    ("PAYLAS", "Projeyi dış komutlara (yapay zeka, minecore-cli) aç / kapat", "toggle_shared"),
    ("KILAVUZ", "Adım adım kullanım kılavuzu", "guide"),
    ("ORNEK", "Örnek sondaj projesini aç", "open_sample"),
    ("SINA", "Kurulumu sına", "self_test"),
)

BUILTIN_WORDS = ("YARDIM", "TEMIZLE")


def workbench_aliases(window: Any) -> AliasTable:
    """The command words of the main window's actions."""
    table = AliasTable()
    for name, description, method in WORKBENCH_COMMANDS:
        table.add(name, description, getattr(window, method))
    return table


def help_text(table: AliasTable) -> str:
    lines = ["Komut sözcükleri (kısaltma yeterli, ör. KOMP):"]
    lines += [f"  {a.name:14} {a.description}" for a in table.items()]
    lines += [
        "  YARDIM         bu liste",
        "  TEMIZLE        çıktıyı temizle",
        "",
        "Tam komutlar açık projede çalışır, proje yolunu yazmayın:",
        "  bm report model au --cutoffs 0.3 0.5 --density 2.7",
        "  drillholes composite assay AU_PPM --length 2",
        "  run C:\\isler\\aylik.txt      (dosyada %PROJE% açık proje olur)",
        "Bir komutun seçenekleri için sonuna --help ekleyin (ör. bm estimate --help).",
        "Yukarı/Aşağı ok: önceki komutlar. Boş Enter: son komutu tekrarla.",
    ]
    return "\n".join(lines)
