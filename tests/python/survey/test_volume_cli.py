"""OC06/OC07 headless: DXF surface import, clipping and volumes from the CLI."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from minecore.cli import main
from minecore.io.dxf import Polyline, write_dxf


def _run(*argv: str) -> str:
    out = io.StringIO()
    assert main(list(argv), out) == 0, out.getvalue()
    return out.getvalue()


def _square_tin(z: float) -> list[tuple[str, np.ndarray]]:
    c = np.array([[0, 0, z], [100, 0, z], [100, 100, z], [0, 100, z]])
    return [("TIN", np.array([[c[0], c[1], c[2]], [c[0], c[2], c[3]]]))]


def test_dxf_import_clip_and_volumes(tmp_path: Path) -> None:
    p = str(tmp_path / "v.mcproj")
    _run("project", "create", p, "--name", "v")
    top, base = tmp_path / "top.dxf", tmp_path / "base.dxf"
    write_dxf(top, faces=_square_tin(10.0))
    write_dxf(base, faces=_square_tin(0.0))
    assert "triangles" in _run("surface", "import", p, "ust", str(top))
    _run("surface", "import", p, "alt", str(base))
    out = _run("surface", "volume", p, "ust", "alt", "--density", "2")
    assert "triangle: cut 100,000.0 m3, fill 0.0 m3" in out
    assert "200,000 t cut" in out
    assert "grid 5 m: cut 100,000.0" in _run(
        "surface", "volume", p, "ust", "alt", "--method", "grid", "--cell", "5"
    )
    axis = tmp_path / "axis.dxf"
    write_dxf(axis, [Polyline("EKSEN", np.array([[0, 50, 0], [100, 50, 0.0]]), False)])
    _run("strings", "import", p, "eksen", str(axis))
    sections_csv = tmp_path / "kesit.csv"
    out = _run(
        "surface", "volume", p, "ust", "alt", "--method", "sections", "--axis", "eksen",
        "--interval", "10", "--half-width", "60", "--sections-csv", str(sections_csv),
    )  # fmt: skip
    assert "end_area: cut 100,000.0" in out
    assert "prismoidal: cut 100,000.0" in out
    assert sections_csv.read_text(encoding="utf-8").startswith("station_m")
    ring = tmp_path / "ring.dxf"
    sq = np.array([[10, 10, 0], [90, 10, 0], [90, 90, 0], [10, 90, 0.0]])
    write_dxf(ring, [Polyline("SINIR", sq, True)])
    _run("strings", "import", p, "sinir", str(ring))
    assert "cut 64,000.0" in _run("surface", "volume", p, "ust", "alt", "--limit", "sinir")
    _run("surface", "clip", p, "ust", "sinir", "--save", "ust_kirpik")
    assert "cut 64,000.0" in _run("surface", "volume", p, "ust_kirpik", "--level", "0")
