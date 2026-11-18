"""OC08/OC09 headless: stockpile and month-end commands from the CLI."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from minecore.blockmodel import BlockModel
from minecore.blockmodel.model import Grid
from minecore.cli import main
from minecore.io.dxf import Polyline, write_dxf
from minecore.project import Project
from minecore.surfaces import Surface, save_surface


def _run(*argv: str) -> str:
    out = io.StringIO()
    assert main(list(argv), out) == 0, out.getvalue()
    return out.getvalue()


def _flat(z: float, size: float = 100.0) -> Surface:
    xy = np.array([[0, 0], [size, 0], [size, size], [0, size]], float)
    return Surface(np.column_stack([xy, np.full(4, z)]))


def test_stockpile_and_month_end_commands(tmp_path: Path) -> None:
    p = str(tmp_path / "s.mcproj")
    _run("project", "create", p, "--name", "s")
    with Project.open(p) as project:
        # A 3 m high pile: 40 x 40 m toe, 20 x 20 m plateau, on ground at 50 m.
        xy = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
        toe = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], float)
        top = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], float)
        z = np.r_[np.full(8, 50.0), np.full(4, 53.0)]
        pts = np.column_stack([np.vstack([xy, toe, top]), z])
        save_surface(project, "yigin_1", Surface(pts))
        save_surface(project, "yigin_0", _flat(50.0))
        save_surface(project, "ocak_ekim", _flat(110.0))
        save_surface(project, "ocak_kasim", _flat(100.0))
        grid = Grid((0.0, 0.0, 80.0), (10.0, 10.0, 10.0), (10, 10, 4))
        BlockModel.create(project, "BM", grid).set_attributes(
            {"au": np.where(np.arange(400) % 10 < 5, 1.0, 0.1)}, {}, "test"
        )
    outline = tmp_path / "toe.dxf"
    ring = np.array([[30, 30, 0], [70, 30, 0], [70, 70, 0], [30, 70, 0.0]])
    write_dxf(outline, [Polyline("TOPUK", ring, True)])
    _run("strings", "import", p, "topuk", str(outline))
    # Every vertex lies in two parallel planes (a prismatoid): V = h/3 (A1 + A2 + sqrt(A1 A2)).
    expected = 3.0 / 3.0 * (40 * 40 + 20 * 20 + 40 * 20)
    out = _run(
        "survey", "stockpile", p, "yigin_1", "topuk", "--base", "surface",
        "--base-surface", "yigin_0", "--density", "2", "--density-date", "2026-11-01",
        "--name", "ROM", "--date", "2026-11-30", "--save",
    )  # fmt: skip
    assert f"ROM: {expected:,.1f} m3 above the surface base" in out
    assert f"tonnes {2 * expected:,.0f}" in out
    assert "stored survey ROM 2026-11-30" in out
    out = _run("survey", "stockpile", p, "yigin_1", "topuk", "--base", "plane", "--spacing", "5")
    assert f"{expected:,.1f} m3 above the plane base" in out
    assert "ROM  2026-11-30" in _run("survey", "stockpile-list", p)
    csv_path = tmp_path / "ay.csv"
    out = _run(
        "survey", "month-end", p, "ocak_ekim", "ocak_kasim", "BM", "--density", "2.5",
        "--grade", "au", "--class", "cevher:0.5", "--truck-loads", "500",
        "--load-factor", "500", "--csv", str(csv_path),
    )  # fmt: skip
    assert "excavated 100,000.0 m3 (exact), in the block model 100,000.0 m3 (100.0%)" in out
    assert "total cevher: 50,000 m3, 125,000 t, grade 1.000" in out
    assert "total pasa: 50,000 m3, 125,000 t, grade 0.100" in out
    assert "survey/truck 1.000" in out
    assert csv_path.read_text(encoding="utf-8").startswith("phase,bench_m")
