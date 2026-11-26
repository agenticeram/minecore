"""Headless commands for the workbench tools (every menu entry has a CLI)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from minecore import cli
from minecore.io.dxf import Polyline, write_dxf

SAMPLE = Path(cli.__file__).resolve().parent / "gui" / "sample"


def _run(*argv: str) -> str:
    out = io.StringIO()
    assert cli.main(list(argv), out) == 0, out.getvalue()
    return out.getvalue()


def test_tool_commands_on_the_sample(tmp_path: Path) -> None:
    p = str(tmp_path / "p.mcproj")
    src = "assay_komp_2m/AU_PPM"
    _run("project", "create", p, "--name", "t")
    _run("drillholes", "import", p, str(SAMPLE))
    assert "lito2: interval, 94 rows" in _run(
        "import", "drillhole-file", p, str(SAMPLE / "lithology.csv"), "--name", "lito2"
    )
    _run("drillholes", "composite", p, "assay", "AU_PPM", "--length", "2")

    stats = json.loads(
        _run("stats", p, src, "--top-cut", "5", "--decluster", "10", "100", "5", "--json")
    )
    assert stats["n"] > 1000
    assert stats["top_cut"]["mean_after"] < stats["mean"]
    assert len(stats["decluster"]["by_cell"]) == 5

    xv = json.loads(_run("xval", p, src, "--radius", "100", "--json"))
    assert xv["n"] == stats["n"]
    assert abs(xv["mean_error"]) < 0.1 * stats["mean"]

    _run("bm", "create", p, "BM", "--source", src, "--size", "20", "20", "10")
    _run("bm", "estimate", p, "BM", src, "au", "--method", "idw", "--radius", "150")
    assert "kabuk_au_0.5" in _run("shell", p, "BM", "au", "0.5")
    for kind, name, target in (
        ("table", "assay_komp_2m", "t.csv"),
        ("block-model", "BM", "bm.csv"),
        ("solid", "kabuk_au_0.5", "s.dxf"),
    ):
        _run("export", kind, p, name, str(tmp_path / target))
        assert (tmp_path / target).stat().st_size > 0
    assert (tmp_path / "bm.csv").read_text(encoding="utf-8-sig").count("\n") == 7714 + 1

    # 200 x 150 m outline at 1000 m: analytic checks on blast and dump
    ring = np.array([[0, 0, 1000], [200, 0, 1000], [200, 150, 1000], [0, 150, 1000]], float)
    write_dxf(tmp_path / "o.dxf", [Polyline("OUT", ring, True)])
    _run("strings", "import", p, "sinir", str(tmp_path / "o.dxf"))
    blast = json.loads(_run("blast", p, "--outline", "sinir", "--json"))
    assert abs(blast["burden"] - 30 * 0.165) < 1e-9  # burden factor x diameter
    assert abs(blast["spacing"] - 1.15 * blast["burden"]) < 1e-9
    holes = blast["layout"]["holes"]
    assert 0.8 < holes * blast["burden"] * blast["spacing"] / (200 * 150) < 1.2
    dump = json.loads(_run("dump", p, "sinir", "--ground-z", "1000", "--json"))
    assert dump["top"] > 1000
    assert abs(dump["area_ha"] - 3.0) < 0.1

    contents = json.loads(_run("contents", p, "--json"))
    assert {t["name"] for t in contents["tables"]} >= {"assay", "lito2", "assay_komp_2m"}
    assert contents["block_models"][0]["attributes"][0] == "au"
    kinds = {(o["kind"], o["name"]) for o in contents["objects"]}
    assert ("solid", "kabuk_au_0.5") in kinds
    assert ("surface", "dokum_sinir") in kinds


def test_command_listing_covers_every_command() -> None:
    listed = {c["command"] for c in json.loads(_run("commands", "--json"))}
    assert {"bm estimate", "blast", "dump", "xval", "export solid", "run", "contents"} <= listed
    assert all(len(c.split()) <= 2 for c in listed)
