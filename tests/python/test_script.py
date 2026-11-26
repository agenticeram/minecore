"""Command scripts (`minecore run`): parsing, checking and running."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from minecore import cli
from minecore.script import ScriptError, parse_script, split_words

SAMPLE = Path(cli.__file__).resolve().parent / "gui" / "sample"


def test_parse_variables_quotes_continuation_and_comments() -> None:
    text = (
        "# note\nrem another note\n\n"
        "set P=C:\\Maden Projeleri\\a.mcproj\n"
        'minecore bm report "%P%" model au ^\n   --cutoffs %CUT% 0.5\n'
        "set cut=9\n"
        "info --json  # trailing words are arguments, not comments\n"
    )
    cmds = parse_script(text, {"CUT": "0.3"})
    assert cmds[0].line == 5
    assert cmds[0].argv == [
        "bm", "report", "C:\\Maden Projeleri\\a.mcproj", "model", "au", "--cutoffs", "0.3", "0.5",
    ]  # fmt: skip
    assert cmds[1].argv[:2] == ["info", "--json"]


def test_parse_errors_name_the_line() -> None:
    with pytest.raises(ScriptError, match="line 2: unknown variable %X%"):
        parse_script("info\nbm report %X%\n")
    with pytest.raises(ScriptError, match="line 1: unclosed"):
        split_words('a "b', 1)
    with pytest.raises(ScriptError, match="cannot run another script"):
        parse_script("run other.txt\n")
    with pytest.raises(ScriptError, match="ends with \\^"):
        parse_script("info ^\n")
    assert parse_script("echo 100%%\n")[0].argv == ["echo", "100%"]


def test_bad_line_stops_the_whole_script_before_anything_runs(tmp_path: Path) -> None:
    proj = tmp_path / "p.mcproj"
    script = tmp_path / "s.txt"
    script.write_text(f'project create "{proj}" --name t\nbm create "{proj}" m --size 1 2\n')
    assert cli.main(["run", str(script)], io.StringIO()) == 2
    assert not proj.exists()


def test_sample_chain_runs_and_stops_at_first_failure(tmp_path: Path) -> None:
    proj = tmp_path / "with space.mcproj"
    script = tmp_path / "s.txt"
    script.write_text(
        f"set P={proj}\n"
        'project create "%P%" --name t\n'
        f'drillholes import "%P%" "{SAMPLE}"\n'
        'drillholes composite "%P%" assay AU_PPM --length 2\n'
        'bm create "%P%" m --source assay_komp_2m/AU_PPM --size 20 20 10\n'
        'bm estimate "%P%" m assay_komp_2m/AU_PPM au --method idw --radius 150\n'
        'bm report "%P%" m au --cutoffs 0 --density 2.5\n',
        encoding="utf-8",
    )
    out = io.StringIO()
    assert cli.main(["run", str(script)], out) == 0, out.getvalue()
    assert "6 of 6 commands succeeded" in out.getvalue()

    bad = tmp_path / "bad.txt"
    bad.write_text(f'bm report "{proj}" m missing --cutoffs 0\ninfo\n')
    out = io.StringIO()
    assert cli.main(["run", str(bad)], out) == 1
    assert "FAILED on line 1" in out.getvalue()
    assert "1 not run" in out.getvalue()
    out = io.StringIO()
    assert cli.main(["run", str(bad), "--keep-going"], out) == 1
    assert "1 of 2 commands succeeded, 1 failed" in out.getvalue()


def test_example_script_is_valid() -> None:
    example = Path(__file__).resolve().parents[2] / "docs" / "examples" / "is_akisi_ornek.txt"
    assert cli.main(["run", str(example), "--check"], io.StringIO()) == 0
