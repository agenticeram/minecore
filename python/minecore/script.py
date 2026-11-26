"""Command scripts: run a list of CLI commands from a text file.

A script holds one ``minecore`` command per line, so a workflow (import,
composite, estimate, report) can be repeated with one call::

    # monthly reconciliation
    set PROJE=C:\\Maden\\maden.mcproj
    drillholes import %PROJE% "C:\\Maden\\sondaj"
    bm report %PROJE% model au --cutoffs 0.3 0.5 ^
        --density 2.7

Rules:

* ``#`` or ``rem`` starts a comment line; blank lines are skipped.
* A line ending in ``^`` continues on the next line (as in cmd.exe).
* A leading ``minecore`` or ``minecore-cli`` word is ignored, so commands can
  be pasted from a terminal.
* ``set NAME=value`` defines a variable; ``%NAME%`` is replaced by a
  ``--var`` value (which wins over ``set``), then the script variable, then
  an environment variable. Names ignore case. An unknown
  name is an error. ``%%`` is a literal percent sign.
* Double quotes group words with spaces; backslashes are literal (Windows
  paths need no escaping).

Every line is checked before anything runs, so a typing error on line 40
does not leave a half-finished project. Commands then run in this process,
in order, and the run stops at the first failure unless ``keep_going``.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

_SET = re.compile(r"^set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$", re.IGNORECASE)
_VAR = re.compile(r"%%|%([A-Za-z_][A-Za-z0-9_]*)%")
_PREFIXES = ("minecore", "minecore-cli", "minecore.exe", "minecore-cli.exe")


class ScriptError(ValueError):
    """A script line that cannot be read or expanded."""


@dataclass(frozen=True)
class ScriptCommand:
    """One command of a script: its first line number and argument vector."""

    line: int
    text: str
    argv: list[str]


@dataclass
class ScriptResult:
    """Outcome of a script run."""

    commands: int = 0
    succeeded: int = 0
    failed: list[tuple[int, str]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed and self.succeeded == self.commands


def read_script_text(path: str | Path) -> str:
    """Read a script as UTF-8 (with or without BOM), else Windows Turkish ANSI."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1254")


def split_words(text: str, line: int) -> list[str]:
    """Split on whitespace, honouring double quotes; backslashes stay literal."""
    words: list[str] = []
    current: list[str] = []
    in_word = False
    quoted = False
    for ch in text:
        if ch == '"':
            quoted = not quoted
            in_word = True
        elif ch.isspace() and not quoted:
            if in_word:
                words.append("".join(current))
                current, in_word = [], False
        else:
            current.append(ch)
            in_word = True
    if quoted:
        raise ScriptError(f"line {line}: unclosed double quote")
    if in_word:
        words.append("".join(current))
    return words


def _expand(text: str, line: int, variables: Mapping[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        if m.group(0) == "%%":
            return "%"
        name = m.group(1)
        for source in (variables, os.environ):
            for key, value in source.items():
                if key.upper() == name.upper():
                    return value
        raise ScriptError(f"line {line}: unknown variable %{name}%")

    return _VAR.sub(repl, text)


def join_words(argv: list[str]) -> str:
    """Show an argument vector the way it would be typed (quotes where needed)."""
    return " ".join(f'"{a}"' if not a or any(c.isspace() for c in a) else a for a in argv)


def parse_script(text: str, variables: Mapping[str, str] | None = None) -> list[ScriptCommand]:
    """Turn script text into commands, expanding ``set`` variables in order."""
    env: dict[str, str] = dict(variables or {})
    fixed = {name.upper() for name in env}  # --var values win over set lines
    logical: list[tuple[int, str]] = []
    pending: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not pending:
            start = number
            if not stripped or stripped.startswith("#") or re.match(r"^rem(\s|$)", stripped, re.I):
                continue
        if stripped.endswith("^"):
            pending.append(stripped[:-1])
            continue
        pending.append(stripped)
        logical.append((start, " ".join(p for p in pending if p)))
        pending = []
    if pending:
        raise ScriptError(f"line {start}: the last line ends with ^ but nothing follows")

    commands: list[ScriptCommand] = []
    for number, body in logical:
        m = _SET.match(body)
        if m:
            if m.group(1).upper() in fixed:
                continue
            value = _expand(m.group(2).strip(), number, env)
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            env = {k: v for k, v in env.items() if k.upper() != m.group(1).upper()}
            env[m.group(1)] = value
            continue
        words = split_words(_expand(body, number, env), number)
        if words and words[0].lower() in _PREFIXES:
            words = words[1:]
        if not words:
            raise ScriptError(f"line {number}: no command after the program name")
        if words[0] == "run":
            raise ScriptError(f"line {number}: a script cannot run another script")
        commands.append(ScriptCommand(number, body, words))
    return commands


def check_commands(
    commands: list[ScriptCommand], validate: Callable[[list[str]], str | None]
) -> list[str]:
    """Problems found by ``validate`` (an argument check), one per bad line."""
    problems = []
    for cmd in commands:
        message = validate(cmd.argv)
        if message is not None:
            problems.append(f"line {cmd.line}: {message}")
    return problems


def run_commands(
    commands: list[ScriptCommand],
    execute: Callable[[list[str], TextIO], int],
    out: TextIO,
    *,
    keep_going: bool = False,
) -> ScriptResult:
    """Run each command with ``execute``; stop at the first failure unless asked not to."""
    result = ScriptResult(commands=len(commands))
    t_all = time.perf_counter()
    for i, cmd in enumerate(commands, start=1):
        out.write(f"[{i}/{len(commands)}] line {cmd.line}: {join_words(cmd.argv)}\n")
        out.flush()
        t0 = time.perf_counter()
        try:
            code = execute(cmd.argv, out)
            reason = f"exit code {code}"
        except SystemExit as exc:  # argparse rejects arguments this way
            code = exc.code if isinstance(exc.code, int) else 1
            reason = f"exit code {code}"
        except Exception as exc:
            code = 1
            reason = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - t0
        if code == 0:
            result.succeeded += 1
            out.write(f"      ok ({seconds:.1f} s)\n")
            continue
        result.failed.append((cmd.line, reason))
        out.write(f"      FAILED on line {cmd.line} ({reason})\n")
        if not keep_going:
            break
    result.seconds = time.perf_counter() - t_all
    skipped = result.commands - result.succeeded - len(result.failed)
    out.write(
        f"script: {result.succeeded} of {result.commands} commands succeeded"
        f"{f', {len(result.failed)} failed' if result.failed else ''}"
        f"{f', {skipped} not run' if skipped else ''}"
        f" in {result.seconds:.1f} s\n"
    )
    return result
