# MineCore

MineCore is an integrated, desktop-first mining engineering platform: drillhole
data, geological modelling, geostatistics, block modelling, resource evaluation,
open-pit and underground design, and scheduling, built on one shared project
model, unit system and geometry kernel.

Open source under the [MIT License](LICENSE). Contributions are welcome.

## Install

```sh
pip install minecore        # Python 3.12+, Windows x64 or Linux x86_64
minecore info               # check the install
minecore-gui                # start the desktop application
```

The desktop application has a CAD-style command line docked under the 3D
view (Görünüm > Komut satırı, Ctrl+9). Command words open the matching tool
(`KOMPOZIT`, `KESTIRIM`, `RAPOR`; a unique abbreviation is enough, `YARDIM`
lists them) and full headless commands run on the open project without its
path (`bm report model au --cutoffs 0.3 0.5`). Every capability also runs
headless from a terminal, e.g. `minecore bm report p.mcproj model au`.
`minecore run workflow.txt` (or `run workflow.txt` in the command line) runs a
file of commands, one per line, with `#` comments, `set NAME=value` /
`%NAME%` variables and `^` continuation; every line is checked before
anything runs (`--check` only checks). See
[docs/examples/is_akisi_ornek.txt](docs/examples/is_akisi_ornek.txt).
The Windows build also ships `minecore-cli.exe` (console) so scripts and AI
agents can drive every tool; see [docs/AUTOMATION.md](docs/AUTOMATION.md).

## Layout

| Path | Contents |
| --- | --- |
| `crates/mc-core` | Rust core engine: contracts, geometry, numerics. No GUI, no Python. |
| `crates/mc-py` | PyO3 bindings, built as `minecore._core`. |
| `python/minecore` | Python application layer, CLI, scripting API. |
| `tests/python` | Python tests (pytest + hypothesis). |
| `tools` | Development tooling, including `verify_submission.py`. |
| `docs` | Architecture, roadmap, ADRs, task briefs, session handoff. |

## Development

Requirements: Rust (pinned in `rust-toolchain.toml`) and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # create .venv and build the native extension
uv run minecore info         # check the Rust-Python bridge
uv run pytest                # Python tests
cargo test --workspace       # Rust tests
```

Quality gates (all must pass before merge):

```sh
uv run ruff check . && uv run ruff format --check . && uv run mypy
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace && uv run pytest
```

Contributions are welcome: see [`CONTRIBUTING.md`](CONTRIBUTING.md)
(Türkçe: [`docs/tr/KATKI_REHBERI.md`](docs/tr/KATKI_REHBERI.md)) and the
[`good first issue`](https://github.com/dersders0120-rgb/minecore/labels/good%20first%20issue)
label. Read `CLAUDE.md` before contributing; it holds the rules every human and
AI contributor follows.
