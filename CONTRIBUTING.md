# Contributing to MineCore

Thank you for helping build an open source mining engineering platform.
Mining engineers, geologists, geostatisticians, surveyors and software
developers are all welcome. You do not need to know Rust to contribute: most
of the application, the CLI and the tests are Python, and many valuable
contributions are not code at all.

Türkçe rehber: [`docs/tr/KATKI_REHBERI.md`](docs/tr/KATKI_REHBERI.md).

## Ways to contribute

| Kind | What it looks like | Label |
| --- | --- | --- |
| Code | A feature group from the roadmap, a bug fix, a CLI command | `phase:P1` … `phase:P14`, `bug` |
| Validation data | A worked example or dataset with a known answer (textbook, paper, hand calculation) that becomes a test | `validation-data` |
| Method review | Check that an algorithm follows the published method; point out wrong assumptions | `needs-domain-expert` |
| Testing | Run the Windows build on your own (anonymised) data and report wrong results | `bug` |
| Documentation and translation | Tutorials, Turkish/English UI text review | `documentation` |

Start here:

- [`good first issue`](https://github.com/dersders0120-rgb/minecore/labels/good%20first%20issue):
  small, self-contained tasks.
- [`help wanted`](https://github.com/dersders0120-rgb/minecore/labels/help%20wanted):
  larger tasks where outside help is especially welcome.
- [`validation-data`](https://github.com/dersders0120-rgb/minecore/labels/validation-data):
  for engineers who can supply reference cases without writing code.
- [`docs/ROADMAP.md`](docs/ROADMAP.md): phases P1–P14 and the order they are
  built in. Work outside the current and next phase is welcome but is
  reviewed after the current phase.
- [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md): what MineCore
  cannot do yet. Each line is a possible contribution.

## Workflow

1. **Pick or open an issue.** Every change starts from an issue. For a new
   engineering feature use the *Feature request* form; for a concrete piece of
   work use the *Task* form; for reference data use the *Validation data*
   form.
2. **Comment before you start.** Say that you are taking it and, for anything
   larger than a small fix, describe your approach in a few lines (method,
   reference, files you expect to touch). A maintainer confirms or suggests
   changes. This avoids two people doing the same work and avoids rewrites.
3. **Fork and branch.** Create a branch from `main` in your fork, for example
   `feat/surface-intersection` or `fix/dxf-arc-import`.
4. **Build it as a vertical slice** (see *Definition of done* below): engine,
   command, GUI entry, tests and documentation in the same pull request.
5. **Run the quality gates locally** (next section).
6. **Open a pull request** that links the issue (`Closes #NN`) and fills in
   the template: what changed, how it was verified, limitations.
7. **CI and review.** CI runs the same gates on Windows and Linux. A
   maintainer reviews the engineering method and the code. Expect questions
   about the reference used for verification.
8. **Merge.** The project owner merges. Keep one change per pull request; a
   small pull request is reviewed faster.

## Definition of done

A feature is done when all of these hold:

- **Engine:** the calculation lives in the domain module (Python) or in
  `crates/mc-core` (Rust) and runs without the GUI.
- **Command:** every tool is reachable as a command. Each GUI menu entry has a
  matching `minecore` CLI command, which also works from the workbench's
  command line and from `minecore-cli.exe`. This lets scripts and AI agents
  drive every tool without screen control. `minecore commands` lists them.
- **GUI:** the workbench has an entry for the tool (menu, dialog or panel).
  User-facing text may be Turkish or English.
- **Verified:** tests check the result against an analytic case, a hand
  calculation, a published worked example or an independent implementation
  (CLAUDE.md rule 4). If a catalogue or issue gives validation numbers, they
  become test assertions.
- **Honest limits:** anything left out is listed in
  `docs/KNOWN_LIMITATIONS.md`. No inert buttons, hardcoded results or stub
  algorithms.

## Engineering rules

Read [`CLAUDE.md`](CLAUDE.md). It holds the rules every human and AI
contributor follows. The ones that most often matter in review:

1. **Evidence over claims.** A change is done when tests you ran pass.
2. **Independent verification** of every numerical algorithm (see above).
3. **Thresholds and regulatory values are user inputs, never defaults.**
   Cut-offs, slope angles, safety factors, vibration limits, air velocity
   limits, QA/QC acceptance limits and similar values are entered by the user
   and stored with the project. Code may offer an example value in
   documentation, never a silent default.
4. **No compliance claims.** Resource classification is never presented as
   JORC, NI 43-101, PERC or similar compliant; that needs a competent person.
5. **Open formats only.** Read and write documented open formats
   (CSV, DXF, OMF, OBJ, STL, LAS, GeoTIFF, Shapefile, E57 and similar). Never
   copy proprietary code, assets or undocumented file formats of other mining
   software vendors (rule 7).
6. **Permissive licences only.** New dependencies must be MIT, BSD, Apache-2.0
   or similar and need maintainer approval (`deny.toml`). A GPL/AGPL tool may
   only be used as a separate process that the user installs, never linked or
   bundled. PySide6 (LGPL) is used through dynamic linking only.
7. **Contracts need an ADR.** Units, coordinates, geometry, block model schema
   and the project format change only through an ADR in `docs/adr/` with a
   compatibility or migration test. Never edit an applied migration.
8. **One-way dependencies:** core ← domain modules ← workbench. The core never
   imports GUI or Python code.
9. **Windows 11 is the target platform.** CI also runs on Linux; keep both
   green, but test GUI work on Windows when you can.
10. **Language:** code, identifiers and documentation in English; issues and
    discussion may be in Turkish or English.

## Development setup

Requirements: Rust (version pinned in `rust-toolchain.toml`, installed by
`rustup`) and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/<you>/minecore.git
cd minecore
uv sync                      # create .venv and build the native extension
uv run minecore info         # check the Rust-Python bridge
uv run minecore-gui          # start the workbench
```

## Quality gates

All must pass before a merge (CI runs them on Windows and Linux):

```sh
uv run ruff check . && uv run ruff format --check . && uv run mypy
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace && uv run pytest
```

While fixing a failure, re-run only the failing tests
(`uv run pytest path/to/test_file.py -k name`), then the full set once before
pushing.

## Roadmap codes

Issues and pull requests cite feature group codes `E-XX-NN` from the
competitor feature inventory (for example `E-CD-02` surface tools) and the
phase that owns them (`phase:P1`). Area prefixes:

| Prefix | Area | Prefix | Area |
| --- | --- | --- | --- |
| VT | Data, drillholes, QA/QC | DP | Drill and blast |
| JM | Geological modelling | NF | Haulage, fleet, production |
| KE | Resource estimation, block model | OL | Survey, point clouds, GIS |
| CD | CAD, visualisation, formats | HV | Ventilation |
| AO | Open-pit optimisation and design | JT | Geotechnics |
| PL | Scheduling and economics | MS, JF | Mine water, geophysics |
| YA | Underground | RH, PF | Closure and environment, platform |

A newly noticed feature of another package goes into the inventory first,
then into the phase that owns its area. Describe the engineering need and the
public method; do not describe or copy a vendor's interface.

## AI-assisted contributions

AI coding agents are welcome and follow the same rules (`AGENTS.md`,
`CLAUDE.md`). The human who opens the pull request is responsible for it:
run the tests yourself and report their real output.

## Licence

By contributing you agree that your contribution is licensed under the
[MIT License](LICENSE).
