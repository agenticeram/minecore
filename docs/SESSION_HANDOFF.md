# Session handoff

Updated: 2026-11-28 (architect, thread "Yüzey araçlarını tamamla").
Resume file for a fresh session: `docs/RESUME_PROMPT.md`.

## Owner direction (binding)
- Windows 11 only. Wants a usable program fast, accepts some bugs, tests the GUI
  himself. Reply in Turkish, short, plain. He is not a GitHub user.
- Ask the owner before any commit or push (owner, 2026-11-18; replaces the
  earlier standing merge permission).
- Overnight 2026-10-25: work without asking until 09:55 TR (06:55 UTC), or 11:55
  TR if he does not answer; skip blockers and report them.
- Target depth: Datamine (Studio RM + NPVS + OP) as one integrated program.
- Open-source libraries welcome if permissive (MIT/BSD/Apache); no GPL.

## State
- 2026-11-28: surface editing completes E-CD-02 (`minecore.surfaceedit`).
  `merge_surfaces` upper/lower envelope or patch (a over b): plan
  arrangement of both TINs (edge crossings, swap points, intersection lines,
  footprint edges) fed to `build_surface` as breaklines, only the pieces of
  the surface that forms the result (consecutive pieces joined), so the
  result is exact on every arrangement face; trimmed to the union footprint
  with rings from the kept triangles (`boundary_rings`). `normal_offset`
  (least-squares meeting point of the moved face planes, fold check),
  `smooth_surface` (z only, towards the neighbour LS plane, user change cap,
  boundary held), `decimate_surface` (greedy insertion to a vertical
  tolerance), `read_ascii_grid` / `grid_surface` (no-data = holes). CLI
  `surface merge|normal-offset|smooth|decimate|from-grid|to-grid`; GUI
  Yüzey > "Yüzey birleştir / yama ..." = new tab "Birleştir / düzenle";
  command YBIRLESTIR. `surfaces.build_surface(tolerate_small_angles=True)`
  leaves tiny unresolvable breakline pieces unconstrained instead of failing;
  `_segment_crossings` now uses a box grid (was quadratic memory with long
  hull edges: OOM at 5k points). Tests `tests/python/survey/test_surfaceedit.py`
  (pyramid pit 83.333 m3, random-TIN envelopes, plane/trough offsets, plane
  smoothing, decimation tolerance, grid round trip, CLI) and the dialog test.
  Next in P1: exact solid booleans and plane sections (E-CD-03), or plan
  sheets (E-CD-05).
- 2026-11-27: surface tools, catalogue OC10 plus part of TP02 / E-CD-02
  (`minecore.surfacetools`): exact TIN contours (chained, closed/open,
  higher ground on the left, index contours, DXF TEXT labels along index
  contours), slope and aspect per triangle with user class limits and
  8-sector aspect table, plan and true area, ESRI ASCII grid rasters of z,
  slope, aspect; profiles of several surfaces along a string (breakpoints at
  every triangle edge, CSV, PNG/PDF/SVG with user vertical exaggeration);
  intersection lines of two surfaces (pit design vs topography = daylight
  line; overlay zero line, a above b on the left); vertical offset. CLI
  `surface contours|slope|profile|intersect|offset`; GUI Yüzey menu four
  entries, one tabbed dialog `gui/surfacetools.py`; command words KONTUR,
  EGIM, PROFIL, KESISIM. `io.dxf.write_dxf` writes TEXT. Tests: pyramid
  (perimeter 8R(1-L/H), slope atan(H/R), true area), planes, random TINs
  (contour/intersection vertices on the surfaces, exact profile). The
  catalogue file `09_olcme_cbs.txt` (owner's PC) could not be read this
  session (device offline), so catalogue validation numbers for OC10 are
  still to be checked against these tests. Next in E-CD-02: upper/lower
  merge and patching, normal offset, smoothing/decimation, grid <-> TIN.
- 2026-11-26: workbench command line (owner asked for an AutoCAD-style
  command bar inside the program, not a separate exe). Bottom dock
  `gui/command_line.py`: Turkish command words open the menu tools (unique
  prefix, accent-insensitive, YARDIM lists), full CLI commands run on the
  open project with its path inserted (`cli.project_argument_position`; the
  GUI closes and reopens the project around the command for the write
  lock), history, Enter repeats. `minecore run SCRIPT` (`minecore.script`)
  runs command files (comments, `set`/`%VAR%`, `--var`, `^`, all lines
  checked first); in the GUI `%PROJE%` is the open project. Owner guide
  (Turkish) in project files `kilavuz/komut-satiri.md`.
  Then (owner: "cli da bulunsun", AI must drive the program without
  computer use): console `minecore-cli.exe` restored in the Windows build;
  `cli_tools.py` gives the menu tools that lacked a command one (`import
  drillhole-file`, `export table|block-model|solid`, `stats`, `xval`,
  `shell`, `blast`, `dump`, `selftest`, `contents`); `commands` lists all;
  `run -` reads stdin; GUI "PAYLAS" mode (read-only between actions, reload on
  change) lets external commands write while the window is open.
  `docs/AUTOMATION.md`.
- 2026-11-19: roadmap replaced by phases P1-P14 (ADR-0011, `docs/ROADMAP.md`),
  built from a competitor feature inventory (owner's `katalog/envanter/`,
  Turkish; 135 feature groups `E-XX-NN`). Shared engines first: CAD kernel
  (P1), scheduling/network/equipment engines (P5). Current phase: P1,
  starting with string editing gaps (E-CD-01).
- 2026-11-19: P1 step 1b (E-CD-01 part 2): `stringops.boolean` (union,
  intersection, difference, xor of closed-string sets), `fillet`, `chamfer`,
  CLI `string boolean` and `edit fillet|chamfer`; editor snapping, typed
  X/Y and bearing/distance/gradient entry, two-point measure, fillet/chamfer
  and booleans against the background set. Next in P1: string attributes,
  then surface tools (E-CD-02).
- 2026-11-19: P1 step 1 (E-CD-01 part 1): `minecore.stringops` string
  toolbox (reverse, split, join, measure, simplify, resample, smooth,
  intersections, trim/extend, offset with mitre/round joins and dz/slope),
  CLI `minecore string list|measure|edit|split|join|intersect`, string
  editor "İşlem" box and "Noktadan böl" mode. Tests: analytic squares,
  L-shape, dumbbell split, circles, hypothesis star polygons. Next in P1:
  polygon booleans, fillet/chamfer, snapping and coordinate entry (E-CD-01
  part 2), then surface tools (E-CD-02).
- 2026-11-18 (later): OC08 + OC09 added. `minecore.survey.stockpiles`: pile
  volume inside a toe outline above a base (previous surface, outline TIN,
  least squares plane with RMS, level), triangle or grid method, density
  with measured date/source, stored surveys (`stockpile_survey` objects)
  and change since the previous survey. `minecore.survey.month_end`: exact
  excavated/filled volume, block share by sampled vertical lines, table by
  phase/bench/material (user grade limits or category attribute), truck
  load factor check. CLI `survey stockpile|stockpile-list|month-end`; GUI
  Açık Ocak > "Ölçme: stok hacmi ve ay sonu ilerleme". Catalogue tests:
  cone 523.6 m3, flat one-block cut = block tonnes. Next per catalogue:
  OC10, OC20.
- 2026-11-18: owner's capability catalogue (on his PC, `textler/`: 00_DIZIN,
  durum.csv, 9 area files, 257 capabilities; add 1-2 at a time, catalogue
  validation numbers become tests, thresholds are user inputs, ask before
  commit/push). OC06 + OC07 added: breakline/boundary/hole/longest-edge
  surfaces and ASCII DXF reading (`surfaces.build_surface`,
  `read_dxf_surface`, `clip_surface`, `io.dxf.read_dxf`); exact
  triangle-overlay, grid and section volumes (`minecore.survey.volumes`;
  CLI `surface import|clip|volume --method triangle|grid|sections`; GUI
  volume method choice and DXF surface import). `reserves.volume_between`
  removed (replaced). Triangulation now runs in local coordinates and is
  cached. Next per catalogue: OC08, OC09 (then UF01-UF07).
- 2026-11-17: PR #59 multi-pass estimation search (`<attr>_pass`), simple
  kriging, slope of regression (`_sor`) and kriging efficiency (`_ke`) for
  OK/SK, confidence criteria on max pass and min slope; GUI, CLI
  (`bm estimate --method sk --pass F:N --mean`, `bm classify --max-pass
  --min-slope`). Analytic Rust tests plus `tests/python/estimation/test_multipass.py`.
  GUI not yet tried by the owner. Next in estimation: kriging neighbourhood
  analysis (KNA), per-pass ellipsoid orientation, local-mean SK.
- 2026-11-11 pause check: `main` = `d7e62fc` (PR #43) holds all work; CI and
  the Windows app build are green on it. Leftover `claude/project-thread-*`
  branches were squash-merged (#26, #37, #43) and hold nothing new. Open PRs
  are Dependabot version bumps only (#50-#57). `docs/RESUME_PROMPT.md` added.
- 2026-11-03: relicensed under MIT and the repository made public
  (ADR-0010); PyPI release workflow (Trusted Publishing, tag `v0.1.0`);
  contributor guide, code of conduct, security policy; CI hardening
  (read-only token, RustSec advisories, Dependabot); #43 OMF export/import
  and haul road network distances feed the fleet; Windows artifacts kept
  7 days.
- 2026-10-25 after #38: #39 fleet haulage cost in the project cash flow and
  a monthly period preset; #40 greedy schedule keeps blending targets; #41
  underground development schedule (decline/level timing, stopes wait for
  ready levels); #42 haul road network shortest routes and travel times.
- Thread "Match Datamine Studio OP and NPVS" (2026-10-25): gap list vs Studio
  OP + NPVS from public knowledge (no inspection of the installed software;
  rule 7). PR #38 closes it: MILP blending targets; variable period lengths
  and per-period capacities (discounting, fixed cost, depreciation, IRR,
  payback in years); MCAF/PCAF cost factors; concentrate TC and penalties;
  bench reserves between surfaces and cut/fill volume (`minecore.reserves`);
  plan string editor (`gui/strings_editor.py`); pit design from toe strings
  (`benchdesign.design_from_toes`, `pit design-toes`); grade control dig
  lines (`minecore.gradecontrol`, CLI `gc`); dump sequencing by haul distance
  feeding the fleet; solid verify/repair/voxel booleans/cut
  (`minecore.solidtools`); connector ramps from side pit bottoms, road edge
  and safety bund strings. Owner asked (14:25 UTC) to delegate simple work
  to Haiku subagents and verify it.
- Same thread, later (2026-10-25): #34 pushbacks chosen by NPV
  (`pitopt/phases.py`); #35 one ramp per separate pit, pit bottoms counted,
  Lane never below the best constant cut-off (hypothesis counterexample
  fixed); #36 pit scenarios (save/compare/reload), underground stope
  economics + schedule + cash flow, truck/loader fleet by period, drill and
  blast design with hole layout, waste dump design, Excel fleet/scenario
  sheets; #37 mineable inventory by pushback and confidence category
  ("Envanter (kategori)" tab). From 11:16 UTC the owner asked for speed over
  tests: #36-#37 features have smoke tests only (listed in
  KNOWN_LIMITATIONS). The owner ran the Windows build of #36 and said every
  feature worked (11:53 UTC).
- Thread "Continue MineCore development" (2026-10-25): PR #27 rotated block
  models (ADR-0009); #28 switchback ramps and technical audit v1
  (`docs/audit/AUDIT_2026-10.md`, known-answer tests A-H, pushback ore-target
  fix); #29 payables, royalty and project cash flow (capital, tax with
  depreciation and loss carry-forward, closure, IRR, payback); #30
  co-products/by-products; #31 stockpile grade bins, capacity, rehandling;
  #32 sensitivity analysis (spider table/chart) and exact Lane balancing
  points; #33 NPV-maximising MILP schedule (HiGHS via scipy, never below the
  greedy schedule). Owner said not to post Windows build links for now.
- `main`: PRs #1-#26 merged (PR #21-#22: process routes in pit optimisation
  and per-route plant capacity in the scheduler; PR #23: ore drives; PR #24:
  self-test covers routes, Lane cut-off, sectors and the decline; PR #26:
  fixes from the owner's first hands-on Windows test, see below).
- Owner's first hands-on test (2026-10-25 morning) and PR #26 fixes: the
  new-project save dialog looked like it wanted an existing file (now folder,
  then name; open accepts the containing folder); about 5 s "not responding" on
  the first import (libraries now warmed up at start, wait cursor); no app
  icon (`tools/make_icon.py` draws it; exe, window and taskbar); tall dialogs
  (variogram, pit optimisation) did not fit the screen (`gui/layout.py`
  scroll panels and `fit_to_screen`); a second block model with the default
  name crashed (free default name, clear message); user messages showed a
  traceback (plain `RuntimeError` now shows as a warning); the validation
  report mixed English and showed 150 rows for about 10 causes (`gui/issues_tr.py`:
  Turkish wording, one row per interval that overlaps many others).
- Overnight additions (PR #15-#20): string import and section linking, user
  guide (F1), re-blocking, Lane cut-off optimisation, partial-block solid
  fractions and weighted reports, solid CLI, CLI swath, spiral haul ramp in
  bench design, implicit RBF domains, section/plan PDF/PNG/SVG plots,
  end-of-period surfaces, decline and level crosscuts, self-test,
  geotechnical sectors in bench design, process routes (mill + leach).
- Bugs fixed overnight: saving a surface/solid/string set under an existing
  name failed (second bench design run crashed); plan plots of tall models.
- Windows build: "Windows app" workflow, artifact `MineCore-windows`; from
  PR #20 on it runs `MineCore.exe --self-test` (whole chain on the sample).
  Its first run found three packaging bugs that earlier exe builds had:
  pyvista steps crashed (half-bundled mypy), the pit optimisation warning
  crashed (no stderr in a windowed exe), PDF/SVG plots failed (matplotlib
  backends not bundled). All fixed; the self-test passes in CI.
- Local gates: run the scratch `gate.sh` equivalent (ruff check, ruff format
  --check, mypy, cargo fmt) and check for the text GATES-OK; piping through
  `tail` hides the exit status.

## Where things live
- Rust: `mc-core::pitopt` (max closure), `mc-core::kriging`.
- Python: `minecore.estimation`, `minecore.blockmodel` (`workflow`,
  `validation`, `classify`, `calc`, `importer`, `reblock`), `minecore.pitopt`
  (`scheduling`, `design`, `width`, `benchdesign`, `cutoff`),
  `minecore.underground` (`stopes`, `development`), `minecore.geology.implicit`,
  `minecore.surfaces`, `minecore.solids`, `minecore.strings`, `minecore.plots`,
  `minecore.selftest`, `minecore.io` (`excel`, `dxf`), `minecore.cli_modelling`.
- GUI: `gui/app.py` (main window, section view, print), `gui/resource.py`,
  `gui/pit.py`, `gui/underground.py`, `gui/geology.py`, `gui/guide.py`,
  `gui/layout.py` (scroll panels), `gui/issues_tr.py` (report wording), `gui/assets/` (icon).
- Limits: `docs/KNOWN_LIMITATIONS.md` (kept current).

## Next actions
1. Owner feedback on the Windows build with real data comes first.
2. Deeper tests for the smoke-tested features of #36-#37 (scenarios, stope
   economics, fleet, drill and blast, dumps, category inventory).
3. Sub-cells (needs its own ADR).
4. Routes and stockpiles in Lane; short-term (monthly shot) scheduling;
   blending in the greedy schedule.
5. Underground: stope shapes beyond grid boxes, footwall drives and
   drawpoints (development schedule done in #41).
6. Dump ramps (haulage cost and road network done in #39, #42).
7. Interoperability: further open formats (OMF done in #43); M12 installer
   and PyPI release.
8. Dependabot PRs #50-#57: merge when CI is green.

## Owner owes
- Try the Windows build with real data and report problems (Yardım → Kurulumu
  sına checks an installation in one click).
