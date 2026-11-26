# Driving MineCore from scripts and AI agents

Every workbench menu entry has a headless command, so a script or an AI agent
can do the work without clicking (no screen control needed).

## Programs

- Windows build: `minecore-cli.exe` next to `MineCore.exe` (console). From
  Python: `minecore` / `python -m minecore`.
- `minecore commands [--json]`: every command with a one-line help.
  `minecore <command> --help`: its arguments.
- `minecore contents PROJECT [--json]`: drillhole tables and fields, block
  models and attributes, solids, surfaces, string sets. Start here.
- Most reporting commands take `--json` for machine-readable output
  (`stats`, `xval`, `blast`, `dump`, `contents`, `bm report`, `info`, ...).
- Exit code 0 is success; errors go to standard error.

## Many commands quickly

Each start of the program costs a second or two. For a series of commands,
send them in one call:

```
minecore-cli run - < steps.txt          # or: run steps.txt
```

One command per line, `#` comments, `set NAME=value` and `%NAME%`, `^`
continues a line, `--var NAME=VALUE` overrides. All lines are checked before
any runs; the run stops at the first failure (line number reported) unless
`--keep-going`.

## Working while the desktop program is open

The workbench holds the project's write lock. To let external commands change
the project while it is open, turn on **Görünüm > Projeyi dış komutlara aç**
(command word `PAYLAS`). The window then holds the lock only while one of its
own tools runs and reloads when the project changes. Reading commands
(`contents`, `stats`, `bm report`, ...) always work.

The same commands can be typed into the workbench's own command line
(bottom dock); there the open project's path is filled in.
