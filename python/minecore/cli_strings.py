"""CLI commands for editing stored string sets (E-CD-01).

    minecore string list P.mcproj [SET]
    minecore string measure P.mcproj SET [--index 0] [--segments] [--csv out.csv]
    minecore string edit P.mcproj SET offset --distance 5 --slope 70 --out crest
    minecore string edit P.mcproj SET simplify --tolerance 0.2 --index 0 --out toe_s
    minecore string split P.mcproj SET --index 0 --vertex 12 --out parts
    minecore string join P.mcproj SET --index 0 --index 1 --out joined
    minecore string intersect P.mcproj SET_A SET_B [--csv out.csv]

``edit`` applies one operation to the chosen strings (all by default) and
saves the whole set, the other strings unchanged, under ``--out`` (which may
be the same name to replace the set).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, TextIO

from minecore import stringops as so
from minecore.project import Project
from minecore.strings import String, StringError, load_strings, save_strings, string_names

OPERATIONS = ("reverse", "simplify", "resample", "smooth", "offset", "trim", "close", "open")


def add_parsers(commands: Any) -> None:
    sp = commands.add_parser("string", help="edit and measure string (polyline) sets")
    cmds = sp.add_subparsers(dest="st_command", metavar="ACTION", required=True)
    ls = cmds.add_parser("list", help="string sets, or the strings of one set")
    ls.add_argument("project")
    ls.add_argument("set", nargs="?")
    me = cmds.add_parser("measure", help="lengths, area, bearings and gradients")
    me.add_argument("project")
    me.add_argument("set")
    me.add_argument("--index", type=int, action="append", help="string number (repeat)")
    me.add_argument("--segments", action="store_true", help="list every segment")
    me.add_argument("--csv", help="write the segment table to a CSV file")
    ed = cmds.add_parser("edit", help="apply one operation and save the set")
    ed.add_argument("project")
    ed.add_argument("set")
    ed.add_argument("operation", choices=OPERATIONS)
    ed.add_argument("--out", required=True, help="name to save the edited set under")
    ed.add_argument("--index", type=int, action="append", help="string number (repeat)")
    ed.add_argument("--tolerance", type=float, help="simplify: 3-D distance, m")
    ed.add_argument("--spacing", type=float, help="resample: point spacing, m")
    ed.add_argument("--iterations", type=int, default=1, help="smooth: passes")
    ed.add_argument(
        "--distance",
        type=float,
        help="offset: plan distance, m (+ outward for closed, left for open)",
    )
    ed.add_argument("--join", choices=("mitre", "round"), default="mitre")
    ed.add_argument("--dz", type=float, default=0.0, help="offset: vertical shift, m")
    ed.add_argument("--slope", type=float, help="offset: slope angle, degrees (+ up)")
    ed.add_argument("--keep", action="store_true", help="offset: keep the source strings too")
    ed.add_argument("--boundary", help="trim: string set to trim or extend to")
    ed.add_argument("--end", choices=("start", "end", "both"), default="end")
    ed.add_argument("--max-extension", type=float, default=math.inf, help="trim: m")
    sl = cmds.add_parser("split", help="cut a string at a vertex")
    sl.add_argument("project")
    sl.add_argument("set")
    sl.add_argument("--index", type=int, required=True)
    sl.add_argument("--vertex", type=int, required=True)
    sl.add_argument("--out", required=True)
    jo = cmds.add_parser("join", help="join two open strings at their nearest ends")
    jo.add_argument("project")
    jo.add_argument("set")
    jo.add_argument("--index", type=int, action="append", required=True, help="give twice")
    jo.add_argument("--tolerance", type=float, default=math.inf, help="largest gap, m")
    jo.add_argument("--out", required=True)
    it = cmds.add_parser("intersect", help="plan crossings between two sets")
    it.add_argument("project")
    it.add_argument("set_a")
    it.add_argument("set_b")
    it.add_argument("--csv", help="write the crossings to a CSV file")


def _pick(strings: list[String], index: list[int] | None) -> list[int]:
    chosen = index if index else list(range(len(strings)))
    for i in chosen:
        if not 0 <= i < len(strings):
            raise StringError(f"string {i} does not exist (the set has {len(strings)})")
    return chosen


def _list(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    if not args.set:
        for name in string_names(project):
            out.write(f"{name}: {len(load_strings(project, name))} strings\n")
        return
    out.write("index  points  closed  plan_m  slope_m  area_m2  z_min  z_max  layer\n")
    for i, s in enumerate(load_strings(project, args.set)):
        m = so.measure(s)
        out.write(
            f"{i}  {m.points}  {'yes' if m.closed else 'no'}  {m.plan_length:.2f}  "
            f"{m.slope_length:.2f}  {m.area:.2f}  {m.z_min:.2f}  {m.z_max:.2f}  {s.layer}\n"
        )


def _measure(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    strings = load_strings(project, args.set)
    rows = []
    for i in _pick(strings, args.index):
        m = so.measure(strings[i])
        out.write(
            f"string {i}: {m.points} points, plan {m.plan_length:.3f} m, slope "
            f"{m.slope_length:.3f} m"
            + (f", area {m.area:.3f} m2" if m.closed else "")
            + f", z {m.z_min:.2f}..{m.z_max:.2f}\n"
        )
        for g in so.segment_table(strings[i]):
            rows.append((i, g))
            if args.segments:
                out.write(
                    f"  {g.start}: plan {g.plan_length:.3f} m, bearing {g.bearing:.2f}, "
                    f"gradient {100 * g.gradient:.2f}%, dip {g.dip:.2f}\n"
                )
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["string", "segment", "plan_m", "slope_m", "bearing_deg", "gradient", "dip_deg"]
            )
            for i, g in rows:
                w.writerow(
                    [
                        i,
                        g.start,
                        f"{g.plan_length:.4f}",
                        f"{g.slope_length:.4f}",
                        f"{g.bearing:.4f}",
                        f"{g.gradient:.6f}",
                        f"{g.dip:.4f}",
                    ]
                )


def _need(value: float | None, flag: str, op: str) -> float:
    if value is None:
        raise StringError(f"{op} needs {flag}")
    return value


def _edit_one(  # noqa: PLR0911 - one return per operation
    s: String, args: argparse.Namespace, boundary: list[String]
) -> list[String]:
    op = args.operation
    if op == "reverse":
        return [so.reverse(s)]
    if op == "simplify":
        return [so.simplify(s, _need(args.tolerance, "--tolerance", op))]
    if op == "resample":
        return [so.resample(s, _need(args.spacing, "--spacing", op))]
    if op == "smooth":
        return [so.smooth(s, args.iterations)]
    if op in ("close", "open"):
        return [so.set_closed(s, op == "close")]
    if op == "offset":
        made = so.offset(
            s,
            _need(args.distance, "--distance", op),
            join=args.join,
            dz=args.dz,
            slope_deg=args.slope,
        )
        return ([s] if args.keep else []) + made
    # trim / extend to the first boundary string that the end reaches
    errors = []
    for b in boundary:
        try:
            return [so.trim_extend(s, b, args.end, args.max_extension)]
        except StringError as exc:
            errors.append(str(exc))
    raise StringError(f"no boundary string could be reached ({'; '.join(sorted(set(errors)))})")


def _edit(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    strings = load_strings(project, args.set)
    chosen = set(_pick(strings, args.index))
    boundary: list[String] = []
    if args.operation == "trim":
        if not args.boundary:
            raise StringError("trim needs --boundary")
        boundary = load_strings(project, args.boundary)
    result: list[String] = []
    for i, s in enumerate(strings):
        if i not in chosen:
            result.append(s)
            continue
        made = _edit_one(s, args, boundary)
        out.write(f"string {i}: {args.operation} -> {len(made)} string(s)\n")
        result += made
    if not result:
        raise StringError("the operation left no strings to save")
    save_strings(project, args.out, result)
    out.write(f"saved '{args.out}': {len(result)} strings\n")


def _split(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    strings = load_strings(project, args.set)
    i = _pick(strings, [args.index])[0]
    result = [*strings[:i], *so.split(strings[i], args.vertex), *strings[i + 1 :]]
    save_strings(project, args.out, result)
    out.write(f"saved '{args.out}': {len(result)} strings\n")


def _join(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    strings = load_strings(project, args.set)
    if len(args.index) != 2 or args.index[0] == args.index[1]:
        raise StringError("give --index twice, for two different strings")
    a, b = _pick(strings, args.index)
    joined = so.join(strings[a], strings[b], args.tolerance)
    result = [s for k, s in enumerate(strings) if k not in (a, b)] + [joined]
    save_strings(project, args.out, result)
    out.write(
        f"saved '{args.out}': {len(result)} strings (joined one has {len(joined.points)} points)\n"
    )


def _intersect(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    rows = []
    for i, a in enumerate(load_strings(project, args.set_a)):
        for j, b in enumerate(load_strings(project, args.set_b)):
            for c in so.intersections(a, b):
                rows.append((i, j, c))
    out.write(f"{len(rows)} crossings\n")
    for i, j, c in rows:
        out.write(f"{i} x {j}: {c.x:.3f} {c.y:.3f}  z {c.z_a:.3f} / {c.z_b:.3f}\n")
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["string_a", "string_b", "x", "y", "z_a", "z_b"])
            for i, j, c in rows:
                w.writerow([i, j, f"{c.x:.4f}", f"{c.y:.4f}", f"{c.z_a:.4f}", f"{c.z_b:.4f}"])


def string_command(args: argparse.Namespace, out: TextIO) -> int:
    handlers = {
        "list": _list,
        "measure": _measure,
        "edit": _edit,
        "split": _split,
        "join": _join,
        "intersect": _intersect,
    }
    with Project.open(args.project) as project:
        handlers[args.st_command](project, args, out)
    return 0


HANDLERS = {"string": string_command}
