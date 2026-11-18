"""CLI commands for mine survey: stockpiles (OC08) and month-end progress (OC09).

    minecore survey stockpile P.mcproj PILE_SURFACE OUTLINE_SET --base plane \\
        --density 1.8 --density-date 2026-11-02 --name ROM-1 --date 2026-11-30 --save
    minecore survey stockpile-list P.mcproj [--name ROM-1]
    minecore survey month-end P.mcproj START END BM --density rho \\
        --grade au --class cevher:0.5 --class "düşük tenör:0.3" --below pasa
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, TextIO

from minecore import strings, surfaces
from minecore.blockmodel import BlockModel
from minecore.project import Project
from minecore.survey import month_end as me
from minecore.survey import stockpiles as sp


def add_parsers(commands: Any) -> None:
    sv = commands.add_parser("survey", help="stockpile volumes and month-end progress")
    cmds = sv.add_subparsers(dest="sv_command", metavar="ACTION", required=True)
    st = cmds.add_parser("stockpile", help="pile volume and tonnes inside a toe outline")
    st.add_argument("project")
    st.add_argument("pile", help="surveyed pile surface")
    st.add_argument("outline", help="string set; its first closed string is the toe outline")
    st.add_argument("--base", choices=sp.BASE_METHODS, required=True, help="base under the pile")
    st.add_argument("--base-surface", help="ground surveyed before the pile (--base surface)")
    st.add_argument("--level", type=float, help="base elevation (--base level)")
    st.add_argument("--spacing", type=float, help="outline point spacing for boundary/plane, m")
    st.add_argument("--cell", type=float, help="grid method cell, m (default: exact triangles)")
    st.add_argument("--density", type=float, help="pile density, t/m3 (measured)")
    st.add_argument("--density-date", default="", help="date the density was measured")
    st.add_argument("--density-source", default="", help="how the density was measured")
    st.add_argument("--name", default="", help="pile name")
    st.add_argument("--material", default="")
    st.add_argument("--date", default="", help="survey date, YYYY-MM-DD")
    st.add_argument("--save", action="store_true", help="store the result in the project")
    ls = cmds.add_parser("stockpile-list", help="stored pile surveys")
    ls.add_argument("project")
    ls.add_argument("--name", help="only this pile")
    ls.add_argument("--csv", help="write the list to a CSV file")
    mo = cmds.add_parser("month-end", help="excavation between two surveys by material")
    mo.add_argument("project")
    mo.add_argument("start", help="surface at the start of the period")
    mo.add_argument("end", help="surface at the end of the period")
    mo.add_argument("model", help="block model")
    mo.add_argument("--density", required=True, help="t/m3 or a density attribute")
    mo.add_argument("--grade", help="grade attribute")
    mo.add_argument(
        "--class",
        dest="classes",
        action="append",
        default=[],
        help="NAME:MIN_GRADE material limit (repeat); needs --grade",
    )
    mo.add_argument("--below", default="pasa", help="material under the lowest limit")
    mo.add_argument("--material", help="material category attribute instead of grade limits")
    mo.add_argument("--phase", help="attribute to split by (pushback, stage)")
    mo.add_argument("--bench-height", type=float, help="bench height, m (default block height)")
    mo.add_argument("--samples", type=int, default=5, help="vertical lines per block side")
    mo.add_argument("--truck-loads", type=int, help="truck loads counted in the period")
    mo.add_argument("--load-factor", type=float, help="t per truck load used for the count")
    mo.add_argument("--csv", help="write the table to a CSV file")


def _outline(project: Project, name: str) -> Any:
    for s in strings.load_strings(project, name):
        if s.closed:
            return s.points
    raise SystemExit(f"string set {name!r} has no closed string")


def _stockpile(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    pile = surfaces.load_surface(project, args.pile)
    base = surfaces.load_surface(project, args.base_surface) if args.base_surface else None
    outline = _outline(project, args.outline)
    if args.base in ("plane", "boundary") and not _is_3d(outline):
        outline = outline[:, :2]  # z from the pile surface
    density = (
        sp.Density(args.density, args.density_date, args.density_source)
        if args.density is not None
        else None
    )
    r = sp.stockpile_volume(
        pile,
        outline,
        base_method=args.base,
        base=base,
        level=args.level,
        spacing=args.spacing,
        cell=args.cell,
        density=density,
        name=args.name,
        material=args.material,
        survey_date=args.date,
    )
    line = (
        f"{r.name or 'pile'}: {r.volume:,.1f} m3 above the {r.base_method} base "
        f"({r.volume_method}); below base {r.below_base:,.1f} m3; "
        f"outline {r.outline_area:,.1f} m2, covered {100 * r.coverage:.2f}%"
    )
    if math.isfinite(r.base_rms):
        line += f"; plane fit RMS {r.base_rms:.3f} m"
    out.write(line + "\n")
    if r.tonnes is not None:
        out.write(f"tonnes {r.tonnes:,.0f} at {r.density:g} t/m3\n")
    if r.coverage < 0.999:
        out.write("warning: the surfaces do not cover the whole outline\n")
    change = sp.change_since_previous(project, r) if r.name and r.survey_date else None
    if change is not None:
        dt = "" if change.tonnes is None else f", {change.tonnes:+,.0f} t"
        out.write(f"change since {change.previous_date}: {change.volume:+,.1f} m3{dt}\n")
    if args.save:
        sp.save_survey(project, r)
        out.write(f"stored survey {r.name} {r.survey_date}\n")


def _is_3d(points: Any) -> bool:
    # Strings drawn in plan carry z = 0 everywhere; treat them as 2D outlines.
    return bool(points.shape[1] == 3 and (points[:, 2] != 0).any())


def _stockpile_list(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    rows = sp.surveys(project, args.name)
    header = ["pile", "date", "material", "volume_m3", "density_t_m3", "tonnes", "base", "method"]
    table = [
        [
            r.name,
            r.survey_date,
            r.material,
            f"{r.volume:.1f}",
            "" if r.density is None else f"{r.density:g}",
            "" if r.tonnes is None else f"{r.tonnes:.0f}",
            r.base_method,
            r.volume_method,
        ]
        for r in rows
    ]
    for t in table:
        out.write("  ".join(t) + "\n")
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(table)


def _classes(specs: list[str]) -> list[me.GradeClass]:
    out = []
    for spec in specs:
        name, sep, value = spec.rpartition(":")
        try:
            out.append(me.GradeClass(name.strip(), float(value)))
        except ValueError:
            raise SystemExit(f"--class expects NAME:MIN_GRADE, got {spec!r}") from None
        if not sep or not name.strip():
            raise SystemExit(f"--class expects NAME:MIN_GRADE, got {spec!r}")
    return out


def _density(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def _month_end(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    r = me.month_end(
        BlockModel.open(project, args.model),
        surfaces.load_surface(project, args.start),
        surfaces.load_surface(project, args.end),
        density=_density(args.density),
        grade=args.grade,
        classes=_classes(args.classes),
        below=args.below,
        material=args.material,
        phase=args.phase,
        bench_height=args.bench_height,
        samples=args.samples,
    )
    out.write(
        f"excavated {r.excavated:,.1f} m3 (exact), in the block model {r.model_volume:,.1f} m3 "
        f"({100 * r.model_coverage:.1f}%); filled {r.filled:,.1f} m3\n"
    )
    out.write("phase  bench  material  volume_m3  tonnes  grade\n")
    for row in r.rows:
        out.write(
            f"{row.phase or '-'}  {row.bench:g}  {row.material}  {row.volume:,.0f}  "
            f"{row.tonnes:,.0f}  {row.grade:.3f}\n"
        )
    for m, (v, t, g) in r.by_material().items():
        out.write(f"total {m}: {v:,.0f} m3, {t:,.0f} t, grade {g:.3f}\n")
    out.write(f"total {r.tonnes:,.0f} t\n")
    if (args.truck_loads is None) != (args.load_factor is None):
        raise SystemExit("give both --truck-loads and --load-factor")
    if args.truck_loads is not None:
        tr = me.truck_reconciliation(r.tonnes, args.truck_loads, args.load_factor)
        out.write(
            f"trucks {tr.loads} x {tr.load_factor:g} t = {tr.truck_tonnes:,.0f} t; "
            f"survey/truck {tr.ratio:.3f}; load factor matching the survey "
            f"{tr.suggested_load_factor:.2f} t/load\n"
        )
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["phase", "bench_m", "material", "volume_m3", "tonnes", "grade", "metal"])
            for row in r.rows:
                w.writerow(
                    [
                        row.phase,
                        f"{row.bench:g}",
                        row.material,
                        f"{row.volume:.1f}",
                        f"{row.tonnes:.1f}",
                        "" if math.isnan(row.grade) else f"{row.grade:.4f}",
                        f"{row.metal:.3f}",
                    ]
                )


def survey_command(args: argparse.Namespace, out: TextIO) -> int:
    with Project.open(args.project) as project:
        if args.sv_command == "stockpile":
            _stockpile(project, args, out)
        elif args.sv_command == "stockpile-list":
            _stockpile_list(project, args, out)
        else:
            _month_end(project, args, out)
    return 0


HANDLERS = {"survey": survey_command}
