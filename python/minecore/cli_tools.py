"""CLI commands for workbench tools that had no headless entry point.

Every menu entry of the desktop workbench must be reachable from the
command line, so scripts and automation (including AI agents) can drive
MineCore without clicking. This module adds: single drillhole file import,
table / block model / solid export, sample statistics and top-cut,
cross-validation, grade shell solid, drill-and-blast design, dump design and
the installation self-test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from minecore import solids, surfaces
from minecore.blockmodel import BlockModel
from minecore.blockmodel.csvexport import export_block_model_csv
from minecore.blockmodel.validation import cross_validate
from minecore.blockmodel.workflow import interval_samples, load_variograms
from minecore.cli_modelling import METHODS, _db, _source
from minecore.drillblast import BlastDesign, annual, layout
from minecore.drillholes import Role
from minecore.drillholes.importer import ImportOptions, import_table, import_workbook
from minecore.estimation import Ellipsoid, Search
from minecore.estimation.statistics import cell_decluster, summary, top_cut_effect
from minecore.io.dxf import Polyline, write_dxf
from minecore.pitopt.benchdesign import BenchGeometry
from minecore.pitopt.dump import capacity_t, design_dump
from minecore.project import Project
from minecore.strings import String, load_strings, save_strings


def _json(out: TextIO, data: object) -> None:
    json.dump(data, out, indent=2, ensure_ascii=False, default=float)
    out.write("\n")


def add_parsers(commands: Any) -> None:  # noqa: PLR0915 - one argument per line
    imp = commands.add_parser("import", help="import single files")
    imp_cmds = imp.add_subparsers(dest="import_command", metavar="ACTION", required=True)
    f = imp_cmds.add_parser("drillhole-file", help="one CSV/TXT table or an Excel workbook")
    f.add_argument("project")
    f.add_argument("file")
    f.add_argument(
        "--role",
        choices=[r.value for r in Role],
        default="interval",
        help="table type (ignored for .xlsx: sheet names decide)",
    )
    f.add_argument("--name", help="table name (default: file name)")
    f.add_argument("--dip-positive-down", action="store_true", help="dips written as 60 = down")

    exp = commands.add_parser("export", help="write project objects to files")
    exp_cmds = exp.add_subparsers(dest="export_command", metavar="ACTION", required=True)
    t = exp_cmds.add_parser("table", help="drillhole table to CSV (; separated)")
    t.add_argument("project")
    t.add_argument("table")
    t.add_argument("out")
    b = exp_cmds.add_parser("block-model", help="block model to CSV (centroids, attributes)")
    b.add_argument("project")
    b.add_argument("model")
    b.add_argument("out")
    s = exp_cmds.add_parser("solid", help="solid to DXF 3DFACE")
    s.add_argument("project")
    s.add_argument("solid")
    s.add_argument("out")

    st = commands.add_parser("stats", help="sample statistics, top-cut effect, declustering")
    st.add_argument("project")
    st.add_argument("source", help="table/field")
    st.add_argument("--top-cut", type=float, help="cap to test")
    st.add_argument(
        "--decluster",
        type=float,
        nargs=3,
        metavar=("MIN", "MAX", "N"),
        help="cell declustering: smallest and largest cell size (m) and count",
    )
    st.add_argument("--json", action="store_true")

    xv = commands.add_parser("xval", help="leave-one-hole-out cross-validation")
    xv.add_argument("project")
    xv.add_argument("source", help="table/field")
    xv.add_argument("--method", choices=["idw", "nn", "ok", "sk"], default="idw")
    xv.add_argument("--variogram", help="saved variogram (kriging)")
    xv.add_argument("--radius", type=float, default=100.0, help="isotropic search radius (m)")
    xv.add_argument("--max", type=int, default=16, help="most samples")
    xv.add_argument("--power", type=float, default=2.0)
    xv.add_argument("--json", action="store_true")

    sh = commands.add_parser("shell", help="grade shell solid from a block model")
    sh.add_argument("project")
    sh.add_argument("model")
    sh.add_argument("attribute")
    sh.add_argument("cutoff", type=float)
    sh.add_argument("--name", help="solid name (default kabuk_<attr>_<cutoff>)")

    bl = commands.add_parser("blast", help="drill-and-blast design; hole layout in an outline")
    bl.add_argument("project")
    bl.add_argument("--diameter", type=float, default=165.0, help="hole diameter (mm)")
    bl.add_argument("--bench", type=float, default=10.0, help="bench height (m)")
    bl.add_argument("--burden-factor", type=float, default=30.0, help="burden / diameter")
    bl.add_argument("--spacing-ratio", type=float, default=1.15)
    bl.add_argument("--subdrill-ratio", type=float, default=0.3)
    bl.add_argument("--stemming-ratio", type=float, default=0.7)
    bl.add_argument("--burden", type=float, help="burden (m), overrides the factor")
    bl.add_argument("--spacing", type=float, help="spacing (m), overrides the ratio")
    bl.add_argument("--explosive-density", type=float, default=0.85, help="g/cm3")
    bl.add_argument("--rock-density", type=float, default=2.7, help="t/m3")
    bl.add_argument("--angle", type=float, default=0.0, help="hole angle from vertical")
    bl.add_argument("--cost-explosive", type=float, default=1.0, help="$/kg")
    bl.add_argument("--cost-drilling", type=float, default=25.0, help="$/m")
    bl.add_argument("--cost-accessory", type=float, default=20.0, help="$/hole")
    bl.add_argument("--annual-mt", type=float, default=0.0, help="material blasted per year")
    bl.add_argument("--outline", help="string set with closed blast outlines: lay out holes")
    bl.add_argument("--azimuth", type=float, default=0.0, help="row direction (deg from N)")
    bl.add_argument("--square", action="store_true", help="square pattern, not staggered")
    bl.add_argument("--dxf", help="write the holes to this DXF")
    bl.add_argument("--json", action="store_true")

    du = commands.add_parser("dump", help="waste dump design inside a closed outline")
    du.add_argument("project")
    du.add_argument("outline", help="string set; its first closed string is the footprint")
    du.add_argument("--ground", help="ground surface (default: flat at --ground-z)")
    du.add_argument("--ground-z", type=float, default=0.0)
    du.add_argument("--lift", type=float, default=10.0, help="lift height (m)")
    du.add_argument("--face", type=float, default=37.0, help="face angle (deg)")
    du.add_argument("--berm", type=float, default=15.0)
    du.add_argument("--max-height", type=float, help="m above the lowest toe")
    du.add_argument("--swell", type=float, default=1.30)
    du.add_argument("--density", type=float, default=2.6, help="in-situ t/m3")
    du.add_argument("--need-mt", type=float, default=0.0, help="waste to place (Mt)")
    du.add_argument("--resolution", type=float, default=5.0)
    du.add_argument("--json", action="store_true")

    ct = commands.add_parser(
        "contents", help="what a project holds: tables and fields, models and attributes, objects"
    )
    ct.add_argument("project")
    ct.add_argument("--json", action="store_true")

    sf = commands.add_parser("selftest", help="whole chain on the bundled sample; writes a report")
    sf.add_argument("report", nargs="?", default="minecore-selftest.txt")


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def import_command(args: argparse.Namespace, out: TextIO) -> int:
    path = Path(args.file)
    with Project.open(args.project) as project:
        db = _db(project)
        if path.suffix.lower() in (".xlsx", ".xlsm"):
            for r in import_workbook(db, path):
                out.write(f"{r.table.name}: {r.table.role}, {r.table.rows} rows\n")
            return 0
        dip = "positive_down" if args.dip_positive_down else "negative_down"
        name = args.name or path.stem
        r = import_table(db, name, Role(args.role), path, ImportOptions(dip_convention=dip))
        out.write(f"{r.table.name}: {r.table.role}, {r.table.rows} rows\n")
    return 0


def export_command(args: argparse.Namespace, out: TextIO) -> int:
    target = Path(args.out)
    with Project.open(args.project, read_only=True) as project:
        if args.export_command == "table":
            table = _db(project).table(args.table)
            cols = [c for c in table.column_names if not c.endswith("__flag")]
            data = {c: table.column(c).to_pylist() for c in cols}
            with target.open("w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh, delimiter=";")
                writer.writerow(cols)
                for i in range(table.num_rows):
                    writer.writerow([_fmt(data[c][i]) for c in cols])
            out.write(f"{table.num_rows:,} rows written to {target}\n")
        elif args.export_command == "block-model":
            rows = export_block_model_csv(target, BlockModel.open(project, args.model))
            out.write(f"{rows:,} blocks written to {target}\n")
        else:
            solid = solids.load_solid(project, args.solid)
            count = write_dxf(target, faces=[(args.solid, solid.vertices[solid.triangles])])
            out.write(f"{count:,} triangles written to {target}\n")
    return 0


def stats_command(args: argparse.Namespace, out: TextIO) -> int:
    with Project.open(args.project, read_only=True) as project:
        samples = interval_samples(_db(project), *_source(args.source))
    s = summary(samples.values)
    result: dict[str, Any] = {
        "n": s.n,
        "mean": s.mean,
        "std": s.std,
        "cv": s.cv,
        "min": s.minimum,
        "median": s.median,
        "max": s.maximum,
        "percentiles": {f"P{p:g}": v for p, v in s.percentiles.items()},
    }
    if args.top_cut is not None:
        e = top_cut_effect(samples.values, args.top_cut)
        result["top_cut"] = {
            "cap": e.cap,
            "capped": e.capped,
            "mean_before": e.mean_before,
            "mean_after": e.mean_after,
            "metal_removed": e.metal_removed,
            "cv_after": e.cv_after,
        }
    if args.decluster is not None:
        lo, hi, n = args.decluster
        d = cell_decluster(samples.xyz, samples.values, np.linspace(lo, hi, int(n)))
        best = float(d.means[list(d.cell_sizes).index(d.best_size)])
        result["decluster"] = {
            "naive_mean": float(np.nanmean(samples.values)),
            "best_cell": float(d.best_size),
            "declustered_mean": best,
            "by_cell": [[float(c), float(m)] for c, m in zip(d.cell_sizes, d.means, strict=True)],
        }
    if args.json:
        _json(out, result)
        return 0
    out.write(
        f"n {s.n:,}  mean {s.mean:.4g}  std {s.std:.4g}  CV {s.cv:.2f}\n"
        f"min {s.minimum:.4g}  median {s.median:.4g}  max {s.maximum:.4g}\n"
        + "  ".join(f"P{p:g} {v:.4g}" for p, v in s.percentiles.items())
        + "\n"
    )
    if "top_cut" in result:
        t = result["top_cut"]
        out.write(
            f"top-cut {t['cap']:g}: {t['capped']} samples capped; mean {t['mean_before']:.4g}"
            f" -> {t['mean_after']:.4g}; metal removed {100 * t['metal_removed']:.1f}%;"
            f" CV after {t['cv_after']:.2f}\n"
        )
    if "decluster" in result:
        d2 = result["decluster"]
        out.write(
            f"declustered mean {d2['declustered_mean']:.4g} with {d2['best_cell']:.1f} m cells"
            f" (naive {d2['naive_mean']:.4g})\n"
        )
    return 0


def xval_command(args: argparse.Namespace, out: TextIO) -> int:
    method = METHODS[args.method]
    with Project.open(args.project, read_only=True) as project:
        samples = interval_samples(_db(project), *_source(args.source))
        variogram = load_variograms(project).get(args.variogram) if args.variogram else None
    if method.is_kriging and variogram is None:
        raise SystemExit("kriging needs --variogram (a saved variogram)")
    search = Search(Ellipsoid.isotropic(args.radius), 1, args.max)
    cv = cross_validate(
        samples, method=method, search=search, variogram=variogram, power=args.power
    )
    result = {
        "n": cv.n,
        "unestimated": cv.unestimated,
        "mean_true": cv.mean_true,
        "mean_estimate": cv.mean_estimate,
        "mean_error": cv.mean_error,
        "mse": cv.mse,
        "correlation": cv.correlation,
        "slope": cv.slope,
    }
    if args.json:
        _json(out, result)
    else:
        for key, value in result.items():
            out.write(f"{key:14}: {_fmt(value)}\n")
    return 0


def shell_command(args: argparse.Namespace, out: TextIO) -> int:
    with Project.open(args.project) as project:
        model = BlockModel.open(project, args.model)
        solid = solids.grade_shell(model, args.attribute, args.cutoff)
        name = args.name or f"kabuk_{args.attribute}_{args.cutoff:g}"
        solids.save_solid(project, name, solid)
    out.write(f"{name}: {len(solid.triangles):,} triangles, volume {solid.volume():,.0f} m3\n")
    return 0


def blast_command(args: argparse.Namespace, out: TextIO) -> int:
    d = BlastDesign(
        diameter_mm=args.diameter,
        bench_height=args.bench,
        burden_factor=args.burden_factor,
        spacing_ratio=args.spacing_ratio,
        subdrill_ratio=args.subdrill_ratio,
        stemming_ratio=args.stemming_ratio,
        explosive_density=args.explosive_density,
        rock_density=args.rock_density,
        hole_angle=args.angle,
        burden=args.burden,
        spacing=args.spacing,
        explosive_cost_kg=args.cost_explosive,
        drilling_cost_m=args.cost_drilling,
        accessory_cost_hole=args.cost_accessory,
    )
    result: dict[str, Any] = {
        "burden": d.b,
        "spacing": d.s,
        "subdrill": d.subdrill,
        "stemming": d.stemming,
        "hole_length": d.hole_length,
        "charge_length": d.charge_length,
        "linear_charge_kg_m": d.linear_charge,
        "charge_kg": d.charge_kg,
        "volume_per_hole": d.volume_per_hole,
        "tonnes_per_hole": d.tonnes_per_hole,
        "powder_factor_kg_m3": d.powder_factor,
        "cost_per_hole": d.cost_per_hole,
        "cost_per_t": d.cost_per_t,
    }
    if args.annual_mt > 0:
        year = annual(d, args.annual_mt * 1e6)
        result["annual"] = {
            "holes": year.holes,
            "drill_m": year.drill_m,
            "explosive_t": year.explosive_t,
            "cost": year.cost,
        }
    if args.outline:
        with Project.open(args.project) as project:
            rings = [s for s in load_strings(project, args.outline) if s.closed]
            if not rings:
                raise SystemExit(f"'{args.outline}' has no closed string")
            parts = [
                layout(d, r.points, azimuth=args.azimuth, staggered=not args.square) for r in rings
            ]
            collars = np.concatenate([c for c, _ in parts])
            toes = np.concatenate([t for _, t in parts])
            holes = [
                String(np.vstack([c, t]), False, "DELIK")
                for c, t in zip(collars, toes, strict=True)
            ]
            name = f"{args.outline}_delikler"
            save_strings(project, name, holes)
        if args.dxf:
            write_dxf(Path(args.dxf), [Polyline("BLAST_HOLES", s.points, False) for s in holes])
        result["layout"] = {
            "holes": len(collars),
            "drill_m": len(collars) * d.hole_length,
            "explosive_t": len(collars) * d.charge_kg / 1000,
            "volume_m3": len(collars) * d.volume_per_hole,
            "string_set": name,
            "dxf": args.dxf,
        }
    if args.json:
        _json(out, result)
        return 0
    out.write(
        f"burden {d.b:.2f} m, spacing {d.s:.2f} m, subdrill {d.subdrill:.2f} m, "
        f"stemming {d.stemming:.2f} m\n"
        f"hole {d.hole_length:.2f} m, charge {d.charge_length:.2f} m, "
        f"{d.linear_charge:.1f} kg/m, {d.charge_kg:,.0f} kg/hole\n"
        f"{d.volume_per_hole:,.0f} m3 ({d.tonnes_per_hole:,.0f} t) per hole, powder factor "
        f"{d.powder_factor:.3f} kg/m3\n"
        f"cost {d.cost_per_hole:,.0f} $/hole, {d.cost_per_t:.3f} $/t\n"
    )
    if "annual" in result:
        a = result["annual"]
        out.write(
            f"per year: {a['holes']:,} holes, {a['drill_m']:,.0f} m drilled, "
            f"{a['explosive_t']:,.0f} t explosive, {a['cost'] / 1e6:,.2f} M$\n"
        )
    if "layout" in result:
        lay = result["layout"]
        out.write(
            f"layout: {lay['holes']} holes, {lay['drill_m']:,.0f} m, "
            f"{lay['explosive_t']:,.1f} t explosive, about {lay['volume_m3']:,.0f} m3; "
            f"saved as string set '{lay['string_set']}'"
            + (f", DXF {lay['dxf']}" if lay["dxf"] else "")
            + "\n"
        )
    return 0


def dump_command(args: argparse.Namespace, out: TextIO) -> int:
    with Project.open(args.project) as project:
        rings = [s for s in load_strings(project, args.outline) if s.closed]
        if not rings:
            raise SystemExit(f"'{args.outline}' has no closed string")
        ground = surfaces.load_surface(project, args.ground) if args.ground else None
        d = design_dump(
            rings[0].points,
            BenchGeometry(args.face, args.lift, args.berm),
            ground=ground,
            ground_z=args.ground_z,
            max_height=args.max_height,
            resolution=args.resolution,
        )
        name = f"dokum_{args.outline}"
        surfaces.save_surface(project, name, d.surface())
    cap = capacity_t(d, args.density, args.swell)
    lifts = []
    cumulative = 0.0
    for level, vol in d.volume_by_lift():
        cumulative += vol / args.swell * args.density
        lifts.append({"toe": level, "loose_m3": vol, "cumulative_t": cumulative})
    result: dict[str, Any] = {
        "loose_volume_m3": d.volume(),
        "area_ha": d.area() / 1e4,
        "top": d.top,
        "capacity_t": cap,
        "surface": name,
        "lifts": lifts,
    }
    need = args.need_mt * 1e6
    if need > 0:
        result["fits"] = cap >= need
        result["shortfall_t"] = max(0.0, need - cap)
    if args.json:
        _json(out, result)
        return 0
    out.write(
        f"loose volume {d.volume() / 1e6:,.2f} Mm3, area {d.area() / 1e4:,.1f} ha, "
        f"top {d.top:,.1f} m, {len(lifts)} lifts, capacity {cap / 1e6:,.2f} Mt in situ\n"
    )
    for row in lifts:
        out.write(
            f"  toe {row['toe']:10,.1f}  {row['loose_m3'] / 1e6:10,.3f} Mm3"
            f"  {row['cumulative_t'] / 1e6:10,.2f} Mt\n"
        )
    if need > 0:
        out.write("fits\n" if cap >= need else f"short by {(need - cap) / 1e6:,.2f} Mt\n")
    out.write(f"surface saved as '{name}'\n")
    return 0


def contents_command(args: argparse.Namespace, out: TextIO) -> int:
    from minecore.blockmodel.model import KIND as BLOCK_MODEL_KIND  # noqa: PLC0415

    with Project.open(args.project, read_only=True) as project:
        tables: list[dict[str, Any]] = []
        if project.objects("drillhole_db"):
            for info in _db(project).tables().values():
                tables.append(
                    {
                        "name": info.name,
                        "role": str(info.role),
                        "rows": info.rows,
                        "fields": [
                            {"name": f.name, "type": str(f.type), "unit": f.unit}
                            for f in info.fields
                        ],
                    }
                )
        models: list[dict[str, Any]] = []
        for rec in project.objects(BLOCK_MODEL_KIND):
            model = BlockModel.open(project, rec.name)
            models.append(
                {
                    "name": rec.name,
                    "blocks": model.grid.blocks,
                    "size": list(model.grid.size),
                    "attributes": list(model.attributes),
                }
            )
        others = [
            {"kind": r.kind, "name": r.name}
            for r in project.objects()
            if r.kind not in (BLOCK_MODEL_KIND, "drillhole_db")
        ]
    if args.json:
        _json(out, {"tables": tables, "block_models": models, "objects": others})
        return 0
    for t in tables:
        fields = ", ".join(f["name"] for f in t["fields"])
        out.write(f"table {t['name']} ({t['role']}, {t['rows']:,} rows): {fields}\n")
    for m in models:
        out.write(
            f"block model {m['name']} ({m['blocks']:,} blocks, "
            f"{' x '.join(f'{v:g}' for v in m['size'])} m): {', '.join(m['attributes'])}\n"
        )
    for o in others:
        out.write(f"{o['kind']} {o['name']}\n")
    return 0


def selftest_command(args: argparse.Namespace, out: TextIO) -> int:
    from minecore.selftest import run  # noqa: PLC0415 - runs the whole chain

    code = run(args.report)
    out.write(Path(args.report).read_text(encoding="utf-8"))
    return code


HANDLERS = {
    "import": import_command,
    "export": export_command,
    "stats": stats_command,
    "xval": xval_command,
    "shell": shell_command,
    "blast": blast_command,
    "dump": dump_command,
    "contents": contents_command,
    "selftest": selftest_command,
}
