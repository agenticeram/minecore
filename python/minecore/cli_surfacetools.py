"""CLI for surface analysis tools (OC10, E-CD-02), under ``minecore surface``.

    minecore surface contours P.mcproj TOPO --interval 5 --index-every 5 \\
        --save kontur_5m --dxf kontur.dxf --label-spacing 200
    minecore surface slope P.mcproj TOPO --classes 0 10 20 30 45 90 \\
        --csv egim.csv --cell 5 --asc-slope egim.asc --asc-aspect baki.asc
    minecore surface profile P.mcproj HAT TOPO PIT --csv profil.csv \\
        --plot profil.png --exaggeration 2
    minecore surface intersect P.mcproj PIT TOPO --save tepe_cizgisi
    minecore surface offset P.mcproj TOPO --dz -0.5 --save topo_eksi50cm
    minecore surface merge P.mcproj PIT TOPO --mode lower --save nihai
    minecore surface normal-offset P.mcproj PIT --distance 2 --save astar
    minecore surface smooth P.mcproj TOPO --passes 3 --max-change 0.2 --save topo_duz
    minecore surface decimate P.mcproj TOPO --tolerance 0.05 --save topo_seyrek
    minecore surface from-grid P.mcproj lidar.asc --save LIDAR --tolerance 0.1
    minecore surface to-grid P.mcproj TOPO --cell 2 --asc topo.asc
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from minecore import strings, surfaces
from minecore import surfaceedit as se
from minecore import surfacetools as st
from minecore.io.dxf import Polyline, Text, write_dxf
from minecore.project import Project

ACTIONS = (
    "contours",
    "slope",
    "profile",
    "intersect",
    "offset",
    "merge",
    "normal-offset",
    "smooth",
    "decimate",
    "from-grid",
    "to-grid",
)


def add_parsers(sf_cmds: Any) -> None:  # noqa: PLR0915 - one parser per action
    co = sf_cmds.add_parser("contours", help="contour lines of a surface (exact on triangles)")
    co.add_argument("project")
    co.add_argument("surface")
    co.add_argument("--interval", type=float, help="contour interval, m")
    co.add_argument("--base", type=float, default=0.0, help="level the interval counts from")
    co.add_argument("--levels", type=float, nargs="+", help="explicit levels instead")
    co.add_argument("--index-every", type=int, default=5, help="every n-th is an index contour")
    co.add_argument("--save", help="store the contours as a string set")
    co.add_argument("--dxf", help="write contours (and labels) to an ASCII DXF")
    co.add_argument("--label-spacing", type=float, help="label index contours every N m (DXF)")
    co.add_argument("--label-height", type=float, default=2.5, help="label text height, m")
    sl = sf_cmds.add_parser("slope", help="slope and aspect map, area by class")
    sl.add_argument("project")
    sl.add_argument("surface")
    sl.add_argument(
        "--classes", type=float, nargs="+", required=True, help="slope class limits, degrees"
    )
    sl.add_argument("--sectors", type=int, default=8, help="aspect sectors (divide 360)")
    sl.add_argument("--csv", help="per-triangle slope/aspect table")
    sl.add_argument("--cell", type=float, help="grid cell for raster output, m")
    sl.add_argument("--asc-slope", help="ESRI ASCII grid of slope, degrees (needs --cell)")
    sl.add_argument("--asc-aspect", help="ESRI ASCII grid of aspect, degrees (needs --cell)")
    sl.add_argument("--asc-z", help="ESRI ASCII grid of elevation (needs --cell)")
    pr = sf_cmds.add_parser("profile", help="profile of surfaces along a line")
    pr.add_argument("project")
    pr.add_argument("line", help="string set; its first string is the profile line")
    pr.add_argument("surfaces", nargs="+", help="surfaces to trace")
    pr.add_argument("--csv", help="write chainage, x, y and one z column per surface")
    pr.add_argument("--plot", help="profile drawing, .png/.pdf/.svg")
    pr.add_argument("--exaggeration", type=float, default=1.0, help="vertical exaggeration")
    it = sf_cmds.add_parser("intersect", help="lines where two surfaces meet (daylight line)")
    it.add_argument("project")
    it.add_argument("a", help="surface a (above on the left of each line)")
    it.add_argument("b", help="surface b")
    it.add_argument("--tolerance", type=float, default=0.001, help="end joining distance, m")
    it.add_argument("--save", help="store the lines as a string set")
    it.add_argument("--dxf", help="write the lines to an ASCII DXF")
    of = sf_cmds.add_parser("offset", help="move a surface vertically")
    of.add_argument("project")
    of.add_argument("surface")
    of.add_argument("--dz", type=float, required=True, help="vertical shift, m (+ up)")
    of.add_argument("--save", required=True, help="name of the new surface")
    me = sf_cmds.add_parser("merge", help="upper/lower envelope of two surfaces, or patch")
    me.add_argument("project")
    me.add_argument("a", help="surface a (the patch in --mode patch)")
    me.add_argument("b", help="surface b")
    me.add_argument(
        "--mode",
        choices=se.MODES,
        required=True,
        help="upper: higher of the two; lower: lower (pit into topography); patch: a over b",
    )
    me.add_argument("--tolerance", type=float, default=0.001, help="vertex merge distance, m")
    me.add_argument("--save", required=True, help="name of the new surface")
    no = sf_cmds.add_parser("normal-offset", help="move a surface along its normals")
    no.add_argument("project")
    no.add_argument("surface")
    no.add_argument("--distance", type=float, required=True, help="m along the normal (+ up)")
    no.add_argument("--max-factor", type=float, default=3.0, help="vertex move limit / distance")
    no.add_argument("--save", required=True, help="name of the new surface")
    sm = sf_cmds.add_parser("smooth", help="smooth elevations (planes stay planar)")
    sm.add_argument("project")
    sm.add_argument("surface")
    sm.add_argument("--passes", type=int, default=1, help="smoothing passes")
    sm.add_argument("--factor", type=float, default=0.5, help="share moved per pass, 0-1")
    sm.add_argument("--max-change", type=float, help="cap on any vertex change, m")
    sm.add_argument("--move-boundary", action="store_true", help="also smooth boundary vertices")
    sm.add_argument("--save", required=True, help="name of the new surface")
    de = sf_cmds.add_parser("decimate", help="fewer points within a vertical tolerance")
    de.add_argument("project")
    de.add_argument("surface")
    de.add_argument("--tolerance", type=float, required=True, help="vertical tolerance, m")
    de.add_argument("--save", required=True, help="name of the new surface")
    fg = sf_cmds.add_parser("from-grid", help="surface from an ESRI ASCII grid (.asc)")
    fg.add_argument("project")
    fg.add_argument("grid", help=".asc file")
    fg.add_argument("--tolerance", type=float, help="decimate to this vertical tolerance, m")
    fg.add_argument("--save", required=True, help="name of the new surface")
    tg = sf_cmds.add_parser("to-grid", help="surface elevations to an ESRI ASCII grid")
    tg.add_argument("project")
    tg.add_argument("surface")
    tg.add_argument("--cell", type=float, required=True, help="cell size, m")
    tg.add_argument("--asc", required=True, help="output .asc file")


def contour_strings(cs: list[st.Contour]) -> list[strings.String]:
    return [strings.String(c.points, c.closed, "KONTUR_ANA" if c.index else "KONTUR") for c in cs]


def contour_dxf(path: Path, cs: list[st.Contour], spacing: float | None, height: float) -> int:
    lines = [Polyline("KONTUR_ANA" if c.index else "KONTUR", c.points, c.closed) for c in cs]
    texts = []
    if spacing:
        for c in cs:
            if c.index:
                texts += [
                    Text("KONTUR_ETIKET", xyz, f"{c.level:g}", height, angle)
                    for xyz, angle in st.label_points(c, spacing)
                ]
    return write_dxf(path, lines, texts=texts)


def _contours(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    s = surfaces.load_surface(project, args.surface)
    cs = st.contours(
        s, args.interval, base=args.base, index_every=args.index_every, levels=args.levels
    )
    if not cs:
        raise SystemExit("no contour level crosses the surface")
    total = sum(c.length for c in cs)
    levels = sorted({c.level for c in cs})
    out.write(
        f"{len(cs)} contour lines on {len(levels)} levels ({levels[0]:g} to {levels[-1]:g} m), "
        f"{sum(c.closed for c in cs)} closed, {sum(c.index for c in cs)} index; "
        f"total length {total:,.1f} m\n"
    )
    if args.save:
        strings.save_strings(project, args.save, contour_strings(cs))
        out.write(f"string set {args.save}: {len(cs)} contours\n")
    if args.dxf:
        n = contour_dxf(Path(args.dxf), cs, args.label_spacing, args.label_height)
        out.write(f"{args.dxf}: {n} entities\n")


def _slope(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    s = surfaces.load_surface(project, args.surface)
    sm = st.slope_map(s)
    a = sm.plan_area
    mean = float((sm.slope * a).sum() / a.sum())
    out.write(
        f"{len(a)} triangles, plan area {a.sum():,.1f} m2, true area {sm.true_area.sum():,.1f} "
        f"m2; slope mean {mean:.2f} deg (area weighted), max {sm.slope.max():.2f} deg\n"
    )
    out.write("slope class  plan_m2  true_m2  share\n")
    for r in st.slope_classes(sm, args.classes):
        out.write(f"{r.label}  {r.plan_area:,.1f}  {r.true_area:,.1f}  {100 * r.share:.1f}%\n")
    out.write("aspect (downslope)  plan_m2  share\n")
    for r in st.aspect_sectors(sm, args.sectors):
        out.write(f"{r.label}  {r.plan_area:,.1f}  {100 * r.share:.1f}%\n")
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["x", "y", "z", "slope_deg", "slope_pct", "aspect_deg", "plan_m2", "true_m2"]
            )
            c = sm.corners.mean(axis=1)
            pct = 100.0 * np.tan(np.radians(sm.slope))
            for k in range(len(c)):
                w.writerow(
                    [
                        f"{c[k, 0]:.3f}",
                        f"{c[k, 1]:.3f}",
                        f"{c[k, 2]:.3f}",
                        f"{sm.slope[k]:.3f}",
                        f"{pct[k]:.2f}",
                        "" if math.isnan(sm.aspect[k]) else f"{sm.aspect[k]:.2f}",
                        f"{sm.plan_area[k]:.3f}",
                        f"{sm.true_area[k]:.3f}",
                    ]
                )
    rasters = [(args.asc_slope, "slope"), (args.asc_aspect, "aspect"), (args.asc_z, "z")]
    if any(p for p, _ in rasters):
        if not args.cell:
            raise SystemExit("raster output needs --cell")
        g = st.surface_grid(s, args.cell)
        for path, field in rasters:
            if path:
                st.write_ascii_grid(Path(path), g, getattr(g, field))
                out.write(f"{path}: {g.z.shape[1]} x {g.z.shape[0]} cells of {g.cell:g} m\n")


def _profile(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    sets = strings.load_strings(project, args.line)
    if not sets:
        raise SystemExit(f"string set {args.line!r} is empty")
    path = sets[0].points
    if sets[0].closed:
        path = np.vstack([path, path[:1]])
    named = {n: surfaces.load_surface(project, n) for n in args.surfaces}
    p = st.profile(named, path)
    out.write(f"profile {p.chainage[-1]:,.2f} m, {len(p.chainage)} points\n")
    for name, z in p.z.items():
        ok = np.isfinite(z)
        if ok.any():
            out.write(
                f"{name}: z {np.nanmin(z):.2f} to {np.nanmax(z):.2f} m over "
                f"{np.diff(p.chainage)[ok[:-1] & ok[1:]].sum():,.1f} m of the line\n"
            )
        else:
            out.write(f"{name}: not under the line\n")
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["chainage_m", "x", "y", *[f"z_{n}" for n in p.z]])
            for k in range(len(p.chainage)):
                zs = ["" if math.isnan(z[k]) else f"{z[k]:.3f}" for z in p.z.values()]
                w.writerow([f"{p.chainage[k]:.3f}", f"{p.xy[k, 0]:.3f}", f"{p.xy[k, 1]:.3f}", *zs])
    if args.plot:
        st.save_profile_plot(Path(args.plot), p, args.exaggeration)
        out.write(f"{args.plot} written\n")


def _intersect(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    a = surfaces.load_surface(project, args.a)
    b = surfaces.load_surface(project, args.b)
    lines = st.intersection_lines(a, b, tolerance=args.tolerance)
    if not lines:
        out.write("the surfaces do not meet\n")
        return
    out.write(
        f"{len(lines)} intersection lines, {sum(x.closed for x in lines)} closed, "
        f"total length {sum(x.length for x in lines):,.1f} m\n"
    )
    if args.save:
        strings.save_strings(
            project, args.save, [strings.String(x.points, x.closed, "KESISIM") for x in lines]
        )
        out.write(f"string set {args.save}: {len(lines)} lines\n")
    if args.dxf:
        n = write_dxf(Path(args.dxf), [Polyline("KESISIM", x.points, x.closed) for x in lines])
        out.write(f"{args.dxf}: {n} entities\n")


def _offset(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    s = st.offset_surface(surfaces.load_surface(project, args.surface), args.dz)
    surfaces.save_surface(project, args.save, s)
    out.write(f"surface {args.save}: {args.surface} moved {args.dz:+g} m\n")


def _merge(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    a = surfaces.load_surface(project, args.a)
    b = surfaces.load_surface(project, args.b)
    s, r = se.merge_surfaces(a, b, args.mode, tolerance=args.tolerance)
    surfaces.save_surface(project, args.save, s)
    out.write(
        f"surface {args.save}: {args.mode} of {args.a} and {args.b}, {r.triangles} triangles; "
        f"both {r.area_both:,.1f} m2 (from {args.a} {r.area_from_a:,.1f}, from {args.b} "
        f"{r.area_from_b:,.1f}), only {args.a} {r.area_a_only:,.1f} m2, only {args.b} "
        f"{r.area_b_only:,.1f} m2\n"
    )


def _normal_offset(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    s = surfaces.load_surface(project, args.surface)
    o, r = se.normal_offset(s, args.distance, max_factor=args.max_factor)
    surfaces.save_surface(project, args.save, o)
    out.write(
        f"surface {args.save}: {args.surface} offset {args.distance:+g} m along the normals, "
        f"largest vertex move {r.max_move:.3f} m, {r.limited} vertices at the move limit\n"
    )


def _smooth(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    s = surfaces.load_surface(project, args.surface)
    o, r = se.smooth_surface(
        s,
        args.passes,
        args.factor,
        max_change=args.max_change,
        hold_boundary=not args.move_boundary,
    )
    surfaces.save_surface(project, args.save, o)
    out.write(
        f"surface {args.save}: {r.iterations} passes, change max {r.max_change:.3f} m, mean "
        f"{r.mean_change:.3f} m, volume change {r.volume_change:+,.1f} m3, "
        f"{r.limited} vertices at the change limit\n"
    )


def _decimate(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    s = surfaces.load_surface(project, args.surface)
    o, r = se.decimate_surface(s, args.tolerance)
    surfaces.save_surface(project, args.save, o)
    out.write(
        f"surface {args.save}: {r.points_before} -> {r.points_after} points, "
        f"{r.triangles} triangles, largest vertical error {r.max_error:.4f} m "
        f"(limit {r.tolerance:g} m)\n"
    )


def _from_grid(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    g = se.read_ascii_grid(Path(args.grid))
    s, n = se.grid_surface(g, args.tolerance)
    surfaces.save_surface(project, args.save, s)
    out.write(
        f"surface {args.save}: {g.z.shape[1]} x {g.z.shape[0]} grid of {g.cell:g} m, {n} cells "
        f"with data, {len(s.points)} points, {len(s.triangles())} triangles\n"
    )


def _to_grid(project: Project, args: argparse.Namespace, out: TextIO) -> None:
    g = st.surface_grid(surfaces.load_surface(project, args.surface), args.cell)
    st.write_ascii_grid(Path(args.asc), g, g.z)
    out.write(f"{args.asc}: {g.z.shape[1]} x {g.z.shape[0]} cells of {g.cell:g} m\n")


def run(project: Project, args: argparse.Namespace, out: TextIO) -> bool:
    """Run a surface tool action; False when the action is not one of these."""
    handlers = {
        "contours": _contours,
        "slope": _slope,
        "profile": _profile,
        "intersect": _intersect,
        "offset": _offset,
        "merge": _merge,
        "normal-offset": _normal_offset,
        "smooth": _smooth,
        "decimate": _decimate,
        "from-grid": _from_grid,
        "to-grid": _to_grid,
    }
    handler = handlers.get(args.sf_command)
    if handler is None:
        return False
    handler(project, args, out)
    return True
