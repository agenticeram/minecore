"""OC06: breaklines, boundaries with holes, longest-edge trimming, DXF surfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minecore.io.dxf import Polyline, read_dxf, write_dxf
from minecore.project import Project
from minecore.surfaces import (
    DxfSurfaceLayers,
    Surface,
    SurfaceError,
    build_surface,
    clip_surface,
    load_surface,
    read_dxf_surface,
    save_surface,
)
from minecore.survey.volumes import triangle_volume


def _plan_area(s: Surface) -> float:
    t = s.triangles()
    p = s.points[:, :2]
    u, v = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]
    return float(0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum())


def _bench_input() -> tuple[np.ndarray, list[np.ndarray]]:
    """One bench: floor z=0 for x<40, face 40..45, top z=10 beyond; 100 m long.

    Sparse spot heights on the floor and the top only, so a plain Delaunay
    cuts across the toe and crest.
    """
    rng = np.random.default_rng(7)
    floor = rng.uniform([0, 0], [38, 100], (25, 2))
    top = rng.uniform([47, 0], [100, 100], (25, 2))
    corners = np.array([[0, 0], [0, 100], [100, 0], [100, 100]], float)
    xy = np.vstack([floor, top, corners])
    z = np.where(xy[:, 0] < 42.5, 0.0, 10.0)
    toe = np.array([[40.0, 0, 0], [40.0, 100, 0]])
    crest = np.array([[45.0, 0, 10], [45.0, 100, 10]])
    return np.column_stack([xy, z]), [toe, crest]


def test_bench_edges_exact_with_breaklines_and_volume_differs_without() -> None:
    points, lines = _bench_input()
    with_lines, report = build_surface(points, lines)
    assert report.inserted > 0
    y = np.linspace(0, 100, 101)
    for x, z in ((40.0, 0.0), (45.0, 10.0), (42.5, 5.0)):
        got, outside = with_lines.elevation(np.column_stack([np.full_like(y, x), y]))
        assert outside == 0
        np.testing.assert_allclose(got, z, atol=1e-9)
    # Exact volume above z = 0: face wedge 5 x 10 / 2 x 100 + top 55 x 10 x 100.
    exact = 2_500.0 + 55_000.0
    assert triangle_volume(with_lines, 0.0).cut == pytest.approx(exact, rel=1e-12)
    plain = Surface(points)
    plain_cut = triangle_volume(plain, 0.0).cut
    assert abs(plain_cut - exact) > 500.0  # plain Delaunay misplaces toe and crest


def test_crossing_breaklines_split_when_z_agree_and_refused_otherwise() -> None:
    pts = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [100, 100, 0.0]])
    a = np.array([[10, 50, 5], [90, 50, 5.0]])
    b = np.array([[50, 10, 5], [50, 90, 5.0]])
    s, report = build_surface(pts, [a, b])
    assert report.crossings_split == 1
    z, _ = s.elevation(np.array([[50.0, 50.0], [30.0, 50.0], [50.0, 70.0]]))
    np.testing.assert_allclose(z, 5.0, atol=1e-12)
    c = np.array([[50, 10, 7], [50, 90, 7.0]])
    with pytest.raises(SurfaceError, match="different elevations"):
        build_surface(pts, [a, c])


def test_boundary_with_hole_and_shape_kept_inside() -> None:
    points, lines = _bench_input()
    full, _ = build_surface(points, lines)
    outer = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float)
    hole = np.array([[30, 30], [60, 30], [60, 60], [30, 60]], float)
    trimmed, _ = build_surface(points, lines, [outer, hole])
    assert _plan_area(trimmed) == pytest.approx(80 * 80 - 30 * 30, rel=1e-12)
    q = np.random.default_rng(3).uniform(10, 90, (3000, 2))
    q = q[~((q[:, 0] > 30) & (q[:, 0] < 60) & (q[:, 1] > 30) & (q[:, 1] < 60))]
    assert trimmed.contains(q).all()
    np.testing.assert_allclose(trimmed.elevation(q)[0], full.elevation(q)[0], atol=1e-9)
    assert not trimmed.contains(np.array([[45.0, 45.0], [5.0, 5.0]])).any()
    clipped = clip_surface(full, [outer])
    assert _plan_area(clipped) == pytest.approx(80 * 80, rel=1e-12)
    np.testing.assert_allclose(clipped.elevation(q)[0], full.elevation(q)[0], atol=1e-9)


def test_longest_edge_removes_spanning_triangles() -> None:
    g = np.arange(0, 51, 10.0)
    gx, gy = np.meshgrid(g, g)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    xy = np.vstack([xy, [[200.0, 25.0]]])  # far point makes long thin triangles
    pts = np.column_stack([xy, np.zeros(len(xy))])
    s, _ = build_surface(pts, max_edge=15.0)
    assert _plan_area(s) == pytest.approx(50 * 50, rel=1e-12)


def test_store_and_reload_keeps_boundary_and_edge_limit(tmp_path: Path) -> None:
    points, lines = _bench_input()
    outer = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float)
    hole = np.array([[30, 30], [60, 30], [60, 60], [30, 60]], float)
    s, _ = build_surface(points, lines, [outer, hole], max_edge=80.0)
    project = Project.create(tmp_path / "p.mcproj", "p")
    save_surface(project, "pit", s)
    back = load_surface(project, "pit")
    assert back.max_edge == 80.0
    assert len(back.boundary) == 2
    np.testing.assert_array_equal(back.triangles(), s.triangles())


def test_dxf_tin_points_contours_and_boundary(tmp_path: Path) -> None:
    # A two-face TIN of a tilted plane, contour polylines and a boundary ring.
    corners = np.array([[0, 0, 0], [100, 0, 10], [100, 100, 10], [0, 100, 0.0]])
    faces = np.array([[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]])
    ring = np.array([[20, 20, 0], [80, 20, 0], [80, 80, 0], [20, 80, 0.0]])
    path = tmp_path / "topo.dxf"
    write_dxf(path, [Polyline("SINIR", ring, True)], faces=[("TIN", faces)])
    ents = read_dxf(path)
    assert ents.layers() == ["SINIR", "TIN"]
    assert len(ents.faces) == 2
    s, report = read_dxf_surface(path, DxfSurfaceLayers(boundary="SINIR"))
    assert report.triangles > 0
    assert _plan_area(s) == pytest.approx(3600.0, rel=1e-12)
    q = np.random.default_rng(5).uniform(20, 80, (500, 2))
    np.testing.assert_allclose(s.elevation(q)[0], q[:, 0] / 10.0, atol=1e-9)
    # LWPOLYLINE contours (group 38 elevation) and POINT entities.
    text = "\n".join(
        [
            "0", "SECTION", "2", "ENTITIES",
            "0", "LWPOLYLINE", "8", "KONTUR", "90", "2", "70", "0", "38", "5.0",
            "10", "50", "20", "0", "10", "50", "20", "100",
            "0", "POINT", "8", "NOKTA", "10", "0", "20", "0", "30", "0",
            "0", "POINT", "8", "NOKTA", "10", "0", "20", "100", "30", "0",
            "0", "POINT", "8", "NOKTA", "10", "100", "20", "0", "30", "10",
            "0", "POINT", "8", "NOKTA", "10", "100", "20", "100", "30", "10",
            "0", "ENDSEC", "0", "EOF",
        ]
    )  # fmt: skip
    path2 = tmp_path / "kontur.dxf"
    path2.write_text(text + "\n", encoding="ascii")
    s2, r2 = read_dxf_surface(path2)
    assert r2.breakline_vertices == 2
    np.testing.assert_allclose(s2.elevation(np.array([[50.0, 30.0]]))[0], 5.0, atol=1e-12)
