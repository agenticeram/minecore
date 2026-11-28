"""Time an exact solid boolean on about one million triangles (roadmap P1 exit).

Two overlapping spheres, each about half a million triangles, placed at
mine-grid coordinates, are joined, intersected and subtracted with
``minecore.solidtools``; the volumes are checked against the analytic
lens formula for two spheres (within the faceting error) and against the
inclusion-exclusion identity. Run: ``uv run python tools/bench_solid_boolean.py``.
"""

from __future__ import annotations

import math
import sys
import time

import manifold3d
import numpy as np

from minecore.solids import Solid
from minecore.solidtools import Boolean, boolean, properties


def sphere(centre: tuple[float, float, float], radius: float, segments: int) -> Solid:
    mesh = manifold3d.Manifold.sphere(radius, segments).to_mesh64()
    v = np.asarray(mesh.vert_properties, np.float64)[:, :3] + np.asarray(centre)
    return Solid(v, np.asarray(mesh.tri_verts, np.int64))


def main() -> int:
    segments = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    r, d = 100.0, 100.0  # radius and centre distance (m)
    e = (500_000.0, 4_500_000.0, 1_000.0)
    a = sphere(e, r, segments)
    b = sphere((e[0] + d, e[1], e[2]), r, segments)
    n = len(a.triangles) + len(b.triangles)
    lens = math.pi * (4 * r + d) * (2 * r - d) ** 2 / 12
    print(f"input: {n:,} triangles; analytic sphere {4 / 3 * math.pi * r**3:,.1f} m3")
    va, vb = properties(a).volume, properties(b).volume
    vols = {}
    for op in Boolean:
        start = time.perf_counter()
        result = boolean(a, b, op)
        took = time.perf_counter() - start
        vols[op] = properties(result).volume
        print(
            f"{op.value:12s} {took:6.2f} s  {len(result.triangles):,} triangles  "
            f"volume {vols[op]:,.2f} m3"
        )
    print(f"lens (analytic): {lens:,.2f} m3; faceted spheres {va:,.2f} + {vb:,.2f} m3")
    gap = vols[Boolean.UNION] + vols[Boolean.INTERSECTION] - va - vb
    print(f"inclusion-exclusion error: {gap:.3e} m3")
    ok = abs(gap) < 1e-6 * va and abs(vols[Boolean.INTERSECTION] - lens) < 1e-3 * lens
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
