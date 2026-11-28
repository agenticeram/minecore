# ADR-0012: Exact solid booleans with the Manifold library

- Status: accepted 2026-11-28
- Deciders: product owner (asked for exact booleans, E-CD-03), architect

## Context

Solid booleans and cuts by a surface were done on a voxel raster: always
closed, but rounded at the voxel size and slow for fine voxels. Phase P1
requires exact booleans: "boolean volumes of cubes, spheres and nested solids
match analytic values; a 1M-triangle boolean completes and is timed".

Robust mesh booleans are a research-grade problem (coplanar faces, shared
edges, touching solids, near-degenerate intersections). Writing a robust
kernel from scratch (exact predicates, arrangement, retriangulation,
classification) would take weeks and still need years of hardening.

Manifold (github.com/elalish/manifold, Apache-2.0, Python wheels
`manifold3d` for Windows x64 and CPython 3.10-3.14) is a maintained,
published boolean kernel used by OpenSCAD and Blender. It works in double
precision (`Mesh64`), resolves coplanar and touching cases by symbolic
perturbation and guarantees a manifold result.

## Decision

1. `manifold3d` (Apache-2.0) is a runtime dependency. It is used only from
   `minecore.solidtools`, loaded on first use.
2. MineCore owns everything around the kernel: input verification with
   named problems (open, non-manifold, flipped, inward), repair with a user
   tolerance, work in a local frame (mine-grid coordinates keep precision),
   removal of zero-thickness shells left by touching solids, section
   outlines from the cut caps, the surface-to-prism construction for
   surface cuts, and volume/area/centroid by the divergence theorem.
3. Results are verified by MineCore's own code (analytic volumes, the
   inclusion-exclusion identity), not by the kernel's own reports.
4. The geometry contracts in `mc-core` are not changed. If the kernel ever
   has to move into Rust, the `solidtools` functions are the boundary.

## Consequences

- Booleans are exact on the triangles; 1M input triangles take about 2 s
  (`tools/bench_solid_boolean.py`).
- The tolerance is a user input: vertex merge distance in repair and the
  kernel's simplification tolerance (features finer than it may be merged;
  volume change bounded by tolerance x area).
- The Windows build bundles one more compiled module (PyInstaller picks it
  up from the import).
- Voxel booleans and voxel surface cuts are removed; `solid boolean|cut`
  lose `--voxel` and take `--tolerance` and `--repair`.
