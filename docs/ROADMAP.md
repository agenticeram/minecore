# MineCore roadmap

Milestones are gates, not dates. A phase closes only when its exit evidence
has been produced and recorded in `project-status.json`. Dates are estimated
after Phase 1, from measured throughput.

## Basis (2026-11-19, ADR-0011)

The plan comes from a feature inventory of the main commercial mining
packages, built from their public product pages, brochures and release notes:

- Datamine (Studio RM/OP/UG/Survey, NPVS, MSO, Supervisor, Isatis.neo,
  DataBlast, MineScape)
- GEOVIA (Surpac, Whittle, MineSched, Minex)
- Maptek (Vulcan and add-ons, PointStudio, BlastLogic, Evolution)
- Deswik
- Micromine (Origin, Geobank, Beyond, Alastri, Spry, Advance, Pitram, Nexus)
- Hexagon MinePlan
- Seequent (Leapfrog, GeoStudio, Oasis montaj)
- Geotechnical packages: Rocscience, Itasca, PLAXIS
- Ventsim, Carlson, Netcad, Esri ArcGIS
- acQuire, RPMGlobal, Austin Powder Paradigm

The inventory and the gap list are kept by the owner, in Turkish, outside the
repository (`katalog/envanter/`, `katalog/eksik_listesi/`).

- They group features into 135 feature groups with codes `E-XX-NN`, for
  example `E-CD-01` for string editing.
- On 2026-11-18, 12 groups were complete, 38 were partial and 85 were
  missing.
- Phases below cite these codes.
- No vendor code, UI, asset or undocumented file format is copied (rule 7).
  The goal is to do the same engineering work, verified independently.

## Principles

1. **Shared engines before modules.** Four engines are built once and reused:
   - CAD kernel: strings, surfaces, solids, exact booleans.
   - Activity scheduling engine: open pit, underground, drill and blast, and
     ventilation staging.
   - Graph/network engine: haul roads, ventilation networks, pipe networks and
     escape routes.
   - Equipment library: haulage, fleet, cycles, and ventilation diesel/heat.
2. **Every phase leaves a usable product.** At the end of a phase the owner
   can do real work in that area from the GUI and the CLI.
3. **Measurable exit evidence.** Each phase is checked against analytic
   cases, hand calculations or an independent implementation (rule 4).
   Catalogue validation numbers become tests.
4. **Small increments in a fixed order.** Work is added one or two feature
   groups at a time, in the order below.
5. **Thresholds and regulatory values are user inputs, never defaults.**
   Resource classification is never presented as compliant with a reporting
   code (rule 8).

## Completed foundation (M0–M2)

| ID | Milestone | Status |
| --- | --- | --- |
| M0 | Foundation: repository, CI, licence policy, verification script | Done |
| M1 | Core contracts: units, CRS, project package, operations, jobs | Done |
| M2 | Drillhole data: import, validation, desurvey, compositing | Done |

Work done under M3–M11 (block models, estimation, pit optimisation, design,
scheduling, underground, implicit geology, OMF/DXF) is carried into the
phases below. Each phase finishes the remaining gaps in its area.

## Phases

| Phase | Scope (feature groups) | Depends on | Exit evidence | Former milestone |
| --- | --- | --- | --- | --- |
| P1 Platform and CAD kernel | String editing (E-CD-01), surface tools (E-CD-02), exact solid booleans and repair (E-CD-03), 3D view (E-CD-04), drawing sheets (E-CD-05), open formats (E-CD-06), macros and scripting (E-CD-07), versioning and audit trail (E-VT-13, E-CD-08 desktop part) | — | Boolean volumes of cubes, spheres and nested solids match analytic values; a 1M-triangle boolean completes and is timed; a bench design can be drawn by hand and plotted | M3, M11 |
| P2 Data, logging, QA/QC | CRM/blank/duplicate/umpire QA/QC (E-VT-07..10), graphic logs (E-VT-11), density (E-VT-12), lab workflow (E-VT-06), drill programme and infill planning (E-VT-05), offline logging forms (E-VT-04), core photos (E-VT-14) | P1 | Synthetic set with a known number of QA/QC failures is flagged exactly; the owner's dataset passes end to end | M2 |
| P3 Geostatistics and resources | Statistics, top cut, contact analysis and soft boundaries (E-KE-01..03); variogram gaps and KNA (E-KE-04, 06); dynamic anisotropy, cokriging, sub-block creation (E-KE-05, 09); indicator kriging, MIK, UC (E-KE-07); SGS and turning bands (E-KE-08); validation, classification, reporting, reconciliation (E-KE-11..14); geometallurgy (E-KE-15) | P2 | Kriging and simulation agree with an independent implementation within recorded tolerance; simulations reproduce histogram and variogram | M5, M6 |
| P4 Geological modelling (parallel with P3) | Section interpretation (E-JM-01), structural data (E-JM-07), implicit model gaps (E-JM-03), fault blocks (E-JM-04), veins (E-JM-05), field mapping (E-JM-08), ML domaining (E-JM-10), advanced structural (E-JM-09) | P1 | Known synthetic surfaces (plane, fold, faulted) recovered within recorded error | M10 |
| P5 Shared engines | Activity scheduling engine with dependencies, resources, calendars, levelling, Gantt view, solid slicing into schedule blocks (E-PL-02, 03); graph/network engine; equipment library (E-NF-01); material movement ledger (E-PL-11) | P1 | Critical path equals hand calculation on small projects; levelled schedules never exceed capacity; network flows match hand calculation | M8 |
| P6 Open-pit planning and haulage | Rimpull travel time, TKPH, fuel, cycle and fleet sizing (E-NF-02..04); short-term planning and grade control (E-PL-04, E-AO-10); dumps, dilution, design reserves (E-AO-07..09); parametric pit design; pit under uncertainty and multi-mine (E-AO-05, E-PL-01); blending, scenarios, cash flow (E-PL-05..08); haulage simulation (E-NF-05); conformance (E-AO-11); mine-to-market (E-PL-10) | P3, P5 | Travel time matches hand calculation from a performance curve; simulation reproduces a known queueing case | M7, M8 |
| P7 Drill and blast | Patterns (E-DP-02), charging (E-DP-03), timing (E-DP-04), fragmentation/vibration/airblast/flyrock prediction (E-DP-05), perimeter and development blasting (E-DP-06), actuals (E-DP-07), explosives inventory (E-DP-08) | P1, P5 | Predictions equal published worked examples; model coefficients are user inputs | — |
| P8 Underground | Development design gaps (E-YA-01), stope optimisation and narrow vein (E-YA-02, 03), ring design (E-YA-04), schedule on the shared engine and underground haulage (E-YA-06, E-NF-06), cavity scans and overbreak (E-YA-10), caving, open-pit-to-underground transition, subsidence (E-YA-05, 07, 09) | P1, P5, P7 | Geometric clash tests; stope results match brute force on small models | M9 |
| P9 Ventilation | Network model, Atkinson resistance, Hardy Cross/Newton solution, fans, regulators, air demand (E-HV-01..06); heat and climate, gas/dust/fumes (E-HV-07, 08); auxiliary ventilation, fire and escape (E-HV-09, 10); measurement and calibration, planning and cost (E-HV-11, 12) | P5, P8 | Textbook networks solved exactly; Kirchhoff laws hold within tolerance at every node and loop | — |
| P10 Survey, point clouds, GIS | Helmert/affine, ED50/ITRF96 (E-OL-02); COGO and adjustment, code lists (E-OL-03, 04); contours, profiles, reports (E-OL-05, 09); point clouds, drone and raster (E-OL-06, 07); GIS, licence areas, spatial analysis (E-OL-08, 10) | P1 | Transformations checked on official control points; adjustments match textbook examples | — |
| P11 Geotechnics, water, geophysics | Stereonets and kinematics, rock mass (E-JT-01, 02); 2D then 3D limit equilibrium, probabilistic (E-JT-03); underground wedges, monitoring (E-JT-05, 06); 2D then 3D numerical analysis, tailings and dump stability (E-JT-04, 07); mine water (E-MS-01); geophysics (E-JF-01) | P1, P4 | Limit equilibrium matches published benchmark problems; elastic stresses around a circular opening match the Kirsch solution | — |
| P12 Coal and stratified deposits | Seam models and quality grids, washability (E-JM-06); coal scheduling (E-PL-09); room-and-pillar and longwall (E-YA-08); dragline and cast blasting (E-AO-12) | P3, P4, P5, P6 | Known thickness and quality grids reproduced | — |
| P13 Closure and environment | Rehabilitation landforms (E-RH-01), catchments and runoff (E-RH-02), acid drainage and waste placement (E-RH-03), emissions and fuel reporting (E-RH-04) | P1, P10 | Landform cut/fill balance and runoff checked on analytic cases | — |
| P14 Commercial readiness | Windows installer, updates, Turkish/English UI (E-PF-04), dashboards (E-PF-03), mobile field app (E-PF-02), cloud sharing (E-PF-01), collaboration (E-CD-08), fleet-management data intake (E-NF-07, 08) | runs alongside; an interim installer follows P1 | Clean-machine install test; published benchmark report | M12 |

Execution order: P1, P2, P3 with P4, P5, P6, P7, P8, P9, P10, P11, P12, P13.
P14 runs alongside the others.

## Tracking

- `project-status.json` lists the phases with their status and evidence.
- Each feature group added is dated in the owner's gap list.
- The owner's capability catalogue (`durum.csv`) gets its `eklendi_tarih`
  only after the tests pass.
- A newly noticed competitor feature goes into the inventory first, then into
  the phase that owns its area. No work is added outside this plan.
