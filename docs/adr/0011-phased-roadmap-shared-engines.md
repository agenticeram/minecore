# ADR-0011: Phased roadmap built from a competitor feature inventory

- Status: accepted 2026-11-19
- Deciders: product owner, architect

## Context

Features were being added in catalogue order, one or two at a time. The owner
asked for a planned roadmap so that MineCore does not grow as a patchwork. The
plan should cover everything the main commercial packages offer.

An inventory of public product information was compiled for the vendors listed
in `docs/ROADMAP.md`. Features were grouped into 135 feature groups with
`E-XX-NN` codes. The inventory itself is kept in Turkish in the owner's
catalogue, outside the repository.

## Decision

1. The milestones M3–M12 are replaced by fourteen phases P1–P14
   (`docs/ROADMAP.md`). M0–M2 stay done. Work already done under M3–M11 counts
   towards the phase that owns its area.
2. Shared engines come before the modules that use them. The phases are
   ordered by these dependencies:
   - CAD kernel (P1)
   - activity scheduling engine, graph/network engine, equipment library (P5)
3. Each phase has measurable exit evidence and leaves a usable product.
4. New features are added one or two feature groups at a time, in phase order.
   A competitor feature found later goes into the inventory and then into its
   phase.

## Consequences

- Underground scheduling and drill and blast build on the P5 engines.
  Ventilation also builds on them; it does not get its own solver
  infrastructure.
- Existing underground development timing moves onto the shared scheduling
  engine in P8.
- Geotechnical numerical analysis (P11) is the longest phase. It may use
  external permissive solvers. GPL solvers may only run as a separate process.
- `project-status.json` tracks phases instead of the old milestones.
