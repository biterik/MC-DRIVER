# Brief for Claude Code — site-restricted (Voronoi-filtered) GCMC insertion in LAMMPS

> Hand this to Claude Code running **inside a LAMMPS source checkout**. It is a **read-only
> source-analysis task**: produce a written report + a change plan. Do **not** modify code yet.
> Record the exact LAMMPS version / git commit you analyze.

## Goal

We want grand-canonical insertion/deletion of H that, instead of random positions in a region,
draws **only from a precomputed list of interstitial site positions** (the list is filtered by a
**minimum Voronoi volume** so only low-insertion-energy cavities are candidates). It must stay
**MPI-parallel** and work with an **EAM** potential (Ni–H). Conceptually this is a **lattice-gas
GCMC on a fixed site list**.

**Reference convention (decided — do NOT reproduce `fix gcmc`'s ideal-gas reservoir).** Adopt the
lattice-gas acceptance natively: with a uniform single-site proposal (pick a site; insert if
empty, delete if occupied), the acceptance is simply `min(1, exp(-β(ΔU ∓ μ)))` — the ideal-gas
factors `zz`, `volume`, and `1/(ngas+1)` are **dropped**, not converted. The fix defines its own
μ zero; it differs from an absolute μ only by a constant we **calibrate once** (the same
calibration the Python reference driver uses). So your job on the acceptance is to *remove* the
ideal-gas normalization and wire in a single lattice-μ keyword — a simplification, not a port.
Determine what must change and where, and whether to patch `fix gcmc` in place or write a new
`fix gcmc/site`.

## Files to read (start here)

- `src/MC/fix_gcmc.cpp` and `src/MC/fix_gcmc.h` — primary target.
- `src/MC/fix_widom.{cpp,h}` and `src/MC/fix_atom_swap.{cpp,h}` — reusable patterns
  (Widom insertion energy; lattice-preserving type swaps).
- `src/VORONOI/compute_voronoi_atom.{cpp,h}` — can it yield interstitial-cavity volumes
  (e.g. via probe atoms)?
- `src/domain.{cpp,h}`, `src/comm*.{cpp,h}`, `src/region.{cpp,h}` — subdomain ownership,
  atom migration, region tests.
- The vcsgc-lammps `fix_sgcmc.{cpp,h}` if present — see "Alternative routes".

## What `fix gcmc` already gives us (verified from `fix_gcmc.h`)

- Synchronized RNG `random_equal` (identical on all ranks) vs per-rank `random_unequal`.
- Per-rank subdomain bounds `sublo`/`subhi`; gas bookkeeping `local_gas_list`,
  `update_gas_atoms_list()`, `ngas`/`ngas_local`/`ngas_before`.
- Region restriction (`region`, `idregion`, `region_volume`).
- Partial vs full energy: `energy(int,int,tagint,double*)` vs `energy_full()`, flag `full_flag`.
- Insertion/deletion entry points: `attempt_atomic_insertion()` / `_full()`,
  `attempt_atomic_deletion()` / `_full()`, plus `pick_random_gas_atom()`.
- Acceptance constants: `beta`, `zz` (activity ≈ e^{βμ}/Λ³), `volume`, `gas_mass`, `tfac_insert`.

## Specific questions to answer (tie each to file:function:line)

1. **Trial position.** In `attempt_atomic_insertion()` / `attempt_atomic_insertion_full()`,
   exactly where is the candidate `coord` generated, and how is the region/`sublo,subhi`
   constraint applied? Is `coord` drawn with `random_equal` (all ranks agree) before the owning
   rank acts? Quote the lines.
2. **Acceptance normalization.** Write the exact insertion and deletion acceptance expressions
   as coded (the roles of `zz`, `volume`, `ngas`, `beta`, `tfac_insert`). Identify precisely
   which factors encode the *continuous-volume ideal-gas reference* (`zz`, `volume`, `1/(ngas+1)`)
   and the minimal edit to **bypass** them in favour of the lattice-gas form
   `min(1, exp(-β(ΔU ∓ μ)))` driven by a single lattice-μ keyword (own reference; calibrated
   once). Do **not** reproduce the ideal-gas normalization — confirm the cleanest cut points.
3. **EAM → full energy.** Confirm that EAM forces the `full_energy` path (default `energy()`
   omits the embedding term). Where is `energy_full()` invoked per move, and what is its cost
   scaling? (This sets the per-move floor; a site list does NOT reduce it.)
4. **MPI distribution of attempts.** How is an insertion/deletion attempt assigned to a rank?
   How do ranks stay consistent (shared RNG, `ngas_before`, reductions)? What breaks if only a
   subset of ranks own candidate sites?
5. **Atom create/delete.** How is an inserted atom created (avec->create_atom, tag assignment,
   image flags, comm/borders) and how is deletion done? What must a site list add: a per-site
   `occupied` flag and a stable `site_id ↔ atom tag` map.
6. **Injecting the site list.** Cleanest mechanism: read a global list of candidate positions
   (+ optional per-site volume/weight); compute per-rank ownership from `sublo/subhi`; keep it
   valid across `pre_exchange`/migration/load-balance. Is drawing a **global site index** with
   `random_equal` (owning rank inserts) the minimal-change route that preserves parallelism?
7. **Voronoi filtering.** Can `compute voronoi/atom` give an empty interstitial cavity's volume
   (probe atom?), or should the filtered list be precomputed offline and only *read* by the fix?
   Where would a `min_volume` (and optional `max_volume`) keyword live?
8. **Deletion symmetry.** For detailed balance the deletion must pick uniformly among *occupied
   listed sites* and use the matching lattice-gas ratio. Confirm `pick_random_gas_atom()` /
   `local_gas_list` can be restricted to listed-and-occupied sites.
9. **Patch vs new fix.** Given the above, recommend: minimal in-place branch of `fix gcmc`
   (guarded by a `sitelist` keyword) vs a new `FixGCMCSite`. Note risk to the existing
   regression tests either way.

## Alternative routes to evaluate (briefly)

- **`fix sgcmc` group restriction.** Since fixes act on a `group`, does restricting `fix sgcmc`
  to a group of candidate atoms already give "only certain sites" — via the ghost-species
  representation (every site carries an atom; swap ghost↔H)? Note the EAM "inert ghost" problem
  (no truly non-interacting species in an `eam/alloy` file). Is dynamic-group update viable for
  a drifting active set?
- **`fix atom/swap`** as a starting point for a fixed-lattice, occupancy-preserving variant.

## Deliverable (write to `SITE_GCMC_ANALYSIS.md` in the LAMMPS tree)

- A table: each required change → `file:function:line`, with effort (S/M/L) and risk.
- The adopted **lattice-gas acceptance** stated explicitly, plus how the single μ₀ reference is
  calibrated (no ideal-gas reproduction).
- A recommended path (patch vs new fix) with rationale.
- A **validation plan**: (a) with `pair_style zero`, reproduce the analytic lattice-gas Langmuir
  isotherm; (b) cross-check ⟨c⟩ and P(N) against the MC-DRIVER reference driver
  (this repo's `tests/test_lammps_native_crosscheck.py`). Because the new fix shares the
  lattice-gas convention, (a) and (b) need **no μ remap**; only (c) — an optional EAM smoke test
  vs `fix gcmc` `full_energy` on a tiny cell — needs μ-reference alignment.
- A list of open risks (load-balance + site ownership, restart/`write_restart` of the site
  state, triclinic boxes, overlap with `region`/`overlap_cutoff` keywords).

## Constraints

- Read-only. No edits. Cite the analyzed commit hash and LAMMPS version.
- Prefer minimal disruption to existing `fix gcmc` behavior/tests.
- Keep everything MPI-correct; call out any step that would serialize the fix.
