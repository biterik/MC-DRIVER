# MC-DRIVER — implementation spec (for Claude Code)

Self-contained spec for a site-list Monte-Carlo driver for H on the octahedral sublattice
of fcc Ni / rock-salt NiH, coupled to LAMMPS for energies. Implemented **outside** the
LLM-LMPS project (standalone repo / script), validated against the acceptance suite at the
end of this document. Parent context: `../project.md`, `../02_MC-DRIVER-AND-CALIBRATION/thread.md`.

---

## 1. Purpose

Compute, for a given interatomic potential, the stress-free relationship between the H
chemical potential μ_H and the H concentration c = N_H / N_Ni at fixed temperature, by
inserting/removing H on the octahedral interstitial sublattice and relaxing the lattice each
MC block. Must support both:

- **Grand-canonical (GC)** sampling at fixed μ_H (for single-phase regions), and
- **Variance-constrained semi-grand-canonical (VC-SGC)** sampling (for points inside the
  α+β miscibility gap, where plain GC/SGC is unstable to phase separation).

The same code drives both the Pezold-EAM and the Ko/Shim/Lee-MEAM runs.

## 2. Physical background (minimal)

- fcc Ni has one octahedral interstitial site per Ni atom. Full occupancy = rock-salt B1
  NiH. The octahedral sites therefore form an fcc sublattice congruent with the Ni lattice.
- The **site list is not computed geometrically** — it is read from a fully-occupied NiH
  structure (the H atom positions). See §4.
- Reference: Korbmacher et al., *Metals* 2018, 8, 280 (same EAM family). Their validation
  numbers are the §10 acceptance targets.

## 3. Inputs / interfaces (provided; do NOT build these)

The structures and potentials are produced by LLM-LMPS thread 01 and provided as files:

1. **Fully-occupied NiH structure** (LAMMPS data file), 10×10×10 conventional cells:
   4000 Ni (type 1) + 4000 H (type 2). The H positions ARE the octahedral site list.
2. **Empty Ni structure** (LAMMPS data file): 4000 Ni, no H — the starting config for the
   dilute branch.
3. **Potential files** + the exact `pair_style`/`pair_coeff` lines for each potential:
   - EAM: `pair_style eam/alloy`, file `ni_h_rcut4.90_rcut2.eam.alloy`.
   - MEAM: `pair_style meam`, files `NiH_KoShimLee_library.meam` + `NiH_KoShimLee.meam`.
   - `POTDIR` env/var points at the potentials directory (locally a copy; on cluster
     `/cmmc/ptmp/biterik/POTENTIALS`).
4. A **run config** (YAML/JSON), see §9.

The driver must accept these as paths/parameters; it must not hard-code structure geometry.

## 4. Data model

- **Site list**: load the provided NiH structure; extract the H-atom coordinates as the
  ordered array `sites[M]` (M = 4000 for 10×10×10). Store each site's fractional + Cartesian
  position and a stable integer `site_id`.
- **Occupancy**: boolean array `occ[M]`. `occ[i]=True` ⇒ an H atom currently sits on site i.
- **LAMMPS atom bookkeeping**: maintain a map `site_id → lammps_atom_id` for occupied sites
  so deletions target the right atom. Insertions use `create_atoms` at the exact site
  Cartesian coordinate (type 2 = H); deletions use `delete_atoms group`/`region` on that atom.
  After create/delete, rebuild neighbor lists before energy evaluation.

## 5. Energy backends (pluggable, same MC layer on top)

Backend interface: `relax_and_energy(state) -> (U_potential, box)`; applied per MC block.

- **(a) statics** — `minimize` at fixed cell OR with `fix box/relax iso 0.0` (stress-free).
  Reproduces Korbmacher (T enters only via the MC acceptance; no vibrational entropy). Use
  predecessor minimization settings: `min_style cg`, tight tolerances.
- **(b) md_npt** — short `fix npt` block at the target T and P=0 (isotropic) for thermal +
  volume relaxation, then average U over the block tail. This is the physical production
  backend. Equilibration vs sampling sub-block lengths are config parameters.

Use the **LAMMPS Python library** (`from lammps import lammps`) — one persistent instance,
driven move-by-move; do not respawn LAMMPS per move. Verify the exact API calls against the
installed LAMMPS version's docs (`create_atoms`, `delete_atoms`, `gather/scatter`, `minimize`,
`run`, extracting `pe` and box).

## 6. MC moves & acceptance

**Move (both ensembles): symmetric single-site flip.** Pick a site i uniformly at random.
- if `occ[i]` is False → propose **insertion** (create H at site i),
- if `occ[i]` is True  → propose **deletion** (remove the H on site i).

Evaluate ΔU via the active backend (the relaxed energy change for the proposed state).

**GC acceptance (lattice-gas grand canonical):**
- insertion: `P_acc = min(1, exp(-β(ΔU - μ)))`
- deletion:  `P_acc = min(1, exp(-β(ΔU + μ)))`

with β = 1/(k_B T), μ = μ_H on the LAMMPS-internal scale (the calibration constant from
thread 02 maps it to Korbmacher's μ₀; the additive reference is absorbed there). Uniform site
selection makes this move detailed-balance correct with no proposal-bias correction. Starting
from full occupancy naturally yields removal-dominated early dynamics (most picks hit occupied
sites) — this is intended, not a bias to correct.

**VC-SGC acceptance:** implement per Sadigh & Erhart et al., *Phys. Rev. B* **85**, 184203
(2012). Use their variance-constrained semi-grand potential with target concentration c₀ and
constraint strength κ̄; the acceptance compares the constrained potential
Φ = U − Δμ·N_H + κ̄·N_Ni·(c − c₀)² before/after the flip. Follow the paper's exact equations
and recommended κ̄ range; expose Δμ, c₀, κ̄ as config. (Implement GC first, validate, then add
VC-SGC.)

## 7. Blocking / cadence

Relaxing after every single flip is too expensive. Structure a **block**: attempt M trial
flips, then run one relaxation/MD block, then record. M, the statics tolerance, and the NPT
sub-block lengths are config parameters tuned on a probe (thread 02). Track and log
acceptance ratio; warn if it drifts below/above sane bounds.

## 8. Outputs (descriptive filenames; explicit precision)

Per the project's output-precision convention:

- **Isotherm CSV**, one row per recorded block, filename
  `isotherm-<POT>-T<T>K-<ENSEMBLE>.csv` (e.g. `isotherm-EAM-T300K-GC.csv`):
  columns `block,n_H,c,mu_eV,U_eV,a_box_Ang,P_bar,acc_ratio` with
  c `%.6f`, mu/U `%.6f` eV, a_box `%.5f` Å, P `%.4g` bar.
- **Run-summary YAML** `run-summary-<POT>-T<T>K-<ENSEMBLE>.yaml`: potential provenance,
  all config, LAMMPS version, git hash of the driver, final ⟨c⟩, ⟨a⟩, std, n_blocks.
- **Occupancy snapshots** (optional, for viz) `occupancy-<POT>-T<T>K-mu<...>.dump`: LAMMPS
  custom dump, coordinates `%.4f` Å (OVITO-grade; below CNA/PTM floors).
- **Log** `mc-driver-<POT>-T<T>K-<ENSEMBLE>.log`: per-block trace, RNG seed, timings.

Record the RNG seed; runs must be reproducible from (seed + config + input structures).

## 9. Config parameters (single file, YAML)

`potential` (eam|meam + files), `structure_full`, `structure_empty`, `start_from`
(full|empty), `ensemble` (gc|vcsgc), `backend` (statics|md_npt), `T_K`, `mu_eV` (gc) or
{`delta_mu_eV`,`c0`,`kappa`} (vcsgc), `n_blocks`, `flips_per_block`, `md_steps_equil`,
`md_steps_sample` (md_npt), `min_etol`,`min_ftol` (statics), `seed`, `out_prefix`.

## 10. Acceptance test suite (the definition of "done")

**Unit / integration (fast, no cluster):**

1. **Analytic non-interacting lattice gas.** Replace the potential with a trivial constant
   per-H site energy ε (no H-H, no relaxation). Then ⟨c⟩(μ,T) must match the closed-form
   Langmuir/Fermi isotherm `θ = 1/(exp((ε−μ)/kT)+1)` across a μ sweep, within MC error
   (e.g. |Δθ| < 0.01 at ~10⁴ blocks). Tests GC acceptance + sampling in isolation.
2. **Small-system exact enumeration.** Tiny lattice (≤ ~16 sites), rigid, with a simple
   tunable nearest-neighbour H-H interaction. Enumerate all 2^M states for the exact ⟨c⟩ and
   full P(c) at chosen (μ,T). MC must reproduce the entire P(c) histogram and ⟨c⟩ within error.
   Rigorous detailed-balance check. Run for both GC and (with matched conditions) VC-SGC.

**Guards (fast):**
- μ → +large ⇒ c → 1; μ → −large ⇒ c → 0.
- Insert then delete the same site ⇒ U and N_H return to baseline (round-trip; create/delete
  + neighbor-rebuild correctness).
- In a single-phase μ window, VC-SGC ⟨c⟩ equals GC ⟨c⟩ within error.

**Real-world anchor (compute-bound; needs thread-01 structures + a working potential):**
3. With the **actual Pezold EAM**, statics backend, 10×10×10, reproduce Korbmacher:
   - homogeneous E(c) fit ⇒ μ₀ = −2.148 eV, α = 0.751 eV, β = 0.355 eV
   - two-phase plateau μ_M = −2.405 eV at 300 K
   - a(c): a_Ni = 3.520 Å → a_NiH = 3.738 Å
   - 0 K Cij (separate calc, thread 01): Ni 250.7/145.5/134.3, NiH 295.3/197.3/33.5 GPa
   Agreement to within a few % validates driver + potential + pipeline jointly.

## 11. Tech stack / non-goals

- Python 3, NumPy, the LAMMPS Python module, PyYAML. No heavyweight frameworks.
- Single-node first (the study cell is 4000–8000 atoms). MPI-parallel insert/delete is a
  non-goal for v1.
- A native C++ LAMMPS fix is explicitly **out of scope for v1** — only revisit if profiling
  shows MC orchestration (not relaxation) dominates. This Python driver is the reference
  implementation / correctness oracle for any future fix.

## 12. Known risk

The LAMMPS-library create/delete + neighbor-rebuild + relaxation loop is the main debugging
surface. MEAM in particular has had quirks under repeated atom create/delete — exercise the
round-trip guard with MEAM early. Confirm the energy after create/delete matches a
from-scratch rebuild of the same configuration (a stronger version of the round-trip guard).
