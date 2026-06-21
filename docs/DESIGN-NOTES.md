# MC-DRIVER — design notes & brainstorming (post-v0.1)

> **Status: BRAINSTORMING, nothing decided.** These are working notes from the design
> discussion *after* the v0.1 first implementation attempt. They are not a spec and not a
> commitment — `MC-DRIVER-SPEC.md` remains the source of truth for what is actually built.
> Items here are candidate directions for a finite-T production version; open questions are
> flagged explicitly. Author of the notes: Erik Bitzek; drafting assistance: Claude Code.

## 1. Where v0.1 stands (the honest baseline)

v0.1 is a **single-process serial reference driver** — deliberately a *correctness oracle*,
not yet a performance tool:

- Per trial flip: an **unrelaxed global `run 0`** energy evaluation (O(N)) at the current,
  fixed box; accept/reject; → one **global** relaxation per MC block.
- Statics backend: per-block `minimize` (optionally `fix box/relax iso 0` = **0 K** box
  relaxation → the Korbmacher / von Pezold reproduction mode, T only in the acceptance).
- md_npt backend: per-block `fix npt … iso P P` → the **barostat sets the volume at
  temperature** (thermal expansion + target P). No `box/relax`, no 0 K minimize here.

**Resolved understanding (the volume question).** Nothing relaxes the volume to 0 K in the
MD-MC path. 0 K box relaxation happens *only* in the statics backend with `cell_relax=True`.
In md_npt the volume evolves during the NPT runs at T, which is what we want. Per flip the box
is never touched. ⇒ Operationally: never enable `cell_relax` in an md_npt run (off by default).

**Speed reality of v0.1.** Its advantages over the native `fix sgcmc` are *functional*, not
yet speed: true insertion/deletion (no ghost species — clean with EAM), and a smaller system
when c < 1 (no placeholder atoms). It is **not** faster *per move* than the native fixes,
because of the global `run 0` per flip + Python + serial chain. Making it actually fast is
what these notes are about.

## 2. Target: finite-T MD-MC, faster *and* faithful

Production goal is a finite-temperature hybrid MD/MC that is faster than the original VC-SGC
while capturing the relaxation physics the original (transmutation, fixed volume) and the
fixed-volume lattice-gas both miss. The template is von Pezold et al., *Acta Mater.* **59**
(2011) 2969 §2.4.

## 3. Idea A — von Pezold-style relaxation cadence (per-flip local + periodic global)

von Pezold §2.4: EAM-MC "**locally relaxed up to the third nearest neighbor shell** whenever
the occupation of an interstitial site changed," with "**global relaxations after every 20th
accepted step**."

Mapping onto MC-DRIVER:

- **Per-flip local relaxation** to ~3NN: freeze atoms beyond a cutoff, relax only the local
  cluster, use that ΔU in the acceptance. Cost O(local), not O(N) — the big-cell win. Recipe:
  `region loc sphere x y z R` → `group mob region loc` → `group froz subtract all mob` →
  `fix hold froz setforce 0 0 0` → `minimize …` → `unfix hold`.
- **Periodic global step** = the **NPT block** (finite-T analog of their global relaxation):
  heals the frozen-shell boundary error *and* sets the volume at temperature.

Open questions / caveats:

- **Minimizer + criterion are OURS to choose — the paper does not state them.** It specifies
  only scope (3NN) and cadence (global every 20 accepted). Candidates: CG or **FIRE**
  (`min_style fire`), force tol ~1e-2…1e-3 eV/Å. FIRE is robust to the sudden local distortion
  an insertion causes. → decide and pin in the SPEC.
- A local *minimize* is a 0 K operation injected into a finite-T scheme — fine as an
  acceptance-energy estimate (von Pezold's intent), but the thermal + volume sampling must come
  from the periodic NPT. This is an approximation to detailed balance (the established, tested
  kind), not exact. → decide how strictly we care, and whether to document/quantify the bias.
- Cutoff radius R for "local": 3NN ≈ ? Å for Ni — tie to the EAM cutoff (changed embedding
  density propagates one shell beyond the pair cutoff). → calibrate.

## 4. Idea B — importance sampling / pre-population (the orders-of-magnitude lever)

von Pezold §2.4.1: most sites are bulk-like and well-predicted by a cheap surrogate, the
**on-site, volume-dependent energy E^H(V)** — H solution energy is a smooth function of the
site's **Voronoi volume** in the H-free lattice (their Fig. 2). Then:

1. **Cheap pass everywhere:** MC using only E^H(V) (table lookup, no EAM, no H–H) over all
   sites → a reasonable global H distribution, fast.
2. **Expensive pass only where it matters:** full EAM MC only on the **active set** — sites
   where E^H(V) is lowered by ≥ ~20 meV vs bulk (low insertion energy → high occupancy → near
   the defect). They cut 70,000 → 1,500 sites, "orders of magnitude."

⇒ **We need a new tool** (candidate module `nih_mc.surrogate` / `tools/`):

- (a) **Voronoi volume per interstitial site.** `compute voronoi/atom` gives volumes of
  *existing* atoms, so for an empty cavity drop a **non-interacting probe** at the site and read
  its Voronoi cell (or tessellate with virtual points at the sites). → implementation detail to
  prototype.
- (b) **Calibrated E^H(V) surrogate**: one-time fit of H solution energy vs site volume for the
  chosen potential (reproduce their Fig. 2). → calibration step + stored fit.
- (c) **Selector**: rank sites by predicted insertion energy, mark low-energy ones active; hold
  the rest at surrogate occupancy. Doubles as an **importance-sampling proposal weight** (propose
  insertions where E^H(V) is low → higher acceptance).

Open questions:

- Threshold (the "≥20 meV" analog) — make it a parameter; study sensitivity.
- The surrogate is calibrated for the **α (fcc) phase**; in a **local hydride** (B1) the
  volume↔energy relation differs → those regions must fall back to full EAM. → need phase
  detection or a conservative "always-EAM near high-c clusters" rule.

## 5. Idea C — dynamic re-evaluation for an evolving lattice

von Pezold had it easy: a **static** dislocation (far field pinned by linear elasticity, core
doesn't migrate), so volumes + active set were computed **once**. In a general finite-T MD-MC
the lattice moves — dislocations glide/multiply, hydrides nucleate and locally expand the
lattice, the box breathes under NPT — so per-site Voronoi volumes and the active set **drift**.

⇒ Periodically (natural cadence: the global/NPT step) **recompute volumes → re-classify active
set → hand off**:

- **Promote** newly-active sites from "frozen at surrogate" to explicit EAM sampling +
  re-equilibrate them locally.
- **Phase-changed** (hydride) regions → fall back to EAM (surrogate invalid there).
- Bookkeeping for sites entering/leaving the active set without breaking the Markov chain
  statistics. → this is the genuinely new part (beyond the paper); design carefully.

Open question: re-classification cadence vs cost vs statistical bias.

## 6. Idea D — parallelism (ranked by ROI)

1. **Parameter / chain parallelism** (best ROI, lowest risk, no kernel change): each (μ, seed)
   is independent → own process / sub-communicator with its own LAMMPS instance. Near-linear
   speedup of the whole isotherm.
2. **MPI within each run** (free via LAMMPS): build with MPI, launch under `mpirun` → the
   NPT/minimize and energy evals are domain-decomposed → large cells, faster blocks. Needs a
   **consistency check** of the Python driver across ranks (numpy `default_rng` same-seed is
   identical per rank ✓; verify the `gather`/create/delete collective paths). Marked a non-goal
   in SPEC §11 → revisit.
3. **Trade-off:** with R ranks, split between (1) more μ-points at once vs (2) bigger/faster
   single runs. Small cells favor (1); large cells favor (2).
4. **Local-ΔU / native C++ fix** (biggest per-move win, biggest effort): a rigorous local ΔU
   that skips the global `run 0` is essentially C++-fix territory (the LAMMPS *library* has no
   cheap partial force eval; EAM is many-body). Deferred; v0.1 stays its correctness oracle.

## 7. Idea E — native site-restricted GCMC fix in LAMMPS (the "hack the fix" route)

Instead of (or alongside) the Python driver, modify LAMMPS so that GCMC insertion draws from a
**precomputed, Voronoi-volume-filtered site list** instead of random positions in a region,
keeping it MPI-parallel. Full read-only source-analysis brief for Claude Code:
**`docs/LAMMPS-SITE-GCMC-BRIEF.md`** (grounded in the real `fix_gcmc.h` function names:
`attempt_atomic_insertion_full()`, `energy_full()`, `random_equal`, `sublo/subhi`, `zz`,
`volume`, …). Probably cleaner as a new `fix gcmc/site` than an in-place branch.

**Reference choice — RESOLVED (the lattice-gas vs ideal-gas thing is NOT a complication).**
Our Python driver already uses the **lattice-gas** acceptance, `min(1, exp(-β(ΔU ∓ μ)))`, with a
uniform single-site proposal (pick any of M sites; insert if empty, delete if occupied). The new
fix should adopt the **same convention natively** and define **its own μ reference** — we do NOT
try to reproduce `fix gcmc`'s ideal-gas reservoir normalization (`zz · volume / (ngas+1)`). This
is *simpler*, not harder: those three factors are simply **dropped**. The lattice μ differs from
an absolute/ideal μ only by a constant (≈ kT·ln(V_site/Λ³) + per-site reference/zero-point),
which we **calibrate once** — exactly the calibration constant the Python driver already defers.
Isotherm shapes and phase plateaus are reference-independent; you only need the absolute zero of
μ when comparing to an external scale (gas pressure, DFT μ₀). Validation then needs **no μ remap**
between the new fix, the analytic lattice-gas Langmuir, and the Python driver — all three share
the convention.

**What is *actually* hard (not the reference):**

- **Parallel simultaneous-move correctness.** Detailed balance with many concurrent
  insertions/deletions across MPI domains needs spatial separation (checkerboard / ≥ interaction
  range apart) so concurrent moves don't share affected-energy regions — the Sadigh/Erhart trick.
  This, not the acceptance formula, is the real care.
- **Per-domain site ownership + occupancy bookkeeping.** Each rank must know which listed sites
  lie in its subdomain (`sublo/subhi`), keep that valid under migration/load-balance, and track a
  per-site occupied/empty flag + stable `site_id ↔ atom tag` map (no double insertion; deletion
  picks uniformly among *occupied listed* sites for proposal symmetry).
- **EAM ⇒ `full_energy` (global per move).** `fix gcmc`'s fast `energy()` omits the EAM embedding
  term, so EAM forces `energy_full()` — a global O(N) energy per move, same floor as the Python
  driver's `run 0`. A site list does **not** lower this. The gains are C++/no-Python overhead,
  parallel full-energy over big cells, and high acceptance from site-targeting (no wasted
  overlapping insertions).

## 8. Open questions to settle before any v2 implementation

- [x] **Reference convention (RESOLVED):** adopt the lattice-gas acceptance natively with our own
      μ zero; calibrate μ₀ once; do not reproduce `fix gcmc`'s ideal-gas normalization. (See §7.)
- [ ] For Idea E: parallel simultaneous-move scheduling (checkerboard / spatial separation) and
      restart handling of the site/occupancy state.
- [ ] Minimizer + force tolerance for the per-flip local relaxation (CG vs FIRE; tol).
- [ ] Local-relaxation cutoff radius (relate to EAM cutoff / NN shells).
- [ ] How strict on detailed balance vs the local-relax/periodic-global approximation? quantify?
- [ ] Surrogate E^H(V): probe-atom Voronoi vs geometric; calibration protocol; α-only validity.
- [ ] Active-set threshold + re-classification cadence; hydride/phase fallback rule.
- [ ] Parallelism split policy (parameter vs domain) and the MPI-driver consistency test.
- [ ] What goes in the SPEC vs stays a runtime option.

## 9. References for these notes

- J. von Pezold, L. Lymperakis, J. Neugebauer, "Hydrogen-enhanced local plasticity at dilute
  bulk H concentrations: the role of H–H interactions and the formation of local hydrides,"
  *Acta Materialia* **59** (2011) 2969–2980. — the §2.4 / §2.4.1 template.
- D. Korbmacher et al., *Metals* **2018**, 8, 280. — 0 K statics anchor (current Test 3).
- B. Sadigh, P. Erhart et al., *Phys. Rev. B* **85**, 184203 (2012). — VC-SGC / parallel MC.
- E. Bitzek, P. Koskinen, F. Gähler, M. Moseler, P. Gumbsch, "Structural Relaxation Made
  Simple," *Phys. Rev. Lett.* **97**, 170201 (2006). — FIRE minimizer (candidate for §3).
