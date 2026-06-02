# MC-DRIVER — implementation progress

## Status (2026-06-01)

| SPEC step | Status | Notes |
|-----------|--------|-------|
| 1. Scaffold (`src/nih_mc/`, `tests/`, CLI) | ✅ done | `pyproject.toml`, `requirements.txt`, `nih-mc` entry point |
| 2. Structure loading + site list (§4) | ✅ done | `nih_mc.structure` |
| 3. LAMMPS plumbing + round-trip guard (§5, §12) | ✅ done | `nih_mc.lammps_runner` + `nih_mc.backends.lammps_statics` |
| 4. GC sampling + statics backend (§5a, §6) | ✅ done | `nih_mc.mc.run_gc`, `LammpsStaticsBackend` |
| 5. Unit tests (§10 tests 1 & 2 + guards) | ✅ all green | 23/23 |
| 6. Finite-T NPT backend + blocking (§5b, §7) | ✅ code done | `LammpsNptBackend`; physical validation deferred to step 8 (needs a real potential) |
| 7. VC-SGC + GC-overlap guard (§6, §10) | ✅ done | `nih_mc.mc.run_vcsgc`, both enumeration agreement and κ̄→0 GC equivalence verified |
| 8. Real-world Korbmacher anchor (§10 test 3) | ⏸ blocked | needs the EAM file `ni_h_rcut4.90_rcut2.eam.alloy` and the RELAXED 10×10×10 structures from LLM-LMPS thread 01 |

## Acceptance-suite results (all from `pytest tests/`)

23 tests, all passing:
- **Langmuir (§10 test 1)**: 7 (μ, T, ε) points + extreme-μ guards on a 200-site lattice; |Δθ| < 0.02.
- **Exact enumeration (§10 test 2)**: GC with attractive coupling on an 8-site ring; GC with repulsive coupling on a 12-site ring; VC-SGC matches its own enumeration on a 10-site ring; VC-SGC with κ̄=0 matches GC exactly (the SPEC's GC-overlap guard, in its mathematically exact form).
- **Round-trip guards (§10/§12)**: trivial backend (insert+delete returns U and N to baseline); LAMMPS API end-to-end via `pair_style zero` on a 2×2×2 lattice (empty→insert→delete, reject path, full→delete→re-insert), all verifying that unrelaxed `run 0` energies after a create/delete cycle reproduce the pre-cycle energy to ≤1e-9 eV.
- **End-to-end (LAMMPS+GC)**: a full GC sweep with `pair_style zero` on a 3×3×3 site list reproduces the Langmuir isotherm.
- **CLI smoke tests**: YAML config → CSV + run-summary YAML + log, for both `backend: trivial` and `backend: statics` with `pair_style zero`.

## Step 8 (Korbmacher anchor) — what's still needed

1. The EAM alloy file `ni_h_rcut4.90_rcut2.eam.alloy` (or whichever file the
   user designates; place it under a `POTDIR` and set `potential.potdir` in
   the config). MEAM is `NiH_KoShimLee.meam` + `NiH_KoShimLee_library.meam`.
2. The RELAXED `NiH-B1-10x10x10.data` and `Ni-fcc-10x10x10.data` from LLM-LMPS
   thread 01 (the ideal `tools/build_structures.py` files in this folder are
   the development/test stand-ins).
3. Run a μ sweep with `ensemble: gc`, `backend: statics`, `T_K: 300`, e.g.
   μ ∈ {-2.5, -2.4, -2.3, ..., -1.6}; fit homogeneous E(c) to extract μ₀, α, β
   and read off the two-phase plateau μ_M = -2.405 eV. Target precision: a few %
   vs. Korbmacher *Metals* 2018, 8, 280.
4. SPEC §12 risk: exercise the round-trip guard with **MEAM** before the full
   sweep — MEAM has had known quirks under repeated create/delete in LAMMPS.

## Design notes / minor surprises

- The LAMMPS Python binding leaked state across multiple `lammps()`
  constructions in the same Python process (bus error in `create_atoms` on
  the second instance, observed on macOS / Darwin 25.5). Workaround:
  `LammpsRunner` keeps a *single* class-level LAMMPS process alive and uses
  `command("clear")` to reset between runs. This is invisible to callers but
  is worth knowing if anyone refactors `lammps_runner.py`.
- For the small-system VC-SGC guard, the spec line "in a single-phase μ window
  VC-SGC ⟨c⟩ equals GC ⟨c⟩ within error" holds in the thermodynamic limit
  but in a 10-site lattice with finite κ̄ the constraint visibly shifts the
  mean (the distribution narrows asymmetrically). The test therefore uses the
  mathematically exact form of the guard: VC-SGC at κ̄=0 *is* GC at μ=Δμ. The
  algorithmic correctness check is separate: MC against an exact VC-SGC
  enumeration at finite κ̄.
- Per-flip ΔU is the unrelaxed `run 0` energy difference; relaxation
  (`minimize` for statics, `fix npt` for md_npt) happens once per MC block via
  the `relax_per_block` hook (SPEC §7). The relaxed energy and box are what
  end up in the per-block CSV row.
