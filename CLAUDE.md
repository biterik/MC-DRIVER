# MC-DRIVER — Claude Code working instructions

You are implementing a **site-list Monte-Carlo driver** for hydrogen on the octahedral
sublattice of fcc Ni / rock-salt NiH, coupled to LAMMPS. This folder is self-contained:
everything you need is here. **Do not read or modify anything outside this folder.**

## Read first

1. `MC-DRIVER-SPEC.md` — the complete spec: data model, energy backends, MC moves &
   acceptance (GC and VC-SGC), outputs, config, and the acceptance test suite. It is the
   source of truth. If anything here and the spec disagree, follow the spec.
2. `config.example.yaml` — an example run configuration.
3. `tools/build_structures.py` — generates the input structures (see below).

## The task, in order (do not skip ahead)

1. **Scaffold** a small Python package (suggested: `src/nih_mc/`, `tests/`, a CLI entry
   point) plus a `requirements.txt`. Keep it dependency-light: numpy, pyyaml, the LAMMPS
   python module. No web services, no heavy frameworks.
2. **Structure loading + site list** (SPEC §4): load a LAMMPS data file, extract the H-atom
   positions as the ordered octahedral site list, build the occupancy state.
3. **LAMMPS plumbing** (SPEC §5): a persistent LAMMPS instance via `from lammps import
   lammps`; create/delete an H atom at a given site; rebuild neighbor lists; read potential
   energy and box. Verify the **round-trip guard** immediately (insert then delete a site
   returns energy + count to baseline) — this is the main integration risk.
4. **GC sampling + statics backend** (SPEC §5a, §6): the symmetric single-site flip move and
   the lattice-gas GC acceptance.
5. **Unit tests** (SPEC §10, tests 1 & 2 + guards): the analytic Langmuir isotherm and the
   small-system exact enumeration. **These require NO real potential and NO cluster** — get
   them green before going further.
6. **Finite-T NPT backend** (SPEC §5b) + blocking/cadence (SPEC §7).
7. **VC-SGC** (SPEC §6, Sadigh & Erhart PRB 85, 184203 (2012)) + the GC-overlap guard.
8. **Real-world anchor** (SPEC §10, test 3): needs the EAM potential file and the structures
   — this is the compute-bound stage; do it last.

## Inputs you must obtain or generate

- **Structures.** For development and unit tests, generate ideal structures yourself:
  ```
  python tools/build_structures.py --structure nih --a 3.738 --n 10 --out NiH-B1-10x10x10.data
  python tools/build_structures.py --structure ni  --a 3.524 --n 10 --out Ni-fcc-10x10x10.data
  ```
  The full-hydride file's H positions are the canonical site list. For the production /
  anchor runs, the user will provide RELAXED structures (from LLM-LMPS thread 01) to swap in.
- **Potential files** (only for the anchor stage, step 8): `ni_h_rcut4.90_rcut2.eam.alloy`
  (EAM), `NiH_KoShimLee.meam` + `NiH_KoShimLee_library.meam` (MEAM). The user provides these
  and a `POTDIR` path. The unit tests in step 5 do not need them.

## House rules (match the parent project)

- **Descriptive filenames** — no `out.dat`, `test.py`, `run1/`. Outputs follow SPEC §8
  (e.g. `isotherm-EAM-T300K-GC.csv`).
- **Explicit precision** in all numeric output per SPEC §8 (c `%.6f`, energies/μ `%.6f` eV,
  lattice constant `%.5f` Å, viz dumps `%.4f` Å).
- **Reproducibility** — every run records its RNG seed; (seed + config + input structure)
  must reproduce a run exactly.
- Verify LAMMPS API calls against the **installed** LAMMPS version's docs; don't assume.

## Definition of done

Steps 1–7 implemented with tests 1 & 2 + all guards green (no cluster needed). Step 8 (the
Korbmacher anchor) validated when the EAM structures + potential are available. Report
acceptance ratios and timings; flag if MEAM misbehaves under repeated create/delete
(SPEC §12).
