"""Energy backends (SPEC §5).

A backend implements:

  energy(state) -> (U_potential_eV, box)        # current total potential energy
  propose_insert(state, site_id) -> ΔU          # without committing
  propose_delete(state, site_id) -> ΔU
  commit_insert(state, site_id) -> ΔU
  commit_delete(state, site_id) -> ΔU
  rollback() -> None                            # discard last proposed move

For lattice-gas-style backends used by the analytic & enumeration tests, the
"relaxation" is a no-op and propose_*/commit_* are cheap. For the LAMMPS
backends, propose_* does the create/delete + minimize/MD; if not accepted, the
move is rolled back (delete then re-insert / re-create + re-minimize).

For correctness simplicity we use the simpler API used in this driver:

  try_flip(state, site_id, kind) -> ΔU          # returns ΔU for the *proposed* state
  accept() -> None                              # keep the proposed state
  reject() -> None                              # restore the prior state

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from .base import EnergyBackend
from .trivial import ConstantSiteEnergyBackend, NNLatticeGasBackend
from .lammps_statics import LammpsStaticsBackend
from .lammps_npt import LammpsNptBackend

__all__ = [
    "EnergyBackend",
    "ConstantSiteEnergyBackend",
    "NNLatticeGasBackend",
    "LammpsStaticsBackend",
    "LammpsNptBackend",
]
