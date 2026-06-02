"""Trivial (no-LAMMPS) energy backends used by the unit tests (SPEC §10 tests 1-2).

* ConstantSiteEnergyBackend: every H carries a fixed site energy ε. No relaxation,
  no H-H interactions. ΔU on insertion is +ε, on deletion is -ε. The GC isotherm
  must then match the Langmuir/Fermi form θ = 1/(exp((ε−μ)/kT)+1).

* NNLatticeGasBackend: site energy ε per H + pair coupling J per occupied
  nearest-neighbor pair. A tiny rigid lattice with an explicit NN list is
  exactly enumerable.

Both backends:
  - own state.occ during try_flip/accept/reject (they flip it forward and revert
    it on reject, so the MC layer never has to snapshot it),
  - report a constant box equal to the input structure's box.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from typing import Tuple

from ..state import State
from ..structure import Box
from .base import EnergyBackend


class ConstantSiteEnergyBackend(EnergyBackend):
    """Lattice gas: U = ε · N_H. Used for the analytic Langmuir test."""

    def __init__(self, eps_eV: float = 0.0):
        self.eps = float(eps_eV)
        self._U: float = 0.0
        self._box: Box | None = None
        self._pending: tuple[int, float, State] | None = None

    def initial_energy(self, state: State) -> Tuple[float, Box]:
        self._U = self.eps * state.n_h
        self._box = state.sites.box
        return self._U, self._box

    def try_flip(self, state: State, site_id: int) -> float:
        if self._pending is not None:
            raise RuntimeError("try_flip called twice without accept/reject")
        i = int(site_id)
        # flip; ΔU = +ε on insertion (now True), -ε on deletion (now False)
        state.occ[i] = ~state.occ[i]
        dU = self.eps if state.occ[i] else -self.eps
        self._pending = (i, dU, state)
        return dU

    def accept(self) -> None:
        if self._pending is None:
            raise RuntimeError("accept without a pending flip")
        _, dU, _ = self._pending
        self._U += dU
        self._pending = None

    def reject(self) -> None:
        if self._pending is None:
            raise RuntimeError("reject without a pending flip")
        i, _dU, state = self._pending
        state.occ[i] = ~state.occ[i]  # revert
        self._pending = None

    def current_energy(self) -> float:
        return self._U

    def current_box(self) -> Box:
        assert self._box is not None
        return self._box


class NNLatticeGasBackend(EnergyBackend):
    """Lattice gas with NN H-H coupling.

    U = ε · N_H + J · Σ_<ij> occ_i occ_j

    The NN list is supplied explicitly (driver-provided), so the backend stays
    agnostic to the underlying lattice type.
    """

    def __init__(self, eps_eV: float, J_eV: float, neighbors: list[list[int]]):
        self.eps = float(eps_eV)
        self.J = float(J_eV)
        self.neighbors = [tuple(int(j) for j in lst) for lst in neighbors]
        self._U: float = 0.0
        self._box: Box | None = None
        self._pending: tuple[int, float, State] | None = None

    def initial_energy(self, state: State) -> Tuple[float, Box]:
        u_site = self.eps * state.n_h
        u_pair = 0.0
        for i, nbrs in enumerate(self.neighbors):
            if not state.occ[i]:
                continue
            for j in nbrs:
                if j > i and state.occ[j]:
                    u_pair += self.J
        self._U = u_site + u_pair
        self._box = state.sites.box
        return self._U, self._box

    def try_flip(self, state: State, site_id: int) -> float:
        if self._pending is not None:
            raise RuntimeError("try_flip called twice without accept/reject")
        i = int(site_id)
        currently = bool(state.occ[i])
        n_occ_nbrs = sum(1 for j in self.neighbors[i] if state.occ[j])
        if currently:
            dU = -self.eps - self.J * n_occ_nbrs   # deletion
        else:
            dU = +self.eps + self.J * n_occ_nbrs   # insertion
        state.occ[i] = ~state.occ[i]
        self._pending = (i, dU, state)
        return dU

    def accept(self) -> None:
        if self._pending is None:
            raise RuntimeError("accept without a pending flip")
        _, dU, _ = self._pending
        self._U += dU
        self._pending = None

    def reject(self) -> None:
        if self._pending is None:
            raise RuntimeError("reject without a pending flip")
        i, _dU, state = self._pending
        state.occ[i] = ~state.occ[i]
        self._pending = None

    def current_energy(self) -> float:
        return self._U

    def current_box(self) -> Box:
        assert self._box is not None
        return self._box
