"""LAMMPS statics backend (SPEC §5a).

Per-flip ΔU uses an unrelaxed `run 0` energy evaluation after create/delete +
neighbor rebuild. A relaxation (`minimize`, optionally with `fix box/relax`)
runs once per MC block (SPEC §7) via `relax_per_block`; the relaxed energy
and (when cell_relax is on) the relaxed box are reported.

`try_flip` mutates `state.occ` and the LAMMPS atom set; `reject()` reverses
both so the state is bit-for-bit restored before the next try.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..state import State
from ..structure import Box
from ..lammps_runner import LammpsRunner
from .base import EnergyBackend


@dataclass
class _Pending:
    state: State
    site_id: int
    was_occupied: bool
    coord: np.ndarray
    new_atom_id: Optional[int]   # set on insertion (the id we created)
    old_atom_id: Optional[int]   # set on deletion (the id we removed)
    pe_before: float
    pe_after: float


class LammpsStaticsBackend(EnergyBackend):
    """Adapter implementing EnergyBackend over a LammpsRunner."""

    def __init__(self, runner: LammpsRunner, round_trip_tol_eV: float = 1.0e-8):
        self.runner = runner
        self.round_trip_tol = float(round_trip_tol_eV)
        self._pe: float = 0.0
        self._pending: Optional[_Pending] = None

    def initial_energy(self, state: State) -> Tuple[float, Box]:
        self._pe = self.runner.potential_energy()
        return self._pe, self.runner.box()

    def try_flip(self, state: State, site_id: int) -> float:
        if self._pending is not None:
            raise RuntimeError("try_flip called twice without accept/reject")
        i = int(site_id)
        coord = state.sites.coords[i].copy()
        was = bool(state.occ[i])
        pe_before = self._pe

        if was:
            aid = state.site_to_atom.pop(i)
            self.runner._delete_h(aid)
            pe_after = self.runner._reset_neighbors_and_eval()
            self._pending = _Pending(state, i, True, coord, None, aid, pe_before, pe_after)
        else:
            new_id = self.runner._create_h_at(coord)
            state.site_to_atom[i] = new_id
            pe_after = self.runner._reset_neighbors_and_eval()
            self._pending = _Pending(state, i, False, coord, new_id, None, pe_before, pe_after)

        state.occ[i] = ~state.occ[i]
        return pe_after - pe_before

    def accept(self) -> None:
        if self._pending is None:
            raise RuntimeError("accept without a pending flip")
        self._pe = self._pending.pe_after
        self._pending = None

    def reject(self) -> None:
        if self._pending is None:
            raise RuntimeError("reject without a pending flip")
        p = self._pending

        if p.was_occupied:
            new_id = self.runner._create_h_at(p.coord)
            p.state.site_to_atom[p.site_id] = new_id
        else:
            assert p.new_atom_id is not None
            self.runner._delete_h(p.new_atom_id)
            p.state.site_to_atom.pop(p.site_id, None)
        pe_check = self.runner._reset_neighbors_and_eval()

        # round-trip guard: unrelaxed energy with the same N_H configuration
        # must be reproducible across a delete/insert cycle.
        if abs(pe_check - p.pe_before) > self.round_trip_tol * max(1.0, abs(p.pe_before)):
            raise RuntimeError(
                f"round-trip reject failed: pe={pe_check:.10g} expected={p.pe_before:.10g}"
            )

        p.state.occ[p.site_id] = ~p.state.occ[p.site_id]
        self._pe = p.pe_before
        self._pending = None

    def relax(self, state: State) -> None:
        """Per-block relaxation (called by the MC loop's relax_per_block hook)."""
        self.runner.minimize()
        self._pe = self.runner.potential_energy()

    def current_energy(self) -> float:
        return self._pe

    def current_box(self) -> Box:
        return self.runner.box()
