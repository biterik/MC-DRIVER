"""LAMMPS NPT (finite-T) backend (SPEC §5b).

Differs from the statics backend only in the per-block "relax" step: an NPT
run (`fix npt`) replaces `minimize`. Per-flip ΔU is still the unrelaxed
`run 0` energy difference; one NPT sub-block per MC block does the thermal +
volume relaxation, and the resulting block-averaged ⟨pe⟩, ⟨P⟩ are reported.

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
from ..lammps_runner import LammpsRunner
from .lammps_statics import LammpsStaticsBackend


class LammpsNptBackend(LammpsStaticsBackend):
    """NPT-block backend; reuses statics ΔU + create/delete bookkeeping."""

    def __init__(self, runner: LammpsRunner, round_trip_tol_eV: float = 1.0e-6):
        super().__init__(runner, round_trip_tol_eV=round_trip_tol_eV)
        self._last_pressure_bar: float = 0.0

    def relax(self, state: State) -> None:
        pe_avg, pr_avg = self.runner.md_npt_block()
        self._pe = pe_avg
        self._last_pressure_bar = pr_avg

    @property
    def last_pressure_bar(self) -> float:
        return self._last_pressure_bar
