"""Energy-backend abstract interface (SPEC §5).

The MC layer calls only this interface; concrete backends (trivial / LAMMPS-statics /
LAMMPS-md_npt) implement it.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from ..state import State
from ..structure import Box


class EnergyBackend(ABC):
    """Pluggable backend; see backends/__init__.py for the contract."""

    @abstractmethod
    def initial_energy(self, state: State) -> Tuple[float, Box]:
        """Energy + box of the current (committed) state."""

    @abstractmethod
    def try_flip(self, state: State, site_id: int) -> float:
        """Tentatively flip occ[site_id] and return ΔU = U_after − U_before.

        Caller must follow up with accept() or reject() before the next try_flip.
        Implementations should leave the *driver-visible* state.occ in the proposed
        configuration so the MC layer can read it on acceptance; on reject(), the
        state must be restored bit-for-bit (occ + LAMMPS atoms + energy).
        """

    @abstractmethod
    def accept(self) -> None:
        """Commit the last proposed flip."""

    @abstractmethod
    def reject(self) -> None:
        """Roll back the last proposed flip."""

    @abstractmethod
    def current_energy(self) -> float:
        """Last committed total potential energy (eV)."""

    @abstractmethod
    def current_box(self) -> Box:
        """Last committed simulation box."""
