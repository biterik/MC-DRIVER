"""Occupancy state on the octahedral site sublattice (SPEC §4).

`State` carries the boolean occupancy array and a stable site_id → lammps_atom_id
map for occupied sites. The actual LAMMPS atom_id is assigned by the energy
backend at insertion time; for the analytic (non-LAMMPS) tests the map is unused.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .structure import SiteList


@dataclass
class State:
    sites: SiteList
    occ: np.ndarray                          # bool array, shape (M,)
    site_to_atom: dict = field(default_factory=dict)  # site_id -> lammps_atom_id

    @classmethod
    def empty(cls, sites: SiteList) -> "State":
        return cls(sites=sites, occ=np.zeros(sites.n_sites, dtype=bool))

    @classmethod
    def full(cls, sites: SiteList) -> "State":
        return cls(sites=sites, occ=np.ones(sites.n_sites, dtype=bool))

    @property
    def n_h(self) -> int:
        return int(self.occ.sum())

    @property
    def n_sites(self) -> int:
        return self.sites.n_sites

    @property
    def concentration(self) -> float:
        """c = N_H / N_Ni (SPEC §1). N_Ni == n_sites for this sublattice."""
        return self.n_h / max(self.sites.n_ni, 1)
