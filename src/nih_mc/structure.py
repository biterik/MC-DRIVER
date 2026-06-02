"""Structure loading and site-list construction (SPEC §4).

The driver does NOT compute octahedral sites geometrically. The H positions of a
fully-occupied NiH (B1) structure ARE the ordered site list; the driver reads
them from a LAMMPS data file.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# atom types are fixed by convention (SPEC §3, tools/build_structures.py)
NI_TYPE = 1
H_TYPE = 2


@dataclass
class Box:
    """Orthogonal LAMMPS box (xlo,xhi,ylo,yhi,zlo,zhi)."""
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    @property
    def lengths(self) -> np.ndarray:
        return np.array(
            [self.xhi - self.xlo, self.yhi - self.ylo, self.zhi - self.zlo]
        )


@dataclass
class LammpsData:
    """Lightweight container for a parsed LAMMPS `atom_style atomic` data file."""
    box: Box
    # atoms is a list of tuples (lammps_id:int, type:int, x:float, y:float, z:float)
    atoms: list
    masses: dict  # type -> mass

    def positions(self, atom_type: int) -> np.ndarray:
        return np.array(
            [(x, y, z) for (_id, t, x, y, z) in self.atoms if t == atom_type],
            dtype=float,
        ).reshape(-1, 3)

    def lammps_ids(self, atom_type: int) -> np.ndarray:
        return np.array(
            [aid for (aid, t, _x, _y, _z) in self.atoms if t == atom_type],
            dtype=int,
        )


def read_lammps_data(path: str) -> LammpsData:
    """Parse a minimal LAMMPS data file written by tools/build_structures.py.

    Supports orthogonal boxes, `atom_style atomic` Atoms section, optional Masses.
    Lines beginning with '#' (after stripping) are comments.
    """
    with open(path, "r") as f:
        raw = f.readlines()

    # strip trailing comments and leading/trailing whitespace
    lines = [ln.split("#", 1)[0].rstrip() for ln in raw]

    n_atoms = None
    xlo = xhi = ylo = yhi = zlo = zhi = None
    masses: dict = {}
    atoms: list = []

    i = 0
    section = None
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue

        # global counts
        if ln.endswith(" atoms"):
            n_atoms = int(ln.split()[0])
            i += 1
            continue
        if ln.endswith(" atom types"):
            i += 1
            continue
        # box bounds
        if ln.endswith("xlo xhi"):
            parts = ln.split()
            xlo, xhi = float(parts[0]), float(parts[1])
            i += 1
            continue
        if ln.endswith("ylo yhi"):
            parts = ln.split()
            ylo, yhi = float(parts[0]), float(parts[1])
            i += 1
            continue
        if ln.endswith("zlo zhi"):
            parts = ln.split()
            zlo, zhi = float(parts[0]), float(parts[1])
            i += 1
            continue

        # section headers
        if ln in ("Masses", "Atoms", "Velocities"):
            section = ln
            i += 1
            # the original-format line "Atoms # atomic" was reduced to "Atoms"
            # after comment stripping above
            continue

        if section == "Masses":
            parts = ln.split()
            if len(parts) >= 2:
                masses[int(parts[0])] = float(parts[1])
            i += 1
            continue

        if section == "Atoms":
            parts = ln.split()
            # atom_style atomic: id type x y z
            aid = int(parts[0])
            atype = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            atoms.append((aid, atype, x, y, z))
            i += 1
            continue

        # unknown line: skip
        i += 1

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError(f"data file {path}: incomplete box bounds")
    if n_atoms is not None and len(atoms) != n_atoms:
        raise ValueError(
            f"data file {path}: header says {n_atoms} atoms, parsed {len(atoms)}"
        )

    return LammpsData(
        box=Box(xlo, xhi, ylo, yhi, zlo, zhi),
        atoms=atoms,
        masses=masses,
    )


@dataclass
class SiteList:
    """Ordered octahedral-site list extracted from a full NiH structure.

    `coords[i]` is the Cartesian position of site i; `site_id == i` is the
    stable index used throughout the driver.
    """
    coords: np.ndarray   # (M, 3)
    box: Box
    n_ni: int            # number of Ni atoms in the source structure

    @property
    def n_sites(self) -> int:
        return self.coords.shape[0]


def build_site_list(full_data: LammpsData) -> SiteList:
    """SPEC §4: H positions of the full hydride define the site list, in order.

    A full hydride must have N_H == N_Ni (one octahedral site per Ni); we check
    that here as a sanity guard.
    """
    h_pos = full_data.positions(H_TYPE)
    n_ni = int((np.array([t for (_id, t, *_xyz) in full_data.atoms]) == NI_TYPE).sum())
    if h_pos.shape[0] == 0:
        raise ValueError("structure_full has no H atoms; cannot build site list")
    if h_pos.shape[0] != n_ni:
        raise ValueError(
            f"full hydride should have N_H == N_Ni, got N_H={h_pos.shape[0]} N_Ni={n_ni}"
        )
    return SiteList(coords=h_pos.copy(), box=full_data.box, n_ni=n_ni)
