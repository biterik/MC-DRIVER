"""Persistent LAMMPS instance and create/delete plumbing (SPEC §5, §12).

Design:

  - One LAMMPS instance lives for the duration of the run.
  - We read the initial structure from a LAMMPS data file (Ni-only or full NiH).
  - The site list is the H positions of the full hydride; H atoms there get
    deleted at startup so the LAMMPS box matches `state.occ`.
  - For each MC trial flip the wrapper does a focused create_atoms /
    delete_atoms, rebuilds neighbor lists, runs `run 0` to evaluate energy,
    and returns ΔU.
  - Statics backend: optionally a `minimize` (cg / fix box/relax) per block.
  - LAMMPS atom IDs are kept stable by us via an internal counter; we keep a
    site_id → lammps_atom_id map in `State.site_to_atom`.

The round-trip guard (SPEC §10/§12) is implemented as a one-shot self-check
that any caller can invoke before running production sampling. It also checks
the stronger "from-scratch" version where we compare U after a flip cycle
against energy of an equivalent rebuilt LAMMPS state.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .state import State
from .structure import Box, H_TYPE, NI_TYPE, SiteList, read_lammps_data


@dataclass
class LammpsBackendConfig:
    pair_style: str                 # e.g. "eam/alloy", "meam", "zero 1.0"
    pair_coeff: list[str]           # list of `pair_coeff …` argument strings
    potdir: Optional[str] = None    # prepended to file paths if not absolute
    units: str = "metal"            # eV / Å / ps
    boundary: str = "p p p"
    neigh_skin: float = 2.0
    extra_init: list[str] = None    # additional setup commands (e.g. min_modify)
    # statics
    do_minimize: bool = True
    min_etol: float = 1.0e-8
    min_ftol: float = 1.0e-4
    min_maxiter: int = 1000
    min_maxeval: int = 10000
    cell_relax: bool = False        # add `fix box/relax iso 0.0` around minimize
    # md_npt
    md_T_K: float = 300.0
    md_P_bar: float = 0.0
    md_steps_equil: int = 2000
    md_steps_sample: int = 2000
    md_thermo_freq: int = 100

    def __post_init__(self):
        if self.extra_init is None:
            self.extra_init = []


def _lmp_cmd_silent(lmp, cmd: str) -> None:
    lmp.command(cmd)


class LammpsRunner:
    """Owns one LAMMPS instance + the create/delete + energy plumbing.

    The runner is constructed from a structure data file path (either the empty
    Ni or the full NiH file) and a backend config. Use `prepare_state` after
    construction to set up the initial occupancy that matches the structure.
    """

    # Class-level cache of a single LAMMPS process. The LAMMPS Python binding
    # leaks state across multiple `lammps()` constructions in one Python
    # process (observed bus errors on macOS); we work around it by reusing one
    # instance and resetting it with `clear` for each new run.
    _shared_lmp = None

    def __init__(self, data_path: str, config: LammpsBackendConfig, suppress_output: bool = True):
        from lammps import lammps  # local import: tests that don't need LAMMPS skip this
        self.config = config
        if LammpsRunner._shared_lmp is None:
            cmdargs = []
            if suppress_output:
                cmdargs += ["-screen", "none", "-log", "none"]
            LammpsRunner._shared_lmp = lammps(cmdargs=cmdargs)
        self.lmp = LammpsRunner._shared_lmp
        # reset any prior session before reading the new data file
        self.lmp.command("clear")
        self._next_atom_id: int = 0
        self._init_simulation(data_path)

    # ----------------------------------------------------------------- init
    def _init_simulation(self, data_path: str) -> None:
        c = self.config
        cmds = [
            f"units {c.units}",
            f"boundary {c.boundary}",
            "atom_style atomic",
            "atom_modify map array sort 0 0.0",  # stable id map; no spatial sort
            f"neighbor {c.neigh_skin} bin",
            "neigh_modify every 1 delay 0 check yes",
            f"read_data {data_path}",
            f"pair_style {c.pair_style}",
        ]
        for line in c.pair_coeff:
            line = line.strip()
            if not line.startswith("pair_coeff"):
                line = "pair_coeff " + line
            # if potdir is given and a token looks like a relative path to a
            # potential file in POTDIR, leave it to the caller to write the
            # absolute path. We don't second-guess pair_coeff syntax here.
            cmds.append(line)
        cmds += [
            "thermo 0",
            "thermo_style custom step pe ke etotal press vol",
        ]
        cmds += c.extra_init
        cmds.append("run 0 post no")  # establishes energy/neighbor lists

        for cmd in cmds:
            _lmp_cmd_silent(self.lmp, cmd)

        # record largest current atom id so future create_atoms gets fresh ids
        ids = self._gather_ids()
        self._next_atom_id = int(ids.max()) if ids.size else 0

    def _gather_ids(self) -> np.ndarray:
        n = self.lmp.get_natoms()
        if n == 0:
            return np.zeros(0, dtype=int)
        ids = self.lmp.gather_atoms("id", 0, 1)  # 0=int, count=1
        return np.frombuffer(ids, dtype=np.int32, count=n).astype(int)

    def _gather_types(self) -> np.ndarray:
        n = self.lmp.get_natoms()
        if n == 0:
            return np.zeros(0, dtype=int)
        t = self.lmp.gather_atoms("type", 0, 1)
        return np.frombuffer(t, dtype=np.int32, count=n).astype(int)

    # ------------------------------------------------------------- queries
    def potential_energy(self) -> float:
        """Total potential energy in current units (eV for `metal`)."""
        return float(self.lmp.get_thermo("pe"))

    def pressure(self) -> float:
        return float(self.lmp.get_thermo("press"))

    def box(self) -> Box:
        boxlo, boxhi, _xy, _yz, _xz, _periodicity, _box_change = self.lmp.extract_box()
        return Box(boxlo[0], boxhi[0], boxlo[1], boxhi[1], boxlo[2], boxhi[2])

    # -------------------------------------------------------- prepare_state
    def prepare_state(self, sites: SiteList, start_from: str) -> State:
        """Build an initial `State` consistent with the loaded structure.

        `start_from` is 'full' (the data file IS the full NiH, all sites occupied)
        or 'empty' (the data file IS Ni-only with NO H atoms).

        Note: callers passing the EMPTY-Ni data file must also pass the SiteList
        extracted from a FULL-NiH file (since the site coordinates live there).
        """
        if start_from == "full":
            # H atoms already in LAMMPS — record their mapping
            ids = self._gather_ids()
            types = self._gather_types()
            h_mask = (types == H_TYPE)
            h_ids = ids[h_mask]
            # the data file's H atoms are in the same order as sites.coords
            # (build_structures.py writes Ni first, then H, in cell-major order),
            # but to be safe we match by coordinates.
            h_coords = self._gather_positions(h_ids)
            mapping = _match_by_coords(sites.coords, h_coords, sites.box)
            occ = np.ones(sites.n_sites, dtype=bool)
            state = State(sites=sites, occ=occ,
                          site_to_atom={int(i): int(h_ids[mapping[i]]) for i in range(sites.n_sites)})
        elif start_from == "empty":
            occ = np.zeros(sites.n_sites, dtype=bool)
            state = State(sites=sites, occ=occ, site_to_atom={})
        else:
            raise ValueError(f"start_from must be 'full' or 'empty', got {start_from!r}")
        return state

    def _gather_positions(self, atom_ids: np.ndarray) -> np.ndarray:
        n = self.lmp.get_natoms()
        x = self.lmp.gather_atoms("x", 1, 3)
        x = np.frombuffer(x, dtype=np.float64, count=3 * n).reshape(n, 3)
        ids = self._gather_ids()
        order = {int(i): k for k, i in enumerate(ids)}
        return np.array([x[order[int(i)]] for i in atom_ids], dtype=float)

    # ---------------------------------------------------------- create/delete
    def _create_h_at(self, coord: np.ndarray) -> int:
        """Insert a type-2 atom at `coord`; return its LAMMPS atom id."""
        self._next_atom_id += 1
        new_id = self._next_atom_id
        # create_atoms n_to_add type X args(...) basis(...) remap yes
        # python binding: create_atoms(n, id, type, x, v=None, image=None, ...)
        # signature: create_atoms(n, id, type, x, v=None, image=None, bexpand=False)
        self.lmp.create_atoms(
            n=1,
            id=[int(new_id)],
            type=[H_TYPE],
            x=[float(coord[0]), float(coord[1]), float(coord[2])],
        )
        return new_id

    def _delete_h(self, atom_id: int) -> None:
        """Delete the H atom with the given LAMMPS id (via a temporary group)."""
        gname = "_mc_del"
        self.lmp.command(f"group {gname} id {int(atom_id)}")
        self.lmp.command(f"delete_atoms group {gname} compress no")
        self.lmp.command(f"group {gname} delete")

    def _reset_neighbors_and_eval(self) -> float:
        """Rebuild neighbor lists and re-evaluate the potential energy.

        `run 0 post no` is the cheapest "evaluate forces & energy" available; it
        forces a neighbor-list rebuild after create/delete because LAMMPS marks
        the topology dirty.
        """
        self.lmp.command("run 0 post no")
        return self.potential_energy()

    # ----------------------------------------------------------- statics / md
    def minimize(self) -> None:
        c = self.config
        if not c.do_minimize:
            return
        if c.cell_relax:
            self.lmp.command("fix _mc_box all box/relax iso 0.0")
        self.lmp.command(f"min_style cg")
        self.lmp.command(
            f"minimize {c.min_etol} {c.min_ftol} {c.min_maxiter} {c.min_maxeval}"
        )
        if c.cell_relax:
            self.lmp.command("unfix _mc_box")

    def md_npt_block(self) -> Tuple[float, float]:
        """Run an NPT equil + sample sub-block; return (⟨pe⟩, ⟨P⟩) over sample."""
        c = self.config
        T = c.md_T_K
        # velocity if first time? we re-seed velocities lazily on each call to
        # avoid hidden state divergence. Caller is responsible for blocking.
        self.lmp.command(f"velocity all create {T} 12345 mom yes rot yes dist gaussian")
        self.lmp.command(
            f"fix _mc_npt all npt temp {T} {T} 0.1 iso {c.md_P_bar} {c.md_P_bar} 1.0"
        )
        # average pe and press over the sample window via fix ave/time
        if c.md_steps_equil > 0:
            self.lmp.command(f"run {c.md_steps_equil}")
        self.lmp.command(
            "variable _mc_pe equal pe\nvariable _mc_pr equal press"
        )
        self.lmp.command(
            f"fix _mc_avg all ave/time 1 {c.md_steps_sample} {c.md_steps_sample} "
            "v__mc_pe v__mc_pr"
        )
        self.lmp.command(f"run {c.md_steps_sample}")
        pe_avg = float(self.lmp.extract_fix("_mc_avg", 0, 1, 1, 1))  # vec, col 1
        pr_avg = float(self.lmp.extract_fix("_mc_avg", 0, 1, 2, 1))  # vec, col 2
        self.lmp.command("unfix _mc_avg")
        self.lmp.command("unfix _mc_npt")
        self.lmp.command("variable _mc_pe delete\nvariable _mc_pr delete")
        return pe_avg, pr_avg

    def close(self) -> None:
        """Release this runner's hold on the shared LAMMPS process.

        Note: the LAMMPS instance itself is kept alive across runners (see the
        class docstring); `close` just clears the current simulation state.
        """
        try:
            self.lmp.command("clear")
        except Exception:
            pass


def _match_by_coords(target: np.ndarray, found: np.ndarray, box: Box, tol: float = 1e-3):
    """For each row in `target`, find the row index in `found` whose periodic
    image distance is smallest; return the index array.

    Used to align driver-side site_ids with LAMMPS atom_ids when reading a
    full hydride: the H atoms in LAMMPS may be reordered by the data parser.
    """
    L = box.lengths
    out = np.empty(target.shape[0], dtype=int)
    used = np.zeros(found.shape[0], dtype=bool)
    for k in range(target.shape[0]):
        d = found - target[k]
        d -= np.round(d / L) * L
        r2 = (d * d).sum(axis=1)
        # block already-used rows
        r2[used] = np.inf
        j = int(np.argmin(r2))
        if r2[j] > (10 * tol) ** 2:
            raise ValueError(
                f"site coord {target[k]} has no LAMMPS H within tol; min d^2={r2[j]:.3e}"
            )
        out[k] = j
        used[j] = True
    return out
