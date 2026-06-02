"""SPEC §10/§12 guard: insert+delete round-trip across the LAMMPS API.

Uses `pair_style zero` so no real potential is needed — this is purely a
correctness check on create/delete/neighbor-rebuild bookkeeping. The stronger
check (from-scratch energy vs. after-cycle energy) also runs here.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nih_mc.backends.lammps_statics import LammpsStaticsBackend
from nih_mc.lammps_runner import LammpsBackendConfig, LammpsRunner
from nih_mc.state import State
from nih_mc.structure import build_site_list, read_lammps_data


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_files(tmp_path_factory):
    """Build 2x2x2 NiH (32 Ni + 32 H) and Ni-only data files for round-trip tests."""
    d = tmp_path_factory.mktemp("rt")
    nih = d / "NiH-B1-2x2x2.data"
    ni = d / "Ni-fcc-2x2x2.data"
    subprocess.check_call([sys.executable, str(REPO / "tools" / "build_structures.py"),
                           "--structure", "nih", "--a", "3.738", "--n", "2", "--out", str(nih)])
    subprocess.check_call([sys.executable, str(REPO / "tools" / "build_structures.py"),
                           "--structure", "ni", "--a", "3.524", "--n", "2", "--out", str(ni)])
    return nih, ni


def _cfg_zero() -> LammpsBackendConfig:
    """`pair_style zero` configuration: trivial potential, just exercises the
    create/delete + neighbor-rebuild plumbing.

    pair_style zero requires a cutoff > 0; pair_coeff is per-type (here `* *`).
    """
    return LammpsBackendConfig(
        pair_style="zero 1.0",
        pair_coeff=["* *"],
        do_minimize=False,
    )


def test_lammps_loads_full_hydride(tiny_files):
    nih, _ = tiny_files
    sites = build_site_list(read_lammps_data(str(nih)))
    runner = LammpsRunner(str(nih), _cfg_zero())
    state = runner.prepare_state(sites, start_from="full")
    assert state.n_h == 32
    assert state.concentration == 1.0
    # all 32 sites should have a mapping to a real LAMMPS atom id
    assert set(state.site_to_atom.keys()) == set(range(32))
    runner.close()


def test_round_trip_empty_then_insert_delete(tiny_files):
    """Start from Ni-only structure, insert one H at site 0, delete it; energy
    and atom count must return to baseline."""
    nih, ni = tiny_files
    sites = build_site_list(read_lammps_data(str(nih)))  # site coords from full
    runner = LammpsRunner(str(ni), _cfg_zero())
    state = runner.prepare_state(sites, start_from="empty")
    backend = LammpsStaticsBackend(runner, round_trip_tol_eV=1.0e-9)
    U0, _ = backend.initial_energy(state)
    n0 = state.n_h

    # insertion (accepted)
    dU_in = backend.try_flip(state, 0)
    backend.accept()
    assert state.occ[0]
    assert state.n_h == n0 + 1
    pe_after_insert = backend.current_energy()
    assert pe_after_insert == pytest.approx(U0 + dU_in, abs=1.0e-9)

    # deletion of the same site (accepted)
    dU_out = backend.try_flip(state, 0)
    backend.accept()
    assert not state.occ[0]
    assert state.n_h == n0
    # back to baseline: U_after == U0 within tolerance
    assert backend.current_energy() == pytest.approx(U0, abs=1.0e-9)
    assert dU_in + dU_out == pytest.approx(0.0, abs=1.0e-9)
    runner.close()


def test_round_trip_reject_path(tiny_files):
    """Propose an insertion and reject it; backend must undo and energy must
    return exactly to pe_before (the reject() guard itself enforces this)."""
    nih, ni = tiny_files
    sites = build_site_list(read_lammps_data(str(nih)))
    runner = LammpsRunner(str(ni), _cfg_zero())
    state = runner.prepare_state(sites, start_from="empty")
    backend = LammpsStaticsBackend(runner, round_trip_tol_eV=1.0e-9)
    U0, _ = backend.initial_energy(state)
    n0 = state.n_h

    _ = backend.try_flip(state, 5)
    # state has occ[5]=True after try_flip
    assert state.occ[5]
    backend.reject()
    assert not state.occ[5]
    assert state.n_h == n0
    assert backend.current_energy() == pytest.approx(U0, abs=1.0e-9)
    runner.close()


def test_round_trip_full_hydride_delete_reinsert(tiny_files):
    """Start from full NiH; delete then re-insert one H; energy returns."""
    nih, _ = tiny_files
    sites = build_site_list(read_lammps_data(str(nih)))
    runner = LammpsRunner(str(nih), _cfg_zero())
    state = runner.prepare_state(sites, start_from="full")
    backend = LammpsStaticsBackend(runner, round_trip_tol_eV=1.0e-9)
    U0, _ = backend.initial_energy(state)
    n0 = state.n_h

    dU1 = backend.try_flip(state, 17)  # deletion
    backend.accept()
    assert not state.occ[17]

    dU2 = backend.try_flip(state, 17)  # re-insertion (accepted)
    backend.accept()
    assert state.occ[17]
    assert state.n_h == n0
    assert backend.current_energy() == pytest.approx(U0, abs=1.0e-9)
    assert dU1 + dU2 == pytest.approx(0.0, abs=1.0e-9)
    runner.close()
