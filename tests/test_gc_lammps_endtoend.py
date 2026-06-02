"""End-to-end GC run through the LAMMPS-statics backend, with pair_style zero.

The trivial potential makes the physics equivalent to a non-interacting lattice
gas (ε = 0). This exercises the full MC loop + LAMMPS create/delete on real
LAMMPS atoms, and the resulting θ(μ,T) must still match Langmuir.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nih_mc.backends.lammps_statics import LammpsStaticsBackend
from nih_mc.lammps_runner import LammpsBackendConfig, LammpsRunner
from nih_mc.mc import GCParams, K_B_EVK, run_gc
from nih_mc.structure import build_site_list, read_lammps_data


REPO = Path(__file__).resolve().parents[1]


def langmuir(mu_eV, eps_eV, T_K):
    return 1.0 / (math.exp((eps_eV - mu_eV) / (K_B_EVK * T_K)) + 1.0)


@pytest.fixture(scope="module")
def small_nih(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e") / "NiH-3x3x3.data"
    subprocess.check_call(
        [sys.executable, str(REPO / "tools" / "build_structures.py"),
         "--structure", "nih", "--a", "3.738", "--n", "3", "--out", str(out)]
    )
    return out


def _cfg_zero():
    return LammpsBackendConfig(
        pair_style="zero 1.0", pair_coeff=["* *"], do_minimize=False
    )


def test_gc_with_lammps_zero_potential_matches_langmuir(small_nih):
    """End-to-end: GC sampling through LAMMPS create/delete reproduces Langmuir."""
    sites = build_site_list(read_lammps_data(str(small_nih)))
    # start from full hydride so all 108 atoms are pre-loaded into LAMMPS
    runner = LammpsRunner(str(small_nih), _cfg_zero())
    state = runner.prepare_state(sites, start_from="full")
    backend = LammpsStaticsBackend(runner)

    T = 400.0
    mu = -0.05  # ε−μ ≈ +1.45 kT ⇒ θ ≈ 0.187
    records = run_gc(
        state=state, backend=backend,
        params=GCParams(T_K=T, mu_eV=mu),
        n_blocks=2000, flips_per_block=20, seed=99,
    )
    burn = 500
    c = float(np.mean([r.c for r in records[burn:]]))
    c_exact = langmuir(mu, 0.0, T)
    assert abs(c - c_exact) < 0.03, f"end-to-end GC θ={c:.4f}, exact Langmuir={c_exact:.4f}"
    runner.close()
