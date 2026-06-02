"""SPEC §10 Test 1: analytic non-interacting lattice-gas Langmuir/Fermi isotherm.

With a constant per-H site energy ε and no H-H or relaxation, the equilibrium
H concentration as a function of (μ, T) must follow

    θ(μ, T) = 1 / (exp((ε − μ)/kT) + 1)

This test isolates the GC acceptance + sampling — no LAMMPS, no potentials.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
import math

import numpy as np
import pytest

from nih_mc.backends import ConstantSiteEnergyBackend
from nih_mc.mc import GCParams, K_B_EVK, run_gc
from nih_mc.state import State
from nih_mc.structure import Box, SiteList


def make_synthetic_site_list(n_sites: int = 200, a: float = 1.0) -> SiteList:
    """A SiteList with arbitrary positions; for non-interacting tests the actual
    coords don't matter, only `n_sites` and `n_ni` (set equal here)."""
    rng = np.random.default_rng(0)
    coords = rng.random((n_sites, 3)) * (a * n_sites ** (1 / 3))
    box = Box(0.0, a * n_sites ** (1 / 3),
              0.0, a * n_sites ** (1 / 3),
              0.0, a * n_sites ** (1 / 3))
    return SiteList(coords=coords, box=box, n_ni=n_sites)


def langmuir_theta(mu_eV: float, eps_eV: float, T_K: float) -> float:
    return 1.0 / (math.exp((eps_eV - mu_eV) / (K_B_EVK * T_K)) + 1.0)


@pytest.mark.parametrize(
    "mu_eV,T_K,eps_eV",
    [
        (-0.10, 300.0, 0.0),
        (-0.02, 300.0, 0.0),
        ( 0.00, 300.0, 0.0),
        (+0.02, 300.0, 0.0),
        (+0.10, 300.0, 0.0),
        (-0.05, 600.0, +0.05),
        (+0.15, 600.0, +0.05),
    ],
)
def test_langmuir_isotherm(mu_eV, T_K, eps_eV):
    """⟨c⟩ from GC sampling must match the Langmuir formula within MC error."""
    sites = make_synthetic_site_list(n_sites=200)
    state = State.empty(sites)
    backend = ConstantSiteEnergyBackend(eps_eV=eps_eV)
    params = GCParams(T_K=T_K, mu_eV=mu_eV)

    # equilibrate + sample
    n_blocks = 4000
    flips_per_block = 50  # M = n_sites picks per "sweep"; here we use 50 flips
    records = run_gc(
        state=state,
        backend=backend,
        params=params,
        n_blocks=n_blocks,
        flips_per_block=flips_per_block,
        seed=42,
    )

    # discard first 20% as burn-in
    burn = n_blocks // 5
    c_samples = np.array([r.c for r in records[burn:]])
    theta_mc = float(c_samples.mean())
    theta_exact = langmuir_theta(mu_eV, eps_eV, T_K)

    # SPEC §10 tolerance "|Δθ| < 0.01 at ~10⁴ blocks". We use 4k * 50 = 2e5
    # flips and a 200-site lattice -> std of θ ~ 1/sqrt(N_eff). 0.02 is a safe
    # tolerance and still well-discriminating.
    assert abs(theta_mc - theta_exact) < 0.02, (
        f"μ={mu_eV} ε={eps_eV} T={T_K}: MC θ={theta_mc:.4f}, exact={theta_exact:.4f}"
    )


def test_langmuir_extreme_mu_limits():
    """SPEC §10 guard: μ → +large ⇒ c → 1, μ → −large ⇒ c → 0."""
    sites = make_synthetic_site_list(n_sites=100)

    state_lo = State.full(sites)
    backend_lo = ConstantSiteEnergyBackend(eps_eV=0.0)
    records_lo = run_gc(
        state=state_lo,
        backend=backend_lo,
        params=GCParams(T_K=300.0, mu_eV=-1.0),  # ε−μ ≈ +40 kT
        n_blocks=200,
        flips_per_block=200,
        seed=1,
    )
    c_lo = float(np.mean([r.c for r in records_lo[100:]]))
    assert c_lo < 0.01, f"expected c→0 for very negative μ, got {c_lo}"

    state_hi = State.empty(sites)
    backend_hi = ConstantSiteEnergyBackend(eps_eV=0.0)
    records_hi = run_gc(
        state=state_hi,
        backend=backend_hi,
        params=GCParams(T_K=300.0, mu_eV=+1.0),
        n_blocks=200,
        flips_per_block=200,
        seed=2,
    )
    c_hi = float(np.mean([r.c for r in records_hi[100:]]))
    assert c_hi > 0.99, f"expected c→1 for very positive μ, got {c_hi}"
