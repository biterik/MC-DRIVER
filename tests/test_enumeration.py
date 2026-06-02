"""SPEC §10 Test 2: small-system exact enumeration.

Tiny rigid lattice (M ≤ 16) with site energy ε and NN H-H coupling J. We
enumerate all 2^M states for the exact ⟨c⟩ and full P(c) at chosen (μ, T) and
require the MC sampler to reproduce them within statistical error. Run for both
GC and (with matched conditions) VC-SGC.

The "matched conditions" guard: in a single-phase μ window, VC-SGC with c₀ = ⟨c⟩_GC
and a moderate κ̄ must give the same ⟨c⟩ as GC.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from nih_mc.backends import NNLatticeGasBackend
from nih_mc.mc import GCParams, K_B_EVK, VCSGCParams, run_gc, run_vcsgc
from nih_mc.state import State
from nih_mc.structure import Box, SiteList


def ring_neighbors(M: int) -> list[list[int]]:
    """Closed-ring NN topology: site i is neighbor of (i-1)%M and (i+1)%M.

    Each pair appears twice (i,j) and (j,i); the NN backend's i<j pair count
    accounts for that.
    """
    return [[(i - 1) % M, (i + 1) % M] for i in range(M)]


def make_ring_site_list(M: int) -> SiteList:
    coords = np.array([[math.cos(2 * math.pi * i / M), math.sin(2 * math.pi * i / M), 0.0]
                       for i in range(M)])
    L = 4.0
    box = Box(0.0, L, 0.0, L, 0.0, L)
    return SiteList(coords=coords, box=box, n_ni=M)


def enumerate_gc(M: int, eps: float, J: float, mu: float, T: float, neighbors):
    """Exact GC partition function over all 2^M states.

    Returns (mean_c, P_n[0..M]) where P_n[n] is the marginal probability of N_H=n.
    """
    beta = 1.0 / (K_B_EVK * T)
    P_n = np.zeros(M + 1)
    logZ_terms: list[float] = []
    args = []
    for s in range(1 << M):
        bits = [(s >> i) & 1 for i in range(M)]
        n = sum(bits)
        u = eps * n
        for i in range(M):
            if bits[i]:
                for j in neighbors[i]:
                    if j > i and bits[j]:
                        u += J
        arg = -beta * (u - mu * n)
        args.append(arg)
    args = np.array(args)
    arg_max = args.max()
    w = np.exp(args - arg_max)
    Z = w.sum()
    for s in range(1 << M):
        n = bin(s).count("1")
        P_n[n] += w[s] / Z
    mean_n = sum(n * P_n[n] for n in range(M + 1))
    return mean_n / M, P_n


def enumerate_vcsgc(M: int, eps: float, J: float, dmu: float, c0: float, kappa: float,
                    T: float, neighbors):
    """Exact VC-SGC partition function (M sites = M Ni atoms here)."""
    beta = 1.0 / (K_B_EVK * T)
    P_n = np.zeros(M + 1)
    args = []
    for s in range(1 << M):
        bits = [(s >> i) & 1 for i in range(M)]
        n = sum(bits)
        u = eps * n
        for i in range(M):
            if bits[i]:
                for j in neighbors[i]:
                    if j > i and bits[j]:
                        u += J
        c = n / M
        Phi = u - dmu * n + kappa * M * (c - c0) ** 2
        args.append(-beta * Phi)
    args = np.array(args)
    arg_max = args.max()
    w = np.exp(args - arg_max)
    Z = w.sum()
    for s in range(1 << M):
        n = bin(s).count("1")
        P_n[n] += w[s] / Z
    mean_n = sum(n * P_n[n] for n in range(M + 1))
    return mean_n / M, P_n


def mc_histogram(records, M: int) -> np.ndarray:
    counts = Counter()
    for r in records:
        counts[r.n_h] += 1
    h = np.zeros(M + 1)
    total = sum(counts.values())
    for n, c in counts.items():
        h[n] = c / total
    return h


def test_gc_enumeration_attractive():
    """GC, M=8 ring, attractive coupling. P(c) and ⟨c⟩ must match exact."""
    M = 8
    eps = -0.05
    J = -0.04
    mu = -0.04
    T = 300.0
    nbrs = ring_neighbors(M)
    sites = make_ring_site_list(M)

    exact_c, exact_P = enumerate_gc(M, eps, J, mu, T, nbrs)

    state = State.empty(sites)
    backend = NNLatticeGasBackend(eps_eV=eps, J_eV=J, neighbors=nbrs)
    records = run_gc(
        state=state, backend=backend,
        params=GCParams(T_K=T, mu_eV=mu),
        n_blocks=20000, flips_per_block=20, seed=7,
    )
    burn = 2000
    sampled = records[burn:]
    mc_c = float(np.mean([r.c for r in sampled]))
    mc_P = mc_histogram(sampled, M)

    assert abs(mc_c - exact_c) < 0.01, f"⟨c⟩ mismatch GC: MC={mc_c:.4f} exact={exact_c:.4f}"
    # full-distribution L1 distance (total variation)
    tv = 0.5 * float(np.abs(mc_P - exact_P).sum())
    assert tv < 0.05, f"GC P(n) total-variation distance too large: {tv:.4f}"


def test_gc_enumeration_repulsive():
    """GC, M=12, repulsive J, broader temperature."""
    M = 12
    eps = 0.0
    J = +0.03
    mu = 0.0
    T = 400.0
    nbrs = ring_neighbors(M)
    sites = make_ring_site_list(M)

    exact_c, exact_P = enumerate_gc(M, eps, J, mu, T, nbrs)

    state = State.empty(sites)
    backend = NNLatticeGasBackend(eps_eV=eps, J_eV=J, neighbors=nbrs)
    records = run_gc(
        state=state, backend=backend,
        params=GCParams(T_K=T, mu_eV=mu),
        n_blocks=20000, flips_per_block=20, seed=13,
    )
    burn = 2000
    sampled = records[burn:]
    mc_c = float(np.mean([r.c for r in sampled]))
    mc_P = mc_histogram(sampled, M)

    assert abs(mc_c - exact_c) < 0.01, f"⟨c⟩ mismatch: MC={mc_c:.4f} exact={exact_c:.4f}"
    tv = 0.5 * float(np.abs(mc_P - exact_P).sum())
    assert tv < 0.05, f"GC P(n) total-variation distance too large: {tv:.4f}"


def test_vcsgc_mc_matches_exact_vcsgc():
    """Direct VC-SGC algorithmic check: MC histogram of N_H must match the exact
    VC-SGC partition-function P(n) for the same parameters."""
    M = 10
    eps = -0.03
    J = -0.02
    dmu = -0.025
    c0 = 0.5
    kappa = 10.0
    T = 500.0
    nbrs = ring_neighbors(M)
    sites = make_ring_site_list(M)

    vc_c_exact, vc_P_exact = enumerate_vcsgc(M, eps, J, dmu, c0, kappa, T, nbrs)

    state = State.empty(sites)
    backend = NNLatticeGasBackend(eps_eV=eps, J_eV=J, neighbors=nbrs)
    records = run_vcsgc(
        state=state, backend=backend,
        params=VCSGCParams(T_K=T, delta_mu_eV=dmu, c0=c0, kappa=kappa),
        n_blocks=20000, flips_per_block=20, seed=31,
    )
    burn = 2000
    sampled = records[burn:]
    mc_c = float(np.mean([r.c for r in sampled]))
    mc_P = mc_histogram(sampled, M)

    assert abs(mc_c - vc_c_exact) < 0.01, (
        f"VC-SGC MC ⟨c⟩={mc_c:.4f} vs exact ⟨c⟩={vc_c_exact:.4f}"
    )
    tv = 0.5 * float(np.abs(mc_P - vc_P_exact).sum())
    assert tv < 0.05, f"VC-SGC P(n) total-variation distance too large: {tv:.4f}"


def test_vcsgc_reduces_to_gc_with_zero_kappa():
    """SPEC §10 guard: with κ̄→0, VC-SGC must coincide with GC at μ=Δμ.

    This is the exact mathematical identity: at κ̄=0, Φ = U − Δμ·N_H is the GC
    grand potential. So VC-SGC MC ⟨c⟩ must match enumerated GC ⟨c⟩ to MC error.
    """
    M = 10
    eps = -0.03
    J = -0.02
    mu = -0.025
    T = 500.0
    nbrs = ring_neighbors(M)
    sites = make_ring_site_list(M)

    gc_c, gc_P = enumerate_gc(M, eps, J, mu, T, nbrs)

    state = State.empty(sites)
    backend = NNLatticeGasBackend(eps_eV=eps, J_eV=J, neighbors=nbrs)
    records = run_vcsgc(
        state=state, backend=backend,
        params=VCSGCParams(T_K=T, delta_mu_eV=mu, c0=0.5, kappa=0.0),
        n_blocks=20000, flips_per_block=20, seed=131,
    )
    burn = 2000
    sampled = records[burn:]
    mc_c = float(np.mean([r.c for r in sampled]))
    mc_P = mc_histogram(sampled, M)

    assert abs(mc_c - gc_c) < 0.01, (
        f"VC-SGC κ̄=0 ⟨c⟩={mc_c:.4f} vs GC ⟨c⟩={gc_c:.4f}"
    )
    tv = 0.5 * float(np.abs(mc_P - gc_P).sum())
    assert tv < 0.05, f"VC-SGC κ̄=0 vs GC P(n) TV distance: {tv:.4f}"


def test_round_trip_trivial_backend():
    """SPEC §10 guard: insert then delete the same site ⇒ N_H and U return to baseline.

    Even on the trivial backend this verifies try/accept/reject bookkeeping.
    """
    M = 8
    nbrs = ring_neighbors(M)
    sites = make_ring_site_list(M)
    state = State.empty(sites)
    backend = NNLatticeGasBackend(eps_eV=-0.05, J_eV=-0.04, neighbors=nbrs)
    U0, _ = backend.initial_energy(state)
    n0 = state.n_h

    # propose+accept insertion at site 3
    dU_in = backend.try_flip(state, 3)
    backend.accept()
    assert state.occ[3]
    assert state.n_h == n0 + 1
    assert math.isclose(backend.current_energy(), U0 + dU_in)

    # propose+accept deletion at site 3
    dU_out = backend.try_flip(state, 3)
    backend.accept()
    assert not state.occ[3]
    assert state.n_h == n0
    assert math.isclose(backend.current_energy(), U0, abs_tol=1e-12), (
        f"round trip: U={backend.current_energy()} U0={U0}"
    )

    # also verify reject() restores everything
    dU_try = backend.try_flip(state, 5)
    backend.reject()
    assert not state.occ[5]
    assert state.n_h == n0
    assert math.isclose(backend.current_energy(), U0)
