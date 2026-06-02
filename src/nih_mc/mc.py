"""Monte-Carlo core: symmetric single-site flip + GC / VC-SGC acceptance (SPEC §6).

The move is the SAME for both ensembles: uniformly pick a site i. If occ[i] is
False, propose insertion; if True, propose deletion. Because the proposal is
symmetric (T(j|i) = T(i|j) = 1/M), no proposal-bias correction is needed and
the lattice-gas GC acceptance is:

    insertion:  P_acc = min(1, exp(-β (ΔU - μ)))
    deletion:   P_acc = min(1, exp(-β (ΔU + μ)))

VC-SGC (Sadigh & Erhart, PRB 85 184203, 2012) uses a different acceptance based
on the variance-constrained semi-grand potential
    Φ = U − Δμ · N_H + κ̄ · N_Ni · (c − c₀)²
where c = N_H / N_Ni. The move is still the single-site flip; the acceptance
compares Φ before/after:
    P_acc = min(1, exp(-β ΔΦ))

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import numpy as np

from .backends.base import EnergyBackend
from .state import State


# Boltzmann constant in eV / K (CODATA 2018, used to one part in 1e7)
K_B_EVK = 8.617333262145178e-5


@dataclass
class GCParams:
    T_K: float
    mu_eV: float

    @property
    def beta(self) -> float:
        return 1.0 / (K_B_EVK * self.T_K)


@dataclass
class VCSGCParams:
    """VC-SGC parameters per Sadigh & Erhart, PRB 85 184203 (2012)."""
    T_K: float
    delta_mu_eV: float    # Δμ
    c0: float             # target concentration
    kappa: float          # κ̄ (constraint strength), in eV per Ni atom

    @property
    def beta(self) -> float:
        return 1.0 / (K_B_EVK * self.T_K)


def gc_log_accept(dU: float, delta_N: int, mu: float, beta: float) -> float:
    """Log of GC acceptance ratio for a single-site flip.

    delta_N is +1 for insertion, -1 for deletion. Acceptance probability is
    min(1, exp(arg)); we return `arg = -β(ΔU − μ·ΔN)`.
    """
    return -beta * (dU - mu * delta_N)


def vcsgc_log_accept(
    dU: float,
    delta_N: int,
    n_h_before: int,
    n_ni: int,
    p: VCSGCParams,
) -> float:
    """Log acceptance for VC-SGC.

    Δ(N_H) = ±1. We compute ΔΦ exactly from N_H_before, N_H_after, and ΔU.
    """
    n_h_after = n_h_before + delta_N
    c_b = n_h_before / n_ni
    c_a = n_h_after / n_ni
    dPhi = dU - p.delta_mu_eV * delta_N + p.kappa * n_ni * ((c_a - p.c0) ** 2 - (c_b - p.c0) ** 2)
    return -p.beta * dPhi


@dataclass
class MCStats:
    n_attempted: int = 0
    n_accepted: int = 0
    n_insertion_attempted: int = 0
    n_insertion_accepted: int = 0
    n_deletion_attempted: int = 0
    n_deletion_accepted: int = 0

    @property
    def acc_ratio(self) -> float:
        return self.n_accepted / max(self.n_attempted, 1)


def attempt_flip_gc(
    state: State,
    backend: EnergyBackend,
    params: GCParams,
    rng: np.random.Generator,
    stats: MCStats,
) -> None:
    """One uniform-site GC flip attempt; updates state, backend, stats in place."""
    i = int(rng.integers(0, state.n_sites))
    was_occupied = bool(state.occ[i])
    delta_N = -1 if was_occupied else +1

    dU = backend.try_flip(state, i)
    log_pacc = gc_log_accept(dU, delta_N, params.mu_eV, params.beta)

    if log_pacc >= 0.0 or rng.random() < math.exp(log_pacc):
        backend.accept()
        stats.n_accepted += 1
        if delta_N == +1:
            stats.n_insertion_accepted += 1
        else:
            stats.n_deletion_accepted += 1
    else:
        backend.reject()

    stats.n_attempted += 1
    if delta_N == +1:
        stats.n_insertion_attempted += 1
    else:
        stats.n_deletion_attempted += 1


def attempt_flip_vcsgc(
    state: State,
    backend: EnergyBackend,
    params: VCSGCParams,
    rng: np.random.Generator,
    stats: MCStats,
) -> None:
    """One uniform-site VC-SGC flip attempt."""
    i = int(rng.integers(0, state.n_sites))
    was_occupied = bool(state.occ[i])
    delta_N = -1 if was_occupied else +1
    n_h_before = state.n_h

    dU = backend.try_flip(state, i)
    log_pacc = vcsgc_log_accept(dU, delta_N, n_h_before, state.sites.n_ni, params)

    if log_pacc >= 0.0 or rng.random() < math.exp(log_pacc):
        backend.accept()
        stats.n_accepted += 1
        if delta_N == +1:
            stats.n_insertion_accepted += 1
        else:
            stats.n_deletion_accepted += 1
    else:
        backend.reject()

    stats.n_attempted += 1
    if delta_N == +1:
        stats.n_insertion_attempted += 1
    else:
        stats.n_deletion_attempted += 1


@dataclass
class BlockRecord:
    block: int
    n_h: int
    c: float
    U_eV: float
    a_box_Ang: float
    P_bar: float
    acc_ratio: float


def run_gc(
    state: State,
    backend: EnergyBackend,
    params: GCParams,
    n_blocks: int,
    flips_per_block: int,
    seed: int,
    relax_per_block: Optional[Callable[[State], None]] = None,
    record_cb: Optional[Callable[[BlockRecord], None]] = None,
) -> list[BlockRecord]:
    """Run GC sampling. Returns the per-block records.

    `relax_per_block` (optional) is called once per block to allow a backend-level
    relaxation/MD pass; the trivial backends don't need it.
    """
    rng = np.random.default_rng(seed)
    backend.initial_energy(state)
    stats = MCStats()
    records: list[BlockRecord] = []

    for b in range(n_blocks):
        for _ in range(flips_per_block):
            attempt_flip_gc(state, backend, params, rng, stats)
        if relax_per_block is not None:
            relax_per_block(state)
        box = backend.current_box()
        a_box = float(box.lengths[0])  # orthogonal cubic by construction
        P_bar = float(getattr(backend, "last_pressure_bar", 0.0))
        rec = BlockRecord(
            block=b,
            n_h=state.n_h,
            c=state.concentration,
            U_eV=backend.current_energy(),
            a_box_Ang=a_box,
            P_bar=P_bar,
            acc_ratio=stats.acc_ratio,
        )
        records.append(rec)
        if record_cb is not None:
            record_cb(rec)

    return records


def run_vcsgc(
    state: State,
    backend: EnergyBackend,
    params: VCSGCParams,
    n_blocks: int,
    flips_per_block: int,
    seed: int,
    relax_per_block: Optional[Callable[[State], None]] = None,
    record_cb: Optional[Callable[[BlockRecord], None]] = None,
) -> list[BlockRecord]:
    """Run VC-SGC sampling."""
    rng = np.random.default_rng(seed)
    backend.initial_energy(state)
    stats = MCStats()
    records: list[BlockRecord] = []

    for b in range(n_blocks):
        for _ in range(flips_per_block):
            attempt_flip_vcsgc(state, backend, params, rng, stats)
        if relax_per_block is not None:
            relax_per_block(state)
        box = backend.current_box()
        a_box = float(box.lengths[0])
        P_bar = float(getattr(backend, "last_pressure_bar", 0.0))
        rec = BlockRecord(
            block=b,
            n_h=state.n_h,
            c=state.concentration,
            U_eV=backend.current_energy(),
            a_box_Ang=a_box,
            P_bar=P_bar,
            acc_ratio=stats.acc_ratio,
        )
        records.append(rec)
        if record_cb is not None:
            record_cb(rec)

    return records
