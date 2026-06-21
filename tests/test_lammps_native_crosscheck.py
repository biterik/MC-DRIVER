"""Cross-checks of MC-DRIVER against the *native* LAMMPS Monte-Carlo fixes.

This module answers a specific question: do MC-DRIVER's grand-canonical (GC) and
variance-constrained semi-grand-canonical (VC-SGC) samplers agree with the
standard LAMMPS implementations of the same statistical ensembles?

  * GC      <->  `fix gcmc`   (LAMMPS MC package)
  * VC-SGC  <->  `fix sgcmc`  (the vcsgc-lammps package of Sadigh & Erhart et al.)

They are NOT drop-in equivalents, and the mappings below are the whole point:

  fix gcmc  exchanges atoms with an *ideal-gas reservoir* and inserts at random
            positions in a region (off-lattice). Its acceptance carries the
            ideal-gas reference (thermal wavelength Lambda, volume V), so for a
            given physical state its chemical potential differs from MC-DRIVER's
            lattice-gas mu by a T-dependent constant
                mu_lattice = mu_gcmc + kT * ln( V / (M * Lambda^3) ).
            To stay reference-free, the GC test below compares a *ratio*
            <N>(mu2)/<N>(mu1), in which Lambda, V and M cancel. In the dilute
            limit both codes give exp(beta*(mu2-mu1)); they differ only by the
            O(theta) lattice saturation (ideal gas never saturates, the lattice
            gas does). The tolerance is sized to absorb that systematic + MC noise.

  fix sgcmc performs *transmutation* (atom type A<->B) at fixed sites; it never
            changes the atom count. MC-DRIVER instead inserts/deletes H on a site
            list. The two are mapped by representing every octahedral site with a
            placeholder ("ghost") species: insert H == transmute ghost->H,
            delete H == transmute H->ghost. With `pair_style zero` (all energies
            zero) the occupation statistics are purely entropic + Delta-mu + kappa,
            so both codes must sample the SAME binomial x Gaussian distribution
                P(N) ~ C(M,N) * exp( beta*[ Delta_mu*N - kappa*M*(N/M - c0)^2 ] ).

IMPORTANT — provenance of this file:
  These cross-checks require a LAMMPS build with the relevant fix (and, for
  VC-SGC, the non-default vcsgc-lammps package). They were authored in an
  environment WITHOUT a runnable LAMMPS, so the *LAMMPS halves* have not been
  executed here; the analytic targets and the MC-DRIVER halves WERE verified.
  The exact `fix sgcmc` argument order / kappa normalization can vary between
  package versions — adjust `_run_fix_sgcmc` to your build if needed. Each test
  skips cleanly when LAMMPS or the required fix is unavailable, so the rest of
  the suite is unaffected.

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

# Whole-module skip if the LAMMPS python module is not importable.
lammps = pytest.importorskip("lammps", reason="native-fix cross-check needs LAMMPS")

from nih_mc.backends import ConstantSiteEnergyBackend  # noqa: E402
from nih_mc.mc import GCParams, VCSGCParams, K_B_EVK, run_gc, run_vcsgc  # noqa: E402
from nih_mc.state import State  # noqa: E402
from nih_mc.structure import Box, SiteList  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _flat_sites(M: int) -> SiteList:
    """A degenerate site list (positions irrelevant for the zero-energy tests)."""
    return SiteList(coords=np.zeros((M, 3)), box=Box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), n_ni=M)


def _new_lmp():
    args = ["-log", "none", "-screen", "none", "-nocite"]
    return lammps.lammps(cmdargs=args)


def _fix_available(fix_name: str) -> bool:
    """Probe whether a given fix style is compiled into this LAMMPS."""
    lmp = _new_lmp()
    try:
        lmp.commands_string(
            f"""
            units metal
            atom_style atomic
            boundary p p p
            region box block 0 10 0 10 0 10
            create_box 2 box
            mass 1 58.69
            mass 2 1.008
            pair_style zero 2.0
            pair_coeff * *
            """
        )
        # create a couple of atoms so a fix can be defined
        lmp.command("create_atoms 1 single 1 1 1")
        try:
            if fix_name == "gcmc":
                lmp.command("fix probe all gcmc 1 1 0 1 12345 300.0 0.0 0.0")
            elif fix_name == "sgcmc":
                # minimal sgcmc; arg order may vary by version
                lmp.command("fix probe all sgcmc 1 0.1 300.0 0.0")
            lmp.command("unfix probe")
            return True
        except Exception:
            return False
    except Exception:
        return False
    finally:
        lmp.close()


def _run_fix_gcmc(mu_eV: float, T_K: float, box_L: float = 12.0,
                  n_equil: int = 2000, n_sample: int = 20000,
                  sample_every: int = 5, seed: int = 12345) -> float:
    """Ideal-gas GCMC with `pair_style zero`; returns the mean atom count <N>."""
    lmp = _new_lmp()
    lmp.commands_string(
        f"""
        units metal
        atom_style atomic
        boundary p p p
        region box block 0 {box_L} 0 {box_L} 0 {box_L}
        create_box 1 box
        mass 1 1.008
        pair_style zero 2.0
        pair_coeff * *
        # average 1 exchange attempt / step, 0 translation moves
        fix gc all gcmc 1 1 0 1 {seed} {T_K} {mu_eV} 0.0
        thermo 100000
        run {n_equil}
        """
    )
    counts = []
    for _ in range(n_sample // sample_every):
        lmp.command(f"run {sample_every} pre no post no")
        counts.append(lmp.get_natoms())
    lmp.close()
    return float(np.mean(counts))


def _run_fix_sgcmc(M: int, dmu_eV: float, c0: float, kappa: float, T_K: float,
                   n_equil: int = 5000, n_sample: int = 60000,
                   sample_every: int = 20, seed: int = 54321):
    """Binary ghost/H transmutation on M fixed sites with `pair_style zero`.

    Returns (mean_concentration, P_N) with P_N[n] = Prob(N_H = n).
    NOTE: `fix sgcmc` argument order and the kappa normalization depend on the
    vcsgc-lammps version; align this invocation with your build's docs.
    """
    # Lay M atoms on a simple cubic grid; start near the target concentration.
    a = 3.0
    n_side = int(round(M ** (1.0 / 3.0)))
    assert n_side ** 3 == M, "use a perfect-cube M for the sgcmc grid"
    L = n_side * a
    lmp = _new_lmp()
    lmp.commands_string(
        f"""
        units metal
        atom_style atomic
        boundary p p p
        region box block 0 {L} 0 {L} 0 {L}
        create_box 2 box
        lattice sc {a}
        create_atoms 1 box
        mass 1 1.008
        mass 2 1.008
        pair_style zero 2.0
        pair_coeff * *
        velocity all create {T_K} {seed} mom yes rot yes
        fix integ all nve
        # variance-constrained semi-grand-canonical transmutation between types 1 and 2.
        # arg order (this version): nevery swap_fraction T deltamu seed variance kappa target_c
        fix mc all sgcmc 1 0.2 {T_K} {dmu_eV} {seed} variance {kappa} {c0}
        thermo 100000
        run {n_equil}
        """
    )
    counts = []
    for _ in range(n_sample // sample_every):
        lmp.command(f"run {sample_every} pre no post no")
        n_h = int(lmp.get_natoms() and
                  np.sum(np.array(lmp.gather_atoms("type", 0, 1)) == 2))
        counts.append(n_h)
    lmp.close()
    P = np.zeros(M + 1)
    c = Counter(counts)
    tot = sum(c.values())
    for n, k in c.items():
        P[n] = k / tot
    mean_c = float(np.mean(counts)) / M
    return mean_c, P


# --------------------------------------------------------------------------- #
# Analytic targets (verified against MC-DRIVER in the authoring environment)
# --------------------------------------------------------------------------- #
def _lattice_theta(mu_eV: float, T_K: float) -> float:
    beta = 1.0 / (K_B_EVK * T_K)
    return 1.0 / (1.0 + math.exp(-beta * mu_eV))


def _vcsgc_zero_energy_PN(M: int, dmu: float, c0: float, kappa: float, T_K: float):
    beta = 1.0 / (K_B_EVK * T_K)
    logw = [math.lgamma(M + 1) - math.lgamma(n + 1) - math.lgamma(M - n + 1)
            + beta * (dmu * n - kappa * M * (n / M - c0) ** 2) for n in range(M + 1)]
    mx = max(logw)
    w = np.array([math.exp(x - mx) for x in logw])
    P = w / w.sum()
    mean_c = sum(n * P[n] for n in range(M + 1)) / M
    return mean_c, P


# --------------------------------------------------------------------------- #
# Test 1: GC  vs  fix gcmc   (reference-free dilute-limit ratio)
# --------------------------------------------------------------------------- #
def test_gc_matches_fix_gcmc_dilute_ratio():
    """In the dilute limit, <N>(mu2)/<N>(mu1) is reference-free and equals
    exp(beta*(mu2-mu1)) for the ideal gas (fix gcmc) and the lattice-gas
    sigmoid ratio for MC-DRIVER; the two coincide to O(theta)."""
    if not _fix_available("gcmc"):
        pytest.skip("this LAMMPS lacks `fix gcmc` (MC package)")

    T = 600.0
    beta = 1.0 / (K_B_EVK * T)
    mu1, mu2 = -0.20, -0.15           # dilute: theta ~ 0.02 .. 0.05
    ratio_ideal = math.exp(beta * (mu2 - mu1))

    # --- MC-DRIVER lattice-gas (eps = 0) ---
    M = 400
    def md_meanc(mu, seed):
        recs = run_gc(State.empty(_flat_sites(M)), ConstantSiteEnergyBackend(0.0),
                      GCParams(T_K=T, mu_eV=mu), n_blocks=20000, flips_per_block=40,
                      seed=seed)
        return float(np.mean([r.c for r in recs[5000:]]))
    md_ratio = md_meanc(mu2, 12) / md_meanc(mu1, 11)
    lattice_ratio = _lattice_theta(mu2, T) / _lattice_theta(mu1, T)

    # --- native fix gcmc ideal gas ---
    n1 = _run_fix_gcmc(mu1, T, seed=101)
    n2 = _run_fix_gcmc(mu2, T, seed=202)
    gcmc_ratio = n2 / n1

    # (a) fix gcmc reproduces the ideal-gas exponential
    assert abs(gcmc_ratio - ratio_ideal) / ratio_ideal < 0.08, (
        f"fix gcmc ratio {gcmc_ratio:.3f} vs ideal {ratio_ideal:.3f}")
    # (b) MC-DRIVER reproduces the lattice-gas sigmoid ratio
    assert abs(md_ratio - lattice_ratio) / lattice_ratio < 0.06, (
        f"MC-DRIVER ratio {md_ratio:.3f} vs lattice {lattice_ratio:.3f}")
    # (c) the two codes agree to O(theta) in the dilute limit
    assert abs(gcmc_ratio - md_ratio) / md_ratio < 0.10, (
        f"GC vs fix gcmc dilute ratio disagree: {md_ratio:.3f} vs {gcmc_ratio:.3f}")


# --------------------------------------------------------------------------- #
# Test 2: VC-SGC  vs  fix sgcmc   (ghost-species mapping, pair_style zero)
# --------------------------------------------------------------------------- #
def test_vcsgc_matches_fix_sgcmc_zero_energy():
    """With all energies zero, both codes must sample the same binomial x Gaussian
    P(N_H); compare means and the full distribution (total-variation distance)."""
    if not _fix_available("sgcmc"):
        pytest.skip("this LAMMPS lacks `fix sgcmc` (vcsgc-lammps package)")

    M = 64                  # perfect cube (4x4x4) for the sgcmc grid
    T = 500.0
    dmu, c0, kappa = -0.02, 0.5, 6.0

    target_c, target_P = _vcsgc_zero_energy_PN(M, dmu, c0, kappa, T)

    # --- MC-DRIVER VC-SGC (eps = J = 0) ---
    recs = run_vcsgc(State.empty(_flat_sites(M)), ConstantSiteEnergyBackend(0.0),
                     VCSGCParams(T_K=T, delta_mu_eV=dmu, c0=c0, kappa=kappa),
                     n_blocks=60000, flips_per_block=20, seed=9)[6000:]
    cc = Counter(r.n_h for r in recs)
    tot = sum(cc.values())
    md_P = np.array([cc.get(n, 0) / tot for n in range(M + 1)])
    md_c = float(np.mean([r.c for r in recs]))

    # --- native fix sgcmc transmutation ---
    sg_c, sg_P = _run_fix_sgcmc(M, dmu, c0, kappa, T)

    tv_md = 0.5 * float(np.abs(md_P - target_P).sum())
    tv_sg = 0.5 * float(np.abs(sg_P - target_P).sum())
    tv_cross = 0.5 * float(np.abs(md_P - sg_P).sum())

    assert abs(md_c - target_c) < 0.02, f"MC-DRIVER <c>={md_c:.4f} target={target_c:.4f}"
    assert tv_md < 0.05, f"MC-DRIVER P(N) TV vs target {tv_md:.4f}"
    assert abs(sg_c - target_c) < 0.03, f"fix sgcmc <c>={sg_c:.4f} target={target_c:.4f}"
    assert tv_sg < 0.07, f"fix sgcmc P(N) TV vs target {tv_sg:.4f}"
    assert tv_cross < 0.08, f"MC-DRIVER vs fix sgcmc P(N) TV {tv_cross:.4f}"
