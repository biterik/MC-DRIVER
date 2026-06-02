#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the validation figures and the numerical results table.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

This script reproduces every pure-Python (analytic + exact-enumeration) result
in ``docs/VALIDATION.md`` from the actual MC-DRIVER package code, so that any
reader can re-run it and cross-check the published figures and numbers:

    python tools/make_validation_figures.py

Outputs (written next to this script, under ``docs/figures/`` and ``docs/``):
    docs/figures/langmuir_isotherm.png
    docs/figures/enumeration_Pn.png
    docs/validation_results.csv

This is licensed under GPL-3.0-or-later (see LICENSE).
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the in-repo package importable without installation.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from nih_mc.backends import ConstantSiteEnergyBackend, NNLatticeGasBackend  # noqa: E402
from nih_mc.mc import GCParams, VCSGCParams, K_B_EVK, run_gc, run_vcsgc  # noqa: E402
from nih_mc.state import State  # noqa: E402
from nih_mc.structure import Box, SiteList  # noqa: E402

FIG_DIR = os.path.join(ROOT, "docs", "figures")
DOC_DIR = os.path.join(ROOT, "docs")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_EXACT = "#1f3b73"   # deep blue   -> theory / exact
C_MC = "#d1495b"      # warm red    -> Monte Carlo
results_rows: list[dict] = []


# --------------------------------------------------------------------------- #
# Helpers (mirror the test suite verbatim)
# --------------------------------------------------------------------------- #
def make_synthetic_site_list(n_sites: int, a: float = 1.0) -> SiteList:
    rng = np.random.default_rng(0)
    L = a * n_sites ** (1 / 3)
    coords = rng.random((n_sites, 3)) * L
    return SiteList(coords=coords, box=Box(0.0, L, 0.0, L, 0.0, L), n_ni=n_sites)


def make_ring_site_list(M: int) -> SiteList:
    coords = np.array([[math.cos(2 * math.pi * i / M),
                        math.sin(2 * math.pi * i / M), 0.0] for i in range(M)])
    return SiteList(coords=coords, box=Box(0.0, 4.0, 0.0, 4.0, 0.0, 4.0), n_ni=M)


def ring_neighbors(M: int) -> list[list[int]]:
    return [[(i - 1) % M, (i + 1) % M] for i in range(M)]


def langmuir_theta(mu, eps, T):
    return 1.0 / (math.exp((eps - mu) / (K_B_EVK * T)) + 1.0)


def enumerate_gc(M, eps, J, mu, T, neighbors):
    beta = 1.0 / (K_B_EVK * T)
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
        args.append(-beta * (u - mu * n))
    args = np.array(args)
    w = np.exp(args - args.max())
    Z = w.sum()
    P_n = np.zeros(M + 1)
    for s in range(1 << M):
        P_n[bin(s).count("1")] += w[s] / Z
    mean_n = sum(n * P_n[n] for n in range(M + 1))
    return mean_n / M, P_n


def enumerate_vcsgc(M, eps, J, dmu, c0, kappa, T, neighbors):
    beta = 1.0 / (K_B_EVK * T)
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
    w = np.exp(args - args.max())
    Z = w.sum()
    P_n = np.zeros(M + 1)
    for s in range(1 << M):
        P_n[bin(s).count("1")] += w[s] / Z
    mean_n = sum(n * P_n[n] for n in range(M + 1))
    return mean_n / M, P_n


def mc_histogram(records, M):
    counts = Counter(r.n_h for r in records)
    h = np.zeros(M + 1)
    total = sum(counts.values())
    for n, c in counts.items():
        h[n] = c / total
    return h


# --------------------------------------------------------------------------- #
# Figure 1 — Langmuir isotherm (SPEC Test 1)
# --------------------------------------------------------------------------- #
def figure_langmuir():
    print("[1/2] Langmuir isotherm ...")
    sites = make_synthetic_site_list(200)
    n_blocks, fpb, burn = 4000, 50, 800

    branches = [
        dict(T=300.0, eps=0.0, label=r"$T=300\,$K, $\varepsilon=0$",
             pts=[-0.10, -0.02, 0.0, 0.02, 0.10]),
        dict(T=600.0, eps=0.05, label=r"$T=600\,$K, $\varepsilon=0.05\,$eV",
             pts=[-0.05, 0.15]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for bi, br in enumerate(branches):
        mu_grid = np.linspace(-0.18, 0.22, 300)
        theta_curve = [langmuir_theta(mu, br["eps"], br["T"]) for mu in mu_grid]
        ax.plot(mu_grid, theta_curve, color=C_EXACT, lw=2,
                ls="-" if bi == 0 else "--",
                label=f"Langmuir (exact): {br['label']}", zorder=2)
        for mu in br["pts"]:
            state = State.empty(sites)
            backend = ConstantSiteEnergyBackend(eps_eV=br["eps"])
            recs = run_gc(state=state, backend=backend,
                          params=GCParams(T_K=br["T"], mu_eV=mu),
                          n_blocks=n_blocks, flips_per_block=fpb, seed=42)
            theta_mc = float(np.mean([r.c for r in recs[burn:]]))
            theta_ex = langmuir_theta(mu, br["eps"], br["T"])
            results_rows.append(dict(test="Langmuir", case=br["label"],
                                     param=f"mu={mu:+.2f}",
                                     mc=f"{theta_mc:.4f}", exact=f"{theta_ex:.4f}",
                                     delta=f"{abs(theta_mc-theta_ex):.4f}",
                                     tol="0.02"))
            ax.scatter([mu], [theta_mc], color=C_MC, s=55, zorder=3,
                       edgecolor="white", linewidth=0.8,
                       label="GC Monte Carlo" if (bi == 0 and mu == br["pts"][0]) else None)
    ax.set_xlabel(r"chemical potential  $\mu$  (eV)")
    ax.set_ylabel(r"H coverage  $\theta = \langle c\rangle = \langle N_\mathrm{H}\rangle/N_\mathrm{Ni}$")
    ax.set_title("Test 1 — Non-interacting lattice gas: GC-MC vs. Langmuir isotherm")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "langmuir_isotherm.png")
    fig.savefig(out)
    plt.close(fig)
    print("      ->", out)


# --------------------------------------------------------------------------- #
# Figure 2 — Exact enumeration P(N_H): four panels (SPEC Test 2)
# --------------------------------------------------------------------------- #
def _panel(ax, title, M, P_exact, P_mc, exact_c, mc_c, exact_label):
    n = np.arange(M + 1)
    width = 0.42
    ax.bar(n - width / 2, P_exact, width, color=C_EXACT, label=exact_label)
    ax.bar(n + width / 2, P_mc, width, color=C_MC, label="MC histogram")
    tv = 0.5 * float(np.abs(P_mc - P_exact).sum())
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$N_\mathrm{H}$")
    ax.set_ylabel(r"$P(N_\mathrm{H})$")
    ax.set_xticks(n)
    ax.text(0.02, 0.97,
            f"$\\langle c\\rangle$ exact = {exact_c:.4f}\n"
            f"$\\langle c\\rangle$ MC = {mc_c:.4f}\n"
            f"TV dist. = {tv:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    return tv


def figure_enumeration():
    print("[2/2] Exact enumeration P(N_H) ...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) GC attractive, M=8
    M, eps, J, mu, T = 8, -0.05, -0.04, -0.04, 300.0
    nbrs = ring_neighbors(M)
    ex_c, ex_P = enumerate_gc(M, eps, J, mu, T, nbrs)
    recs = run_gc(state=State.empty(make_ring_site_list(M)),
                  backend=NNLatticeGasBackend(eps_eV=eps, J_eV=J, neighbors=nbrs),
                  params=GCParams(T_K=T, mu_eV=mu),
                  n_blocks=20000, flips_per_block=20, seed=7)[2000:]
    mc_c, mc_P = float(np.mean([r.c for r in recs])), mc_histogram(recs, M)
    tv = _panel(axes[0, 0], "(a) GC, attractive  (M=8, J=-0.04 eV, T=300 K)",
                M, ex_P, mc_P, ex_c, mc_c, "GC enumeration")
    results_rows.append(dict(test="GC enum (attractive)", case="M=8", param="TV",
                             mc=f"{mc_c:.4f}", exact=f"{ex_c:.4f}",
                             delta=f"{tv:.4f}", tol="0.05"))

    # (b) GC repulsive, M=12
    M, eps, J, mu, T = 12, 0.0, 0.03, 0.0, 400.0
    nbrs = ring_neighbors(M)
    ex_c, ex_P = enumerate_gc(M, eps, J, mu, T, nbrs)
    recs = run_gc(state=State.empty(make_ring_site_list(M)),
                  backend=NNLatticeGasBackend(eps_eV=eps, J_eV=J, neighbors=nbrs),
                  params=GCParams(T_K=T, mu_eV=mu),
                  n_blocks=20000, flips_per_block=20, seed=13)[2000:]
    mc_c, mc_P = float(np.mean([r.c for r in recs])), mc_histogram(recs, M)
    tv = _panel(axes[0, 1], "(b) GC, repulsive  (M=12, J=+0.03 eV, T=400 K)",
                M, ex_P, mc_P, ex_c, mc_c, "GC enumeration")
    results_rows.append(dict(test="GC enum (repulsive)", case="M=12", param="TV",
                             mc=f"{mc_c:.4f}", exact=f"{ex_c:.4f}",
                             delta=f"{tv:.4f}", tol="0.05"))

    # (c) VC-SGC finite kappa, M=10
    M, eps, J, dmu, c0, kappa, T = 10, -0.03, -0.02, -0.025, 0.5, 10.0, 500.0
    nbrs = ring_neighbors(M)
    ex_c, ex_P = enumerate_vcsgc(M, eps, J, dmu, c0, kappa, T, nbrs)
    recs = run_vcsgc(state=State.empty(make_ring_site_list(M)),
                     backend=NNLatticeGasBackend(eps_eV=eps, J_eV=J, neighbors=nbrs),
                     params=VCSGCParams(T_K=T, delta_mu_eV=dmu, c0=c0, kappa=kappa),
                     n_blocks=20000, flips_per_block=20, seed=31)[2000:]
    mc_c, mc_P = float(np.mean([r.c for r in recs])), mc_histogram(recs, M)
    tv = _panel(axes[1, 0],
                r"(c) VC-SGC, finite $\bar\kappa$  (M=10, $\bar\kappa$=10 eV, T=500 K)",
                M, ex_P, mc_P, ex_c, mc_c, "VC-SGC enumeration")
    results_rows.append(dict(test="VC-SGC enum", case="M=10", param="TV",
                             mc=f"{mc_c:.4f}", exact=f"{ex_c:.4f}",
                             delta=f"{tv:.4f}", tol="0.05"))

    # (d) VC-SGC kappa=0 reduces to GC, M=10
    M, eps, J, mu, T = 10, -0.03, -0.02, -0.025, 500.0
    nbrs = ring_neighbors(M)
    ex_c, ex_P = enumerate_gc(M, eps, J, mu, T, nbrs)
    recs = run_vcsgc(state=State.empty(make_ring_site_list(M)),
                     backend=NNLatticeGasBackend(eps_eV=eps, J_eV=J, neighbors=nbrs),
                     params=VCSGCParams(T_K=T, delta_mu_eV=mu, c0=0.5, kappa=0.0),
                     n_blocks=20000, flips_per_block=20, seed=131)[2000:]
    mc_c, mc_P = float(np.mean([r.c for r in recs])), mc_histogram(recs, M)
    tv = _panel(axes[1, 1],
                r"(d) VC-SGC $\bar\kappa\to0$ = GC  (M=10, $\mu=\Delta\mu$, T=500 K)",
                M, ex_P, mc_P, ex_c, mc_c, "GC enumeration")
    results_rows.append(dict(test="VC-SGC kappa=0 -> GC", case="M=10", param="TV",
                             mc=f"{mc_c:.4f}", exact=f"{ex_c:.4f}",
                             delta=f"{tv:.4f}", tol="0.05"))

    fig.suptitle("Test 2 — Small-system exact enumeration: MC histograms vs. "
                 "exact partition-function P(N_H)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG_DIR, "enumeration_Pn.png")
    fig.savefig(out)
    plt.close(fig)
    print("      ->", out)


def write_results_csv():
    out = os.path.join(DOC_DIR, "validation_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "case", "param", "mc", "exact",
                                          "delta", "tol"])
        w.writeheader()
        w.writerows(results_rows)
    print("      ->", out)


if __name__ == "__main__":
    figure_langmuir()
    figure_enumeration()
    write_results_csv()
    print("\nDone. Figures in docs/figures/, numbers in docs/validation_results.csv")
