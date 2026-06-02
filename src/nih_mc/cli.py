"""Command-line driver. Usage:

    nih-mc run config.yaml

Reads the YAML config (SPEC §9), wires up the chosen backend + ensemble, and
writes outputs per SPEC §8.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .config import RunConfig, expand_pair_coeff, load_yaml_config
from .mc import BlockRecord, GCParams, VCSGCParams, run_gc, run_vcsgc
from .outputs import IsothermCsvWriter, write_run_summary
from .state import State
from .structure import build_site_list, read_lammps_data


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("nih_mc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(sh)
    return logger


def _make_backend_and_state(cfg: RunConfig):
    """Construct the backend + initial state for a real run.

    Returns (state, backend, lammps_version_string_or_None).
    """
    sites = build_site_list(read_lammps_data(cfg.structure_full))

    if cfg.backend == "trivial":
        from .backends import ConstantSiteEnergyBackend
        backend = ConstantSiteEnergyBackend(eps_eV=cfg.eps_eV)
        state = State.empty(sites) if cfg.start_from == "empty" else State.full(sites)
        backend.initial_energy(state)
        return state, backend, None

    from .backends import LammpsNptBackend, LammpsStaticsBackend
    from .lammps_runner import LammpsBackendConfig, LammpsRunner

    data_path = (
        cfg.structure_empty if cfg.start_from == "empty" and cfg.structure_empty
        else cfg.structure_full
    )
    pair_coeff = expand_pair_coeff(cfg.potential, data_path)
    lcfg = LammpsBackendConfig(
        pair_style=cfg.potential.pair_style,
        pair_coeff=pair_coeff,
        potdir=cfg.potential.potdir,
        do_minimize=(cfg.backend == "statics"),
        min_etol=cfg.min_etol,
        min_ftol=cfg.min_ftol,
        cell_relax=cfg.cell_relax,
        md_T_K=cfg.T_K,
        md_P_bar=cfg.md_P_bar,
        md_steps_equil=cfg.md_steps_equil,
        md_steps_sample=cfg.md_steps_sample,
    )
    runner = LammpsRunner(data_path, lcfg)
    state = runner.prepare_state(sites, start_from=cfg.start_from)
    if cfg.backend == "statics":
        backend = LammpsStaticsBackend(runner)
    elif cfg.backend == "md_npt":
        backend = LammpsNptBackend(runner)
    else:
        raise ValueError(f"unknown backend {cfg.backend!r}")
    backend.initial_energy(state)
    lmp_version = str(runner.lmp.version())
    return state, backend, lmp_version


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_yaml_config(args.config)
    out_dir = Path(args.outdir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.out_prefix
    csv_path = out_dir / f"{prefix}.csv"
    log_path = out_dir / f"{prefix}.log"
    yaml_path = out_dir / f"{prefix}-run-summary.yaml"

    logger = _setup_logging(log_path)
    logger.info(f"nih_mc {__version__} starting run")
    logger.info(f"config: {args.config}")
    logger.info(f"seed: {cfg.seed}")

    state, backend, lmp_version = _make_backend_and_state(cfg)
    logger.info(
        f"sites: {state.n_sites}, n_ni: {state.sites.n_ni}, start_from: {cfg.start_from}"
    )
    if lmp_version:
        logger.info(f"LAMMPS version: {lmp_version}")

    csv = IsothermCsvWriter(path=csv_path,
                            mu_eV=(cfg.mu_eV if cfg.ensemble == "gc"
                                   else (cfg.delta_mu_eV or 0.0)))
    csv.open()

    relax_cb = None
    if hasattr(backend, "relax"):
        relax_cb = backend.relax

    t0 = time.time()
    if cfg.ensemble == "gc":
        if cfg.mu_eV is None:
            raise SystemExit("GC ensemble requires mu_eV in the config")
        records = run_gc(
            state=state, backend=backend,
            params=GCParams(T_K=cfg.T_K, mu_eV=cfg.mu_eV),
            n_blocks=cfg.n_blocks, flips_per_block=cfg.flips_per_block,
            seed=cfg.seed, relax_per_block=relax_cb,
            record_cb=csv.write,
        )
    elif cfg.ensemble == "vcsgc":
        if None in (cfg.delta_mu_eV, cfg.c0, cfg.kappa):
            raise SystemExit("VC-SGC ensemble requires delta_mu_eV, c0, kappa")
        records = run_vcsgc(
            state=state, backend=backend,
            params=VCSGCParams(
                T_K=cfg.T_K,
                delta_mu_eV=cfg.delta_mu_eV,
                c0=cfg.c0,
                kappa=cfg.kappa,
            ),
            n_blocks=cfg.n_blocks, flips_per_block=cfg.flips_per_block,
            seed=cfg.seed, relax_per_block=relax_cb,
            record_cb=csv.write,
        )
    else:
        raise SystemExit(f"unknown ensemble {cfg.ensemble!r}")
    elapsed = time.time() - t0
    csv.close()

    c_vals = np.array([r.c for r in records])
    a_vals = np.array([r.a_box_Ang for r in records])
    summary = {
        "driver_version": __version__,
        "lammps_version": lmp_version,
        "config_path": str(args.config),
        "seed": cfg.seed,
        "ensemble": cfg.ensemble,
        "backend": cfg.backend,
        "T_K": cfg.T_K,
        "mu_eV": cfg.mu_eV,
        "delta_mu_eV": cfg.delta_mu_eV,
        "c0": cfg.c0,
        "kappa": cfg.kappa,
        "n_blocks": cfg.n_blocks,
        "flips_per_block": cfg.flips_per_block,
        "structure_full": cfg.structure_full,
        "structure_empty": cfg.structure_empty,
        "potential_kind": cfg.potential.kind,
        "potential_files": cfg.potential.files,
        "mean_c": float(c_vals.mean()),
        "std_c": float(c_vals.std(ddof=1)),
        "mean_a_Ang": float(a_vals.mean()),
        "std_a_Ang": float(a_vals.std(ddof=1)),
        "elapsed_s": elapsed,
    }
    write_run_summary(yaml_path, summary)
    logger.info(
        f"done: mean c = {summary['mean_c']:.6f} ± {summary['std_c']:.6f}, "
        f"⟨a⟩ = {summary['mean_a_Ang']:.5f} Å, elapsed = {elapsed:.1f} s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nih-mc", description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run an isotherm point from a config file")
    p_run.add_argument("config", help="YAML config file (see config.example.yaml)")
    p_run.add_argument("--outdir", default=".", help="directory for outputs (default: .)")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
