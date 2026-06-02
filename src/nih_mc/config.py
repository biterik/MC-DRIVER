"""Config loader (SPEC §9). Maps a YAML config to the per-component dataclasses.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PotentialConfig:
    kind: str                    # "eam" | "meam" | "zero" | "trivial"
    pair_style: str = ""         # e.g. "eam/alloy", "meam", "zero 1.0"
    files: list = field(default_factory=list)
    potdir: Optional[str] = None


@dataclass
class RunConfig:
    potential: PotentialConfig
    structure_full: str
    structure_empty: Optional[str]
    start_from: str              # "full" | "empty"
    ensemble: str                # "gc" | "vcsgc"
    backend: str                 # "statics" | "md_npt" | "trivial"
    T_K: float
    # GC
    mu_eV: Optional[float] = None
    # VC-SGC
    delta_mu_eV: Optional[float] = None
    c0: Optional[float] = None
    kappa: Optional[float] = None
    # MC cadence
    n_blocks: int = 2000
    flips_per_block: int = 50
    # statics
    min_etol: float = 1.0e-8
    min_ftol: float = 1.0e-4
    cell_relax: bool = False
    # md_npt
    md_steps_equil: int = 2000
    md_steps_sample: int = 2000
    md_P_bar: float = 0.0
    # bookkeeping
    seed: int = 12345
    out_prefix: str = "isotherm-run"
    # trivial (unit-test) backend params; ignored unless backend == 'trivial'
    eps_eV: float = 0.0


def load_yaml_config(path: str) -> RunConfig:
    import yaml
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    pot = raw.get("potential", {})
    p = PotentialConfig(
        kind=pot.get("kind", "trivial"),
        pair_style=pot.get("pair_style", ""),
        files=list(pot.get("files", []) or []),
        potdir=pot.get("potdir"),
    )
    return RunConfig(
        potential=p,
        structure_full=raw.get("structure_full"),
        structure_empty=raw.get("structure_empty"),
        start_from=raw.get("start_from", "full"),
        ensemble=raw.get("ensemble", "gc"),
        backend=raw.get("backend", "statics"),
        T_K=float(raw.get("T_K", 300.0)),
        mu_eV=raw.get("mu_eV"),
        delta_mu_eV=raw.get("delta_mu_eV"),
        c0=raw.get("c0"),
        kappa=raw.get("kappa"),
        n_blocks=int(raw.get("n_blocks", 2000)),
        flips_per_block=int(raw.get("flips_per_block", 50)),
        min_etol=float(raw.get("min_etol", 1.0e-8)),
        min_ftol=float(raw.get("min_ftol", 1.0e-4)),
        cell_relax=bool(raw.get("cell_relax", False)),
        md_steps_equil=int(raw.get("md_steps_equil", 2000)),
        md_steps_sample=int(raw.get("md_steps_sample", 2000)),
        md_P_bar=float(raw.get("md_P_bar", 0.0)),
        seed=int(raw.get("seed", 12345)),
        out_prefix=raw.get("out_prefix", "isotherm-run"),
        eps_eV=float(raw.get("eps_eV", 0.0)),
    )


def expand_pair_coeff(p: PotentialConfig, structure_path: str) -> list[str]:
    """Build pair_coeff lines for the chosen potential family.

    SPEC §3 lists the canonical forms:
      EAM:   `pair_coeff * * <potdir>/<eam.alloy> Ni H`
      MEAM:  `pair_coeff * * <potdir>/<lib.meam> Ni H <potdir>/<param.meam> Ni H`
    """
    if not p.potdir:
        prefix = ""
    else:
        prefix = p.potdir.rstrip("/") + "/"
    files_abs = [prefix + f if not f.startswith("/") else f for f in p.files]

    if p.kind == "eam":
        if len(files_abs) != 1:
            raise ValueError("EAM expects exactly one .eam.alloy file")
        return [f"pair_coeff * * {files_abs[0]} Ni H"]
    if p.kind == "meam":
        if len(files_abs) != 2:
            raise ValueError("MEAM expects library + parameter files")
        lib, par = files_abs
        return [f"pair_coeff * * {lib} Ni H {par} Ni H"]
    if p.kind == "zero":
        return ["pair_coeff * *"]
    raise ValueError(f"unknown potential kind {p.kind!r}")
