"""Smoke test for the CLI on a trivial-backend config (no LAMMPS).

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]


def test_cli_trivial_run(tmp_path):
    # build a tiny NiH structure to provide a site list
    nih = tmp_path / "NiH.data"
    subprocess.check_call(
        [sys.executable, str(REPO / "tools" / "build_structures.py"),
         "--structure", "nih", "--a", "3.738", "--n", "3", "--out", str(nih)]
    )
    cfg = {
        "potential": {"kind": "trivial"},
        "structure_full": str(nih),
        "structure_empty": None,
        "start_from": "empty",
        "ensemble": "gc",
        "backend": "trivial",
        "T_K": 400.0,
        "mu_eV": -0.02,
        "n_blocks": 500,
        "flips_per_block": 20,
        "seed": 7,
        "out_prefix": "cli-smoke",
        "eps_eV": 0.0,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    env = {**__import__("os").environ, "PYTHONPATH": str(REPO / "src")}
    res = subprocess.run(
        [sys.executable, "-m", "nih_mc.cli", "run", str(cfg_path), "--outdir", str(tmp_path)],
        check=True, capture_output=True, text=True, env=env,
    )
    csv = tmp_path / "cli-smoke.csv"
    log = tmp_path / "cli-smoke.log"
    ysum = tmp_path / "cli-smoke-run-summary.yaml"
    assert csv.exists() and log.exists() and ysum.exists()
    # CSV should have header + 500 rows
    lines = csv.read_text().splitlines()
    assert lines[0] == "block,n_H,c,mu_eV,U_eV,a_box_Ang,P_bar,acc_ratio"
    assert len(lines) == 1 + 500
    # summary should contain mean_c near Langmuir(μ=-0.02 ε=0 T=400)
    summary = yaml.safe_load(ysum.read_text())
    assert "mean_c" in summary
    assert 0 < summary["mean_c"] < 1
