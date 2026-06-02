"""CLI smoke test on `pair_style zero` — exercises the LAMMPS-statics path
through the YAML loader without needing a real potential file.

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


def test_cli_lammps_zero(tmp_path):
    nih = tmp_path / "NiH.data"
    subprocess.check_call(
        [sys.executable, str(REPO / "tools" / "build_structures.py"),
         "--structure", "nih", "--a", "3.738", "--n", "3", "--out", str(nih)]
    )
    cfg = {
        "potential": {
            "kind": "zero",
            "pair_style": "zero 1.0",
            "files": [],
        },
        "structure_full": str(nih),
        "structure_empty": None,
        "start_from": "full",
        "ensemble": "gc",
        "backend": "statics",
        "T_K": 400.0,
        "mu_eV": -0.02,
        "n_blocks": 200,
        "flips_per_block": 10,
        "seed": 11,
        "out_prefix": "cli-zero",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    env = {**__import__("os").environ, "PYTHONPATH": str(REPO / "src")}
    res = subprocess.run(
        [sys.executable, "-m", "nih_mc.cli", "run", str(cfg_path), "--outdir", str(tmp_path)],
        check=True, capture_output=True, text=True, env=env,
    )
    csv = tmp_path / "cli-zero.csv"
    ysum = tmp_path / "cli-zero-run-summary.yaml"
    assert csv.exists() and ysum.exists()
    lines = csv.read_text().splitlines()
    assert len(lines) == 1 + 200
    summary = yaml.safe_load(ysum.read_text())
    assert summary["lammps_version"] is not None
