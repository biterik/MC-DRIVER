"""Output writers (SPEC §8): isotherm CSV, run-summary YAML, log file.

Precision conventions are fixed by SPEC §8:
  c             %.6f
  mu, U         %.6f eV
  a_box         %.5f Å
  P             %.4g bar

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, TextIO

from .mc import BlockRecord


CSV_HEADER = "block,n_H,c,mu_eV,U_eV,a_box_Ang,P_bar,acc_ratio\n"


def format_block_row(rec: BlockRecord, mu_eV: float) -> str:
    return (
        f"{rec.block},{rec.n_h},{rec.c:.6f},{mu_eV:.6f},{rec.U_eV:.6f},"
        f"{rec.a_box_Ang:.5f},{rec.P_bar:.4g},{rec.acc_ratio:.6f}\n"
    )


@dataclass
class IsothermCsvWriter:
    path: Path
    mu_eV: float
    _fh: Optional[TextIO] = field(default=None, repr=False)

    def open(self) -> None:
        self._fh = open(self.path, "w")
        self._fh.write(CSV_HEADER)
        self._fh.flush()

    def write(self, rec: BlockRecord) -> None:
        assert self._fh is not None
        self._fh.write(format_block_row(rec, self.mu_eV))
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def write_run_summary(path: Path, summary: dict) -> None:
    """Write a YAML run-summary (see SPEC §8). Falls back to a tiny inline writer
    if pyyaml isn't available, so the package doesn't hard-require yaml at runtime."""
    try:
        import yaml  # type: ignore
        with open(path, "w") as f:
            yaml.safe_dump(summary, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        with open(path, "w") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")


def block_records_to_csv(path: Path, records: Iterable[BlockRecord], mu_eV: float) -> None:
    w = IsothermCsvWriter(path=Path(path), mu_eV=mu_eV)
    w.open()
    for r in records:
        w.write(r)
    w.close()
