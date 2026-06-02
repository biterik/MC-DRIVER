"""Round-trip test for the LAMMPS data-file parser and site-list extractor.

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nih_mc.structure import (
    H_TYPE,
    NI_TYPE,
    build_site_list,
    read_lammps_data,
)


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_nih(tmp_path_factory):
    """A 2×2×2 NiH structure built by tools/build_structures.py."""
    out = tmp_path_factory.mktemp("struct") / "NiH-2x2x2.data"
    subprocess.check_call(
        [
            sys.executable,
            str(REPO / "tools" / "build_structures.py"),
            "--structure", "nih",
            "--a", "3.738",
            "--n", "2",
            "--out", str(out),
        ]
    )
    return out


def test_read_lammps_data_counts(tiny_nih):
    data = read_lammps_data(str(tiny_nih))
    # 2x2x2 NiH: 4*2^3 Ni + 4*2^3 H = 32 + 32 = 64 atoms
    assert len(data.atoms) == 64
    n_ni = (data.positions(NI_TYPE)).shape[0]
    n_h = (data.positions(H_TYPE)).shape[0]
    assert n_ni == 32
    assert n_h == 32
    # box should be 2*a = 7.476 Å on each side
    L = data.box.lengths
    np.testing.assert_allclose(L, [7.476, 7.476, 7.476], atol=1e-6)


def test_build_site_list_matches_H_positions(tiny_nih):
    data = read_lammps_data(str(tiny_nih))
    sites = build_site_list(data)
    h_pos = data.positions(H_TYPE)
    assert sites.n_sites == 32
    assert sites.n_ni == 32
    np.testing.assert_allclose(sites.coords, h_pos)


def test_build_site_list_requires_full_hydride(tmp_path):
    """Empty Ni structure (no H) should fail site-list extraction."""
    out = tmp_path / "Ni-2x2x2.data"
    subprocess.check_call(
        [
            sys.executable,
            str(REPO / "tools" / "build_structures.py"),
            "--structure", "ni",
            "--a", "3.524",
            "--n", "2",
            "--out", str(out),
        ]
    )
    data = read_lammps_data(str(out))
    with pytest.raises(ValueError, match="no H atoms"):
        build_site_list(data)
