#!/usr/bin/env python3
"""
build_structures.py — write LAMMPS data files for fcc Ni and rock-salt B1 NiH.

These are IDEAL (unrelaxed) structures, sufficient for developing and unit-testing
the MC driver (site-list extraction, create/delete plumbing, the analytic and
enumeration tests). For the production / Korbmacher-anchor runs, swap in the
RELAXED structures produced by LLM-LMPS thread 01.

Conventions (must match the driver's expectations):
  - atom types: Ni = 1, H = 2
  - atom_style atomic ("Atoms # atomic": id type x y z)
  - orthogonal periodic box 0..n*a
  - In the full hydride, the H atoms ARE the octahedral site list, in the order
    emitted here (cell-major, then the 4-atom basis below).

fcc Ni conventional-cell basis (fractional):
  (0,0,0) (1/2,1/2,0) (1/2,0,1/2) (0,1/2,1/2)
rock-salt H sublattice = octahedral sites (fractional):
  (1/2,0,0) (0,1/2,0) (0,0,1/2) (1/2,1/2,1/2)

Usage:
  python build_structures.py --structure nih --a 3.738 --n 10 --out NiH-B1-10x10x10.data
  python build_structures.py --structure ni  --a 3.524 --n 10 --out Ni-fcc-10x10x10.data

Author: Erik Bitzek <e.bitzek@mpi-susmat.de>
Implementation and testing by Claude Code (Anthropic).

Part of MC-DRIVER. Copyright (C) 2026 Erik Bitzek.
Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
See the LICENSE file at the repository root for the full license text.
"""
import argparse

NI_BASIS = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
H_BASIS  = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]
MASS = {1: 58.6934, 2: 1.008}


def lattice(basis, n, a):
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for (bx, by, bz) in basis:
                    pts.append(((i + bx) * a, (j + by) * a, (k + bz) * a))
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", choices=["ni", "nih"], required=True)
    p.add_argument("--a", type=float, required=True, help="lattice constant (Angstrom)")
    p.add_argument("--n", type=int, default=10, help="cells per side (n x n x n)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    L = args.n * args.a
    ni = lattice(NI_BASIS, args.n, args.a)
    h = lattice(H_BASIS, args.n, args.a) if args.structure == "nih" else []

    atoms = [(1, x, y, z) for (x, y, z) in ni] + [(2, x, y, z) for (x, y, z) in h]

    with open(args.out, "w") as f:
        f.write(f"# {args.structure} ideal, a={args.a} A, {args.n}x{args.n}x{args.n}, "
                f"Ni=type1 H=type2 (H positions = octahedral site list)\n\n")
        f.write(f"{len(atoms)} atoms\n2 atom types\n\n")
        f.write(f"0.0 {L:.8f} xlo xhi\n0.0 {L:.8f} ylo yhi\n0.0 {L:.8f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t in (1, 2):
            f.write(f"{t} {MASS[t]}\n")
        f.write("\nAtoms # atomic\n\n")
        for idx, (t, x, y, z) in enumerate(atoms, start=1):
            f.write(f"{idx} {t} {x:.8f} {y:.8f} {z:.8f}\n")

    n_ni, n_h = len(ni), len(h)
    print(f"wrote {args.out}: {len(atoms)} atoms (Ni={n_ni}, H={n_h}), box {L:.5f} A")


if __name__ == "__main__":
    main()
