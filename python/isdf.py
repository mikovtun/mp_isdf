#!/usr/bin/env python3
"""Minimal grid ISDF demo adapted from ../isdf/isdf.py.

PySCF builds the molecular grid; isdf_ao evaluates AOs and selects anchors
with local QRCP (MAGMA on CUDA, LAPACK++ otherwise). The remaining ISDF fit
and spherical-cell Coulomb metric are NumPy reference implementations.

Build first: cmake --build build --target isdf_ao
Example: OMP_PROC_BIND=false python python/isdf.py --level 0 --rank 4
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from pyscf import dft, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import isdf_ao


AXES = {
    "ao": isdf_ao.AxisType.AtomicOrbital,
    "dx": isdf_ao.AxisType.AtomicOrbitalGradX,
    "dy": isdf_ao.AxisType.AtomicOrbitalGradY,
    "dz": isdf_ao.AxisType.AtomicOrbitalGradZ,
    "occ": isdf_ao.AxisType.OccupiedOrbital,
}


def coulomb_metric(interp_vecs, coords, weights):
    """Spherical-cell grid Coulomb metric (NumPy reference; O(Ngrid**2)).

    interp_vecs has shape (Nanchor, Ngrid) and is NOT weight-scaled.
    """
    c_sphere = (32.0 / 15.0) * np.pi**2 * (3.0 / (4.0 * np.pi)) ** (5.0 / 3.0)
    weighted = interp_vecs.T * weights[:, None]
    potential = np.empty_like(weighted)
    ngrid = len(weights)
    block_size = max(1, min(256, 8_000_000 // max(ngrid, 1)))
    for first in range(0, ngrid, block_size):
        last = min(first + block_size, ngrid)
        delta = coords[first:last, None, :] - coords[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        kernel = np.zeros_like(distance)
        np.divide(1.0, distance, out=kernel, where=distance > 0)
        rows = np.arange(first, last)
        kernel[np.arange(last - first), rows] = c_sphere * np.abs(weights[rows]) ** (-1.0 / 3.0)
        potential[first:last] = kernel @ weighted
    metric = weighted.T @ potential
    return 0.5 * (metric + metric.T)


def build_isdf(mol, mf=None, *, level=0, max_rank=0, tolerance=1e-10,
               axes=("ao",), compute_metric=True):
    """Build AO-pair ISDF, using this project's C++ evaluator and selector.

    `axes` selects the anchor-search features (ao/dx/dy/dz/occ), concatenated
    before forming their outer product. The fitted tensor remains AO-pair ISDF.
    `max_rank=0` leaves rank to the QR diagonal tolerance. Anchor indices are
    relative to the original, unpruned PySCF grid.
    """
    started = time.perf_counter()
    if isinstance(axes, str):
        axes = (axes,)
    try:
        axis_types = [AXES[axis] if isinstance(axis, str) else axis for axis in axes]
    except KeyError as exc:
        raise ValueError(f"unknown axis: {exc.args[0]!r}; choose from {list(AXES)}") from exc

    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    all_coords = np.ascontiguousarray(grids.coords, dtype=np.float64)
    all_weights = np.ascontiguousarray(grids.weights, dtype=np.float64)
    keep = all_weights != 0
    if not np.any(keep):
        raise ValueError("grid has no nonzero weights")

    context = isdf_ao.LibcintContext(mol._atm, mol._bas, mol._env)
    selector = isdf_ao.AnchorSelection(context)
    needs_occ = isdf_ao.AxisType.OccupiedOrbital in axis_types
    if needs_occ and (mf is None or not isinstance(mf, scf.hf.RHF) or
                      mol.spin != 0 or mf.mo_coeff is None):
        raise ValueError("'occ' requires a converged, closed-shell RHF mean-field object")
    if needs_occ and not mf.converged:
        raise ValueError("RHF mean-field object has not converged")
    coefficients = np.ascontiguousarray(mf.mo_coeff) if needs_occ else None
    nocc = mol.nelectron // 2 if needs_occ else 0
    anchor_indices = np.asarray(selector.select(
        all_coords, all_weights, axis_types, tolerance=tolerance,
        max_rank=max_rank, mo_coeff=coefficients, nocc=nocc,
    ), dtype=np.intp)
    if len(anchor_indices) == 0:
        raise ValueError("QRCP found no significant anchors; adjust tolerance or grid")

    # Only nonzero-weight points contribute to the ISDF fit and Coulomb metric.
    coords = all_coords[keep]
    weights = all_weights[keep]
    ao = context.eval_ao(coords)
    anchor_coords = all_coords[anchor_indices]
    anchor_weights = all_weights[anchor_indices]
    anchor_ao = context.eval_ao(anchor_coords)
    rank, nao = anchor_ao.shape
    products = (ao[:, :, None] * ao[:, None, :]).reshape(len(coords), nao * nao)
    anchor_pairs = (anchor_ao[:, :, None] * anchor_ao[:, None, :]).reshape(rank, nao * nao)
    weighted_pairs = anchor_pairs * anchor_weights[:, None]
    gram = weighted_pairs @ weighted_pairs.T
    transform = np.linalg.pinv(gram, rcond=1e-12) @ weighted_pairs
    # Preserve the original script's quadrature-weighted interpolation vectors.
    interp_vecs = (products @ transform.T * weights[:, None]).T

    metric = None
    if compute_metric:
        metric = coulomb_metric(interp_vecs / weights[None, :], coords, weights)
        metric = anchor_weights[:, None] * metric * anchor_weights[None, :]
    return {
        "anchor_indices": anchor_indices,
        "anchor_coords": anchor_coords,
        "anchor_weights": anchor_weights,
        "coordinates": coords,
        "weights": weights,
        "interp_vecs": interp_vecs,
        "coulomb_metric": metric,
        "X1": anchor_ao, "X2": anchor_ao, "X3": anchor_ao, "X4": anchor_ao,
        "elapsed": time.perf_counter() - started,
    }


def _read_xyz(path):
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f"empty XYZ file: {path}")
    try:
        count = int(lines[0].strip())
    except ValueError:
        atom_lines = lines  # Also accept headerless XYZ.
    else:
        atom_lines = lines[2:2 + count]
        if len(atom_lines) != count:
            raise ValueError(f"XYZ atom count does not match file: {path}")
    atoms = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid XYZ atom line: {line!r}")
        atoms.append((fields[0], tuple(map(float, fields[1:]))))
    if not atoms:
        raise ValueError("XYZ file contains no atoms")
    return atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--molecule", type=Path, help="XYZ file (default: H2)")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--rank", type=int, default=4, help="maximum rank (0: tolerance only)")
    parser.add_argument("--tolerance", type=float, default=1e-10,
                        help="relative QR diagonal cutoff")
    parser.add_argument("--axes", default="ao",
                        help="comma-separated anchor-search features: ao,dx,dy,dz,occ")
    parser.add_argument("--no-metric", action="store_true",
                        help="skip NumPy grid Coulomb metric")
    args = parser.parse_args()
    if args.rank < 0:
        parser.error("--rank must be nonnegative")
    if args.level < 0:
        parser.error("--level must be nonnegative")
    atom = _read_xyz(args.molecule) if args.molecule else "H 0 0 0; H 0 0 0.74"
    mol = gto.M(atom=atom, basis=args.basis, unit="Angstrom", verbose=0)
    axes = [axis.strip().lower() for axis in args.axes.split(",")]
    mf = None
    if "occ" in axes:
        mf = scf.RHF(mol).run(verbose=0)
        if not mf.converged:
            raise RuntimeError("RHF did not converge")
    result = build_isdf(mol, mf, level=args.level, max_rank=args.rank,
                        tolerance=args.tolerance, axes=axes,
                        compute_metric=not args.no_metric)
    print(f"Grid points: {len(result['coordinates'])}; AOs: {mol.nao_nr()}; "
          f"anchors: {len(result['anchor_indices'])}")
    print("Original grid anchor indices:", result["anchor_indices"])
    if result["coulomb_metric"] is not None:
        print("Coulomb metric shape:", result["coulomb_metric"].shape)
    print(f"ISDF build time: {result['elapsed']:.3f} s")


if __name__ == "__main__":
    main()
