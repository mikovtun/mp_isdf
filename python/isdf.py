#!/usr/bin/env python3
"""Minimal grid ISDF demo adapted from ../isdf/isdf.py.

PySCF builds the molecular grid; mp_isdf evaluates AOs and selects anchors
with local QRCP (MAGMA on device, LAPACK++ on host). The ISDF fit remains
in NumPy; the Coulomb metric uses integral-direct AO-pair Cholesky in C++.

Build first: cmake --build build --target mp_isdf_python
Example: OMP_PROC_BIND=false python python/isdf.py --level 0 --anchor-rank 0
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from pyscf import dft, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import mp_isdf


AXES = {
    "ao": mp_isdf.AxisType.AtomicOrbital,
    "dx": mp_isdf.AxisType.AtomicOrbitalGradX,
    "dy": mp_isdf.AxisType.AtomicOrbitalGradY,
    "dz": mp_isdf.AxisType.AtomicOrbitalGradZ,
    "occ": mp_isdf.AxisType.OccupiedOrbital,
}


def grid_coulomb_metric(interp_vecs, coords, weights):
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


def build_isdf(mol, mf=None, *, level=0, anchor_rank=0, anchor_tol=1e-10,
               axes=("ao",), compute_metric=True, space="auto",
               metric_algorithm="unblocked_greedy", metric_tol=1e-12,
               metric_rank=0, metric_batch_size=64, report_timings=False):
    """Build AO-pair ISDF, using this project's C++ evaluator and selector.

    `axes` selects the anchor-search features (ao/dx/dy/dz/occ), concatenated
    before forming their outer product. The fitted tensor remains AO-pair ISDF.
    `anchor_rank=0` leaves rank to the QR diagonal `anchor_tol`. Anchor indices
    are relative to the original, unpruned PySCF grid. `space` selects where AO
    evaluation and QRCP run (auto, host, device); the NumPy fit stays on host.
    The integral-direct metric always runs on the host; `metric_tol` is relative
    to the largest initial AO-pair ERI diagonal. `report_timings` prints each
    completed stage immediately; timings are also returned in the result.
    """
    started = time.perf_counter()
    last = started
    timings = {}

    def record(stage):
        nonlocal last
        now = time.perf_counter()
        timings[stage] = now - last
        last = now
        if report_timings:
            print(f"  {stage:18s} {timings[stage]:10.6f} s  "
                  f"(elapsed {last - started:.6f} s)", flush=True)

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
    record("grid")

    context = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
    selector = mp_isdf.AnchorSelection(context)
    record("context")
    needs_occ = mp_isdf.AxisType.OccupiedOrbital in axis_types
    if needs_occ and (mf is None or not isinstance(mf, scf.hf.RHF) or
                      mol.spin != 0 or mf.mo_coeff is None):
        raise ValueError("'occ' requires a converged, closed-shell RHF mean-field object")
    if needs_occ and not mf.converged:
        raise ValueError("RHF mean-field object has not converged")
    coefficients = np.ascontiguousarray(mf.mo_coeff) if needs_occ else None
    nocc = mol.nelectron // 2 if needs_occ else 0
    anchor_indices = np.asarray(selector.select(
        all_coords, all_weights, axis_types, tolerance=anchor_tol,
        max_rank=anchor_rank, mo_coeff=coefficients, nocc=nocc, space=space,
    ), dtype=np.intp)
    if len(anchor_indices) == 0:
        raise ValueError("QRCP found no significant anchors; adjust tolerance or grid")
    record("anchor_selection")

    # Only nonzero-weight points contribute to the ISDF fit and Coulomb metric.
    coords = all_coords[keep]
    weights = all_weights[keep]
    ao = context.eval_ao(coords, space=space)
    anchor_coords = all_coords[anchor_indices]
    anchor_weights = all_weights[anchor_indices]
    anchor_ao = context.eval_ao(anchor_coords, space=space)
    record("ao_evaluation")
    rank, nao = anchor_ao.shape
    products = (ao[:, :, None] * ao[:, None, :]).reshape(len(coords), nao * nao)
    anchor_pairs = (anchor_ao[:, :, None] * anchor_ao[:, None, :]).reshape(rank, nao * nao)
    weighted_pairs = anchor_pairs * anchor_weights[:, None]
    gram = weighted_pairs @ weighted_pairs.T
    transform = np.linalg.pinv(gram, rcond=1e-12) @ weighted_pairs
    # Unweighted interpolation vectors; quadrature weights are applied only
    # where an integral is evaluated, not stored in the vectors themselves.
    interp_vecs = (products @ transform.T).T
    record("isdf_fit")

    overlap = mol.intor("int1e_ovlp")
    overlap_approx = np.einsum(
        "ku,kv,k->uv", anchor_ao, anchor_ao,
        anchor_weights * (interp_vecs @ weights), optimize=True,
    )
    overlap_relative_error = (np.linalg.norm(overlap_approx - overlap) /
                              max(np.linalg.norm(overlap), 1e-30))
    record("overlap_check")

    metric = None
    if compute_metric:
        metric = mp_isdf.coulomb_metric(
            context, transform, algorithm=metric_algorithm,
            tolerance=metric_tol, max_rank=metric_rank,
            batch_size=metric_batch_size,
        )
        metric = anchor_weights[:, None] * metric * anchor_weights[None, :]
        record("coulomb_metric")
    else:
        timings["coulomb_metric"] = 0.0
        if report_timings:
            print("  coulomb_metric     skipped", flush=True)
    return {
        "anchor_indices": anchor_indices,
        "anchor_coords": anchor_coords,
        "anchor_weights": anchor_weights,
        "coordinates": coords,
        "weights": weights,
        "interp_vecs": interp_vecs,
        "overlap_reconstruction": overlap_approx,
        "overlap_relative_error": overlap_relative_error,
        "coulomb_metric": metric,
        "X1": anchor_ao, "X2": anchor_ao, "X3": anchor_ao, "X4": anchor_ao,
        "elapsed": last - started,
        "timings": timings,
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
    parser.add_argument("--anchor-rank", type=int, default=0,
                        help="maximum anchor rank (0: tolerance only; default: 0)")
    parser.add_argument("--anchor-tol", type=float, default=1e-10,
                        help="relative QR diagonal cutoff")
    parser.add_argument("--axes", default="ao",
                        help="comma-separated anchor-search features: ao,dx,dy,dz,occ")
    parser.add_argument("--space", choices=("auto", "host", "device"), default="auto",
                        help="where AO evaluation and QRCP run")
    parser.add_argument("--metric-tol", type=float, default=1e-12,
                        help="relative residual-diagonal cutoff for integral-direct Cholesky")
    parser.add_argument("--metric-rank", type=int, default=0,
                        help="maximum ERI Cholesky rank (0: tolerance only)")
    parser.add_argument("--no-metric", action="store_true",
                        help="skip integral-direct Coulomb metric")
    args = parser.parse_args()
    if args.anchor_rank < 0:
        parser.error("--anchor-rank must be nonnegative")
    if not np.isfinite(args.anchor_tol) or args.anchor_tol < 0:
        parser.error("--anchor-tol must be finite and nonnegative")
    if args.level < 0:
        parser.error("--level must be nonnegative")
    if not np.isfinite(args.metric_tol) or args.metric_tol < 0:
        parser.error("--metric-tol must be finite and nonnegative")
    if args.metric_rank < 0:
        parser.error("--metric-rank must be nonnegative")
    atom = _read_xyz(args.molecule) if args.molecule else "H 0 0 0; H 0 0 0.74"
    mol = gto.M(atom=atom, basis=args.basis, unit="Angstrom", verbose=0)
    axes = [axis.strip().lower() for axis in args.axes.split(",")]
    mf = None
    if "occ" in axes:
        mf = scf.RHF(mol).run(verbose=0)
        if not mf.converged:
            raise RuntimeError("RHF did not converge")
    result = build_isdf(mol, mf, level=args.level, anchor_rank=args.anchor_rank,
                        anchor_tol=args.anchor_tol, axes=axes,
                        compute_metric=not args.no_metric, space=args.space,
                        metric_tol=args.metric_tol, metric_rank=args.metric_rank,
                        report_timings=True)
    print(f"Grid points: {len(result['coordinates'])}; AOs: {mol.nao_nr()}; "
          f"anchors: {len(result['anchor_indices'])}")
    print("Original grid anchor indices:", result["anchor_indices"])
    if args.anchor_rank and len(result["anchor_indices"]) == args.anchor_rank:
        print(f"Anchor count equals --anchor-rank {args.anchor_rank}; use --anchor-rank 0 "
              "for tolerance-only selection or a larger value for a higher cap.")
    print(f"Overlap reconstruction relative error: {result['overlap_relative_error']:.3e}")
    if result["coulomb_metric"] is not None:
        print("Coulomb metric shape:", result["coulomb_metric"].shape)
    print(f"ISDF build time: {result['elapsed']:.6f} s")


if __name__ == "__main__":
    main()
