#!/usr/bin/env python3
"""Minimal grid ISDF demo adapted from ../isdf/isdf.py.

PySCF builds the molecular grid; mp_isdf evaluates AOs and selects anchors
with local QRCP (MAGMA on device, LAPACK++ on host). The ISDF fit remains
in NumPy; the Coulomb metric uses integral-direct AO-pair Cholesky in C++.
Derivative targets are spatial electronic-coordinate AO derivatives (not
nuclear gradients or assembled four-component spinor integrals).

Build first: cmake --build build --target mp_isdf_python
Example: OMP_PROC_BIND=false python python/isdf.py --level 0 --anchor-rank 0
"""

import argparse
from itertools import combinations, product
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


def _target_labels(target):
    """Four AO slots (u,v,w,x); zero or two carry first derivatives."""
    if len(target) != 4 or any(c not in "0xyz" for c in target) or \
            sum(c != "0" for c in target) not in (0, 2):
        raise ValueError("target must be '0000' or four 0/x/y/z slots "
                         "with exactly two derivatives")
    return target[:2], target[2:]


def grid_cross_metric(left, right, coords, weights):
    """Grid Coulomb metric between two independently fitted channel sets."""
    c_sphere = (32.0 / 15.0) * np.pi**2 * (3.0 / (4.0 * np.pi)) ** (5.0 / 3.0)
    weighted_left = left.T * weights[:, None]
    weighted_right = right.T * weights[:, None]
    ngrid = len(weights)
    result = np.zeros((len(left), len(right)))
    block_size = max(1, min(256, 8_000_000 // max(ngrid, 1)))
    for first in range(0, ngrid, block_size):
        last = min(first + block_size, ngrid)
        distance = np.linalg.norm(coords[first:last, None] - coords[None, :], axis=-1)
        kernel = np.zeros_like(distance)
        np.divide(1.0, distance, out=kernel, where=distance > 0)
        rows = np.arange(first, last)
        kernel[np.arange(last - first), rows] = c_sphere * np.abs(weights[rows]) ** (-1.0 / 3.0)
        result += weighted_left[first:last].T @ (kernel @ weighted_right)
    return result


def _reference_target(mol, target, cache=None):
    """PySCF scalar or two-slot derivative ERI in requested u,v,w,x order."""
    _target_labels(target)
    if target == "0000":
        if cache is not None and "int2e" in cache:
            return cache["int2e"]
        tensor = mol.intor("int2e", aosym="s1")
        if cache is not None:
            cache["int2e"] = tensor
        return tensor
    slots = [i for i, c in enumerate(target) if c != "0"]
    if slots in ([0, 1], [2, 3]):
        kernel = "int2e_ipvip1"
        perm = [0, 1, 2, 3] if slots == [0, 1] else [2, 3, 0, 1]
    else:
        kernel = "int2e_ip1ip2"
        perm = [0, 1, 2, 3]
        if slots[0] == 1:
            perm[0], perm[1] = perm[1], perm[0]
        if slots[1] == 3:
            perm[2], perm[3] = perm[3], perm[2]
    component = 3 * "xyz".index(target[perm[0]]) + "xyz".index(target[perm[1] if kernel == "int2e_ipvip1" else perm[2]])
    if cache is not None and kernel in cache:
        tensor = cache[kernel]
    else:
        tensor = mol.intor(kernel, comp=9, aosym="s1")
        if cache is not None:
            cache[kernel] = tensor
    return tensor[component].transpose(np.argsort(perm))


def evaluate_isdf_jk(result, dm):
    """Return (J, K) for a real symmetric AO density matrix.

    The metric in `result` already includes both anchor quadrature weights.
    X[p,mu] = AO_mu(anchor_p), and (mu nu|lambda sigma) is approximated by
    sum_pq X[p,mu] X[p,nu] Z[p,q] X[q,lambda] X[q,sigma].
    """
    x = np.asarray(result["X1"])
    z = result["coulomb_metric"]
    if z is None:
        raise ValueError("a Coulomb metric is required to evaluate J and K")
    density = np.asarray(dm, dtype=np.float64)
    if density.shape != (x.shape[1], x.shape[1]) or not np.isfinite(density).all():
        raise ValueError("dm must be a finite AO density matrix of shape (nao,nao)")
    if not np.allclose(density, density.T, rtol=1e-12, atol=1e-12):
        raise ValueError("dm must be symmetric")
    if np.shape(z) != (len(x), len(x)):
        raise ValueError("Coulomb metric has the wrong anchor dimensions")
    projected_density = x @ density @ x.T
    j = x.T @ ((z @ np.diag(projected_density))[:, None] * x)
    k = x.T @ (z * projected_density) @ x
    return 0.5 * (j + j.T), 0.5 * (k + k.T)


def pair_space_error_norms(pair_grid, pair_anchor, anchor_weights, theta, grid_weights,
                           block_size=256):
    """Absolute and relative grid-weighted L2 error of a fitted pair channel.

    Pair columns retain their ordered AO indices. Absolute grid weights define
    a positive norm even when the quadrature contains negative weights.
    """
    weighted_anchor = pair_anchor * anchor_weights[:, None]
    residual_sq = exact_sq = 0.0
    for first in range(0, len(grid_weights), block_size):
        last = min(first + block_size, len(grid_weights))
        exact = pair_grid[first:last]
        reconstructed = theta[:, first:last].T @ weighted_anchor
        weight = np.abs(grid_weights[first:last])
        residual_sq += np.einsum("g,gi,gi->", weight, exact - reconstructed,
                                 exact - reconstructed, optimize=True)
        exact_sq += np.einsum("g,gi,gi->", weight, exact, exact, optimize=True)
    absolute = np.sqrt(residual_sq)
    relative = absolute / np.sqrt(exact_sq) if exact_sq else (0.0 if absolute == 0 else np.inf)
    return {"l2": absolute, "relative": relative}


def build_isdf(mol, mf=None, *, level=0, anchor_rank=0, anchor_tol=1e-10,
               axes=("ao",), compute_metric=True, space="auto",
               metric_algorithm="unblocked_greedy", metric_tol=1e-12,
               metric_rank=0, metric_batch_size=64, report_timings=False,
               derivative_targets=(), selection_only=False, debug=False,
               metric_seed=0, pair_space_error=False):
    """Build AO-pair ISDF, using this project's C++ evaluator and selector.

    `axes` selects the anchor-search features (ao/dx/dy/dz/occ), concatenated
    before forming their outer product. The fitted tensor remains AO-pair ISDF.
    `anchor_rank=0` leaves rank to the QR diagonal `anchor_tol`. Anchor indices
    are relative to the original, unpruned PySCF grid. `space` selects where AO
    evaluation and QRCP run (auto, host, device); the NumPy fit stays on host.
    `metric_algorithm="unblocked_greedy"` uses the host integral-direct
    Cholesky metric (`metric_tol` is relative to the largest initial ERI
    diagonal); `"unblocked_uniform"` samples pivots uniformly among pairs
    with residual above the cutoff, reproducibly with `metric_seed`.
    `"grid"` uses the spherical-cell grid approximation instead.
    `derivative_targets` lists four-slot strings such as '00xy' and 'x00y',
    or 'all' for all 54 two-derivative spatial ERIs. '0000' selects the
    ordinary AO ISDF path; with `--compare` it compares the full scalar ERI.
    The four slots are (u,v,w,x);
    'x', 'y', 'z' mean electronic-coordinate first derivatives on individual
    AOs, not nuclear derivatives or assembled spinor integrals. Derivative
    channels are fitted separately from `axes` on the selected anchors.
    `target_factors[target]` exposes the four target-specific anchor matrices
    X1..X4 and their anchor-weighted cross Coulomb metric. The top-level
    X1..X4 and `coulomb_metric` always describe ordinary AO ISDF.
    `compute_metric=False` skips both scalar and derivative metrics.
    `selection_only=True` returns immediately after anchor selection (and
    requires compute_metric=False). `pair_space_error=True` checks the grid-
    weighted L2 reconstruction of ordinary AO pairs and each distinct requested
    derivative pair channel, independently of `compute_metric`. `debug` logs
    greedy Cholesky pivots and suppresses live timing output; timings remain
    available in the result.
    Otherwise `report_timings` prints completed stages immediately.
    """
    started = time.perf_counter()
    last = started
    timings = {}

    def record(stage):
        nonlocal last
        now = time.perf_counter()
        timings[stage] = now - last
        last = now
        if report_timings and not debug:
            print(f"  {stage:18s} {timings[stage]:10.6f} s  "
                  f"(elapsed {last - started:.6f} s)", flush=True)

    if isinstance(axes, str):
        axes = (axes,)
    try:
        axis_types = [AXES[axis] if isinstance(axis, str) else axis for axis in axes]
    except KeyError as exc:
        raise ValueError(f"unknown axis: {exc.args[0]!r}; choose from {list(AXES)}") from exc
    if metric_algorithm not in ("grid", "unblocked_greedy", "unblocked_uniform"):
        raise ValueError(f"unknown metric algorithm: {metric_algorithm!r}")
    if not isinstance(metric_seed, (int, np.integer)) or not 0 <= metric_seed < 2**64:
        raise ValueError("metric_seed must be an unsigned 64-bit integer")
    if selection_only and compute_metric:
        raise ValueError("selection_only requires compute_metric=False")
    if selection_only and pair_space_error:
        raise ValueError("pair_space_error requires fitting; disable selection_only")
    if isinstance(derivative_targets, str):
        derivative_targets = (derivative_targets,)
    derivative_targets = tuple(dict.fromkeys(derivative_targets))
    if "all" in derivative_targets:
        if derivative_targets != ("all",):
            raise ValueError("'all' cannot be combined with individual derivative targets")
        derivative_targets = tuple("".join("xyz"[d[0]] if i == slots[0] else
                                           "xyz"[d[1]] if i == slots[1] else "0"
                                           for i in range(4))
                                   for slots in combinations(range(4), 2)
                                   for d in product(range(3), repeat=2))
    for target in derivative_targets:
        _target_labels(target)
    # The scalar channel already belongs to the ordinary ISDF result.
    scalar_target_requested = "0000" in derivative_targets
    derivative_targets = tuple(t for t in derivative_targets if t != "0000")

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
    if selection_only:
        return {
            "anchor_indices": anchor_indices,
            "anchor_coords": all_coords[anchor_indices],
            "anchor_weights": all_weights[anchor_indices],
            "coordinates": all_coords[keep],
            "weights": all_weights[keep],
            "interp_vecs": None,
            "overlap_reconstruction": None,
            "overlap_relative_error": None,
            "pair_space_errors": None,
            "coulomb_metric": None,
            "derivative_integrals": {},
            "target_factors": {},
            "X1": None, "X2": None, "X3": None, "X4": None,
            "elapsed": last - started,
            "timings": timings,
        }

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
    pair_space_errors = None
    if pair_space_error:
        pair_space_errors = {
            "scalar": pair_space_error_norms(products, anchor_pairs, anchor_weights,
                                             interp_vecs, weights)
        }
        record("pair_space_check")

    overlap = mol.intor("int1e_ovlp")
    overlap_approx = np.einsum(
        "ku,kv,k->uv", anchor_ao, anchor_ao,
        anchor_weights * (interp_vecs @ weights), optimize=True,
    )
    overlap_relative_error = (np.linalg.norm(overlap_approx - overlap) /
                              max(np.linalg.norm(overlap), 1e-30))
    record("overlap_check")

    def scalar_metric():
        if metric_algorithm == "grid":
            raw = grid_coulomb_metric(interp_vecs, coords, weights)
        else:
            raw = mp_isdf.coulomb_metric(
                context, transform, algorithm=metric_algorithm,
                tolerance=metric_tol, max_rank=metric_rank,
                batch_size=metric_batch_size, debug=debug, seed=metric_seed,
            )
        return anchor_weights[:, None] * raw * anchor_weights[None, :]

    metric = None
    if compute_metric:
        metric = scalar_metric()
        record("coulomb_metric")
    else:
        timings["coulomb_metric"] = 0.0
        if report_timings and not debug:
            print("  coulomb_metric     skipped", flush=True)

    derivative_integrals = {}
    target_factors = {}
    if metric is not None and scalar_target_requested:
        target_factors["0000"] = {
            "X1": anchor_ao, "X2": anchor_ao,
            "X3": anchor_ao, "X4": anchor_ao,
            "coulomb_metric": metric,
        }
    if derivative_targets and (compute_metric or pair_space_error):
        gradient = context.eval_ao_gradient(coords, space=space)
        anchor_gradient = context.eval_ao_gradient(anchor_coords, space=space)
        fields = {"0": ao, **dict(zip("xyz", gradient))}
        anchor_fields = {"0": anchor_ao, **dict(zip("xyz", anchor_gradient))}
        channels = {}
        cross_metrics = {}

        def channel(label):
            if label not in channels:
                left, right = label
                pair_grid = (fields[left][:, :, None] *
                             fields[right][:, None, :]).reshape(len(coords), nao*nao)
                pair_anchor = (anchor_fields[left][:, :, None] *
                               anchor_fields[right][:, None, :]).reshape(rank, nao*nao)
                # Fit each ordered channel independently, even with shared anchors.
                coefficients = np.linalg.pinv(pair_anchor * anchor_weights[:, None],
                                              rcond=1e-12).T
                theta = (pair_grid @ coefficients.T).T
                channels[label] = pair_anchor, coefficients, theta
            return channels[label]

        if pair_space_error:
            labels = dict.fromkeys(label for target in derivative_targets
                                   for label in _target_labels(target))
            for label in labels:
                pair_anchor, _, theta = channel(label)
                left, right = label
                pair_grid = (fields[left][:, :, None] *
                             fields[right][:, None, :]).reshape(len(coords), nao*nao)
                pair_space_errors[label] = pair_space_error_norms(
                    pair_grid, pair_anchor, anchor_weights, theta, weights)
            record("derivative_pair_space_check")

        if compute_metric:
            for target in derivative_targets:
                target_started = time.perf_counter()
                left_label, right_label = _target_labels(target)
                left_pairs, left_coeff, left_theta = channel(left_label)
                right_pairs, right_coeff, right_theta = channel(right_label)
                pair = left_label, right_label
                if pair not in cross_metrics:
                    if metric_algorithm == "grid":
                        raw = grid_cross_metric(left_theta, right_theta, coords, weights)
                    else:
                        labels = {"0": 0, "x": 1, "y": 2, "z": 3}
                        if debug:
                            print(f"[cholesky target {target}]", file=sys.stderr, flush=True)
                        raw = mp_isdf.derivative_cross_metric(
                            context, left_coeff, tuple(labels[c] for c in left_label),
                            right_coeff, tuple(labels[c] for c in right_label),
                            tolerance=metric_tol, max_rank=metric_rank, debug=debug,
                            algorithm=metric_algorithm, seed=metric_seed,
                        )
                    cross_metrics[pair] = anchor_weights[:, None] * raw * anchor_weights[None, :]
                target_factors[target] = {
                    "X1": anchor_fields[left_label[0]],
                    "X2": anchor_fields[left_label[1]],
                    "X3": anchor_fields[right_label[0]],
                    "X4": anchor_fields[right_label[1]],
                    "coulomb_metric": cross_metrics[pair],
                }
                derivative_integrals[target] = (left_pairs.T @ cross_metrics[pair] @ right_pairs
                                                ).reshape(nao, nao, nao, nao)
                if report_timings and not debug:
                    print(f"    target {target:<4s} {time.perf_counter()-target_started:10.6f} s",
                          flush=True)
            record("derivative_integrals")
    return {
        "anchor_indices": anchor_indices,
        "anchor_coords": anchor_coords,
        "anchor_weights": anchor_weights,
        "coordinates": coords,
        "weights": weights,
        "interp_vecs": interp_vecs,
        "overlap_reconstruction": overlap_approx,
        "overlap_relative_error": overlap_relative_error,
        "pair_space_errors": pair_space_errors,
        "coulomb_metric": metric,
        "derivative_integrals": derivative_integrals,
        "target_factors": target_factors,
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
    parser.add_argument("--metric", choices=("unblocked_greedy", "unblocked_uniform", "grid"),
                        default="unblocked_greedy",
                        help="Coulomb metric algorithm (default: unblocked_greedy)")
    parser.add_argument("--metric-tol", type=float, default=1e-12,
                        help="relative Cholesky residual cutoff (ignored for --metric grid)")
    parser.add_argument("--metric-rank", type=int, default=0,
                        help="maximum ERI Cholesky rank (0: tolerance only; ignored for grid)")
    parser.add_argument("--metric-seed", type=int, default=0,
                        help="random seed for unblocked_uniform (default: 0)")
    parser.add_argument("--metric-batch-size", type=int, default=64,
                        help="AO projection rows per BLAS++ batch (default: 64)")
    parser.add_argument("--derivative-target", action="append", default=[],
                        help="ERI slots (0000 for ordinary AOs; e.g. 00xy or x00y for derivatives); "
                             "repeat, or use 'all' for two derivatives")
    parser.add_argument("--pair-space-error", action="store_true",
                        help="check grid-weighted L2 pair-product reconstruction, including requested derivative channels")
    parser.add_argument("--no-metric", action="store_true",
                        help="skip scalar and derivative metrics; keep the AO fit and overlap check")
    parser.add_argument("--compare", action="store_true",
                        help="compare full four-index ERIs for explicit targets; "
                             "otherwise compare ISDF J/K with PySCF")
    parser.add_argument("--debug", action="store_true",
                        help="log each greedy Cholesky pivot and print timings at the end")
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
    if args.metric_batch_size <= 0:
        parser.error("--metric-batch-size must be positive")
    if not 0 <= args.metric_seed < 2**64:
        parser.error("--metric-seed must be an unsigned 64-bit integer")
    if args.compare and args.no_metric:
        parser.error("--compare requires a metric (omit --no-metric)")
    runner_started = time.perf_counter()
    comparison_timings = []
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
                        metric_algorithm=args.metric, metric_tol=args.metric_tol,
                        metric_rank=args.metric_rank,
                        metric_batch_size=args.metric_batch_size,
                        metric_seed=args.metric_seed,
                        derivative_targets=args.derivative_target,
                        pair_space_error=args.pair_space_error,
                        report_timings=not args.debug, debug=args.debug)
    print(f"Grid points: {len(result['coordinates'])}; AOs: {mol.nao_nr()}; "
          f"anchors: {len(result['anchor_indices'])}")
    if args.anchor_rank and len(result["anchor_indices"]) == args.anchor_rank:
        print(f"Anchor count equals --anchor-rank {args.anchor_rank}; use --anchor-rank 0 "
              "for tolerance-only selection or a larger value for a higher cap.")
    if result["overlap_relative_error"] is not None:
        print(f"Overlap reconstruction relative error: {result['overlap_relative_error']:.3e}")
    if result["pair_space_errors"] is not None:
        for label, error in result["pair_space_errors"].items():
            print(f"Pair-space {label} grid L2 error: {error['l2']:.3e}; "
                  f"relative error: {error['relative']:.3e}")
    if result["coulomb_metric"] is not None:
        print("Coulomb metric shape:", result["coulomb_metric"].shape)
    if not args.debug:
        print(f"ISDF build time: {result['elapsed']:.6f} s")
    if result["derivative_integrals"]:
        print(f"Derivative ERI targets: {', '.join(result['derivative_integrals'])}")
    if args.compare and not args.derivative_target:
        started = time.perf_counter()
        if mf is not None:
            dm = mf.make_rdm1()
            density_source = "converged RHF"
        else:
            dm = scf.hf.init_guess_by_minao(mol)
            density_source = "PySCF MINAO initial guess"
        seconds = time.perf_counter() - started
        comparison_timings.append(("density_matrix", seconds))
        if not args.debug:
            print(f"  density_matrix       {seconds:10.6f} s "
                  f"({density_source})", flush=True)

        started = time.perf_counter()
        j_isdf, k_isdf = evaluate_isdf_jk(result, dm)
        seconds = time.perf_counter() - started
        comparison_timings.append(("isdf_jk", seconds))
        if not args.debug:
            print(f"  isdf_jk              {seconds:10.6f} s", flush=True)

        started = time.perf_counter()
        j_exact, k_exact = scf.hf.get_jk(mol, dm)
        seconds = time.perf_counter() - started
        comparison_timings.append(("pyscf_jk", seconds))
        if not args.debug:
            print(f"  pyscf_jk             {seconds:10.6f} s", flush=True)
        for label, approximate, exact in (("J", j_isdf, j_exact),
                                          ("K", k_isdf, k_exact)):
            error = np.linalg.norm(approximate - exact)
            relative = error / max(np.linalg.norm(exact), 1e-30)
            print(f"{label} ISDF vs PySCF: absolute Frobenius error {error:.3e}; "
                  f"relative error {relative:.3e}")
    if args.compare and args.derivative_target:
        targets = dict(result["derivative_integrals"])
        if "0000" in args.derivative_target:
            x = result["X1"]
            pairs = (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)
            nao = mol.nao_nr()
            targets["0000"] = (pairs.T @ result["coulomb_metric"] @ pairs
                               ).reshape(nao, nao, nao, nao)
        exact_cache = {}
        for target, approximate in targets.items():
            started = time.perf_counter()
            exact = _reference_target(mol, target, exact_cache)
            error = np.linalg.norm(approximate - exact)
            relative = error / max(np.linalg.norm(exact), 1e-30)
            seconds = time.perf_counter() - started
            comparison_timings.append((f"pyscf_target_{target}", seconds))
            if args.debug:
                print(f"  PySCF target {target}: absolute error {error:.3e}; "
                      f"relative error {relative:.3e}", flush=True)
            else:
                print(f"  PySCF target {target} {seconds:.6f} s; "
                      f"absolute error {error:.3e}; relative error {relative:.3e}", flush=True)
    if args.debug:
        print("Timing report:")
        for stage, seconds in result["timings"].items():
            fraction = 100 * seconds / result["elapsed"] if result["elapsed"] else 0.0
            print(f"  {stage:22s} {seconds:10.6f} s  ({fraction:5.1f}% of build)")
        print(f"  ISDF build total       {result['elapsed']:10.6f} s")
        for stage, seconds in comparison_timings:
            print(f"  {stage:22s} {seconds:10.6f} s")
        print(f"  Runner total           {time.perf_counter()-runner_started:10.6f} s")


if __name__ == "__main__":
    main()
