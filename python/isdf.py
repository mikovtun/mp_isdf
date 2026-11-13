#!/usr/bin/env python3
"""Minimal grid ISDF demo adapted from ../isdf/isdf.py.

PySCF builds the molecular grid; mp_isdf evaluates AOs and selects anchors
with local QRCP (MAGMA on device, LAPACK++ on host). The default ISDF fit remains in NumPy; rbrp-id instead reuses its RBRP ID.
The Coulomb metric uses integral-direct AO-pair Cholesky in C++.
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


def _stream_pair_space_error_norms(left, right, pair_anchor, anchor_weights,
                                   theta, grid_weights):
    """Grid-weighted pair error with only one AO-pair matrix of scratch."""
    anchor = (pair_anchor * anchor_weights[:, None]).reshape(
        len(anchor_weights), left.shape[1], right.shape[1])
    residual_sq = exact_sq = 0.0
    for g in range(len(grid_weights)):
        exact = np.outer(left[g], right[g])
        reconstructed = np.einsum("k,kij->ij", theta[:, g], anchor,
                                  optimize=True)
        weight = abs(grid_weights[g])
        residual_sq += weight * np.sum((exact - reconstructed)**2)
        exact_sq += weight * np.sum(exact**2)
    absolute = np.sqrt(residual_sq)
    relative = absolute / np.sqrt(exact_sq) if exact_sq else (0.0 if absolute == 0 else np.inf)
    return {"l2": absolute, "relative": relative}


def build_isdf(mol, mf=None, *, level=0, anchor_rank=0, anchor_tol=None,
               axes=("ao",), compute_metric=True, space="auto",
               anchor_algorithm="dense", anchor_batch_size=128, fit_batch_size=2048,
               anchor_sketch_size=128, anchor_oversampling=16,
               anchor_seed=7, anchor_audit=False,
               rbrp_block_size=16, rbrp_tau_block=None,
               rbrp_rank_rtol=1e-12, rbrp_refresh_every=8,
               rbrp_svd_rtol=1e-12,
               cvt_weight="electron_density", cvt_max_iter=100, cvt_tol=1e-5,
               metric_algorithm="unblocked_greedy", metric_tol=1e-12,
               metric_rank=0, metric_batch_size=64, report_timings=False,
               derivative_targets=(), selection_only=False, debug=False,
               metric_seed=0, pair_space_error=False, report_dimensions=False):
    """Build AO-pair ISDF, using this project's C++ evaluator and selector.

    `axes` selects the anchor-search features (ao/dx/dy/dz/occ), concatenated
    before forming their outer product. The fitted tensor remains AO-pair ISDF.
    A positive `anchor_rank` forces that many pivots for rand sketch selection;
    it cannot be combined with an explicit `anchor_tol`. With neither set,
    the default tolerance is 1e-10. `anchor_tol` is a relative
    QR-diagonal/max-column cutoff for dense/online selection and a sketch
    Frobenius cutoff for rand sketch selection. Anchor indices are
    relative to the original, unpruned PySCF grid. `space` selects where AO
    evaluation and QRCP run (auto, host, device); the NumPy fit stays on host.
    `anchor_algorithm` is `dense` (default), `online` (direct multipass QRCP),
    `rbrp` (host AO-only residual block sampling followed by the usual NumPy
    fit), `rbrp-id` (same anchors with Algorithm 5.1 ID coefficients instead
    of the NumPy scalar fit), `rand_feature_kr` (Khatri-Rao sketch of concatenated features, signed
    weights treated by absolute value), `rand_ao_kr_quadrature` (AO-only
    Khatri-Rao sketch weighted by absolute quadrature weights), or `cvt`.
    CVT requires a positive `anchor_rank`; it ignores
    `anchor_tol` and `axes`, uses `anchor_seed`, `cvt_max_iter`, and `cvt_tol`,
    and uses either spin-summed RKS B3LYP electron density (default) or
    absolute quadrature weights as its importance vector, not their product.
    Zero quadrature-weight and zero-importance sites are excluded from
    clustering and snapping.
    `anchor_batch_size` controls selection streaming; `fit_batch_size` controls
    interpolation-vector batches without materializing grid-wide AO pairs.
    RBRP uses sqrt(abs(grid_weight)) in its AO-pair rows, skips zero weights,
    and interprets anchor_tol as a relative Frobenius norm tolerance (anchor_rank
    is a cap). Only AO features are currently supported. space='device' uses
    bounded Kokkos CUDA AO evaluation and MAGMA QR/BLAS in a CUDA build;
    space='auto' uses the device backend in a CUDA build and host otherwise.
    rbrp-id constructs both
    scalar interpolation vectors and metric coefficients from the same
    SVD-stabilized small L1 factor, without the NumPy scalar fit.
    Rand sketch selection uses `anchor_tol` as sketch Frobenius
    tolerance and `anchor_seed`. With an explicit rank, sketch rows equal
    `anchor_rank + anchor_oversampling`; otherwise `anchor_sketch_size` sets
    the sketch rows (and `sketch_size - oversampling` caps the rank).
    `anchor_audit=True` compares actual pair-space projection residuals against
    dense QRCP at equal rank, only on small problems (not an ID-error audit).
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
    sketch_algorithms = ("rand_feature_kr", "rand_ao_kr_quadrature")
    rbrp_mode = anchor_algorithm in ("rbrp", "rbrp-id")
    if anchor_algorithm not in ("dense", "online", "cvt", "rbrp", "rbrp-id", *sketch_algorithms):
        raise ValueError(f"unknown anchor algorithm: {anchor_algorithm!r}")
    if rbrp_mode:
        if axis_types != [mp_isdf.AxisType.AtomicOrbital]:
            raise ValueError("RBRP currently requires axes=('ao',)")
        if (not isinstance(rbrp_block_size, (int, np.integer)) or rbrp_block_size < 1 or
            not isinstance(rbrp_refresh_every, (int, np.integer)) or rbrp_refresh_every < 0 or
            not np.isfinite(rbrp_rank_rtol) or rbrp_rank_rtol <= 0 or
            not np.isfinite(rbrp_svd_rtol) or rbrp_svd_rtol < 0 or
            (rbrp_tau_block is not None and
             (not np.isfinite(rbrp_tau_block) or not 0 <= rbrp_tau_block < 1))):
            raise ValueError("invalid RBRP block size, refresh interval or tolerances")
    if cvt_weight not in ("electron_density", "quadrature_weights"):
        raise ValueError("cvt_weight must be 'electron_density' or 'quadrature_weights'")
    if not isinstance(cvt_max_iter, (int, np.integer)) or cvt_max_iter < 1 or not np.isfinite(cvt_tol) or cvt_tol < 0:
        raise ValueError("cvt_max_iter must be positive and cvt_tol finite and nonnegative")
    if anchor_algorithm == "cvt" and anchor_rank < 1:
        raise ValueError("CVT requires a positive anchor_rank")
    if not isinstance(anchor_batch_size, (int, np.integer)) or anchor_batch_size <= 0:
        raise ValueError("anchor_batch_size must be a positive integer")
    if not isinstance(fit_batch_size, (int, np.integer)) or fit_batch_size <= 0:
        raise ValueError("fit_batch_size must be a positive integer")
    if anchor_rank != 0 and anchor_tol is not None:
        raise ValueError("anchor_rank and anchor_tol are mutually exclusive")
    if anchor_tol is None:
        anchor_tol = 1e-10
    if not np.isfinite(anchor_tol) or anchor_tol < 0:
        raise ValueError("anchor_tol must be finite and nonnegative")
    if (not isinstance(anchor_sketch_size, (int, np.integer)) or
        not isinstance(anchor_oversampling, (int, np.integer)) or
        anchor_oversampling < 1 or anchor_sketch_size < 1 or
        (anchor_algorithm in sketch_algorithms and anchor_rank == 0 and
         anchor_sketch_size <= anchor_oversampling)):
        raise ValueError("randomized anchor sketch_size must exceed oversampling when anchor_rank=0")
    if not isinstance(anchor_seed, (int, np.integer)) or not 0 <= anchor_seed < 2**64:
        raise ValueError("anchor_seed must be an unsigned 64-bit integer")
    if anchor_audit and anchor_algorithm not in sketch_algorithms:
        raise ValueError("anchor_audit requires a rand sketch algorithm")
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
    nao = mol.nao_nr()
    axis_width = sum(nocc if axis == mp_isdf.AxisType.OccupiedOrbital else nao
                     for axis in axis_types)
    pair_rank_ceiling = axis_width * (axis_width + 1) // 2
    if report_dimensions:
        dimensions = (
            f"[ISDF dimensions] AOs={nao} axis_features={axis_width} "
            f"symmetric_pair_rank_ceiling={pair_rank_ceiling} "
            f"grid_points={len(all_coords)} nonzero_weight_points={int(np.count_nonzero(keep))}"
        )
        if anchor_algorithm in sketch_algorithms:
            ell = (anchor_rank + anchor_oversampling
                   if anchor_rank > 0 else anchor_sketch_size)
            sketch_cap = min(ell - anchor_oversampling, pair_rank_ceiling,
                             int(np.count_nonzero(keep)))
            dimensions += (f" sketch_rows={ell} oversampling={anchor_oversampling}"
                           f" sketch_rank_cap={sketch_cap}"
                           f" sketch_matrix={ell}x{int(np.count_nonzero(keep))}"
                           f" sketch_storage={ell * int(np.count_nonzero(keep)) * 8 / 2**20:.1f}MiB")
            if anchor_rank > 0:
                dimensions += f" requested_rank={anchor_rank}"
                if anchor_rank > sketch_cap:
                    dimensions += " requested_rank_exceeds_structural_cap=yes"
            if anchor_rank > 0:
                dimensions += " (anchor_sketch_size ignored)"
        print(dimensions, flush=True)
    anchor_choices = {
        "dense": "QRColumnPivoting",
        "online": "OnlineQRColumnPivoting",
        "rand_feature_kr": "RandomizedFeatureKhatriRao",
        "rand_ao_kr_quadrature": "RandomizedAOQuadratureKhatriRao",
        "cvt": "CentroidalVoronoi",
    }
    if not rbrp_mode:
        try:
            anchor_choice = getattr(mp_isdf.LocalAnchorSelectionAlgorithm,
                                    anchor_choices[anchor_algorithm])
        except AttributeError as exc:
            raise RuntimeError("rebuild the mp_isdf extension to use this anchor algorithm") from exc
    cvt_weights = None
    if anchor_algorithm == "cvt":
        cvt_weights = np.zeros(len(all_coords), dtype=np.float64)
        if cvt_weight == "quadrature_weights":
            cvt_weights[keep] = np.abs(all_weights[keep])
        else:
            rks = dft.RKS(mol)
            rks.xc = "b3lyp"
            rks.grids.level = level
            rks.kernel()
            if not rks.converged:
                raise RuntimeError("RKS B3LYP did not converge for CVT density")
            # Spin-summed electron density on the same PySCF grid as selection.
            ao_cvt = dft.numint.eval_ao(mol, all_coords[keep])
            cvt_weights[keep] = np.maximum(
                dft.numint.eval_rho(mol, ao_cvt, rks.make_rdm1()), 0.0)
        record("cvt_weights")
    rbrp_result = None
    rbrp_ao = None
    if rbrp_mode:
        settings = dict(batch_size=anchor_batch_size, block_size=rbrp_block_size,
                        tolerance=anchor_tol, max_rank=anchor_rank, seed=anchor_seed,
                        tau_block=rbrp_tau_block, rank_rtol=rbrp_rank_rtol,
                        refresh_every=rbrp_refresh_every, svd_rtol=rbrp_svd_rtol,
                        compute_id=(anchor_algorithm == "rbrp-id" and not selection_only))
        device_backend = getattr(mp_isdf, "rbrp_ao_device", None)
        if space == "device" and device_backend is None:
            raise ValueError("RBRP device selection requires a CUDA build")
        if space == "device" or (space == "auto" and device_backend is not None):
            rbrp_result = device_backend(selector, all_coords[keep], all_weights[keep],
                                         **settings)
        else:
            # Host backend reuses the AO array subsequently needed by the driver.
            rbrp_ao = context.eval_ao(all_coords[keep], space="host")
            try:
                rbrp_backend = mp_isdf.rbrp_ao_cpu
            except AttributeError as exc:
                raise RuntimeError("rebuild the mp_isdf extension for the C++ RBRP backend") from exc
            rbrp_result = rbrp_backend(rbrp_ao, all_weights[keep], **settings)
        anchor_indices = np.flatnonzero(keep)[rbrp_result["indices"]]
        if debug:
            print(f"[rbrp] rank={len(anchor_indices)} residual={rbrp_result['relative_residual']:.6e} "
                  f"status={rbrp_result['status']} blocks={len(rbrp_result['history'])}", flush=True)
    else:
        anchor_indices = np.asarray(selector.select(
            all_coords, all_weights, axis_types, tolerance=anchor_tol,
            max_rank=anchor_rank, algorithm=anchor_choice,
            mo_coeff=coefficients, nocc=nocc, space=space,
            batch_size=anchor_batch_size, debug=debug,
            sketch_size=anchor_sketch_size, oversampling=anchor_oversampling,
            seed=anchor_seed, audit=anchor_audit,
            cvt_weights=cvt_weights, cvt_max_iter=cvt_max_iter, cvt_tol=cvt_tol,
        ), dtype=np.intp)
    if len(anchor_indices) == 0:
        raise ValueError("anchor selection found no anchors; adjust settings or grid")
    record("anchor_selection")
    if selection_only:
        return {
            "anchor_indices": anchor_indices,
            "rbrp_diagnostics": ({key: rbrp_result[key] for key in
                                  ("relative_residual", "status", "history")}
                                 if rbrp_mode else None),
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
    ao = rbrp_ao if rbrp_ao is not None else context.eval_ao(coords, space=space)
    anchor_coords = all_coords[anchor_indices]
    anchor_weights = all_weights[anchor_indices]
    anchor_ao = ao[rbrp_result["indices"]] if rbrp_mode else context.eval_ao(anchor_coords, space=space)
    record("ao_evaluation")
    rank, nao = anchor_ao.shape
    anchor_pairs = (anchor_ao[:, :, None] * anchor_ao[:, None, :]).reshape(rank, nao * nao)
    id_info = None
    if anchor_algorithm == "rbrp-id":
        # Algorithm 5.1: reuse L and Q; no NumPy AO-pair Gram fit.
        interp_vecs = rbrp_result["interp_vecs"]
        transform = rbrp_result["transform"]
        id_info = {key: rbrp_result[key] for key in
                   ("retained_rank", "condition", "relative_residual", "status")}
        if debug:
            print(f"[rbrp-id] {id_info}", flush=True)
    else:
        weighted_pairs = anchor_pairs * anchor_weights[:, None]
        gram = weighted_pairs @ weighted_pairs.T
        gram_inverse = np.linalg.pinv(gram, rcond=1e-12)
        transform = gram_inverse @ weighted_pairs
        # P(grid) @ (W P(anchor)).T equals (AO(grid) @ AO(anchor).T)**2 * W.
        interp_vecs = np.empty((rank, len(coords)), dtype=np.float64)
        for first in range(0, len(coords), fit_batch_size):
            stop = min(first + fit_batch_size, len(coords))
            cross = (ao[first:stop] @ anchor_ao.T)**2
            cross *= anchor_weights[None, :]
            interp_vecs[:, first:stop] = gram_inverse @ cross.T
    record("isdf_fit")
    pair_space_errors = None
    if pair_space_error:
        pair_space_errors = {
            "scalar": _stream_pair_space_error_norms(
                ao, ao, anchor_pairs, anchor_weights, interp_vecs, weights)
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
                pair_anchor = (anchor_fields[left][:, :, None] *
                               anchor_fields[right][:, None, :]).reshape(rank, nao*nao)
                # Fit each ordered channel independently, even with shared anchors.
                # Keep the rectangular pseudoinverse's cutoff (not a squared Gram cutoff).
                coefficients = np.linalg.pinv(pair_anchor * anchor_weights[:, None],
                                              rcond=1e-12).T
                theta = np.empty((rank, len(coords)), dtype=np.float64)
                coefficient_tensor = coefficients.reshape(rank, nao, nao)
                for first in range(0, len(coords), fit_batch_size):
                    stop = min(first + fit_batch_size, len(coords))
                    contracted = np.einsum("gi,kij->gkj", fields[left][first:stop],
                                           coefficient_tensor, optimize=True)
                    theta[:, first:stop] = np.einsum(
                        "gkj,gj->kg", contracted, fields[right][first:stop],
                        optimize=True)
                channels[label] = pair_anchor, coefficients, theta
            return channels[label]

        if pair_space_error:
            labels = dict.fromkeys(label for target in derivative_targets
                                   for label in _target_labels(target))
            for label in labels:
                pair_anchor, _, theta = channel(label)
                left, right = label
                pair_space_errors[label] = _stream_pair_space_error_norms(
                    fields[left], fields[right], pair_anchor, anchor_weights,
                    theta, weights)
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
        "rbrp_diagnostics": ({"relative_residual": rbrp_result["relative_residual"],
                              "status": rbrp_result["status"],
                              "history": rbrp_result["history"],
                              "id": id_info}
                             if rbrp_mode else None),
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
                        help="force this anchor rank (mutually exclusive with --anchor-tol)")
    parser.add_argument("--anchor-tol", type=float, default=None,
                        help="stop at this relative QR/sketch residual tolerance (default: 1e-10 if rank omitted)")
    parser.add_argument("--anchor-algorithm",
                        choices=("dense", "online", "rand_feature_kr", "rand_ao_kr_quadrature", "cvt", "rbrp", "rbrp-id"),
                        default="dense", help="rbrp selects anchors and uses the existing fit; rbrp-id uses Algorithm 5.1 ID coefficients")
    parser.add_argument("--rbrp-block-size", type=int, default=16)
    parser.add_argument("--rbrp-tau-block", type=float, default=None,
                        help="local QR trailing-energy threshold (default: 1/block_size)")
    parser.add_argument("--rbrp-rank-rtol", type=float, default=1e-12)
    parser.add_argument("--rbrp-refresh-every", type=int, default=8)
    parser.add_argument("--rbrp-svd-rtol", type=float, default=1e-12)
    parser.add_argument("--anchor-batch-size", type=int, default=128,
                        help="grid points per AO batch for online or rand sketch anchor selection")
    parser.add_argument("--fit-batch-size", type=int, default=2048,
                        help="grid points per interpolation-vector fit batch")
    parser.add_argument("--anchor-sketch-size", type=int, default=128,
                        help="rand sketch rows when --anchor-rank=0 (default: 128)")
    parser.add_argument("--anchor-oversampling", type=int, default=16,
                        help="rand sketch extra rows beyond explicit rank (default: 16)")
    parser.add_argument("--anchor-seed", type=int, default=7,
                        help="rand sketch/CVT anchor seed (default: 7)")
    parser.add_argument("--cvt-weight", choices=("electron_density", "quadrature_weights"),
                        default="electron_density",
                        help="CVT importance: RKS B3LYP electron density or absolute quadrature weights")
    parser.add_argument("--cvt-max-iter", type=int, default=100)
    parser.add_argument("--cvt-tol", type=float, default=1e-5,
                        help="CVT relative centroid-shift stopping tolerance")
    parser.add_argument("--anchor-audit", action="store_true",
                        help="compare rand sketch pivots to dense QRCP at equal rank (small grids only)")
    parser.add_argument("--axes", default="ao",
                        help="comma-separated anchor-search features: ao,dx,dy,dz,occ")
    parser.add_argument("--space", choices=("auto", "host", "device"), default="auto",
                        help="where AO evaluation and anchor selection run")
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
    if args.anchor_algorithm == "cvt" and args.anchor_rank == 0:
        parser.error("--anchor-algorithm cvt requires --anchor-rank > 0")
    if args.cvt_max_iter < 1 or not np.isfinite(args.cvt_tol) or args.cvt_tol < 0:
        parser.error("--cvt-max-iter must be positive and --cvt-tol finite and nonnegative")
    if args.anchor_batch_size <= 0:
        parser.error("--anchor-batch-size must be positive")
    if args.fit_batch_size <= 0:
        parser.error("--fit-batch-size must be positive")
    if args.anchor_sketch_size < 1 or args.anchor_oversampling < 1 or (
            args.anchor_algorithm in ("rand_feature_kr", "rand_ao_kr_quadrature") and args.anchor_rank == 0 and
            args.anchor_sketch_size <= args.anchor_oversampling):
        parser.error("randomized sketch size must exceed oversampling when --anchor-rank=0")
    if not 0 <= args.anchor_seed < 2**64:
        parser.error("--anchor-seed must be an unsigned 64-bit integer")
    if args.anchor_audit and args.anchor_algorithm not in ("rand_feature_kr", "rand_ao_kr_quadrature"):
        parser.error("--anchor-audit requires a rand sketch anchor algorithm")
    if args.anchor_rank != 0 and args.anchor_tol is not None:
        parser.error("--anchor-rank and --anchor-tol are mutually exclusive")
    if args.anchor_tol is not None and (not np.isfinite(args.anchor_tol) or args.anchor_tol < 0):
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
                        anchor_algorithm=args.anchor_algorithm,
                        anchor_batch_size=args.anchor_batch_size,
                        fit_batch_size=args.fit_batch_size,
                        anchor_sketch_size=args.anchor_sketch_size,
                        anchor_oversampling=args.anchor_oversampling,
                        anchor_seed=args.anchor_seed, anchor_audit=args.anchor_audit,
                        rbrp_block_size=args.rbrp_block_size,
                        rbrp_tau_block=args.rbrp_tau_block,
                        rbrp_rank_rtol=args.rbrp_rank_rtol,
                        rbrp_refresh_every=args.rbrp_refresh_every,
                        rbrp_svd_rtol=args.rbrp_svd_rtol,
                        cvt_weight=args.cvt_weight, cvt_max_iter=args.cvt_max_iter,
                        cvt_tol=args.cvt_tol,
                        metric_algorithm=args.metric, metric_tol=args.metric_tol,
                        metric_rank=args.metric_rank,
                        metric_batch_size=args.metric_batch_size,
                        metric_seed=args.metric_seed,
                        derivative_targets=args.derivative_target,
                        pair_space_error=args.pair_space_error,
                        report_timings=not args.debug, debug=args.debug,
                        report_dimensions=True)
    print(f"Grid points: {len(result['coordinates'])}; AOs: {mol.nao_nr()}; "
          f"anchors: {len(result['anchor_indices'])}")
    if result["rbrp_diagnostics"] is not None:
        diagnostics = result["rbrp_diagnostics"]
        print(f"RBRP projection residual: {diagnostics['relative_residual']:.3e}; "
              f"status: {diagnostics['status']}")
        if diagnostics.get("id") is not None:
            print(f"RBRP ID retained SVD rank: {diagnostics['id']['retained_rank']}")
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
