#!/usr/bin/env python3
"""Controlled CVT/raw-density/normalized-density ISDF comparison; writes JSON."""
import argparse
import json
from pathlib import Path

import numpy as np
from pyscf import dft, gto, scf

from isdf import (DENSITY_SKETCHES, _read_xyz, build_isdf, evaluate_isdf_jk,
                  mp_isdf, prepare_point_importance)


def compare(mol, *, rank, level=0, oversampling=16, seeds=(7, 11, 23),
            thresholds=(1e-24,), convention="electron_density", space="host",
            metric="unblocked_greedy"):
    """One grid, importance, and RHF comparison density for every variant."""
    grid = dft.gen_grid.Grids(mol)
    grid.level = level
    grid.build()
    importance = prepare_point_importance(mol, grid.coords, grid.weights,
                                          level=level, convention=convention)
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise RuntimeError("comparison RHF did not converge")
    dm = mf.make_rdm1()
    j_ref, k_ref = scf.hf.get_jk(mol, dm)
    rows = []
    evaluator = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
    q = np.empty(np.count_nonzero(grid.weights), dtype=np.float64)
    fit_coords = grid.coords[grid.weights != 0]
    for first in range(0, len(q), 128):
        ao = evaluator.eval_ao(fit_coords[first:first + 128], space=space)
        q[first:first + len(ao)] = np.sum(ao**2, axis=1)
    active_mass = importance[grid.weights != 0]
    # CVT has no AO screening; repeat its initialization seeds, but not thresholds.
    for algorithm in ("cvt", *DENSITY_SKETCHES):
        for threshold in ((thresholds[0],) if algorithm == "cvt" else thresholds):
            for seed in seeds:
                result = build_isdf(
                    mol, level=level, grid=grid, point_importance=importance,
                    anchor_algorithm=algorithm, anchor_rank=rank,
                    anchor_oversampling=oversampling, anchor_seed=seed,
                    ao_screening_threshold=threshold, density_weight=convention,
                    metric_algorithm=metric, space=space, pair_space_error=True)
                j, k = evaluate_isdf_jk(result, dm)
                errors = {}
                for label, approx, exact, factor in (("J", j, j_ref, .5),
                                                      ("K", k, k_ref, -.25)):
                    delta = approx - exact
                    errors[label] = {
                        "frobenius": float(np.linalg.norm(delta)),
                        "relative_frobenius": float(np.linalg.norm(delta) /
                                                     max(np.linalg.norm(exact), 1e-30)),
                        # Spin-summed restricted density: E_J=Tr(DJ)/2,
                        # E_K=-Tr(DK)/4. Keep signed and absolute errors.
                        "energy_error": float(factor * np.einsum("ij,ji->", dm, delta)),
                        "absolute_energy_error": float(abs(factor * np.einsum("ij,ji->", dm, delta))),
                    }
                rank_actual = len(result["anchor_indices"])
                rows.append({
                    "algorithm": algorithm, "seed": int(seed), "rank": rank_actual,
                    "sketch_size": rank + oversampling if algorithm != "cvt" else None,
                    "importance_convention": convention, "ao_norm2_threshold": threshold,
                    "positive_importance_candidates": int(np.count_nonzero(active_mass > 0)),
                    "ao_screened_candidates": int(np.count_nonzero((active_mass > 0) & (q <= threshold)))
                                              if algorithm != "cvt" else 0,
                    "errors": errors, "pair_space_errors": result["pair_space_errors"],
                    "fit_diagnostics": result["fit_diagnostics"],
                    "anchor_indices": result["anchor_indices"],
                    "anchor_coords": result["anchor_coords"],
                    "timings": result["timings"],
                })
                print(f"{algorithm} seed={seed} threshold={threshold:g}: "
                      f"J={errors['J']['relative_frobenius']:.3e} "
                      f"K={errors['K']['relative_frobenius']:.3e}", flush=True)
    return rows


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", type=Path)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--oversampling", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1e-24])
    parser.add_argument("--density-weight", choices=("electron_density", "quadrature_density"),
                        default="electron_density")
    parser.add_argument("--space", choices=("host", "auto", "device"), default="host")
    parser.add_argument("--metric", choices=("unblocked_greedy", "grid"), default="unblocked_greedy")
    parser.add_argument("--output", type=Path, default=Path("density_sketch_comparison.json"))
    args = parser.parse_args()
    if args.rank < 1 or args.oversampling < 1 or args.level < 0:
        parser.error("rank and oversampling must be positive; level must be nonnegative")
    if any(not np.isfinite(t) or t < 0 for t in args.thresholds):
        parser.error("thresholds must be finite and nonnegative")
    if any(not 0 <= seed < 2**64 for seed in args.seeds):
        parser.error("seeds must be unsigned 64-bit integers")
    atom = _read_xyz(args.molecule) if args.molecule else "H 0 0 0; H 0 0 0.74"
    mol = gto.M(atom=atom, basis=args.basis, verbose=0)
    rows = compare(mol, rank=args.rank, level=args.level, oversampling=args.oversampling,
                   seeds=args.seeds, thresholds=args.thresholds, convention=args.density_weight,
                   space=args.space, metric=args.metric)
    args.output.write_text(json.dumps(_json_value({"basis": args.basis, "level": args.level,
                                                   "results": rows}), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
