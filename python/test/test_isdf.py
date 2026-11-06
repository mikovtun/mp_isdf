"""Integration tests for the minimal Python ISDF driver."""

from pathlib import Path
from contextlib import redirect_stdout
import io
import os
import subprocess
import sys
import unittest

import numpy as np
from pyscf import dft, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import (_reference_target, build_isdf, evaluate_isdf_jk,
                  grid_coulomb_metric, grid_cross_metric, main, mp_isdf,
                  pair_space_error_norms)


class TestMinimalISDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

    def test_ao_anchor_selection_and_metric(self):
        result = build_isdf(self.mol, level=0, anchor_rank=4)
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = 0
        grids.build()
        context = mp_isdf.LibcintContext(self.mol._atm, self.mol._bas, self.mol._env)
        expected = mp_isdf.AnchorSelection(context).select(
            grids.coords, grids.weights, mp_isdf.AxisType.AtomicOrbital, max_rank=4
        )
        np.testing.assert_array_equal(result["anchor_indices"], expected)
        rank = len(expected)
        self.assertEqual(result["X1"].shape, (rank, self.mol.nao_nr()))
        self.assertEqual(result["interp_vecs"].shape, (rank, len(result["weights"])))
        self.assertEqual(result["coulomb_metric"].shape, (rank, rank))
        np.testing.assert_allclose(result["coulomb_metric"],
                                   result["coulomb_metric"].T, atol=1e-12)
        np.testing.assert_allclose(result["X1"], context.eval_ao(result["anchor_coords"]))
        ao = context.eval_ao(result["coordinates"])
        products = (ao[:, :, None] * ao[:, None, :]).reshape(len(ao), -1)
        anchor_ao = result["X1"]
        weighted_pairs = (anchor_ao[:, :, None] * anchor_ao[:, None, :]).reshape(rank, -1)
        weighted_pairs *= result["anchor_weights"][:, None]
        transform = np.linalg.pinv(weighted_pairs @ weighted_pairs.T, rcond=1e-12) @ weighted_pairs
        expected_interp = (products @ transform.T).T
        np.testing.assert_allclose(result["interp_vecs"], expected_interp, rtol=1e-12, atol=1e-12)
        approx = anchor_ao.T @ (
            (result["anchor_weights"] * (result["interp_vecs"] @ result["weights"]))[:, None]
            * anchor_ao
        )
        np.testing.assert_allclose(result["overlap_reconstruction"], approx)
        overlap = self.mol.intor("int1e_ovlp")
        np.testing.assert_allclose(
            result["overlap_relative_error"],
            np.linalg.norm(approx - overlap) / np.linalg.norm(overlap),
        )
        self.assertTrue(np.isfinite(result["overlap_relative_error"]))
        self.assertIsNone(result["pair_space_errors"])
        self.assertEqual(list(result["timings"]), [
            "grid", "context", "anchor_selection", "ao_evaluation",
            "isdf_fit", "overlap_check", "coulomb_metric",
        ])
        self.assertTrue(all(value >= 0 for value in result["timings"].values()))
        np.testing.assert_allclose(sum(result["timings"].values()), result["elapsed"],
                                   rtol=1e-12, atol=1e-12)

    def test_pair_space_error_independent_of_metric(self):
        result = build_isdf(self.mol, level=0, anchor_rank=2,
                            compute_metric=False, pair_space_error=True,
                            derivative_targets=("x00y", "x0y0"))
        self.assertIsNone(result["coulomb_metric"])
        self.assertEqual(result["derivative_integrals"], {})
        self.assertEqual(set(result["pair_space_errors"]), {"scalar", "x0", "0y", "y0"})
        self.assertIn("pair_space_check", result["timings"])
        self.assertIn("derivative_pair_space_check", result["timings"])
        context = mp_isdf.LibcintContext(self.mol._atm, self.mol._bas, self.mol._env)
        ao = context.eval_ao(result["coordinates"])
        gradient = context.eval_ao_gradient(result["coordinates"])
        anchor_ao = context.eval_ao(result["anchor_coords"])
        anchor_gradient = context.eval_ao_gradient(result["anchor_coords"])
        fields = {"0": ao, **dict(zip("xyz", gradient))}
        anchor_fields = {"0": anchor_ao, **dict(zip("xyz", anchor_gradient))}
        for label, error in result["pair_space_errors"].items():
            left, right = ("0", "0") if label == "scalar" else label
            exact = (fields[left][:, :, None] * fields[right][:, None, :]).reshape(len(ao), -1)
            anchors = (anchor_fields[left][:, :, None] *
                       anchor_fields[right][:, None, :]).reshape(len(anchor_ao), -1)
            if label == "scalar":
                theta = result["interp_vecs"]
            else:
                theta = (exact @ np.linalg.pinv(
                    anchors * result["anchor_weights"][:, None], rcond=1e-12)).T
            reconstructed = theta.T @ (anchors * result["anchor_weights"][:, None])
            weights = np.abs(result["weights"])[:, None]
            expected_l2 = np.sqrt(np.sum(weights * (exact - reconstructed)**2))
            expected_rel = expected_l2 / np.sqrt(np.sum(weights * exact**2))
            np.testing.assert_allclose(error["l2"], expected_l2, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(error["relative"], expected_rel, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result["pair_space_errors"]["scalar"]["l2"],
                                   pair_space_error_norms(
                             (ao[:, :, None] * ao[:, None, :]).reshape(len(ao), -1),
                             (anchor_ao[:, :, None] * anchor_ao[:, None, :]).reshape(len(anchor_ao), -1),
                             result["anchor_weights"], result["interp_vecs"],
                             result["weights"], block_size=1)["l2"])

    def test_pair_space_error_requires_fit(self):
        with self.assertRaisesRegex(ValueError, "pair_space_error requires fitting"):
            build_isdf(self.mol, compute_metric=False, selection_only=True,
                       pair_space_error=True)

    def test_pair_space_error_with_signed_grid_weights(self):
        exact = np.array([[1., 2.], [3., 4.]])
        anchor = np.array([[1., 0.]])
        theta = np.array([[1., 3.]])
        error = pair_space_error_norms(exact, anchor, np.array([1.]), theta,
                                       np.array([2., -1.]), block_size=1)
        np.testing.assert_allclose(error["l2"], np.sqrt(24.))
        np.testing.assert_allclose(error["relative"], np.sqrt(24. / 35.))

    def test_timings_are_reported_during_build(self):
        output = io.StringIO()
        original = mp_isdf.coulomb_metric

        def check_progress(*args, **kwargs):
            self.assertIn("overlap_check", output.getvalue())
            self.assertNotIn("coulomb_metric", output.getvalue())
            return original(*args, **kwargs)

        from unittest.mock import patch
        with redirect_stdout(output), patch.object(mp_isdf, "coulomb_metric", check_progress):
            result = build_isdf(self.mol, level=0, anchor_rank=2,
                                report_timings=True)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), len(result["timings"]))
        self.assertIn("grid", lines[0])
        self.assertIn("coulomb_metric", lines[-1])

    def test_debug_cholesky_and_deferred_timings(self):
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "isdf.py"),
                   "--anchor-rank", "3", "--metric-rank", "2",
                   "--metric-batch-size", "1", "--derivative-target", "00zz",
                   "--debug", "--compare",
                   "--space", "host"]
        environment = {**os.environ, "OMP_NUM_THREADS": "2", "OMP_PROC_BIND": "false",
                       "OPENBLAS_NUM_THREADS": "1"}
        completed = subprocess.run(command, check=True, capture_output=True, text=True,
                                   env=environment, timeout=60)
        for text in ("[cholesky ao] rank=1", "[cholesky derivative] rank=1",
                     "column_quartets=", "max_remaining=", "reason=rank_limit",
                     "projection_seconds=", "metric_gemm_seconds="):
            self.assertIn(text, completed.stderr)
        self.assertIn("Timing report:", completed.stdout)
        self.assertLess(completed.stdout.index("PySCF target 00zz:"),
                        completed.stdout.index("Timing report:"))
        self.assertIn("pyscf_target_00zz", completed.stdout)
        self.assertTrue(completed.stdout.rstrip().splitlines()[-1].strip().startswith(
            "Runner total"))

    def test_grid_cross_metric(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        weights = np.array([0.3, -0.2, 0.5])
        left = np.array([[1., 2., -1.]])
        right = np.array([[0., 1., 3.], [2., 0., 1.]])
        stacked = grid_coulomb_metric(np.concatenate((left, right)), coords, weights)
        np.testing.assert_allclose(grid_cross_metric(left, right, coords, weights),
                                   stacked[:1, 1:])
        np.testing.assert_allclose(grid_cross_metric(right, left, coords, weights),
                                   stacked[1:, :1])

    def test_grid_metric_selection(self):
        from unittest.mock import patch
        with patch.object(mp_isdf, "coulomb_metric", side_effect=AssertionError(
                "grid metric must not call libcint Cholesky")):
            result = build_isdf(self.mol, level=0, anchor_rank=2,
                                metric_algorithm="grid", metric_rank=1)
        expected = grid_coulomb_metric(result["interp_vecs"],
                                       result["coordinates"], result["weights"])
        weights = result["anchor_weights"]
        np.testing.assert_allclose(result["coulomb_metric"],
                                   weights[:, None] * expected * weights[None, :])
        self.assertGreaterEqual(result["timings"]["coulomb_metric"], 0)

    def test_jk_contractions_and_pyscf_reference(self):
        result = build_isdf(self.mol, level=0, anchor_rank=3)
        x, z = result["X1"], result["coulomb_metric"]
        dm = scf.hf.init_guess_by_minao(self.mol)
        j, k = evaluate_isdf_jk(result, dm)
        eri = np.einsum("pu,pv,pq,qw,qx->uvwx", x, x, z, x, x)
        np.testing.assert_allclose(j, np.einsum("uvwx,wx->uv", eri, dm))
        np.testing.assert_allclose(k, np.einsum("uwvx,wx->uv", eri, dm))
        j_ref, k_ref = scf.hf.get_jk(self.mol, dm)
        exact = self.mol.intor("int2e")
        np.testing.assert_allclose(j_ref, np.einsum("uvwx,wx->uv", exact, dm))
        np.testing.assert_allclose(k_ref, np.einsum("uwvx,wx->uv", exact, dm))
        self.assertTrue(np.isfinite(np.linalg.norm(j - j_ref)))
        self.assertTrue(np.isfinite(np.linalg.norm(k - k_ref)))
        with self.assertRaisesRegex(ValueError, "density"):
            evaluate_isdf_jk(result, np.ones((3, 3)))
        with self.assertRaisesRegex(ValueError, "symmetric"):
            evaluate_isdf_jk(result, np.array([[1., 1.], [0., 1.]]))
        with self.assertRaisesRegex(ValueError, "metric"):
            evaluate_isdf_jk({**result, "coulomb_metric": None}, dm)

    def test_cli_comparison_is_opt_in(self):
        from unittest.mock import patch
        args = ["isdf.py", "--anchor-rank", "2", "--space", "host"]
        output = io.StringIO()
        with patch.object(sys, "argv", args), redirect_stdout(output), \
             patch.object(scf.hf, "get_jk", side_effect=AssertionError(
                 "exact J/K must not run by default")):
            main()
        self.assertNotIn("ISDF vs PySCF", output.getvalue())
        output = io.StringIO()
        with patch.object(sys, "argv", args + ["--compare"]), redirect_stdout(output):
            main()
        self.assertIn("J ISDF vs PySCF", output.getvalue())
        self.assertIn("K ISDF vs PySCF", output.getvalue())
        output = io.StringIO()
        with patch.object(sys, "argv", args + ["--derivative-target", "0000",
                                               "--compare"]), redirect_stdout(output), \
             patch.object(scf.hf, "get_jk", side_effect=AssertionError(
                 "explicit 0000 comparison must use full ERIs")):
            main()
        self.assertIn("PySCF target 0000", output.getvalue())
        self.assertNotIn("ISDF vs PySCF", output.getvalue())
        output = io.StringIO()
        with patch.object(sys, "argv", args + ["--derivative-target", "00zz",
                                               "--compare"]), redirect_stdout(output), \
             patch.object(scf.hf, "get_jk", side_effect=AssertionError(
                 "derivative comparison must not compute PySCF J/K")):
            main()
        self.assertIn("PySCF target 00zz", output.getvalue())
        self.assertNotIn("ISDF vs PySCF", output.getvalue())
        output = io.StringIO()
        with patch.object(sys, "argv", args + ["--no-metric", "--derivative-target", "00zz"]), \
             redirect_stdout(output), \
             patch.object(mp_isdf, "coulomb_metric", side_effect=AssertionError(
                 "no metric should run")), \
             patch.object(mp_isdf, "derivative_cross_metric", side_effect=AssertionError(
                 "no derivative metric should run")):
            main()
        self.assertIn("overlap_check", output.getvalue())
        self.assertIn("Overlap reconstruction relative error", output.getvalue())
        self.assertNotIn("Derivative ERI targets", output.getvalue())

    def test_0000_uses_ordinary_ao_isdf(self):
        from unittest.mock import patch
        for algorithm in ("unblocked_greedy", "grid"):
            with patch.object(mp_isdf.LibcintContext, "eval_ao_gradient",
                              side_effect=AssertionError("0000 must not evaluate AO gradients")):
                result = build_isdf(self.mol, level=0, anchor_rank=3,
                                    derivative_targets=("0000",), metric_algorithm=algorithm)
            baseline = build_isdf(self.mol, level=0, anchor_rank=3,
                                  metric_algorithm=algorithm)
            np.testing.assert_allclose(result["coulomb_metric"], baseline["coulomb_metric"])
            self.assertEqual(result["derivative_integrals"], {})
            self.assertEqual(set(result["target_factors"]), {"0000"})
            ordinary = result["target_factors"]["0000"]
            for slot in ("X1", "X2", "X3", "X4"):
                np.testing.assert_allclose(ordinary[slot], result["X1"])
            np.testing.assert_allclose(ordinary["coulomb_metric"], result["coulomb_metric"])
            self.assertNotIn("derivative_integrals", result["timings"])
        np.testing.assert_allclose(_reference_target(self.mol, "0000"),
                                   self.mol.intor("int2e", aosym="s1"))

    def test_two_gradient_targets_independent_of_anchor_axes(self):
        from itertools import combinations
        mol = gto.M(atom="H 0 0 0; H 0.3 0.2 0.74", basis="sto-3g", verbose=0)
        ctx = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        identity = np.eye(mol.nao_nr()**2)
        for slots in combinations(range(4), 2):
            labels = ["0"] * 4
            labels[slots[0]], labels[slots[1]] = "x", "z"
            target = "".join(labels)
            a, b = target[:2], target[2:]
            digits = {"0": 0, "x": 1, "y": 2, "z": 3}
            actual = mp_isdf.derivative_cross_metric(
                ctx, identity, tuple(digits[v] for v in a),
                identity, tuple(digits[v] for v in b), tolerance=1e-13,
            ).reshape((mol.nao_nr(),) * 4)
            np.testing.assert_allclose(actual, _reference_target(mol, target),
                                       atol=1e-10, rtol=1e-10,
                                       err_msg=f"target {target}")
        for axes, metric in (("ao", "unblocked_greedy"),
                             ("ao", "unblocked_uniform"),
                             (("ao", "dz"), "grid")):
            result = build_isdf(mol, level=0, anchor_rank=3, axes=axes,
                                metric_algorithm=metric, metric_seed=7,
                                derivative_targets=("00zz", "z00z"))
            self.assertIsNotNone(result["coulomb_metric"])
            self.assertEqual(set(result["derivative_integrals"]), {"00zz", "z00z"})
            self.assertEqual(set(result["target_factors"]), {"00zz", "z00z"})
            anchor_grad_z = ctx.eval_ao_gradient(result["anchor_coords"])[2]
            for target, integral in result["derivative_integrals"].items():
                factors = result["target_factors"][target]
                self.assertEqual(integral.shape, (mol.nao_nr(),)*4)
                self.assertTrue(np.isfinite(integral).all())
                for slot, label in zip(("X1", "X2", "X3", "X4"), target):
                    expected = result["X1"] if label == "0" else anchor_grad_z
                    np.testing.assert_allclose(factors[slot], expected)
                np.testing.assert_allclose(
                    np.einsum("pu,pv,pq,qw,qx->uvwx", factors["X1"], factors["X2"],
                              factors["coulomb_metric"], factors["X3"], factors["X4"],
                              optimize=True), integral, atol=1e-10)
            self.assertIn("derivative_integrals", result["timings"])
        with self.assertRaisesRegex(ValueError, "two derivatives"):
            build_isdf(mol, derivative_targets=("x000",))
        exact_fit = build_isdf(self.mol, level=0, anchor_rank=3,
                               derivative_targets=("00zz",))
        np.testing.assert_allclose(exact_fit["derivative_integrals"]["00zz"],
                                   _reference_target(self.mol, "00zz"), atol=1e-10)

    def test_host_space_driver(self):
        result = build_isdf(self.mol, level=0, anchor_rank=2,
                            compute_metric=False, space="host")
        self.assertGreater(len(result["anchor_indices"]), 0)
        self.assertTrue(np.isfinite(result["overlap_relative_error"]))
        self.assertEqual(result["timings"]["coulomb_metric"], 0.0)
        from unittest.mock import patch
        with patch.object(mp_isdf, "coulomb_metric", side_effect=AssertionError(
                "no metric should run")), \
             patch.object(mp_isdf, "derivative_cross_metric", side_effect=AssertionError(
                 "no derivative metric should run")), \
             patch.object(mp_isdf.LibcintContext, "eval_ao_gradient",
                          side_effect=AssertionError("no gradient AO should run")):
            skipped = build_isdf(self.mol, level=0, anchor_rank=2,
                                 compute_metric=False, derivative_targets=("0000", "00zz"))
        self.assertIsNone(skipped["coulomb_metric"])
        self.assertEqual(skipped["derivative_integrals"], {})
        self.assertEqual(skipped["target_factors"], {})
        with patch.object(mp_isdf.LibcintContext, "eval_ao",
                          side_effect=AssertionError("selection only must skip AO fit")):
            selected = build_isdf(self.mol, level=0, anchor_rank=2,
                                  compute_metric=False, derivative_targets=("00zz",),
                                  selection_only=True)
        self.assertIsNone(selected["interp_vecs"])
        self.assertEqual(selected["target_factors"], {})
        self.assertIsNone(selected["overlap_relative_error"])
        self.assertEqual(list(selected["timings"]), ["grid", "context", "anchor_selection"])

    def test_occupied_feature_selection(self):
        mf = scf.RHF(self.mol).run(verbose=0)
        result = build_isdf(self.mol, mf, level=0, anchor_rank=2,
                            axes=("ao", "occ"), compute_metric=False)
        self.assertGreater(len(result["anchor_indices"]), 0)
        self.assertLessEqual(len(result["anchor_indices"]), 2)
        self.assertIsNone(result["coulomb_metric"])
        self.assertTrue(np.isfinite(result["overlap_relative_error"]))


if __name__ == "__main__":
    unittest.main()
