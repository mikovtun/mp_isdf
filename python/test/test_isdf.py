"""Integration tests for the minimal Python ISDF driver."""

from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
import unittest

import numpy as np
from pyscf import dft, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import build_isdf, evaluate_isdf_jk, grid_coulomb_metric, main, mp_isdf


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
        self.assertEqual(list(result["timings"]), [
            "grid", "context", "anchor_selection", "ao_evaluation",
            "isdf_fit", "overlap_check", "coulomb_metric",
        ])
        self.assertTrue(all(value >= 0 for value in result["timings"].values()))
        np.testing.assert_allclose(sum(result["timings"].values()), result["elapsed"],
                                   rtol=1e-12, atol=1e-12)

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

    def test_host_space_driver(self):
        result = build_isdf(self.mol, level=0, anchor_rank=2,
                            compute_metric=False, space="host")
        self.assertGreater(len(result["anchor_indices"]), 0)
        self.assertTrue(np.isfinite(result["overlap_relative_error"]))
        self.assertEqual(result["timings"]["coulomb_metric"], 0.0)

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
