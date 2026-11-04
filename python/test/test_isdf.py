"""Integration tests for the minimal Python ISDF driver."""

from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
import unittest

import numpy as np
from pyscf import dft, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import build_isdf, grid_coulomb_metric, mp_isdf


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
