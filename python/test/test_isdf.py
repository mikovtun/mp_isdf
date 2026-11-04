"""Integration tests for the minimal Python ISDF driver."""

from pathlib import Path
import sys
import unittest

import numpy as np
from pyscf import dft, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import build_isdf, mp_isdf


class TestMinimalISDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

    def test_ao_anchor_selection_and_metric(self):
        result = build_isdf(self.mol, level=0, max_rank=4)
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

    def test_host_space_driver(self):
        result = build_isdf(self.mol, level=0, max_rank=2,
                            compute_metric=False, space="host")
        self.assertGreater(len(result["anchor_indices"]), 0)
        self.assertTrue(np.isfinite(result["overlap_relative_error"]))

    def test_occupied_feature_selection(self):
        mf = scf.RHF(self.mol).run(verbose=0)
        result = build_isdf(self.mol, mf, level=0, max_rank=2,
                            axes=("ao", "occ"), compute_metric=False)
        self.assertGreater(len(result["anchor_indices"]), 0)
        self.assertLessEqual(len(result["anchor_indices"]), 2)
        self.assertIsNone(result["coulomb_metric"])
        self.assertTrue(np.isfinite(result["overlap_relative_error"]))


if __name__ == "__main__":
    unittest.main()
