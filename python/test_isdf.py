"""Integration tests for the minimal Python ISDF driver."""

from pathlib import Path
import sys
import unittest

import numpy as np
from pyscf import dft, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isdf import build_isdf, isdf_ao


class TestMinimalISDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

    def test_ao_anchor_selection_and_metric(self):
        result = build_isdf(self.mol, level=0, max_rank=4)
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = 0
        grids.build()
        context = isdf_ao.LibcintContext(self.mol._atm, self.mol._bas, self.mol._env)
        expected = isdf_ao.AnchorSelection(context).select(
            grids.coords, grids.weights, isdf_ao.AxisType.AtomicOrbital, max_rank=4
        )
        np.testing.assert_array_equal(result["anchor_indices"], expected)
        rank = len(expected)
        self.assertEqual(result["X1"].shape, (rank, self.mol.nao_nr()))
        self.assertEqual(result["interp_vecs"].shape, (rank, len(result["weights"])))
        self.assertEqual(result["coulomb_metric"].shape, (rank, rank))
        np.testing.assert_allclose(result["coulomb_metric"],
                                   result["coulomb_metric"].T, atol=1e-12)
        np.testing.assert_allclose(result["X1"], context.eval_ao(result["anchor_coords"]))

    def test_occupied_feature_selection(self):
        mf = scf.RHF(self.mol).run(verbose=0)
        result = build_isdf(self.mol, mf, level=0, max_rank=2,
                            axes=("ao", "occ"), compute_metric=False)
        self.assertGreater(len(result["anchor_indices"]), 0)
        self.assertLessEqual(len(result["anchor_indices"]), 2)
        self.assertIsNone(result["coulomb_metric"])


if __name__ == "__main__":
    unittest.main()
