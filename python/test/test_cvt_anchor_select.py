"""CVT anchor selection and driver integration tests."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build"))
import mp_isdf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import build_isdf


class TestCVT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        cls.context = mp_isdf.LibcintContext(cls.mol._atm, cls.mol._bas, cls.mol._env)
        cls.selector = mp_isdf.AnchorSelection(cls.context)
        cls.axis = mp_isdf.AxisType.AtomicOrbital
        cls.algorithm = mp_isdf.LocalAnchorSelectionAlgorithm.CentroidalVoronoi

    def select(self, points, weights, cvt_weights, **kwargs):
        return self.selector.select(points, weights, self.axis,
                                    max_rank=kwargs.pop("max_rank", 2),
                                    algorithm=self.algorithm, cvt_weights=cvt_weights,
                                    space="host", **kwargs)

    def test_weighted_clusters_and_pruned_snap(self):
        points = np.array([[0., 0, 0], [0.1, 0, 0], [99., 0, 0],
                           [10., 0, 0], [10.1, 0, 0]])
        weights = np.array([1., 1., 0., -1., 1.])
        density = np.ones(5)
        result = self.select(points, weights, density, seed=3)
        self.assertEqual(len(result), 2)
        self.assertIn(result[0], (0, 1))
        self.assertIn(result[1], (3, 4))
        self.assertEqual(result, self.select(points, weights, density, seed=3))
        self.assertNotIn(2, result)

    def test_validation_and_degenerate_points(self):
        points = np.zeros((4, 3))
        weights = np.array([1., 0., 2., 3.])
        self.assertEqual(len(set(self.select(points, weights, np.ones(4), max_rank=3))), 3)
        for density, rank in ((np.ones(3), 2), (np.array([1., -1., 1., 1.]), 2),
                              (np.ones(4), 4)):
            with self.subTest(density=density, rank=rank), self.assertRaises(ValueError):
                self.select(points, weights, density, max_rank=rank)
        with self.assertRaises(ValueError):
            self.select(points, weights, np.ones(4), cvt_max_iter=0)
        with self.assertRaises(ValueError):
            self.select(points, weights, np.ones(4), cvt_tol=-1)
        with self.assertRaises(ValueError):
            self.select(points, weights, np.ones(4), max_rank=0)
        with self.assertRaises(ValueError):
            self.selector.select(points, weights, self.axis, max_rank=2,
                                 algorithm=self.algorithm)
        if not mp_isdf.cuda_enabled:
            with self.assertRaises(ValueError):
                self.selector.select(points, weights, self.axis, max_rank=2,
                                     algorithm=self.algorithm, cvt_weights=np.ones(4),
                                     space="device")

    def test_exclusive_weight_modes(self):
        points = np.array([[0., 0, 0], [5., 0, 0], [10., 0, 0], [20., 0, 0]])
        weights = np.array([1., 1., 100., 0.])
        density = np.array([100., 1., 1., 1000000.])
        self.assertEqual(self.select(points, weights, density, max_rank=1, seed=2), [0])
        self.assertEqual(self.select(points, weights, np.abs(weights), max_rank=1, seed=2), [2])

    def test_driver_weight_modes(self):
        for weight in ("quadrature_weights", "electron_density"):
            with self.subTest(weight=weight):
                result = build_isdf(self.mol, level=0, anchor_rank=2,
                                    anchor_algorithm="cvt", cvt_weight=weight,
                                    selection_only=True, compute_metric=False,
                                    space="host")
                self.assertEqual(len(result["anchor_indices"]), 2)
                self.assertTrue(np.all(result["anchor_weights"] != 0))


if __name__ == "__main__":
    unittest.main()
