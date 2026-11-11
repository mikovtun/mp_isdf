"""Small host/CUDA checks for the structured randomized pair-product sketch."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import qr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build"))
import mp_isdf


class TestRandomizedAnchorSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atm = np.zeros((3, 6), dtype=np.int32)
        atm[:, 1] = [20, 23, 26]
        bas = np.zeros((3, 8), dtype=np.int32)
        bas[:, 0] = [0, 1, 2]
        bas[:, 2:4] = 1
        bas[:, 5] = [29, 31, 33]
        bas[:, 6] = [30, 32, 34]
        env = np.zeros(35)
        env[20:29] = [0, 0, 0, 0, 0, 1, 0, 1, 0]
        env[29:35] = [.8, 1, 1.1, 1, 1.3, 1]
        cls.context = mp_isdf.LibcintContext(atm, bas, env)
        cls.selector = mp_isdf.AnchorSelection(cls.context)
        cls.points = np.random.default_rng(23).normal(size=(19, 3))
        cls.weights = np.linspace(.4, 1.5, 19)
        cls.weights[2] = 0
        cls.weights[4] = -1
        cls.axis = mp_isdf.AxisType.AtomicOrbital

    def select(self, **kwargs):
        return self.selector.select(
            self.points, self.weights, self.axis,
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedQRColumnPivoting,
            sketch_size=12, oversampling=3, seed=31, **kwargs)

    def test_pivots_and_real_pair_projection(self):
        mask = self.weights != 0
        phi = self.selector.evaluate_axis(self.points[mask], self.axis, space="host")
        a = (np.einsum("gi,gj->gij", phi, phi).reshape(sum(mask), -1)
             * np.sqrt(np.abs(self.weights[mask]))[:, None]).T
        original = np.flatnonzero(mask)
        for space in (("host", "device") if mp_isdf.cuda_enabled else ("host",)):
            for batch in (1, 4, 19):
                with self.subTest(space=space, batch=batch):
                    pivots = self.select(space=space, batch_size=batch, max_rank=3)
                    self.assertEqual(pivots, self.select(space=space, batch_size=2, max_rank=3))
                    self.assertEqual(len(pivots), 3)
                    self.assertNotIn(2, pivots)
                    positions = np.searchsorted(original, pivots)
                    q = qr(a[:, positions], mode="economic")[0]
                    error = np.linalg.norm(a-q @ (q.T @ a))/np.linalg.norm(a)
                    self.assertLess(error, 0.4)
            complete = self.select(space=space, batch_size=4)
            self.assertEqual(len(complete), 6)  # packed-symmetric physical rank
            self.assertEqual(self.select(space=space, tolerance=1, max_rank=0), [])

    def test_concatenated_axes_and_occupied(self):
        axes = [self.axis, mp_isdf.AxisType.AtomicOrbitalGradX,
                mp_isdf.AxisType.OccupiedOrbital]
        opts = dict(mo_coeff=np.eye(3), nocc=1, max_rank=3,
                    sketch_size=12, oversampling=3, seed=31, batch_size=2)
        for space in (("host", "device") if mp_isdf.cuda_enabled else ("host",)):
            result = self.selector.select(
                self.points, self.weights, axes,
                algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedQRColumnPivoting,
                space=space, **opts)
            self.assertEqual(len(result), 3)
            self.assertEqual(len(set(result)), 3)
            self.assertNotIn(2, result)

    def test_validations_and_optional_audit(self):
        self.assertEqual(len(self.select(space="host", max_rank=2, audit=True)), 2)
        # Explicit rank determines sketch rows independently of sketch_size.
        explicit = self.selector.select(
            self.points, self.weights, self.axis,
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedQRColumnPivoting,
            sketch_size=2, oversampling=3, max_rank=3, seed=31, space="host")
        self.assertEqual(explicit, self.select(space="host", max_rank=3))
        for kw in ({"batch_size": 0}, {"oversampling": 0}):
            opts = dict(kw)
            if "oversampling" in opts:
                with self.assertRaises(ValueError):
                    self.selector.select(self.points, self.weights, self.axis,
                        algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedQRColumnPivoting,
                        sketch_size=12, **opts)
            else:
                with self.assertRaises(ValueError):
                    self.select(space="host", **opts)
        self.assertEqual(self.selector.select(
            self.points, np.zeros_like(self.weights), self.axis,
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedQRColumnPivoting,
            sketch_size=12, oversampling=3, space="host"), [])
        if not mp_isdf.cuda_enabled:
            with self.assertRaises(ValueError):
                self.select(space="device")


if __name__ == "__main__":
    unittest.main()
