"""Independent small-matrix checks of the bounded-batch Kokkos QRCP selector.

Uses hand-built libcint tables so this test does not require PySCF.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import qr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build"))
import mp_isdf


class TestOnlineAnchorSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Three unit-coefficient s shells, each on a different atom.
        atm = np.zeros((3, 6), dtype=np.int32)
        atm[:, 1] = [20, 23, 26]
        bas = np.zeros((3, 8), dtype=np.int32)
        bas[:, 0] = [0, 1, 2]
        bas[:, 2:4] = 1
        bas[:, 5] = [29, 31, 33]
        bas[:, 6] = [30, 32, 34]
        env = np.zeros(35)
        env[20:29] = [0, 0, 0, 0, 0, 1.0, 0, 1.0, 0]
        env[29:35] = [0.8, 1, 1.1, 1, 1.3, 1]
        cls.context = mp_isdf.LibcintContext(atm, bas, env)
        cls.selector = mp_isdf.AnchorSelection(cls.context)
        cls.points = np.random.default_rng(23).normal(size=(17, 3))
        cls.weights = np.linspace(0.5, 2, 17)
        cls.weights[2] = 0
        cls.weights[4] *= -1
        cls.axis = mp_isdf.AxisType.AtomicOrbital

    def online(self, **kwargs):
        return self.selector.select(
            self.points, self.weights, self.axis,
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.OnlineQRColumnPivoting,
            **kwargs)

    def test_greedy_and_batch_invariance(self):
        mask = self.weights != 0
        features = self.selector.evaluate_axis(self.points[mask], self.axis, space="host")
        matrix = (np.einsum("gi,gj->gij", features, features).reshape(sum(mask), -1)
                  * np.sqrt(np.abs(self.weights[mask]))[:, None]).T
        expected = np.flatnonzero(mask)[qr(matrix, pivoting=True)[2]]
        for space in (("host", "device") if mp_isdf.cuda_enabled else ("host",)):
            for batch in (1, 3, 17):
                with self.subTest(space=space, batch=batch):
                    actual = self.online(space=space, batch_size=batch)
                    np.testing.assert_array_equal(actual, expected[:len(actual)])
                    self.assertEqual(len(actual), 6)  # symmetric AO pairs
                    np.testing.assert_array_equal(self.online(
                        space=space, batch_size=batch, max_rank=3), expected[:3])
                    # Independently verify the greedy residual at every rank.
                    remaining = np.arange(matrix.shape[1])
                    for k, pivot in enumerate(actual):
                        if k:
                            q = qr(matrix[:, np.searchsorted(np.flatnonzero(mask), actual[:k])],
                                   mode="economic")[0]
                            residual = matrix - q @ (q.T @ matrix)
                        else:
                            residual = matrix
                        self.assertAlmostEqual(
                            np.linalg.norm(residual[:, np.searchsorted(np.flatnonzero(mask), pivot)]),
                            max(np.linalg.norm(residual, axis=0)), delta=1e-9)

    def test_gradient_and_occupied_axis_on_both_spaces(self):
        axes = [self.axis, mp_isdf.AxisType.AtomicOrbitalGradX,
                mp_isdf.AxisType.OccupiedOrbital]
        coefficients = np.eye(3, dtype=float)
        for space in (("host", "device") if mp_isdf.cuda_enabled else ("host",)):
            with self.subTest(space=space):
                options = dict(mo_coeff=coefficients, nocc=1, space=space, max_rank=3)
                dense = self.selector.select(self.points, self.weights, axes, **options)
                online = self.selector.select(
                    self.points, self.weights, axes,
                    algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.OnlineQRColumnPivoting,
                    batch_size=2, **options)
                self.assertEqual(online, dense)

    def test_cutoff_and_invalid_inputs(self):
        self.assertEqual(self.online(tolerance=1, space="host"), [])
        for cutoff in (0.1, 0.5):
            self.assertEqual(self.online(tolerance=cutoff, batch_size=2, space="host"),
                             self.selector.select(self.points, self.weights, self.axis,
                                                  tolerance=cutoff, space="host"))
        self.assertEqual(self.online(max_rank=1, batch_size=2, space="host"),
                         self.online(batch_size=2, space="host")[:1])
        self.assertEqual(self.selector.select(
            self.points, np.zeros_like(self.weights), self.axis,
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.OnlineQRColumnPivoting,
            space="host"), [])
        with self.assertRaises(ValueError):
            self.online(batch_size=0, space="host")
        if not mp_isdf.cuda_enabled:
            with self.assertRaises(ValueError):
                self.online(space="device")


if __name__ == "__main__":
    unittest.main()
