"""Tests for local ISDF anchor selection. Build: cmake --build build --target isdf_ao."""

from pathlib import Path
import gc
import sys
import unittest

import numpy as np
from scipy.linalg import qr
from pyscf import gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build"))
import isdf_ao


class TestAnchorSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)
        cls.mean_field = scf.RHF(cls.mol).run(verbose=0)
        cls.context = isdf_ao.LibcintContext(cls.mol._atm, cls.mol._bas, cls.mol._env)
        cls.selector = isdf_ao.AnchorSelection(cls.context)
        cls.points = np.random.default_rng(12).normal(size=(8, 3))
        cls.weights = np.array([1., 2., 0., -1., 1., 3., 1., 1.])

    def check_reference(self, axes, tolerance=1e-10, max_rank=0, occupied=False):
        good = self.weights != 0
        points = self.points[good]
        ao = self.mol.eval_gto("GTOval_sph", points)
        gradient = self.mol.eval_gto("GTOval_ip_sph", points)
        nocc = self.mol.nelectron // 2
        c = self.mean_field.mo_coeff
        pieces = {
            isdf_ao.AxisType.AtomicOrbital: ao,
            isdf_ao.AxisType.AtomicOrbitalGradX: gradient[0],
            isdf_ao.AxisType.AtomicOrbitalGradY: gradient[1],
            isdf_ao.AxisType.AtomicOrbitalGradZ: gradient[2],
            isdf_ao.AxisType.OccupiedOrbital: ao @ c[:, :nocc],
        }
        columns = np.hstack([pieces[axis] for axis in axes])
        outer = (columns[:, :, None] * columns[:, None, :]).reshape(len(points), -1)
        matrix = (outer * np.sqrt(np.abs(self.weights[good]))[:, None]).T
        _, r, pivots = qr(matrix, pivoting=True, mode="economic")
        diag = np.abs(np.diag(r))
        count = int(np.sum(diag > tolerance * diag[0])) if len(diag) and diag[0] else 0
        if max_rank:
            count = min(count, max_rank)
        expected = np.flatnonzero(good)[pivots[:count]].tolist()
        actual = self.selector.select(
            self.points, self.weights, axes, tolerance=tolerance,
            max_rank=max_rank, mo_coeff=c if occupied else None,
            nocc=nocc if occupied else 0,
        )
        self.assertEqual(actual, expected)
        self.assertNotIn(2, actual)  # Original index of the removed zero-weight point.

    def test_ao_and_gradient(self):
        self.check_reference([isdf_ao.AxisType.AtomicOrbital])
        self.check_reference([isdf_ao.AxisType.AtomicOrbitalGradX,
                              isdf_ao.AxisType.AtomicOrbitalGradZ], max_rank=4)

    def test_axis_concatenation_with_repeated_types(self):
        self.check_reference([isdf_ao.AxisType.AtomicOrbitalGradZ,
                              isdf_ao.AxisType.AtomicOrbital,
                              isdf_ao.AxisType.AtomicOrbitalGradZ], max_rank=3)

    def test_occupied_rhf(self):
        self.check_reference([isdf_ao.AxisType.OccupiedOrbital], occupied=True)
        self.check_reference([isdf_ao.AxisType.OccupiedOrbital,
                              isdf_ao.AxisType.AtomicOrbitalGradY], occupied=True)

    def test_single_axis_and_all_zero_weights(self):
        axis = isdf_ao.AxisType.AtomicOrbital
        selected = self.selector.select(self.points, self.weights, axis, max_rank=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(self.selector.select(self.points, np.zeros(8), axis), [])

    def test_selector_keeps_context_alive(self):
        temporary = isdf_ao.LibcintContext(self.mol._atm, self.mol._bas, self.mol._env)
        selector = isdf_ao.AnchorSelection(temporary)
        del temporary
        gc.collect()
        self.assertTrue(selector.select(self.points, self.weights,
                                        isdf_ao.AxisType.AtomicOrbital))

    def test_invalid_and_unimplemented(self):
        axis = isdf_ao.AxisType.OccupiedOrbital
        with self.assertRaises(ValueError):
            self.selector.select(self.points, self.weights, axis)
        with self.assertRaises(ValueError):
            self.selector.select(np.empty((0, 3)), np.empty(0), axis)
        with self.assertRaises(ValueError):
            self.selector.select(self.points, self.weights, axis,
                                 mo_coeff=np.ones((2, 1)), nocc=1)
        with self.assertRaises(RuntimeError):
            self.selector.select(self.points, self.weights,
                                 isdf_ao.AxisType.AtomicOrbital,
                                 algorithm=isdf_ao.LocalAnchorSelectionAlgorithm.RandomizedQRColumnPivoting)
        distributed = isdf_ao.DistributedAnchorSelection(self.context)
        with self.assertRaises(RuntimeError):
            distributed.select(self.points, self.weights, isdf_ao.AxisType.AtomicOrbital)


if __name__ == "__main__":
    unittest.main()
