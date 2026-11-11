"""Tests for local ISDF anchor selection. Build: cmake --build build --target mp_isdf_python."""

from pathlib import Path
import gc
import sys
import unittest

import numpy as np
from scipy.linalg import qr
from pyscf import gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build"))
import mp_isdf


class TestAnchorSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)
        cls.mean_field = scf.RHF(cls.mol).run(verbose=0)
        cls.context = mp_isdf.LibcintContext(cls.mol._atm, cls.mol._bas, cls.mol._env)
        cls.selector = mp_isdf.AnchorSelection(cls.context)
        cls.points = np.random.default_rng(12).normal(size=(8, 3))
        cls.weights = np.array([1., 2., 0., -1., 1., 3., 1., 1.])

    def check_reference(self, axes, tolerance=1e-10, max_rank=0, occupied=False,
                        space="auto"):
        good = self.weights != 0
        points = self.points[good]
        ao = self.mol.eval_gto("GTOval_sph", points)
        gradient = self.mol.eval_gto("GTOval_ip_sph", points)
        nocc = self.mol.nelectron // 2
        c = self.mean_field.mo_coeff
        pieces = {
            mp_isdf.AxisType.AtomicOrbital: ao,
            mp_isdf.AxisType.AtomicOrbitalGradX: gradient[0],
            mp_isdf.AxisType.AtomicOrbitalGradY: gradient[1],
            mp_isdf.AxisType.AtomicOrbitalGradZ: gradient[2],
            mp_isdf.AxisType.OccupiedOrbital: ao @ c[:, :nocc],
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
            nocc=nocc if occupied else 0, space=space,
        )
        self.assertEqual(actual, expected)
        self.assertNotIn(2, actual)  # Original index of the removed zero-weight point.

    def test_ao_and_gradient(self):
        self.check_reference([mp_isdf.AxisType.AtomicOrbital])
        self.check_reference([mp_isdf.AxisType.AtomicOrbitalGradX,
                              mp_isdf.AxisType.AtomicOrbitalGradZ], max_rank=4)

    def test_axis_concatenation_with_repeated_types(self):
        self.check_reference([mp_isdf.AxisType.AtomicOrbitalGradZ,
                              mp_isdf.AxisType.AtomicOrbital,
                              mp_isdf.AxisType.AtomicOrbitalGradZ], max_rank=3)

    def test_selection_runtime_dispatch(self):
        axes = [mp_isdf.AxisType.AtomicOrbital,
                mp_isdf.AxisType.AtomicOrbitalGradX]
        spaces = ("host", "auto", "device") if mp_isdf.cuda_enabled else ("host", "auto")
        for space in spaces:
            with self.subTest(space=space):
                self.check_reference(axes, max_rank=3, space=space)
        if not mp_isdf.cuda_enabled:
            with self.assertRaises(ValueError):
                self.selector.select(self.points, self.weights, axes, space="device")
        self.assertEqual(self.selector.select(self.points, np.zeros(8), axes,
                                              space="host"), [])
        with self.assertRaises(ValueError):
            self.selector.select(self.points, self.weights, axes, space="invalid")

    def test_axis_evaluation_on_host(self):
        # Exercises the host AO/gradient kernels, MO projection, and axis
        # assembly—not just compilation of the host-space template.
        axes = [mp_isdf.AxisType.AtomicOrbitalGradZ,
                mp_isdf.AxisType.OccupiedOrbital,
                mp_isdf.AxisType.AtomicOrbital,
                mp_isdf.AxisType.AtomicOrbitalGradZ]
        c = self.mean_field.mo_coeff
        nocc = self.mol.nelectron // 2
        ao = self.mol.eval_gto("GTOval_sph", self.points)
        grad_z = self.mol.eval_gto("GTOval_ip_sph", self.points)[2]
        expected = np.hstack((grad_z, ao @ c[:, :nocc], ao, grad_z))
        host = self.selector.evaluate_axis(self.points, axes, mo_coeff=c,
                                           nocc=nocc, space="host")
        default = self.selector.evaluate_axis(self.points, axes, mo_coeff=c,
                                              nocc=nocc, space="auto")
        if mp_isdf.cuda_enabled:
            device = self.selector.evaluate_axis(self.points, axes, mo_coeff=c,
                                                 nocc=nocc, space="device")
        np.testing.assert_allclose(host, expected, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(host, default, rtol=1e-11, atol=1e-12)
        if mp_isdf.cuda_enabled:
            np.testing.assert_allclose(host, device, rtol=1e-11, atol=1e-12)

    def test_occupied_rhf(self):
        self.check_reference([mp_isdf.AxisType.OccupiedOrbital], occupied=True)
        self.check_reference([mp_isdf.AxisType.OccupiedOrbital,
                              mp_isdf.AxisType.AtomicOrbitalGradY], occupied=True)

    def test_single_axis_and_all_zero_weights(self):
        axis = mp_isdf.AxisType.AtomicOrbital
        selected = self.selector.select(self.points, self.weights, axis, max_rank=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(self.selector.select(self.points, np.zeros(8), axis), [])

    def test_selector_keeps_context_alive(self):
        temporary = mp_isdf.LibcintContext(self.mol._atm, self.mol._bas, self.mol._env)
        selector = mp_isdf.AnchorSelection(temporary)
        del temporary
        gc.collect()
        self.assertTrue(selector.select(self.points, self.weights,
                                        mp_isdf.AxisType.AtomicOrbital))

    def test_invalid_and_unimplemented(self):
        axis = mp_isdf.AxisType.OccupiedOrbital
        with self.assertRaises(ValueError):
            self.selector.select(self.points, self.weights, axis)
        with self.assertRaises(ValueError):
            self.selector.select(np.empty((0, 3)), np.empty(0), axis)
        with self.assertRaises(ValueError):
            self.selector.select(self.points, self.weights, axis,
                                 mo_coeff=np.ones((2, 1)), nocc=1)
        randomized = self.selector.select(
            self.points, self.weights, mp_isdf.AxisType.AtomicOrbital,
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedFeatureKhatriRao,
            sketch_size=12, oversampling=3, max_rank=2, seed=7)
        self.assertEqual(len(randomized), 2)
        self.assertEqual(len(set(randomized)), 2)
        self.assertNotIn(2, randomized)
        distributed = mp_isdf.DistributedAnchorSelection(self.context)
        with self.assertRaises(RuntimeError):
            distributed.select(self.points, self.weights, mp_isdf.AxisType.AtomicOrbital)


if __name__ == "__main__":
    unittest.main()
