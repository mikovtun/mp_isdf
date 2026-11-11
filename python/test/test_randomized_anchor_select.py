"""Small host/CUDA checks for the structured randomized pair-product sketch."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from pyscf import dft, gto
from scipy.linalg import qr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build"))
import mp_isdf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import build_isdf


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
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedFeatureKhatriRao,
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
            self.assertEqual(len(self.select(space=space, tolerance=1, max_rank=2)), 2)
            self.assertEqual(len(self.select(space=space, tolerance=0, max_rank=2)), 2)

    def test_quadrature_ao_mode(self):
        positive = np.abs(self.weights)
        for space in (("host", "device") if mp_isdf.cuda_enabled else ("host",)):
            for batch in (1, 5, 19):
                opts = dict(max_rank=3, oversampling=4, sketch_size=1,
                            seed=31, space=space, batch_size=batch)
                strict = self.selector.select(
                    self.points, positive, self.axis,
                    algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedAOQuadratureKhatriRao,
                    **opts)
                general = self.selector.select(
                    self.points, positive, self.axis,
                    algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedFeatureKhatriRao,
                    **opts)
                self.assertEqual(strict, general)
                self.assertEqual(len(strict), 3)
                self.assertNotIn(2, strict)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.selector.select(self.points, self.weights, self.axis,
                algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedAOQuadratureKhatriRao,
                max_rank=3, space="host")
        with self.assertRaisesRegex(ValueError, "axes"):
            self.selector.select(self.points, positive,
                [self.axis, mp_isdf.AxisType.AtomicOrbitalGradX],
                algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedAOQuadratureKhatriRao,
                max_rank=3, space="host")

    def test_explicit_khatri_rao_identity(self):
        rng = np.random.default_rng(7)
        phi = rng.normal(size=(4, 9))
        w = rng.uniform(.1, 2., size=9)
        g = rng.normal(size=(6, 4))
        h = rng.normal(size=(6, 4))
        structured = (g @ phi) * (h @ phi) * (np.sqrt(w)/np.sqrt(6))[None, :]
        pairs = np.einsum("ig,jg->ijg", phi, phi).reshape(16, 9)
        projection = np.einsum("si,sj->sij", g, h).reshape(6, 16)/np.sqrt(6)
        np.testing.assert_allclose(structured, projection @ (pairs*np.sqrt(w)))

    def test_driver_rank_tolerance_are_exclusive(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_isdf(mol, anchor_algorithm="rand_feature_kr", anchor_rank=2,
                       anchor_tol=1e-6, compute_metric=False, selection_only=True)

    def test_driver_positive_quadrature(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        original_grids = dft.gen_grid.Grids

        class PositiveGrids(original_grids):
            def build(self, *args, **kwargs):
                result = super().build(*args, **kwargs)
                self.weights = np.abs(self.weights)
                return result

        with patch.object(dft.gen_grid, "Grids", PositiveGrids):
            result = build_isdf(mol, anchor_algorithm="rand_ao_kr_quadrature",
                                anchor_rank=2, anchor_oversampling=3,
                                compute_metric=False, selection_only=True, space="host")
        raw_grid = original_grids(mol)
        raw_grid.level = 0
        raw_grid.build()
        if np.any(raw_grid.weights < 0):
            with self.assertRaisesRegex(ValueError, "nonnegative quadrature"):
                build_isdf(mol, anchor_algorithm="rand_ao_kr_quadrature",
                           anchor_rank=2, compute_metric=False, selection_only=True,
                           space="host")
        feature = build_isdf(mol, anchor_algorithm="rand_feature_kr",
                             anchor_rank=2, anchor_oversampling=3,
                             compute_metric=False, selection_only=True, space="host")
        self.assertEqual(len(feature["anchor_indices"]), 2)
        self.assertEqual(len(result["anchor_indices"]), 2)
        self.assertTrue(np.all(result["anchor_weights"] > 0))

    def test_concatenated_axes_and_occupied(self):
        axes = [self.axis, mp_isdf.AxisType.AtomicOrbitalGradX,
                mp_isdf.AxisType.OccupiedOrbital]
        opts = dict(mo_coeff=np.eye(3), nocc=1, max_rank=3,
                    sketch_size=12, oversampling=3, seed=31, batch_size=2)
        for space in (("host", "device") if mp_isdf.cuda_enabled else ("host",)):
            result = self.selector.select(
                self.points, self.weights, axes,
                algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedFeatureKhatriRao,
                space=space, **opts)
            self.assertEqual(len(result), 3)
            self.assertEqual(len(set(result)), 3)
            self.assertNotIn(2, result)

    def test_validations_and_optional_audit(self):
        self.assertEqual(len(self.select(space="host", max_rank=2, audit=True)), 2)
        # Explicit rank determines sketch rows independently of sketch_size.
        explicit = self.selector.select(
            self.points, self.weights, self.axis,
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedFeatureKhatriRao,
            sketch_size=2, oversampling=3, max_rank=3, seed=31, space="host")
        self.assertEqual(explicit, self.select(space="host", max_rank=3))
        for kw in ({"batch_size": 0}, {"oversampling": 0}):
            opts = dict(kw)
            if "oversampling" in opts:
                with self.assertRaises(ValueError):
                    self.selector.select(self.points, self.weights, self.axis,
                        algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedFeatureKhatriRao,
                        sketch_size=12, **opts)
            else:
                with self.assertRaises(ValueError):
                    self.select(space="host", **opts)
        self.assertEqual(self.selector.select(
            self.points, np.zeros_like(self.weights), self.axis,
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedFeatureKhatriRao,
            sketch_size=12, oversampling=3, space="host"), [])
        if not mp_isdf.cuda_enabled:
            with self.assertRaises(ValueError):
                self.select(space="device")


if __name__ == "__main__":
    unittest.main()
