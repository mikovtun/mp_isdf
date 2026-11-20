"""Density sketch algebra, host/device selection, and physical refit regression."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from pyscf import dft, gto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import (DENSITY_SKETCHES, _normalized_pair_error, _physical_pair_refit,
                  build_isdf, mp_isdf, prepare_point_importance)


class TestDensitySketch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
        cls.context = mp_isdf.LibcintContext(cls.mol._atm, cls.mol._bas, cls.mol._env)
        cls.selector = mp_isdf.AnchorSelection(cls.context)
        cls.points = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0], [3., 0, 0], [1e3, 0, 0]])
        cls.weights = np.array([1., -2., 4., 0., 1.])
        cls.mass = np.array([1., 2., 3., 1e6, 1e6])
        cls.axis = mp_isdf.AxisType.AtomicOrbital

    def select(self, normalized=True, **kwargs):
        opts = dict(algorithm=getattr(mp_isdf.LocalAnchorSelectionAlgorithm,
                    "RandomizedAODensityNormalizedKhatriRao" if normalized else
                    "RandomizedAODensityKhatriRao"),
                    max_rank=1, oversampling=3, point_importance=self.mass,
                    space="host", seed=31)
        opts.update(kwargs)
        return self.selector.select(self.points, self.weights, self.axis, **opts)

    def test_normalized_rank_one_removes_amplitude(self):
        # All nonzero one-AO pair columns have exactly the same direction.
        # Normalized selection must pick the greatest mass, independent of q.
        for space in (("host", "device") if mp_isdf.cuda_enabled else ("host",)):
            for batch_size in (1, 2, 5):
                with self.subTest(space=space, batch_size=batch_size):
                    self.assertEqual(self.select(space=space, batch_size=batch_size), [2])
                    raw = self.select(False, space=space, batch_size=batch_size)
                    q = np.sum(self.context.eval_ao(self.points, space="host")**2, axis=1)
                    expected = np.argmax((self.mass * q**2) * (self.weights != 0))
                    self.assertEqual(raw, [expected])
                    self.assertNotEqual(raw, [2])
        self.assertEqual(self.select(point_importance=self.mass * 100), [2])
        # A finite sketch has exactly density-driven norms in this rank-one case.
        self.assertEqual(self.select(audit=True), [2])

    def test_no_hidden_quadrature_factor(self):
        altered = self.weights.copy()
        altered[:3] = [1e9, -1e-9, 1e-20]
        result = self.selector.select(
            self.points, altered, self.axis, max_rank=1, oversampling=3,
            point_importance=self.mass, space="host",
            algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedAODensityNormalizedKhatriRao)
        self.assertEqual(result, [2])
        m = self.mass.copy()
        m[2] = 0
        self.assertEqual(self.select(point_importance=m), [1])

    def test_screening_and_validations(self):
        q = np.sum(self.context.eval_ao(self.points, space="host")**2, axis=1)
        self.assertEqual(self.select(ao_screening_threshold=(q[1] + q[2]) / 2), [1])
        self.assertEqual(self.select(max_rank=0, ao_screening_threshold=1e10), [])
        self.assertEqual(self.select(max_rank=0, point_importance=np.zeros(5)), [])
        for kwargs in ({"point_importance": None}, {"point_importance": np.ones(4)},
                       {"point_importance": [1., -1., 1., 1., 1.]},
                       {"point_importance": [np.nan] * 5},
                       {"point_importance": [np.inf] * 5},
                       {"ao_screening_threshold": -1}, {"ao_screening_threshold": np.nan},
                       {"ao_screening_threshold": 1e10}, {"max_rank": 2},
                       {"point_importance": np.zeros(5)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select(**kwargs)
        with self.assertRaisesRegex(ValueError, "axes"):
            self.selector.select(
                self.points, self.weights, [self.axis, mp_isdf.AxisType.AtomicOrbitalGradX],
                algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedAODensityNormalizedKhatriRao,
                point_importance=self.mass)
        if not mp_isdf.cuda_enabled:
            with self.assertRaises(ValueError):
                self.select(space="device")

    def test_raw_mode_matches_legacy_with_importance_as_weights(self):
        # Same Gaussian draws and complete scale, without hidden physical weights.
        mass = self.mass.copy()
        mass[self.weights == 0] = 0
        old = self.selector.select(
            self.points, mass, self.axis, max_rank=1, oversampling=3, seed=31,
            space="host", algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedAOQuadratureKhatriRao)
        self.assertEqual(self.select(False), old)

    def test_multi_ao_matches_independent_legacy_scaling(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        context = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        selector = mp_isdf.AnchorSelection(context)
        points = np.random.default_rng(74).normal(size=(23, 3))
        phi = context.eval_ao(points, space="host")
        q = np.sum(phi**2, axis=1)
        mass = np.linspace(.1, 2, len(points))
        weights = np.linspace(-1, 2, len(points))
        weights[2] = 0
        mass[5] = 0
        reference_weights = mass / q**2
        reference_weights[weights == 0] = 0
        for space in (("host", "device") if mp_isdf.cuda_enabled else ("host",)):
            for seed in (7, 31):
                opts = dict(max_rank=2, oversampling=4, seed=seed, space=space)
                reference = selector.select(
                    points, reference_weights, self.axis, batch_size=3,
                    algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedAOQuadratureKhatriRao,
                    **opts)
                for batch_size in (1, 11):
                    actual = selector.select(
                        points, weights, self.axis, batch_size=batch_size,
                        point_importance=mass,
                        algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.RandomizedAODensityNormalizedKhatriRao,
                        **opts)
                    self.assertEqual(actual, reference)

    def test_explicit_sketch_identity_packing_and_rescaling(self):
        rng = np.random.default_rng(41)
        phi = rng.normal(size=(7, 4))
        m = rng.uniform(.1, 2, 7)
        ell = 13
        u, v = rng.normal(size=(2, 4, ell))
        q = np.sum(phi**2, axis=1)
        pairs = np.einsum("gi,gj->gij", phi, phi).reshape(7, -1).T
        probes = np.einsum("it,jt->tij", u, v).reshape(ell, -1) / np.sqrt(ell)
        structured = ((phi @ u) * (phi @ v)).T * (np.sqrt(m) / q)[None, :] / np.sqrt(ell)
        np.testing.assert_allclose(structured, probes @ (pairs * (np.sqrt(m) / q)), atol=1e-14)
        i, j = np.triu_indices(4)
        packed = phi[:, i] * phi[:, j] * np.where(i == j, 1., np.sqrt(2))
        np.testing.assert_allclose(np.linalg.norm(packed, axis=1), q)
        scaled_phi = phi * np.exp(rng.normal(size=7))[:, None]
        scaled = ((scaled_phi @ u) * (scaled_phi @ v)).T
        scaled *= (np.sqrt(m / ell) / np.sum(scaled_phi**2, axis=1))[None, :]
        np.testing.assert_allclose(structured, scaled, atol=1e-14)
        # Isotropy of independent factored probes, checked statistically.
        u, v = rng.normal(size=(2, 4, 100_000))
        scores = np.mean(((phi @ u) * (phi @ v))**2, axis=1) * m / q**2
        np.testing.assert_allclose(scores, m, rtol=.04)

    def test_stable_refit_weighted_storage_and_rank_deficiency(self):
        rng = np.random.default_rng(12)
        anchors = rng.normal(size=(3, 2))
        anchors[2] = anchors[0]  # redundant selected column
        pairs = np.einsum("gi,gj->gij", anchors, anchors).reshape(3, -1)
        ao = rng.normal(size=(11, 2))
        weights = np.array([1e-8, -2., 3.])
        theta, transform, diagnostics = _physical_pair_refit(pairs, weights, ao, 2)
        exact = np.einsum("gi,gj->gij", ao, ao).reshape(11, -1)
        physical = np.linalg.lstsq(pairs.T, exact.T, rcond=1e-12)[0]
        np.testing.assert_allclose(weights[:, None] * theta, physical, atol=1e-14)
        np.testing.assert_allclose(transform @ exact.T, theta, rtol=1e-13)
        self.assertEqual(diagnostics["retained_rank"], 2)
        # Normalized residual is explicitly the physical relative-column loss.
        mass = np.linspace(.1, 1, len(ao))
        error = _normalized_pair_error(ao, pairs, weights, theta, mass, 0)
        residual = exact.T - pairs.T @ physical
        q = np.sum(ao**2, axis=1)
        expected = np.sqrt(np.sum(mass * np.sum(residual**2, axis=0) / q**2) / sum(mass))
        self.assertAlmostEqual(error["relative"], expected)

    def test_driver_shared_mass_and_refit_keeps_zero_mass_points(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        grid = dft.gen_grid.Grids(mol)
        grid.level = 0
        grid.build()
        importance = np.ones(len(grid.coords))
        importance[::2] = 0
        for algorithm in ("cvt", *DENSITY_SKETCHES):
            with self.subTest(algorithm=algorithm):
                result = build_isdf(mol, grid=grid, anchor_algorithm=algorithm,
                                    point_importance=importance, anchor_rank=2,
                                    anchor_oversampling=3, compute_metric=False,
                                    pair_space_error=True, space="host")
                self.assertTrue(np.all(importance[result["anchor_indices"]] > 0))
                self.assertEqual(result["interp_vecs"].shape[1], np.count_nonzero(grid.weights))
                self.assertTrue(np.isfinite(result["interp_vecs"]).all())
                self.assertIn("density_normalized", result["pair_space_errors"])
                self.assertIsNotNone(result["fit_diagnostics"])
        with patch("isdf.prepare_point_importance", return_value=importance) as prepare:
            result = build_isdf(mol, grid=grid, anchor_algorithm=DENSITY_SKETCHES[1],
                                anchor_rank=2, compute_metric=False, selection_only=True,
                                density_weight="quadrature_density", space="host")
            self.assertEqual(prepare.call_args.kwargs["convention"], "quadrature_density")
            np.testing.assert_array_equal(result["point_importance"], importance)

    def test_comparison_runner_and_json(self):
        import json
        from compare_density_sketches import compare, _json_value
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        rows = compare(mol, rank=2, oversampling=3, seeds=(7,), thresholds=(1e-24,))
        self.assertEqual([row["algorithm"] for row in rows], ["cvt", *DENSITY_SKETCHES])
        for row in rows:
            self.assertIn("energy_error", row["errors"]["J"])
            self.assertIn("energy_error", row["errors"]["K"])
            self.assertIn("density_normalized", row["pair_space_errors"])
        encoded = json.dumps(_json_value(rows), allow_nan=False)
        self.assertEqual(len(json.loads(encoded)), 3)

    def test_importance_generation_conventions(self):
        # Density generation itself is covered by the CVT integration tests;
        # use deterministic synthetic density to verify exact batching/scaling.
        fake_rks = unittest.mock.Mock()
        fake_rks.converged = True
        fake_rks.make_rdm1.return_value = np.eye(1)
        with patch("isdf.dft.RKS", return_value=fake_rks), \
             patch("isdf.dft.numint.eval_ao", side_effect=lambda mol, coords: np.ones((len(coords), 1))), \
             patch("isdf.dft.numint.eval_rho", side_effect=lambda mol, ao, dm: np.full(len(ao), 2.)):
            rho = prepare_point_importance(self.mol, self.points, self.weights, batch_size=2)
            mass = prepare_point_importance(self.mol, self.points, self.weights,
                                            convention="quadrature_density", batch_size=1)
        np.testing.assert_array_equal(rho, [2., 2., 2., 0., 2.])
        np.testing.assert_array_equal(mass, rho * np.abs(self.weights))


if __name__ == "__main__":
    unittest.main()
