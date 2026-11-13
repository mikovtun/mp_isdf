"""RBRP/Algorithm 5.1 regression and driver integration tests."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from pyscf import dft, gto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import build_isdf, mp_isdf
from rbrp_id import AOPairs, interpolate, select


class TestRBRPID(unittest.TestCase):
    def test_absolute_weighted_selection_and_signed_id(self):
        rng = np.random.default_rng(19)
        ao = rng.normal(size=(40, 4))
        w = np.exp(rng.normal(size=40)) * rng.choice([-1., 1.], size=40)
        op = AOPairs(ao, w, batch_size=7)
        res = select(op, block_size=4, max_rank=6, rtol=0., seed=3)
        other = select(AOPairs(ao, np.abs(w), batch_size=11), block_size=4,
                       max_rank=6, rtol=0., seed=3)
        np.testing.assert_array_equal(res.indices, other.indices)
        np.testing.assert_allclose(op.rows(np.arange(40)) @ op.rows(np.arange(40)).T,
            (ao @ ao.T)**2 * op.sqrtw[:, None] * op.sqrtw[None, :], atol=1e-12)
        theta, transform, info = interpolate(op, res)
        pairs = (ao[:, :, None] * ao[:, None, :]).reshape(40, -1)
        j = res.indices
        direct_transform = np.linalg.pinv(w[j, None] * pairs[j], rcond=1e-12).T
        np.testing.assert_allclose(transform, direct_transform, atol=1e-11)
        np.testing.assert_allclose(theta, (pairs @ direct_transform.T).T, atol=1e-11)
        np.testing.assert_allclose(theta[:, j] * w[j][None, :], np.eye(len(j)), atol=1e-12)
        err = np.linalg.norm((pairs - theta.T @ (w[j, None] * pairs[j])) *
                             op.sqrtw[:, None])
        self.assertAlmostEqual(err / np.sqrt(op.norm2().sum()), res.relative_residual, places=12)
        self.assertEqual(info["retained_rank"], len(j))

    def test_cpp_backend_matches_direct_fit_at_its_pivots(self):
        rng = np.random.default_rng(52)
        ao = rng.normal(size=(36, 5))
        weights = np.exp(rng.normal(size=36)) * rng.choice([-1., 1.], size=36)
        opts = dict(block_size=4, batch_size=8, max_rank=8, seed=12,
                    tolerance=0., compute_id=True)
        result = mp_isdf.rbrp_ao_cpu(ao, weights, **opts)
        j = np.asarray(result["indices"])
        self.assertEqual(len(np.unique(j)), len(j))
        np.testing.assert_array_equal(j, mp_isdf.rbrp_ao_cpu(ao, weights, **opts)["indices"])
        pairs = (ao[:, :, None] * ao[:, None, :]).reshape(len(ao), -1)
        direct = np.linalg.pinv(weights[j, None] * pairs[j], rcond=1e-12).T
        np.testing.assert_allclose(result["transform"], direct, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result["interp_vecs"], (pairs @ direct.T).T,
                                   rtol=1e-10, atol=1e-10)
        residual = (pairs - result["interp_vecs"].T @ (weights[j, None]*pairs[j]))
        relative = (np.linalg.norm(residual * np.sqrt(np.abs(weights))[:, None]) /
                    np.linalg.norm(pairs * np.sqrt(np.abs(weights))[:, None]))
        self.assertAlmostEqual(relative, result["relative_residual"], places=10)
        self.assertEqual(result["retained_rank"], len(j))
        self.assertEqual(result["history"].shape[1], 4)
        only = mp_isdf.rbrp_ao_cpu(ao, weights, **(opts | {"compute_id": False}))
        self.assertNotIn("interp_vecs", only)
        with self.assertRaisesRegex(ValueError, "nonzero weights"):
            mp_isdf.rbrp_ao_cpu(ao, np.zeros(len(ao)))

    def test_rank_deficiency_zero_and_validation(self):
        rng = np.random.default_rng(11)
        ao = np.tile(rng.normal(size=(3, 3)), (8, 1))
        op = AOPairs(ao, -np.ones(len(ao)))
        res = select(op, block_size=8, rtol=1e-10)
        self.assertEqual(len(res.indices), 3)
        self.assertLess(res.relative_residual, 1e-10)
        self.assertEqual(select(AOPairs(np.zeros((4, 2)), np.ones(4))).status, "tolerance")
        with self.assertRaises(ValueError):
            AOPairs(ao, np.zeros(len(ao)))

    def test_driver_id_and_selector_only_against_refit(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        opts = dict(level=0, anchor_rank=2, anchor_seed=7, space="host",
                    pair_space_error=True)
        with patch("isdf.np.linalg.pinv", side_effect=AssertionError("NumPy fit used")):
            fit_id = build_isdf(mol, anchor_algorithm="rbrp-id", **opts)
        fit_old = build_isdf(mol, anchor_algorithm="rbrp", **opts)
        np.testing.assert_array_equal(fit_id["anchor_indices"], fit_old["anchor_indices"])
        np.testing.assert_allclose(fit_id["interp_vecs"], fit_old["interp_vecs"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(fit_id["coulomb_metric"], fit_old["coulomb_metric"], rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(fit_id["pair_space_errors"]["scalar"]["relative"],
                               fit_old["pair_space_errors"]["scalar"]["relative"], places=9)
        only = build_isdf(mol, anchor_algorithm="rbrp-id", level=0, anchor_rank=2,
                          compute_metric=False, selection_only=True)
        self.assertIsNone(only["interp_vecs"])
        np.testing.assert_array_equal(only["anchor_indices"], fit_id["anchor_indices"])

    def test_driver_signed_grid_anchor_weights(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        grid = dft.gen_grid.Grids(mol)
        grid.level = 0
        grid.build()
        coords, weights = grid.coords.copy(), -np.abs(grid.weights.copy())

        def make_grid(_mol):
            return SimpleNamespace(level=0, build=lambda: None,
                                   coords=coords.copy(), weights=weights.copy())

        with patch("isdf.dft.gen_grid.Grids", side_effect=make_grid):
            result = build_isdf(mol, level=0, anchor_rank=2,
                                anchor_algorithm="rbrp-id", space="host",
                                compute_metric=False, pair_space_error=True)
        self.assertTrue(np.all(result["anchor_weights"] < 0))
        j = result["anchor_indices"]
        active = np.flatnonzero(weights)
        local = np.searchsorted(active, j)
        np.testing.assert_allclose(result["interp_vecs"][:, local] *
                                   result["anchor_weights"][None, :],
                                   np.eye(len(j)), atol=1e-12)
        self.assertAlmostEqual(result["pair_space_errors"]["scalar"]["relative"],
                               result["rbrp_diagnostics"]["relative_residual"], places=10)

    def test_driver_rejects_unsupported_axes(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        with self.assertRaisesRegex(ValueError, "axes"):
            build_isdf(mol, anchor_algorithm="rbrp-id", axes=("dx",))

    @unittest.skipIf(hasattr(mp_isdf, "rbrp_ao_device"), "requires CPU-only build")
    def test_device_rejected_without_cuda(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        with self.assertRaisesRegex(ValueError, "requires a CUDA build"):
            build_isdf(mol, anchor_algorithm="rbrp-id", space="device")

    @unittest.skipUnless(hasattr(mp_isdf, "rbrp_ao_device"), "requires CUDA build")
    def test_device_rbrp_id_and_signed_weights(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        grid = dft.gen_grid.Grids(mol)
        grid.level = 0
        grid.build()
        mask = grid.weights != 0
        coords = np.ascontiguousarray(grid.coords[mask])
        weights = np.ascontiguousarray(grid.weights[mask])
        weights[::2] *= -1
        context = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        selector = mp_isdf.AnchorSelection(context)
        ao = context.eval_ao(coords, space="host")
        pairs = (ao[:, :, None] * ao[:, None, :]).reshape(len(ao), -1)
        for block in (1, 2, 4):
            device = mp_isdf.rbrp_ao_device(selector, coords, weights,
                                            max_rank=2, block_size=block,
                                            batch_size=37, compute_id=True)
            j = device["indices"]
            ref = np.linalg.pinv(weights[j, None]*pairs[j], rcond=1e-12).T
            np.testing.assert_allclose(device["transform"], ref, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(device["interp_vecs"], (pairs @ ref.T).T,
                                       rtol=1e-9, atol=1e-9)
            exact = pairs - device["interp_vecs"].T @ (weights[j, None]*pairs[j])
            rel = (np.linalg.norm(exact*np.sqrt(np.abs(weights))[:, None]) /
                   np.linalg.norm(pairs*np.sqrt(np.abs(weights))[:, None]))
            self.assertAlmostEqual(rel, device["relative_residual"], places=10)
        dev_fit = build_isdf(mol, anchor_algorithm="rbrp-id", anchor_rank=2,
                             space="device", pair_space_error=True,
                             compute_metric=False)
        self.assertAlmostEqual(dev_fit["pair_space_errors"]["scalar"]["relative"],
                               dev_fit["rbrp_diagnostics"]["relative_residual"], places=10)
        selected = build_isdf(mol, anchor_algorithm="rbrp", anchor_rank=2,
                              space="device", compute_metric=False, selection_only=True)
        self.assertEqual(len(selected["anchor_indices"]), 2)
        with patch.object(mp_isdf, "rbrp_ao_device", wraps=mp_isdf.rbrp_ao_device) as gpu:
            automatic = build_isdf(mol, anchor_algorithm="rbrp", anchor_rank=2,
                                   space="auto", compute_metric=False, selection_only=True)
            gpu.assert_called_once()
            np.testing.assert_array_equal(automatic["anchor_indices"],
                                          selected["anchor_indices"])
            build_isdf(mol, anchor_algorithm="rbrp", anchor_rank=2,
                       space="host", compute_metric=False, selection_only=True)
            gpu.assert_called_once()


if __name__ == "__main__":
    unittest.main()
