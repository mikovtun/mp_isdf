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

    def test_driver_rejects_unsupported_axes_and_device(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        with self.assertRaisesRegex(ValueError, "axes"):
            build_isdf(mol, anchor_algorithm="rbrp-id", axes=("dx",))
        with self.assertRaisesRegex(ValueError, "host"):
            build_isdf(mol, anchor_algorithm="rbrp-id", space="device")


if __name__ == "__main__":
    unittest.main()
