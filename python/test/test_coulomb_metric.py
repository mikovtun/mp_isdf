"""Integral-direct Cholesky: diagonal, pivot, residual, pairs, contraction and driver."""
from pathlib import Path
import sys
import unittest
import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isdf import build_isdf, mp_isdf


class TestCoulombMetric(unittest.TestCase):
    def check_molecule(self, mol):
        ctx = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        nao = mol.nao_nr()
        mu, nu = np.tril_indices(nao)
        npair = len(mu)
        eri = mol.intor('int2e', aosym='s1')
        packed = eri[mu[:, None], nu[:, None], mu[None, :], nu[None, :]]
        factor1, pivots1, residual1 = mp_isdf.cholesky_eri(ctx, tolerance=0, max_rank=1)
        self.assertEqual(factor1.shape, (1, npair))
        self.assertEqual(pivots1[0], np.argmax(np.diag(packed)))
        np.testing.assert_allclose(factor1[0], packed[:, pivots1[0]] /
                                   np.sqrt(packed[pivots1[0], pivots1[0]]), atol=1e-12)
        np.testing.assert_allclose(residual1, np.maximum(0, np.diag(packed)-factor1[0]**2), atol=1e-12)
        factor, pivots, residual = mp_isdf.cholesky_eri(ctx, tolerance=1e-12)
        self.assertEqual(len(set(pivots.tolist())), len(pivots))
        np.testing.assert_allclose(factor.T @ factor, packed, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(residual, np.maximum(0, np.diag(packed-factor.T@factor)), atol=2e-10)
        self.assertLessEqual(residual.max(), 1e-12*np.diag(packed).max()+1e-12)
        rng = np.random.default_rng(42)
        transform = rng.normal(size=(3, nao*nao))
        full = transform.reshape(3, nao, nao)
        expected = np.einsum('aij,ijkl,bkl->ab', full, eri, full)
        for batch in (1, 2, 64):
            np.testing.assert_allclose(mp_isdf.coulomb_metric(ctx, transform, batch_size=batch),
                                       expected, atol=2e-10, rtol=2e-10)
        coeff = full[:, mu, nu].copy()
        coeff[:, mu != nu] += full[:, nu[mu != nu], mu[mu != nu]]
        proj = coeff @ factor1[0]
        np.testing.assert_allclose(mp_isdf.coulomb_metric(ctx, transform, max_rank=1),
                                   np.outer(proj,proj), atol=1e-12)
        np.testing.assert_array_equal(mp_isdf.coulomb_metric(ctx, transform, tolerance=2),
                                      np.zeros((3,3)))
        with self.assertRaises(ValueError):
            mp_isdf.coulomb_metric(ctx, transform, algorithm='dense')
        with self.assertRaises(ValueError):
            mp_isdf.cholesky_eri(ctx, max_rank=npair+1)
        with self.assertRaises(ValueError):
            mp_isdf.coulomb_metric(ctx, transform, batch_size=0)

    def test_s_shells(self):
        self.check_molecule(gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', verbose=0))

    def test_p_shell(self):
        self.check_molecule(gto.M(atom='He 0 0 0', basis='cc-pvdz', verbose=0))

    def test_parallel_derivative_shell_quartets(self):
        # cc-pVDZ has multiple s and p shells: both channels have enough
        # ordered shell-pair tasks to run the parallel host path.
        mol = gto.M(atom='He 0 0 0', basis='cc-pvdz', verbose=0)
        ctx = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        nao = mol.nao_nr()
        identity = np.eye(nao * nao)
        actual = mp_isdf.derivative_cross_metric(
            ctx, identity, (0, 0), identity, (3, 3), tolerance=1e-12)
        expected = (mol.intor('int2e_ipvip1', comp=9, aosym='s1')[8]
                    .transpose(2, 3, 0, 1).reshape(nao*nao, nao*nao))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_driver_uses_ao_coefficients_and_anchor_weights(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        result = build_isdf(mol, level=0, anchor_rank=3)
        anchor = result['X1']
        weighted = (anchor[:,:,None]*anchor[:,None,:]).reshape(len(anchor), -1)
        weighted *= result['anchor_weights'][:,None]
        transform = np.linalg.pinv(weighted@weighted.T, rcond=1e-12) @ weighted
        coeff = transform.reshape(len(anchor), mol.nao_nr(),mol.nao_nr())
        eri = mol.intor('int2e', aosym='s1')
        expected = np.einsum('aij,ijkl,bkl->ab', coeff, eri, coeff)
        w = result['anchor_weights']
        np.testing.assert_allclose(result['coulomb_metric'], w[:,None]*expected*w[None,:],
                                   rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
