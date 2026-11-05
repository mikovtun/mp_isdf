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

    @staticmethod
    def reference_greedy(matrix, max_rank):
        residual = np.maximum(0.0, np.diag(matrix).copy())
        factors, pivots = [], []
        for _ in range(max_rank):
            pivot = int(np.argmax(residual))  # first index on an exact tie
            maximum = residual[pivot]
            if maximum <= 0:
                break
            column = matrix[:, pivot].copy()
            for factor in factors:
                column -= factor * factor[pivot]
            column /= np.sqrt(maximum)
            factors.append(column)
            pivots.append(pivot)
            residual = np.maximum(0.0, residual - column**2)
            residual[pivot] = 0.0
        return np.array(factors), pivots, residual

    def test_parallel_correction_and_pivot_reduction(self):
        # 14 AOs give 105 packed pairs, above the host reduction threshold.
        mol = gto.M(atom='He 0 0 0', basis='cc-pvtz', verbose=0)
        ctx = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        mu, nu = np.tril_indices(mol.nao_nr())
        eri = mol.intor('int2e', aosym='s1')
        matrix = eri[mu[:, None], nu[:, None], mu[None, :], nu[None, :]]
        expected, pivots, residual = self.reference_greedy(matrix, 4)
        actual, selected, remainder = mp_isdf.cholesky_eri(ctx, tolerance=0, max_rank=4)
        np.testing.assert_array_equal(selected, pivots)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(remainder, residual, atol=1e-12, rtol=1e-12)
        # Many small batches exercise the parallel BLAS++ projection and Gram
        # paths; the ordered AO transform must fold off-diagonal pairs once.
        transform = np.random.default_rng(91).normal(size=(19, mol.nao_nr()**2))
        ordered = transform.reshape(19, mol.nao_nr(), mol.nao_nr())
        packed = ordered[:, mu, nu].copy()
        packed[:, mu != nu] += ordered[:, nu[mu != nu], mu[mu != nu]]
        projected = packed @ actual.T
        expected_metric = projected @ projected.T
        for batch in (1, 2, 64):
            metric = mp_isdf.coulomb_metric(ctx, transform, max_rank=4,
                                            tolerance=0, batch_size=batch)
            np.testing.assert_allclose(metric, expected_metric, atol=2e-10, rtol=2e-10)
            np.testing.assert_allclose(metric, metric.T, atol=1e-12)

    def test_parallel_derivative_correction_and_reduction(self):
        # Two 10x10 ordered channels give 200 pairs, above the threshold.
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
        ctx = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        npair = mol.nao_nr()**2
        identity = np.eye(npair)
        ordinary = mol.intor('int2e', aosym='s1').reshape(npair, npair)
        mixed = (mol.intor('int2e_ipvip1', comp=9, aosym='s1')[8]
                 .transpose(2, 3, 0, 1).reshape(npair, npair))
        derivative = (mol.intor('int2e_ipvip1ipvip2', comp=81, aosym='s1')[80]
                      .reshape(npair, npair))
        matrix = np.block([[ordinary, mixed], [mixed.T, derivative]])
        factors, _, _ = self.reference_greedy(matrix, 3)
        expected = factors[:, :npair].T @ factors[:, npair:]
        actual = mp_isdf.derivative_cross_metric(
            ctx, identity, (0, 0), identity, (3, 3), tolerance=0, max_rank=3)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

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
