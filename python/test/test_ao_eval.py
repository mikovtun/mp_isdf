"""Compare Kokkos AO values and gradients with PySCF's native evaluator.

Build the extension first: cmake --build build --target mp_isdf_python
Run: OMP_PROC_BIND=false python -m unittest discover -s python/test -p 'test_*.py'
"""

from pathlib import Path
import sys
import unittest

import numpy as np
from pyscf import gto

# The CMake extension is built in the project build directory by default.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build"))
import mp_isdf


class TestAOEvaluation(unittest.TestCase):
    def test_against_pyscf(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(17, 3))
        for basis in ("sto-3g", "cc-pvdz", "cc-pvtz"):
            with self.subTest(basis=basis):
                mol = gto.M(
                    atom="O 0 0 0; H 0 0.75 0.5; H 0 -0.75 0.5",
                    basis=basis,
                    verbose=0,
                )
                actual = mp_isdf.eval_ao(mol._atm, mol._bas, mol._env, points)
                expected = mol.eval_gto("GTOval_sph", points)
                self.assertEqual(actual.shape, (len(points), mol.nao_nr()))
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    def test_gradients_against_pyscf(self):
        rng = np.random.default_rng(11)
        # Include the nuclear center to exercise zero-coordinate monomials.
        points = np.vstack(([0.0, 0.0, 0.0], rng.normal(size=(17, 3))))
        for basis in ("sto-3g", "cc-pvdz", "cc-pvtz"):
            with self.subTest(basis=basis):
                mol = gto.M(
                    atom="O 0 0 0; H 0 0.75 0.5; H 0 -0.75 0.5",
                    basis=basis,
                    verbose=0,
                )
                actual = mp_isdf.eval_ao_gradient(mol._atm, mol._bas, mol._env, points)
                expected = mol.eval_gto("GTOval_ip_sph", points)
                self.assertEqual(actual.shape, (3, len(points), mol.nao_nr()))
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    def test_context_reuse_and_owned_tables(self):
        mol = gto.M(
            atom="O 0 0 0; H 0 0.75 0.5; H 0 -0.75 0.5",
            basis="cc-pvtz", verbose=0,
        )
        points = np.random.default_rng(7).normal(size=(9, 3))
        expected = mol.eval_gto("GTOval_sph", points)
        expected_grad = mol.eval_gto("GTOval_ip_sph", points)
        atm, bas, env = mol._atm.copy(), mol._bas.copy(), mol._env.copy()
        context = mp_isdf.LibcintContext(atm, bas, env)
        self.assertEqual(context.nao, mol.nao_nr())
        # The context must not retain pointers into mutable Python input arrays.
        atm.fill(0)
        bas.fill(0)
        env.fill(0)
        for _ in range(2):
            np.testing.assert_allclose(context.eval_ao(points), expected, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(
                context.eval_ao_gradient(points), expected_grad, rtol=1e-11, atol=1e-12
            )

    def test_runtime_space_dispatch(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="cc-pvdz", verbose=0)
        points = np.random.default_rng(17).normal(size=(7, 3))
        ctx = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        expected = mol.eval_gto("GTOval_sph", points)
        expected_grad = mol.eval_gto("GTOval_ip_sph", points)
        spaces = ("host", "auto", "device") if mp_isdf.cuda_enabled else ("host", "auto")
        for space in spaces:
            with self.subTest(space=space):
                np.testing.assert_allclose(ctx.eval_ao(points, space=space), expected,
                                           rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(ctx.eval_ao_gradient(points, space=space),
                                           expected_grad, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(
                    mp_isdf.eval_ao(mol._atm, mol._bas, mol._env, points, space=space),
                    expected, rtol=1e-11, atol=1e-12)
        if not mp_isdf.cuda_enabled:
            with self.assertRaises(ValueError):
                ctx.eval_ao(points, space="device")
        with self.assertRaises(ValueError):
            ctx.eval_ao(points, space="invalid")

    def test_empty_grid(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="cc-pvdz", verbose=0)
        empty = np.empty((0, 3))
        actual = mp_isdf.eval_ao(mol._atm, mol._bas, mol._env, empty)
        self.assertEqual(actual.shape, (0, mol.nao_nr()))
        gradient = mp_isdf.eval_ao_gradient(mol._atm, mol._bas, mol._env, empty)
        self.assertEqual(gradient.shape, (3, 0, mol.nao_nr()))
        context = mp_isdf.LibcintContext(mol._atm, mol._bas, mol._env)
        self.assertEqual(context.eval_ao(empty).shape, (0, mol.nao_nr()))
        self.assertEqual(context.eval_ao_gradient(empty).shape, (3, 0, mol.nao_nr()))


if __name__ == "__main__":
    unittest.main()
