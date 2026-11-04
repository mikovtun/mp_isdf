"""Compare the Kokkos AO evaluator with PySCF's native spherical AO values.

Build the extension first: cmake --build build --target isdf_ao
Run: OMP_PROC_BIND=false python python/test_ao_eval.py
"""

from pathlib import Path
import sys
import unittest

import numpy as np
from pyscf import gto

# The CMake extension is built in the project build directory by default.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import isdf_ao


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
                actual = isdf_ao.eval_ao(mol._atm, mol._bas, mol._env, points)
                expected = mol.eval_gto("GTOval_sph", points)
                self.assertEqual(actual.shape, (len(points), mol.nao_nr()))
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    def test_empty_grid(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="cc-pvdz", verbose=0)
        actual = isdf_ao.eval_ao(mol._atm, mol._bas, mol._env, np.empty((0, 3)))
        self.assertEqual(actual.shape, (0, mol.nao_nr()))


if __name__ == "__main__":
    unittest.main()
