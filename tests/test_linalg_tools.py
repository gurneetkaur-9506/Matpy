import unittest

import numpy as np

from specialist_lib import eig, svd


class TestEig(unittest.TestCase):
    def test_returns_eigenvectors_then_diagonal_matrix(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        V, D = eig(A)
        w, v = np.linalg.eig(A)
        # MATLAB's [V, D] = eig(A) puts the eigenvectors first and the
        # eigenvalues in a diagonal matrix.
        np.testing.assert_allclose(V, v)
        np.testing.assert_allclose(D, np.diag(w))

    def test_eigendecomposition_holds(self):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((4, 4))
        A = (A + A.T) / 2
        V, D = eig(A)
        np.testing.assert_allclose(A @ V, V @ D, atol=1e-12)

    def test_works_with_complex_input(self):
        A = np.array([[0.0, -1.0], [1.0, 0.0]])
        V, D = eig(A)
        np.testing.assert_allclose(A @ V, V @ D, atol=1e-12)


class TestSvd(unittest.TestCase):
    def test_returns_plain_V_and_diagonal_S(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        U, S, V = svd(A)
        u, s, vh = np.linalg.svd(A)
        np.testing.assert_allclose(U, u)
        np.testing.assert_allclose(S, np.diag(s))
        # numpy returns Vh (conjugate transpose); the wrapper must return V.
        np.testing.assert_allclose(V, vh.conj().T)

    def test_reconstruction(self):
        rng = np.random.default_rng(11)
        A = rng.standard_normal((5, 3))
        U, S, V = svd(A)
        # A = U * S * V' for the reduced decomposition.
        np.testing.assert_allclose(U @ S @ V.T, A, atol=1e-12)

    def test_works_with_complex_input(self):
        A = np.array([[1.0, 2.0j], [3.0, 4.0]])
        U, S, V = svd(A)
        np.testing.assert_allclose(U @ S @ V.conj().T, A, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
