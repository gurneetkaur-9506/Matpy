"""Linear-algebra wrappers that restore MATLAB's output conventions.

The raw numpy call does not match MATLAB's calling contract in two common
places, so a thin translation layer is needed:

- ``eig(A)`` -- MATLAB's ``[V, D] = eig(A)`` returns the eigenvector matrix
  first and the eigenvalues as a *diagonal matrix* ``D``; ``numpy.linalg.eig``
  returns ``(w, v)`` with ``w`` a plain vector.
- ``svd(A)`` -- MATLAB's ``[U, S, V] = svd(A)`` returns ``V`` (not the
  conjugate transpose) and ``S`` as a *diagonal matrix*; ``numpy.linalg.svd``
  returns ``(U, s, Vh)`` with ``s`` a vector and ``Vh`` the conjugate
  transpose.  The wrapper restores MATLAB's order, matrix ``S`` and plain
  ``V`` so downstream code that indexes ``V(:, k)`` keeps working.
"""

import numpy as np

__all__ = ["eig", "svd"]


def eig(A):
    """Right eigenvectors and eigenvalues, MATLAB-style.

    Returns ``(V, D)`` matching MATLAB's ``[V, D] = eig(A)``: the columns of
    ``V`` are the right eigenvectors and ``D`` is the diagonal matrix of
    eigenvalues.  ``numpy.linalg.eig`` returns the eigenvalues as a vector
    with the eigenvectors second, so both the order and the diagonal matrix
    are restored here.
    """
    w, v = np.linalg.eig(A)
    return v, np.diag(w)


def svd(A):
    """Singular value decomposition, MATLAB-style.

    Returns ``(U, S, V)`` matching MATLAB's ``[U, S, V] = svd(A, 'econ')``:
    ``U`` and ``V`` are the left and right singular vectors and ``S`` is the
    diagonal matrix of singular values, so ``A = U @ S @ V.T`` holds for
    rectangular inputs as well as square ones.  ``numpy.linalg.svd`` returns
    the singular values as a vector and the right singular vectors as the
    conjugate transpose ``Vh``, so the vector is turned back into a diagonal
    matrix and ``Vh`` into the plain ``V`` MATLAB callers expect.
    """
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    return u, np.diag(s), vh.conj().T
