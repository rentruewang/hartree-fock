import numpy as np
from numpy import linalg
from numpy.typing import NDArray


def square(mat: NDArray) -> bool:
    """
    Checks if the input matrix is square. A square matrix must be 2D.
    """

    return mat.ndim == 2 and mat.shape[0] == mat.shape[1]


def symmetric(mat: NDArray) -> bool:
    """
    Checks if the input matrix is symmetric. A symmetric matrix must be square.
    """

    return square(mat) and np.allclose(mat, mat.T)


def pos_def(mat: NDArray) -> bool:
    """
    Checks if the input matrix is positive definite. The matrix must be square.
    """

    if not square(mat):
        return False

    val, _ = linalg.eig(mat)
    return (val > 0).all().item()
