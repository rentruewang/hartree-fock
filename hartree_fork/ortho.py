from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg
from numpy.typing import NDArray

from . import checks


class Ortho(ABC):
    def __call__(self, S: NDArray) -> NDArray:
        """
        Parameter
        ---------

        S:
            The overlap matrix.
            Must be positive definite and symmetric.

        Returns
        -------

        A matrix X such that X.T @ S @ X = I
        """

        assert checks.pos_def(S)
        assert checks.symmetric(S)

        # s is guarenteed to be > 0 because it's positive definite.
        s, U = linalg.eigh(S)

        # Perform diagonalization using different strategies.
        X = self.diagonalize(s, U)

        assert np.allclose(X.T @ U @ X, np.eye(X.shape[1]))

        return X

    @abstractmethod
    def diagonalize(self, s: NDArray, U: NDArray) -> NDArray:
        ...


class Symmetric(Ortho):
    def diagonalize(self, s: NDArray, U: NDArray) -> NDArray:
        return U @ np.diag(s**-0.5) @ U.T


class Canonical(Ortho):
    def diagonalize(self, s: NDArray, U: NDArray) -> NDArray:
        return U @ np.diag(s**-0.5)
