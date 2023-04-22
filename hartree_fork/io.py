import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import jit
from numpy import ndarray
from numpy.typing import NDArray
from typing_extensions import Self
import itertools
from . import checks


@dataclass(frozen=True)
class HFInput:
    """
    HartreeForkInput is the input terms of the HF algorithm.
    """

    orbitals: int
    """
    The number of orbitals. This is equal to the dimension of the matrices.
    """

    electrons: int
    """
    The number of electrons in the system.
    """

    converge: float
    """
    The threshold below which the program deems to have converged.
    """

    vnn: float
    """
    The nuclear-nuclear repulsion energy.
    Will be a scalar because of Born-Oppenheimer approximation.
    """

    kinetic: NDArray
    """
    The kinetic energy for each orbital.
    2D matrix that goes into the hamiltonian.
    """

    potential: NDArray
    """
    The potential energy for each orbital.
    2D matrix that goes into the hamiltonian.
    """

    overlap: NDArray
    """
    The overlap matrix for orbitals.
    2D matrix because this is the overlap between matrices.
    """

    ijkl: NDArray
    """
    The (ij|kl) integral terms (used for Coulomb and exchange).
    4D matrix because there are 4 parameters (uses Yoshimine sort).
    """

    def __post_init__(self):
        assert isinstance(self.orbitals, int)
        assert isinstance(self.vnn, float)
        assert isinstance(self.kinetic, ndarray)
        assert isinstance(self.potential, ndarray)
        assert isinstance(self.overlap, ndarray)
        assert isinstance(self.ijkl, ndarray)

        assert self.kinetic.ndim == 2, self.kinetic.ndim
        assert self.potential.ndim == 2, self.potential.ndim
        assert self.overlap.ndim == 2, self.overlap.ndim
        assert self.ijkl.ndim == 4, self.ijkl.ndim

        assert (
            self.orbitals
            == len(self.kinetic)
            == len(self.potential)
            == len(self.overlap)
        )

    @staticmethod
    def make_symmetric(mat: NDArray):
        assert checks.square(mat)
        return mat + mat.T - mat.diagonal()

    @staticmethod
    @jit
    def yoshimine(x: int, y: int):
        if x < y:
            x, y = y, x
        return x * (x + 1) / 2 + y

    @staticmethod
    @jit
    def yoshimine_4(a: int, b: int, c: int, d: int) -> int:
        ab = HFInput.yoshimine(a, b)
        cd = HFInput.yoshimine(c, d)
        abcd = HFInput.yoshimine(ab, cd)
        return abcd

    def from_yoshimine(self, raw_data: list[tuple[tuple[int, int, int, int], float]]):
        indices = [tuple(r[0]) for r in raw_data]
        values = [float(r[1]) for r in raw_data]

        # Since all of the following permutations
        # (ab|cd) (ba|cd) (ab|dc) (ba|dc) (cd|ab) (cd|ba) (dc|ab) (dc|ba)
        # are equal, hash them with yoshimine.

        yoshimine_dict = {
            HFInput.yoshimine_4(idx): val for (idx, val) in zip(indices, values)
        }

        mat = np.zeros(shape=[self.orbitals] * 4)

        for i, j, k, l in itertools.product(*([range(self.orbitals)] * 4)):
            yoshimine = self.yoshimine_4(i, j, k, l)
            mat[i, j, k, l] = yoshimine_dict[yoshimine]

        return mat

    @classmethod
    def parse(cls, fname: str) -> Self:
        with open(fname) as f:
            data = json.load(f)

        return cls(
            orbitals=int(data["orbitals"]),
            electrons=int(data["electrons"]),
            converge=float(data["converge"]),
            vnn=float(data["vnn"]),
            kinetic=cls.make_symmetric(np.array(data["kinetic"])),
            potential=cls.make_symmetric(np.array(data["potential"])),
            overlap=cls.make_symmetric(np.array(data["overlap"])),
            ijkl=cls.from_yoshimine(data["ijkl"]),
        )
