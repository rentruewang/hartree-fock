import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import float64, ndarray
from numpy.typing import NDArray
from omegaconf import DictConfig
from typing_extensions import Self

from . import checks, paths
from .paths import skip_if_none


@dataclass(frozen=True)
class HFInput:
    """
    HartreeForkInput is the input terms of the HF algorithm.
    """

    electrons: int
    """
    The number of electrons/orbitals in the system.
    This is equal to the dimension of the matrices.
    """

    converge: float
    """
    The threshold below which the program deems to have converged.
    """

    iterations: int
    """
    The maximum iterations to run.
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

    density_init: NDArray | None
    """
    The initial density matrix.
    If None, zeros would be used.
    """

    ijkl: NDArray
    """
    The (ij|kl) integral terms (used for Coulomb and exchange).
    4D matrix because there are 4 parameters (uses Yoshimine sort).
    """

    def __post_init__(self):
        assert isinstance(self.electrons, int)
        assert isinstance(self.vnn, float)
        assert isinstance(self.kinetic, ndarray)
        assert isinstance(self.potential, ndarray)
        assert isinstance(self.overlap, ndarray)

        assert len(set(self.kinetic.shape)) == 1, self.kinetic.shape
        assert len(set(self.potential.shape)) == 1, self.potential.shape
        assert len(set(self.overlap.shape)) == 1, self.overlap.shape
        assert len(set(self.ijkl.shape)) == 1, self.ijkl.shape

        assert self.kinetic.ndim == 2, self.kinetic.ndim
        assert self.potential.ndim == 2, self.potential.ndim
        assert self.overlap.ndim == 2, self.overlap.ndim
        assert self.ijkl.ndim == 4, self.ijkl.ndim

        assert (
            self.electrons
            == len(self.kinetic)
            == len(self.potential)
            == len(self.overlap)
            == len(self.ijkl)
        )

    @staticmethod
    def parse_txt_to_symmetric(fname: Path):
        mat = np.loadtxt(fname)
        return HFInput.make_symmetric(mat)

    @staticmethod
    def make_symmetric(mat: NDArray[float64]):
        if checks.symmetric(mat):
            return mat

        assert checks.square(mat)
        return mat + mat.T - mat.diagonal()

    @staticmethod
    def yoshimine(x: int, y: int) -> int:
        if x < y:
            x, y = y, x
        return x * (x + 1) // 2 + y

    @staticmethod
    def yoshimine_4(a: int, b: int, c: int, d: int) -> int:
        ab = HFInput.yoshimine(a, b)
        cd = HFInput.yoshimine(c, d)
        abcd = HFInput.yoshimine(ab, cd)
        return abcd

    @staticmethod
    def from_yoshimine(mapping: dict[tuple[int, int, int, int], float], orbitals: int):
        # Since all of the following permutations
        # (ab|cd) (ba|cd) (ab|dc) (ba|dc) (cd|ab) (cd|ba) (dc|ab) (dc|ba)
        # are equal, hash them with yoshimine.

        yoshimine_dict = {
            HFInput.yoshimine_4(*tuple_4): val for tuple_4, val in mapping.items()
        }

        mat = np.zeros(shape=[orbitals] * 4)

        for i, j, k, l in itertools.product(*[range(orbitals) for _ in range(4)]):
            yoshimine = HFInput.yoshimine_4(i, j, k, l)
            mat[i, j, k, l] = yoshimine_dict[yoshimine]

        return mat

    @staticmethod
    def parse_ijkl(fname: str | Path) -> dict[tuple[int, int, int, int], float]:
        with open(fname) as f:
            data = f.readlines()

        result = {}
        for line in data:
            i, j, k, l, val = map(float, line.split())
            i, j, k, l = map(int, [i, j, k, l])
            result[i, j, k, l] = val
        return result

    @classmethod
    def from_config(cls, name: str, cfg: DictConfig) -> Self:
        orbitals = int(cfg["electrons"])

        data_path = Path(paths.DATA) / name

        return cls(
            electrons=orbitals,
            converge=float(cfg["converge"]),
            iterations=int(cfg["iterations"]),
            vnn=float(cfg["vnn"]),
            kinetic=cls.parse_txt_to_symmetric(data_path / "kinetic.txt"),
            potential=cls.parse_txt_to_symmetric(data_path / "potential.txt"),
            overlap=cls.parse_txt_to_symmetric(data_path / "overlap.txt"),
            density_init=skip_if_none(cls.parse_txt_to_symmetric)(
                paths.exist_or_none(data_path / "density.txt")
            ),
            ijkl=cls.from_yoshimine(cls.parse_ijkl(data_path / "ijkl.txt"), orbitals),
        )
