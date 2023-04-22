from .io import HFInput
import numpy as np
from numpy.typing import NDArray
from . import checks


def density(C: NDArray, electrons: int):
    assert checks.square(C)
    occupied = C[:, electrons / 2]

    # D_uv = Sigma_i C_ui C_vi
    return np.einsum("ui,vi->uv", occupied, occupied)


def fork(H: NDArray, D: NDArray, ijkl: NDArray):
    J = ijkl
    K = ijkl.transpose(0, 2, 1, 3)

    two_j_sub_k = 2 * J - K

    # Calculate the two electron terms.
    # s: sigma, l: lambda, u: mu, v: nu
    two_electron = np.einsum("sl,uvls->uv", D, two_j_sub_k)

    return H + two_electron


def hartree_fork(hf_input: HFInput) -> float:
    # Number of orbitals.
    N = hf_input.orbitals

    # Hamiltonian.
    H = hf_input.kinetic + hf_input.potential

    # The 4 integrals (ij|kl).
    ijkl = hf_input.ijkl

    # The overlap matrix.
    S = hf_input.overlap

    # Density is initialized to 0
    D = np.zeros(shape=[H, H])

    threshold = hf_input.converge

    E = 0

    return E
