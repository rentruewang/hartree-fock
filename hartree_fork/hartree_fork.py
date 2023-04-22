import alive_progress
import numpy as np
from numpy import linalg
from numpy.typing import NDArray

from . import checks, ortho
from .inputs import HFInput


def density(C: NDArray, electrons: int):
    assert checks.square(C)
    occupied = C[:, : electrons // 2]

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


def energy(D: NDArray, H: NDArray, F: NDArray) -> float:
    return (D * (H + F)).sum()


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
    D = np.zeros(shape=[N, N])

    vnn = hf_input.vnn

    orthogonalizer = ortho.get(hf_input.orthogonalizer)

    threshold = hf_input.converge
    iterations = hf_input.iterations
    electrons = hf_input.electrons

    E = 0

    with alive_progress.alive_bar(iterations) as bar:
        for _ in range(iterations):
            bar()

            F = fork(H, D, ijkl)
            E = energy(D, H, F)
            X = orthogonalizer(S)

            # Fp: F' = X.T @ F @ X
            Fp = X.T @ F @ X

            # Cp: C', since F'C' = eC', C' is the eigen vectors of F'
            (_, Cp) = linalg.eig(Fp)

            C = X @ Cp

            D_new = density(C, electrons)

            if ((D - D_new) ** 2).sum() < threshold:
                bar(skipped=True)
                break

            D = D_new

    return E + vnn
