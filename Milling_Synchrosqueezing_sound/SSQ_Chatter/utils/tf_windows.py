# Comentario: utilidades TF (ventanas locales) y SVD en batch
from typing import Tuple, Optional
import numpy as np


def extract_local_windows(
    S1: np.ndarray,
    K: int,
    time_vector: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 3D tensor of local time windows A_i from a TF matrix S1.
    """
    # Comentario: validaciones básicas
    if not isinstance(S1, np.ndarray):
        raise TypeError("S1 must be a NumPy array")
    M, N = S1.shape
    if K <= 0 or K > N:
        raise ValueError("K must be in [1, N]")

    if time_vector is None:
        time_vector = np.arange(N, dtype=float)

    num_windows = N - K + 1
    A_all = np.empty((num_windows, M, K), dtype=S1.dtype)
    for i in range(num_windows):
        A_all[i] = S1[:, i : i + K]

    t_Ai = time_vector[K - 1 :]
    return A_all, t_Ai


def compute_svd(
    A: np.ndarray,
    ensure_real: bool = True,
):
    """
    Compute SVD for 2D or batched 3D arrays: returns (U, S, Vh).
    """
    # Comentario: soporta A 2D (M,K) o 3D (B,M,K)
    if A.ndim == 2:
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
    elif A.ndim == 3:
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
    else:
        raise ValueError("A must be 2D or 3D")

    if ensure_real:
        U = np.real_if_close(U)
        S = np.real_if_close(S)
        Vh = np.real_if_close(Vh)
    return U, S, Vh
