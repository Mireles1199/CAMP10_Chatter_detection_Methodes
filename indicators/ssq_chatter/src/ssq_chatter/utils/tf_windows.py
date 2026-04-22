from __future__ import annotations
# Comentario: extracción de subventanas y SVD (SRP: responsabilidad única)
from typing import Tuple, Optional, Literal
import numpy as np

class WindowExtractor:
    """
    WindowExtractor: Time-Frequency Window Extraction and SVD Computation Utility.

    A utility class providing static methods for extracting local windows from
    time-frequency representations and computing their singular value decompositions.
    """

    @staticmethod
    def extract_local_windows(        S1: np.ndarray,
        K: int,
        time_vector: Optional[np.ndarray] = None,
        mode: Literal["center", "causal_inclusive", "forward_inclusive"] = "causal_inclusive",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fixed-size local windows from a 2D time-frequency representation.
        This function slides through the time dimension of a 2D array (frequency x time)
        and extracts windows of fixed width K centered or aligned at each time step.
        Windows at boundaries are padded to maintain consistent size.
        Parameters
        ----------
        S1 : np.ndarray
            Input 2D array of shape (F, T) where F is the number of frequencies
            and T is the number of time steps.
        K : int
            Window width (number of time samples per window). Must be > 0.
        time_vector : Optional[np.ndarray], default=None
            1D array of time values corresponding to columns of S1. If None,
            integer indices are used instead.
        mode : Literal["center", "causal_inclusive", "forward_inclusive"], default="causal_inclusive"
            Window alignment mode:
            - "center": Window centered at each time step (K//2 left, K-K//2-1 right).
            - "causal_inclusive": Window includes K past samples ending at current step [i-K+1, ..., i].
            - "forward_inclusive": Window includes current and K-1 future samples [i, ..., i+K-1].
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A_out : np.ndarray
                Stacked windows of shape (T, F, K) where T is the number of windows,
                F is the number of frequencies, and K is the window width.
            t_out : np.ndarray
                Time values (or indices) corresponding to each window, shape (T,).
        Raises
        ------
        TypeError
            If S1 is not a NumPy array.
        ValueError
            If S1 is not 2D, K is not positive, or mode is not recognized.
        Notes
        -----
        Boundary windows are padded with zeros. Left-padding is preferred at the start,
        right-padding at the end, and centered padding for center mode.
        """

        if not isinstance(S1, np.ndarray):
            raise TypeError("S1 must be a NumPy array")
        if S1.ndim != 2:
            raise ValueError("S1 must be 2D (freq x time)")
        if K <= 0:
            raise ValueError("K must be > 0")

        F, T = S1.shape
        A = []
        times = []

        times_test = S1[:,0]  # para debug: revisar primeros K bloques para cada modo

        # Comentario: helper: pad seguro (sin negativos)
        def pad_block(block: np.ndarray, want_K: int, pad_left_hint: int) -> np.ndarray:
            width = block.shape[1]
            if width >= want_K:
                return block
            deficit = want_K - width
            left_pad = min(deficit, max(0, pad_left_hint))
            right_pad = deficit - left_pad
            return np.pad(block, ((0, 0), (left_pad, right_pad)), mode="constant")

        for i in range(T):
            if mode == "center":
                # Comentario: alrededor de i (K par/impar OK)
                pad_left = K // 2
                pad_right = K - pad_left - 1
                i0 = max(0, i - pad_left)
                i1 = min(T, i + pad_right + 1)  # exclusivo

                block = S1[:, i0:i1]
                block = pad_block(block, K, pad_left_hint=pad_left)

            elif mode == "causal_inclusive":
                # Comentario: exactamente lo que pides:
                # toma K columnas del pasado **incluyendo i**: [i-K+1, ..., i]
                i0 = max(0, i - (K - 1))
                i1 = i + 1  # exclusivo, incluye i
                block = S1[:, i0:i1]
                # Comentario: falta por la izquierda si estamos cerca del inicio
                have_left = i - i0  # columnas reales a la izquierda de i
                need_left = (K - 1) - have_left
                left_hint = max(0, need_left)
                block = pad_block(block, K, pad_left_hint=left_hint)


            elif mode == "forward_inclusive":
                # Comentario: futuro incluyendo i: [i, ..., i+K-1]
                i0 = i
                i1 = min(T, i + K)
                block = S1[:, i0:i1]
                # Comentario: falta por la derecha si estamos cerca del final
                left_hint = 0  # todo el déficit va a la derecha en este modo
                block = pad_block(block, K, pad_left_hint=left_hint)
            elif mode == 'last':
                # Comentario: ultimo tiempo de bloque = i, bloque de K columnas hacia atrás
                # bloque causal que termina en i, incluyendo i,
                # pero SOLO si está completo; si no, no inventa nada
                if i < (K-1):
                    continue

                i0 = i - (K-1)
                i1 = i   # exclusivo, incluye i
                block = S1[:, i0:i1]


            else:
                raise ValueError("mode must be 'center', 'causal_inclusive', 'forward_inclusive', or 'last'")

            A.append(block)
            times.append(time_vector[i] if time_vector is not None else i)

        A_out = np.stack(A, axis=0)  # (B, F, K)
        t_out = np.asarray(times)
        return A_out, t_out

    @staticmethod
    def compute_svd(A: np.ndarray, ensure_real: bool = True):
        """
        Compute Singular Value Decomposition (SVD) for 2D or 3D arrays.
        Performs SVD on a 2D matrix or applies SVD to each batch in a 3D array.
        For 3D inputs, the SVD is computed independently for each batch along
        the first dimension.
        Parameters
        ----------
        A : np.ndarray
            Input array for SVD computation. Must be 2D (matrix) or 3D (batch of matrices).
            - If 2D: shape (F, K) where F is number of features/rows and K is number of columns
            - If 3D: shape (B, F, K) where B is batch size, F is features/rows, K is columns
        ensure_real : bool, optional
            If True, converts complex numbers to real when imaginary parts are negligible.
            Default is True.
        Returns
        -------
        U : np.ndarray
            Left singular vectors.
            - If 2D input: shape (F, min(F, K))
            - If 3D input: shape (B, F, min(F, K))
        S : np.ndarray
            Singular values in descending order.
            - If 2D input: shape (min(F, K),)
            - If 3D input: shape (B, min(F, K))
        Vh : np.ndarray
            Right singular vectors (conjugate transpose of V).
            - If 2D input: shape (min(F, K), K)
            - If 3D input: shape (B, min(F, K), K)
        Raises
        ------
        ValueError
            If input array is not 2D or 3D.
        """

        # SVD in batches if A is 3D; if 2D, standard SVD
        if A.ndim == 2:
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
        elif A.ndim == 3:
            # Decompose by batch
            B = A.shape[0]
            F, K = A.shape[1], A.shape[2]
            U = np.zeros((B, F, min(F, K)), dtype=A.dtype)
            S = np.zeros((B, min(F, K)), dtype=A.dtype)
            Vh = np.zeros((B, min(F, K), K), dtype=A.dtype)
            for b in range(B):
                Ub, Sb, Vhb = np.linalg.svd(A[b], full_matrices=False)
                U[b], S[b], Vh[b] = Ub, Sb, Vhb
        else:
            raise ValueError("A must be 2D or 3D")

        if ensure_real:
            U = np.real_if_close(U)
            S = np.real_if_close(S)
            Vh = np.real_if_close(Vh)
        return U, S, Vh
