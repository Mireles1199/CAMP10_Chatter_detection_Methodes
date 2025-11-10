from __future__ import annotations
# Comentario: extracción de subventanas y SVD (SRP: responsabilidad única)
from typing import Tuple, Optional, Literal
import numpy as np

class WindowExtractor:
    # Comentario: clase utilitaria estática para extraer ventanas y SVD

    @staticmethod
    def extract_local_windows(
        S1: np.ndarray,
        K: int,
        time_vector: Optional[np.ndarray] = None,
        mode: Literal["center", "causal_inclusive", "forward_inclusive"] = "causal_inclusive",
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Comentario: validaciones
        if not isinstance(S1, np.ndarray):
            raise TypeError("S1 must be a NumPy array")
        if S1.ndim != 2:
            raise ValueError("S1 must be 2D (freq x time)")
        if K <= 0:
            raise ValueError("K must be > 0")

        F, T = S1.shape
        A = []
        times = []

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

            else:
                raise ValueError("mode must be 'center', 'causal_inclusive', or 'forward_inclusive'")

            A.append(block)
            times.append(time_vector[i] if time_vector is not None else i)

        A_out = np.stack(A, axis=0)  # (B, F, K)
        t_out = np.asarray(times)
        return A_out, t_out

    @staticmethod
    def compute_svd(A: np.ndarray, ensure_real: bool = True):
        # Comentario: SVD por lotes si A es 3D; si es 2D, SVD estándar
        if A.ndim == 2:
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
        elif A.ndim == 3:
            # Comentario: descompone por batch B
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
