from __future__ import annotations
import numpy as np
from typing import Tuple

def select_imf_near_band(imfs: np.ndarray , fs: float, band: Tuple[float, float]) -> int:
    """"Selecciona el índice del IMF con mayor energía en una banda de frecuencia.

    Args:
        imfs (np.ndarray): Matriz (K, N) de IMFs (cada fila es un IMF).
        fs (float): Frecuencia de muestreo (Hz).
        band (Tuple[float, float]): Banda objetivo (lo, hi) en Hz.

    Returns:
        int: Índice k del IMF con mayor energía espectral dentro de la banda.

    Raises:
        ValueError: Si `imfs` no es 2D.

    Notas:
        - Usa ventana Hann y FFT de tamaño `n` (siguiente potencia de 2 >= N) para estimar energía.
        - La energía en banda se calcula como suma de |X|^2 con máscara [lo, hi].
    """
    lo, hi = band
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
    K, N = imfs.shape
    n: int = int(1 << (N - 1).bit_length())
    freqs: np.ndarray  = np.fft.rfftfreq(n, d=1.0/fs)
    mask: np.ndarray  = (freqs >= lo) & (freqs <= hi)
    best_k: int = 0
    best_e: float = -np.inf
    win: np.ndarray  = np.hanning(N)
    for k in range(K):
        x = imfs[k]
        X: np.ndarray  = np.fft.rfft(x * win, n=n)
        E: float = float(np.sum(np.abs(X[mask])**2))
        if E > best_e:
            best_e, best_k = E, k
    return best_k
