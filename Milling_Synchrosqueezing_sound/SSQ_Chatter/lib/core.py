# Comentario: núcleo del pipeline: STFT/SSQ-STFT -> submatrices -> SVD -> criterio chatter
from typing import Tuple, Dict, Any
import numpy as np
from scipy.signal import get_window
from ssqueezepy import ssq_stft

from ..utils.tf_windows import extract_local_windows, compute_svd
from .detection import detectar_chatter_3sigma


def sqq_chatter(
    signal: np.ndarray,
    fs: float,
    win_length_ms: float,
    hop_ms: float,
    n_fft: int,
    sigma: float = 4,
    Ai_length: int = 4,
    frac_stable: float = 0.25,
    SSQ: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Run the full chatter detection pipeline with SSQ-STFT and SVD.

    Returns (in this exact order):
        Tsx, Sx, fs, t, A_i, D, d1, res
    """
    # Comentario: sanitizar entradas
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if win_length_ms <= 0 or hop_ms <= 0:
        raise ValueError("win_length_ms and hop_ms must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if Ai_length <= 0:
        raise ValueError("Ai_length must be > 0")
    if frac_stable <= 0.0 or frac_stable > 1.0:
        raise ValueError("frac_stable must be in (0, 1]")

    # Comentario: convertir milisegundos a muestras
    win_length = int(round(win_length_ms * 1e-3 * fs))
    hop_length = int(round(hop_ms * 1e-3 * fs))
    if win_length < 3:
        raise ValueError("win_length too small")
    if hop_length < 1:
        raise ValueError("hop_length too small")

    # Comentario: preparar ventana gaussiana (sigma en muestras)
    sigma_samples = win_length / sigma
    window = get_window(("gaussian", sigma_samples), win_length)

    # Comentario: SSQ-STFT (obtenemos Tsx y Sx)
    Tsx, Sx, _, _, _, _ = ssq_stft(
        x,
        window=window,
        n_fft=n_fft,
        win_len=win_length,
        hop_len=hop_length,
        fs=fs,
        get_dWx=True,
        get_w=True,
    )

    # Comentario: vector de tiempo para el dominio TF
    t = np.arange(Sx.shape[1]) * (hop_length / fs)

    # Comentario: extraer submatrices locales A_i a lo largo del tiempo
    if SSQ:
        A_i, t_i = extract_local_windows(Tsx, K=Ai_length, time_vector=t)
    else:
        A_i, t_i = extract_local_windows(Sx, K=Ai_length, time_vector=t)

    # Comentario: SVD en batch para todas las submatrices
    U, D, Vt = compute_svd(A_i, ensure_real=True)
    d1 = D[:, 0]  # Comentario: primer valor singular por ventana

    # Comentario: criterio de chatter (3σ con fallback MAD)
    res = detectar_chatter_3sigma(
        d1=d1,
        fraccion_estable=frac_stable,
        alpha=0.05,
        z=3.0,
    )

    # Comentario: devolver en el orden solicitado
    return Tsx, Sx, fs, t, A_i, t_i, D, d1, res
