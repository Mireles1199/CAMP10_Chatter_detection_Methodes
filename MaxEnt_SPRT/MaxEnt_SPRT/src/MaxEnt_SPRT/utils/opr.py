
from __future__ import annotations
from typing import Tuple, List
import numpy as np


def sample_opr(y: np.ndarray, t: np.ndarray, fs: float, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae datos once-per-revolution (1 muestra por revolución).

    ratio = fs/fr debe ser entero. Se toma el primer índice de cada revolución.
    """
    ratio = fs / fr
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("fs/fr debe ser entero para muestreo OPR exacto.")
    step = int(round(ratio))
    return y[::step], t[::step]

def segment_opr(opr: np.ndarray, opr_t: np.ndarray, N_seg: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Divide la señal OPR en segmentos consecutivos de longitud N_seg.
    Se descarta el sobrante final si no completa un segmento.
    """
    n_total = len(opr)
    n_segments = n_total // N_seg
    segments: List[np.ndarray] = []
    segments_t: List[np.ndarray] = []
    for k in range(n_segments):
        start = k * N_seg
        end = start + N_seg
        segments.append(opr[start:end])
        segments_t.append(opr_t[start:end])
    return segments, segments_t
