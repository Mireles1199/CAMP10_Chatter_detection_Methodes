from __future__ import annotations
import numpy as np
from typing import Dict, Literal, Optional, Tuple, Any

def band_counts_over_time(
    A: np.ndarray ,
    f_inst: np.ndarray ,
    band: Tuple[float, float] = (450.0, 600.0),
    energy_mode: Literal["A2", "A"] = "A2",
    thr_mode: Optional[Literal["mad", "percentile", "none"]] = "mad",
    thr_k_mad: float = 3.0,
    thr_percentile: float = 95.0,
    count_win_samples: int = 100,
    count_step_samples: Optional[int] = None,
) -> Tuple[np.ndarray , np.ndarray , Dict[str, Any]]:
    """Cuenta, por ventana, muestras que caen en banda y (opcional) superan un umbral de energía.

    Args:
        A (np.ndarray): Amplitud instantánea A(t).
        f_inst (np.ndarray): Frecuencia instantánea f_inst(t) en Hz.
        band (Tuple[float, float]): Banda (lo, hi) de interés en Hz.
        energy_mode (Literal["A2","A"]): Modo para energía/umbral (A^2 o A).
        thr_mode (Optional[Literal["mad","percentile","none"]]): Estrategia de umbral; None/'none' desactiva.
        thr_k_mad (float): Multiplicador del MAD cuando `thr_mode="mad"`.
        thr_percentile (float): Percentil cuando `thr_mode="percentile"`.
        count_win_samples (int): Tamaño de ventana (muestras) para el conteo.
        count_step_samples (Optional[int]): Paso entre ventanas; por defecto igual a la ventana.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: (counts, centers, debug_meta).

    Detalles:
        - Si `thr_mode` es None/'none', se cuenta solo pertenencia a banda; si no,
          además se exige energía > umbral.
        - `centers` contiene el índice de muestra del centro de cada ventana.
    """
    if count_step_samples is None:
        count_step_samples = count_win_samples

    N: int = len(A)
    lo, hi = band
    E: np.ndarray  = A**2 if energy_mode == "A2" else A

    thr: Optional[float]
    if thr_mode in ("mad", "percentile"):
        if thr_mode == "mad":
            med: float = float(np.median(E))
            mad: float = float(np.median(np.abs(E - med))) + 1e-12
            thr = med + thr_k_mad * mad
        else:
            thr = float(np.percentile(E, thr_percentile))
    elif thr_mode in ("none", None):
        thr = None
    else:
        raise ValueError("thr_mode debe ser 'mad', 'percentile' o 'none'")

    counts: list[int] = []
    centers: list[int] = []
    for start in range(0, N - count_win_samples + 1, count_step_samples):
        sl = slice(start, start + count_win_samples)
        mask_band: np.ndarray  = (f_inst[sl] >= lo) & (f_inst[sl] <= hi)

        if thr is None:
            hits: int = int(np.count_nonzero(mask_band))
        else:
            mask_energy: np.ndarray  = (E[sl] > thr)
            hits = int(np.count_nonzero(mask_band & mask_energy))

        counts.append(hits)
        centers.append(start + count_win_samples // 2)

    debug: Dict[str, Any] = {"thr": thr, "thr_mode": thr_mode, "band": band, "energy_mode": energy_mode}
    return np.asarray(counts), np.asarray(centers), debug

