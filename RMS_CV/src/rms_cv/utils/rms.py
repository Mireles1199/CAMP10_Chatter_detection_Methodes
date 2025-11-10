# Comentario: cálculo de secuencias RMS por ventanas
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

def rms_sequence(
    signal: Union[np.ndarray, list],
    fs: float,
    *,
    # Ventaneo en tiempo
    window_sec: Optional[float] = None,
    step_sec: Optional[float] = None,
    overlap_pct: Optional[float] = None,  # si se da, prioridad sobre step_sec
    # Ventaneo en muestras (tiene prioridad sobre *_sec)
    N: Optional[int] = None,
    hop: Optional[int] = None,
    # Pretratamientos
    detrend: bool = False,
    bandpass: Optional[Tuple] = None,  # Hook no implementado
    clip: Optional[Tuple[float, float]] = None,  # (vmin, vmax)
    # Salidas / bordes
    return_times: bool = True,
    return_indices: bool = False,
    pad_mode: str = "none",  # "none" | "reflect" | "constant"
    # Aliases por errores de tecleo
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calcula la secuencia RMS por ventanas para monitoreo en línea.
    """
    # --- Corrección de tecleos
    if "derend" in kwargs:
        detrend = kwargs["derend"]
    if "bandpas" in kwargs:
        bandpass = kwargs["bandpas"]

    x = np.asarray(signal)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError("`signal` debe ser 1D (T,) o 2D (T, C) con tiempo en eje 0.")

    if not (isinstance(fs, (int, float)) and fs > 0):
        raise ValueError("`fs` debe ser > 0.")

    T, C = x.shape

    # --- Ventana en muestras
    if N is not None:
        if not (isinstance(N, (int, np.integer)) and N >= 1):
            raise ValueError("`N` debe ser entero >= 1.")
        win = int(N)
    else:
        if window_sec is None or window_sec <= 0:
            raise ValueError("Debe especificar `window_sec > 0` si no se da `N`.")
        win = int(round(window_sec * fs))
        if win < 1:
            raise ValueError("`window_sec * fs` < 1: ventana demasiado pequeña.")

    # --- Salto entre ventanas
    if hop is not None:
        if not (isinstance(hop, (int, np.integer)) and hop >= 1):
            raise ValueError("`hop` debe ser entero >= 1.")
        step = int(hop)
    else:
        if overlap_pct is not None:
            if not (0 <= overlap_pct < 1):
                raise ValueError("`overlap_pct` debe estar en [0, 1).")
            step_sec_eff = (1.0 - overlap_pct) * (win / fs if N is not None else window_sec)
            step = max(1, int(round(step_sec_eff * fs)))
        elif step_sec is not None:
            if step_sec <= 0:
                raise ValueError("`step_sec` debe ser > 0.")
            step = max(1, int(round(step_sec * fs)))
        else:
            step = win

    if bandpass is not None:
        raise NotImplementedError("`bandpass` es un hook, no implementado aquí.")

    pad_mode = str(pad_mode).lower()
    if pad_mode not in ("none", "reflect", "constant"):
        raise ValueError("`pad_mode` debe ser 'none', 'reflect' o 'constant'.")

    if T < win:
        if pad_mode == "none":
            return {"rms": np.empty((0, C)), "pad_mode": pad_mode}
        else:
            pad_needed = win - T
    else:
        if pad_mode == "none":
            pad_needed = 0
        else:
            frames = int(np.ceil((T - win) / step)) + 1
            total_needed = (frames - 1) * step + win
            pad_needed = max(0, total_needed - T)

    if pad_needed > 0:
        if pad_mode == "reflect":
            x = np.pad(x, ((0, pad_needed), (0, 0)), mode="reflect")
        elif pad_mode == "constant":
            x = np.pad(x, ((0, pad_needed), (0, 0)), mode="constant", constant_values=0)

    T_eff = x.shape[0]

    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except Exception as e:
        raise RuntimeError("Se requiere NumPy >= 1.20 para sliding_window_view.") from e

    if pad_mode == "none":
        starts = np.arange(0, max(0, T - win + 1), step, dtype=int)
    else:
        starts = np.arange(0, T_eff - win + 1, step, dtype=int)

    if starts.size == 0:
        out = {"rms": np.empty((0, C)), "pad_mode": pad_mode}
        if return_times:
            out["times"] = np.empty((0,), dtype=float)
        if return_indices:
            out["indices"] = np.empty((0, 2), dtype=int)
        return out

    sw = sliding_window_view(x, window_shape=win, axis=0)  # (T_eff - win + 1, win, C)
    windows = sw[starts]  # (F, win, C)

    if detrend:
        mean_win = windows.mean(axis=1, keepdims=True)
        windows = windows - mean_win

    if clip is not None:
        vmin, vmax = clip
        windows = np.clip(windows, vmin, vmax)

    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))  # (F, C)

    if np.asarray(signal).ndim == 1:
        rms = rms[:, 0]

    results: Dict[str, Any] = {"rms": rms, "pad_mode": pad_mode}

    if return_indices:
        idx = np.stack([starts, starts + win], axis=1)
        results["indices"] = idx

    if return_times:
        centers = starts + (win / 2.0)
        times = centers / fs
        results["times"] = times

    return results
