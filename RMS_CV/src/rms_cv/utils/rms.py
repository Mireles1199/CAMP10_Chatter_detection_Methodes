# Comentario: cálculo de secuencias RMS por ventanas
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np

def rms_sequence(
    signal: Union[np.ndarray, list],
    fs: float,
    *,
    # Time windowing
    window_sec: Optional[float] = None,
    step_sec: Optional[float] = None,
    overlap_pct: Optional[float] = None,  # if given, takes priority over step_sec
    # Sample-based windowing (takes priority over *_sec)
    N: Optional[int] = None,
    hop: Optional[int] = None,
    # Preprocessing
    detrend: bool = False,
    bandpass: Optional[Tuple] = None,  # Hook not implemented
    clip: Optional[Tuple[float, float]] = None,  # (vmin, vmax)
    # Outputs / edges
    return_times: bool = True,
    return_indices: bool = False,
    pad_mode: str = "none",  # "none" | "reflect" | "constant"
    # Aliases for typos
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compute Root Mean Square (RMS) values over a sequence of windowed frames.
    This function segments a signal into overlapping or non-overlapping windows and
    computes the RMS value for each window across all channels.
    Parameters
    ----------
    signal : Union[np.ndarray, list]
        Input signal. Can be 1D with shape (T,) for single channel or 2D with
        shape (T, C) for multi-channel, where T is the number of samples and
        C is the number of channels.
    fs : float
        Sampling frequency in Hz. Must be > 0.
    window_sec : Optional[float], default=None
        Window duration in seconds. Required if `N` is not provided. Must be > 0.
    step_sec : Optional[float], default=None
        Step/hop duration in seconds between consecutive windows. Must be > 0.
        Ignored if `overlap_pct` or `hop` is provided.
    overlap_pct : Optional[float], default=None
        Overlap percentage in range [0, 1). Takes priority over `step_sec`.
        Computed as: step = window_duration * (1 - overlap_pct).
    N : Optional[int], default=None
        Window length in samples. Takes priority over `window_sec`. Must be >= 1.
    hop : Optional[int], default=None
        Hop length in samples. Takes priority over `step_sec` and `overlap_pct`.
        Must be >= 1.
    detrend : bool, default=False
        If True, removes the mean from each window before RMS computation.
    bandpass : Optional[Tuple], default=None
        Bandpass filter parameters (not implemented; reserved for future use).
    clip : Optional[Tuple[float, float]], default=None
        Clipping range (vmin, vmax). If provided, all samples are clipped to
        this range before RMS computation.
    return_times : bool, default=True
        If True, returns the center time of each window in seconds.
    return_indices : bool, default=False
        If True, returns the start and end sample indices of each window.
    pad_mode : str, default="none"
        Padding mode for handling edges. Options:
        - "none": No padding; may return fewer frames
        - "reflect": Reflect padding at signal boundaries
        - "constant": Zero-padding at signal boundaries
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "rms" : np.ndarray
            RMS values with shape (F,) for 1D input or (F, C) for 2D input,
            where F is the number of frames.
        - "pad_mode" : str
            The padding mode used.
        - "times" : np.ndarray (if return_times=True)
            Center time of each frame in seconds, shape (F,).
        - "indices" : np.ndarray (if return_indices=True)
            Start and end sample indices for each frame, shape (F, 2).
    Raises
    ------
    ValueError
        If signal dimensions are invalid, fs <= 0, window parameters are invalid,
        or overlap_pct is not in [0, 1).
    NotImplementedError
        If `bandpass` parameter is provided (hook not yet implemented).
    RuntimeError
        If NumPy version < 1.20 (sliding_window_view not available).
    Notes
    -----
    - Sample-based parameters (N, hop) take priority over time-based parameters
      (window_sec, step_sec, overlap_pct).
    - Typos in keyword arguments are automatically corrected ("derend" → detrend,
      "bandpas" → bandpass).
    - RMS computation uses float64 internally for numerical stability.
    """

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

    # --- Window in samples
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

    # --- Hop length in samples
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
