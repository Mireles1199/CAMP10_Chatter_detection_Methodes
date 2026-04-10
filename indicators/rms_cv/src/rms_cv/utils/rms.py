"""Windowed Root Mean Square (RMS) sequence computation.

Exposes a single public function, :func:`rms_sequence`, which partitions a
uni- or multi-channel time series into overlapping or non-overlapping frames
and computes the RMS value of every frame.  Both time-based and sample-based
parameterisation are supported, as well as optional edge padding, per-frame
mean detrending, and amplitude clipping before the power computation.

The heavy lifting is delegated to :func:`numpy.lib.stride_tricks.sliding_window_view`
(requires NumPy \u2265 1.20) so that no Python-level loop is needed.
"""

# Cálculo de secuencias RMS por ventanas deslizantes
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
    """Compute Root Mean Square (RMS) values over a sliding window of frames.

    Segments *signal* into consecutive frames and computes the RMS of every
    frame.  Parameterisation can be time-based (seconds) or sample-based;
    sample-based parameters always take priority.  The output dictionary
    always contains the ``"rms"`` array and may optionally include per-frame
    centre times and start/end sample indices.

    Args:
        signal (Union[np.ndarray, list]): Input signal.  Can be:

            * 1-D array/list of shape ``(T,)`` for a single channel.
            * 2-D array of shape ``(T, C)`` for *C* simultaneous channels
              (time on axis 0).
        fs (float): Sampling frequency [Hz].  Must be > 0.
        window_sec (Optional[float], optional): Window duration [s].  Required
            if *N* is not provided.  Must be > 0.
        step_sec (Optional[float], optional): Step between consecutive windows
            [s].  Must be > 0.  Ignored when *overlap_pct* or *hop* is given.
        overlap_pct (Optional[float], optional): Fractional overlap in
            ``[0, 1)``.  Takes priority over *step_sec*.
            Computed as ``step = window_duration \u00d7 (1 - overlap_pct)``.
        N (Optional[int], optional): Window length [samples].  Takes priority
            over *window_sec*.  Must be \u2265 1.
        hop (Optional[int], optional): Hop length [samples].  Takes priority
            over *step_sec* and *overlap_pct*.  Must be \u2265 1.
        detrend (bool, optional): If ``True``, subtract the per-frame mean
            before computing power.  Defaults to ``False``.
        bandpass (Optional[Tuple], optional): Bandpass filter parameters
            (reserved; not yet implemented).
        clip (Optional[Tuple[float, float]], optional): Amplitude clipping
            range ``(v_min, v_max)`` applied to every frame before RMS
            computation.  ``None`` disables clipping.
        return_times (bool, optional): Include centre-of-frame timestamps
            (``"times"`` key) in the output.  Defaults to ``True``.
        return_indices (bool, optional): Include start/end sample indices
            (``"indices"`` key) in the output.  Defaults to ``False``.
        pad_mode (str, optional): Edge-handling strategy.  One of:

            * ``"none"`` — no padding; may return fewer frames than expected.
            * ``"reflect"`` — reflect the signal at both boundaries.
            * ``"constant"`` — zero-pad at the right boundary only.

            Defaults to ``"none"``.
        **kwargs: Accepted typo aliases (``derend`` \u2192 *detrend*,
            ``bandpas`` \u2192 *bandpass*).  Unknown keys are silently ignored.

    Returns:
        Dict[str, Any]: Result dictionary with the following keys:

        * ``"rms"`` (*np.ndarray*) — RMS values, shape ``(F,)`` for 1-D
          input or ``(F, C)`` for 2-D input, where *F* is the frame count.
        * ``"pad_mode"`` (*str*) — the value of *pad_mode* that was used.
        * ``"times"`` (*np.ndarray*) — centre time of each frame [s], shape
          ``(F,)``.  Present only when *return_times* is ``True``.
        * ``"indices"`` (*np.ndarray*) — start and end sample index of each
          frame, shape ``(F, 2)``.  Present only when *return_indices* is
          ``True``.

    Raises:
        ValueError: If *signal* is not 1-D or 2-D; if ``fs <= 0``; if window
            parameters result in a zero-length window; if ``overlap_pct`` is
            outside ``[0, 1)``; or if *pad_mode* is not one of the accepted
            strings.
        NotImplementedError: If *bandpass* is provided (hook not yet
            implemented).
        RuntimeError: If NumPy < 1.20 (``sliding_window_view`` unavailable).

    Note:
        Sample-based parameters (*N*, *hop*) always take priority over
        their time-based counterparts (*window_sec*, *step_sec*,
        *overlap_pct*).  RMS computation internally casts to ``float64`` for
        numerical stability.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal(10_000)
        >>> out = rms_sequence(x, fs=1000.0, window_sec=0.1, overlap_pct=0.5)
        >>> out["rms"].shape
        (199,)
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
