"""Signal filtering helpers used inside analysis windows."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, detrend
from scipy.ndimage import uniform_filter1d


def compute_window_length(n: int) -> int:
    """Return an odd Savitzky–Golay window length suitable for *n* samples.

    Length is 5 % of *n*, rounded up to the nearest odd number, with a
    minimum of 7.
    """
    wl = int(n * 0.05)
    if wl % 2 == 0:
        wl += 1
    return max(wl, 7)


def savgol_filter_window(signal: np.ndarray, polyorder: int = 3) -> np.ndarray:
    """Apply a Savitzky–Golay filter with auto-computed window length.

    Parameters
    ----------
    signal   : 1-D input array.
    polyorder: polynomial order (default 3).

    Returns
    -------
    Filtered array of the same length as *signal*.
    """
    wl = compute_window_length(len(signal))
    return savgol_filter(signal, window_length=wl, polyorder=polyorder)


def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Uniform moving average via :func:`scipy.ndimage.uniform_filter1d`.

    Parameters
    ----------
    signal      : 1-D input array.
    window_size : filter width in samples (>= 1).
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    return uniform_filter1d(signal, size=window_size, mode="nearest")


def filter_window_signals(
    q: np.ndarray,
    q_o: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Savitzky–Golay pre-filtering to displacement *q* and velocity *q_o*.

    Returns
    -------
    (q_filtered, velocity_for_crossing)
    """
    q_filt = savgol_filter_window(q)
    q_o_filt = savgol_filter_window(q_o)
    return q_filt, q_o_filt
