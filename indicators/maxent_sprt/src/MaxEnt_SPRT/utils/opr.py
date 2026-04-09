
from __future__ import annotations
from typing import Tuple, List
import numpy as np


def sample_opr(y: np.ndarray, t: np.ndarray, fs: float, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform exact once-per-revolution downsampling by integer decimation.

    The function assumes the desired OPR resampling can be achieved through an
    integer ratio ``fs / fr``. Under that assumption, it selects every ``step``
    sample from the signal and time arrays, preserving time alignment.

    :param y: One-dimensional signal samples to be resampled.
    :param t: Time vector aligned sample-by-sample with ``y``.
    :param fs: Original sampling frequency in Hz of the input signal.
    :param fr: Target once-per-revolution sampling frequency in Hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled signal array and its aligned
        resampled time vector.

    Raises:
        ValueError: If ``fs / fr`` is not an integer within numerical tolerance,
        because exact OPR decimation would not be possible.
    """

    ratio = fs / fr
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("fs/fr must be an integer for exact OPR sampling.")
    step = int(round(ratio))
    return y[::step], t[::step]

def segment_opr(opr: np.ndarray, opr_t: np.ndarray, N_seg: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split OPR data into fixed-length segments.

    :param opr: One-dimensional OPR-resampled signal to partition.
    :param opr_t: Time vector aligned with ``opr``.
    :param N_seg: Number of OPR samples per segment.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: List of OPR segments and list
        of matching time segments, both preserving the original order.

    Notes:
        If ``len(opr)`` is not divisible by ``N_seg``, trailing samples that do
        not complete a full segment are discarded.
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
