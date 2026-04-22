
from __future__ import annotations
import math
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
    # if abs(ratio - round(ratio)) > 1e-9:
    #     raise ValueError("fs/fr must be an integer for exact OPR sampling.")
    step = int(math.ceil(ratio))
    return y[step::step], t[step::step]

def segment_opr(opr: np.ndarray, opr_t: np.ndarray, N_seg: int, step: int | None = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split OPR data into fixed-length segments with optional overlap.

    :param opr: One-dimensional OPR-resampled signal to partition.
    :param opr_t: Time vector aligned with ``opr``.
    :param N_seg: Number of OPR samples per segment.
    :param step: Hop size in OPR samples between consecutive segment starts.
        ``None`` (default) is equivalent to ``step = N_seg`` (no overlap).
        Values smaller than ``N_seg`` produce overlapping segments.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: List of OPR segments and list
        of matching time segments, both preserving the original order.

    Notes:
        Trailing samples that do not complete a full segment are discarded.
    """

    if step is None:
        step = N_seg
    segments: List[np.ndarray] = []
    segments_t: List[np.ndarray] = []
    start = 0
    while start + N_seg <= len(opr):
        segments.append(opr[start:start + N_seg])
        segments_t.append(opr_t[start:start + N_seg])
        start += step
    return segments, segments_t


def segment_signal_raw(
    y: np.ndarray,
    t: np.ndarray,
    N_samples_per_seg: int,
    step: int | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split a raw (non-decimated) signal into fixed-length blocks.

    Unlike :func:`segment_opr`, this function operates directly on the
    full-rate signal without prior OPR decimation.  It is the natural
    companion to :func:`segment_opr` for the ``segmentation="raw"`` mode.

    :param y: One-dimensional signal samples.
    :param t: Time vector aligned with ``y``.
    :param N_samples_per_seg: Number of raw samples per block.
    :param step: Hop size in raw samples between consecutive block starts.
        ``None`` (default) is equivalent to ``step = N_samples_per_seg``
        (no overlap).  Values smaller than ``N_samples_per_seg`` produce
        overlapping blocks.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: List of signal blocks and
        list of matching time blocks, both preserving the original order.

    Notes:
        Trailing samples that do not complete a full block are discarded.
    """
    if step is None:
        step = N_samples_per_seg
    segments: List[np.ndarray] = []
    segments_t: List[np.ndarray] = []
    start = 0
    while start + N_samples_per_seg <= len(y):
        segments.append(y[start:start + N_samples_per_seg])
        segments_t.append(t[start:start + N_samples_per_seg])
        start += step
    return segments, segments_t
