
from __future__ import annotations
from typing import Tuple, List
import numpy as np


def sample_opr(y: np.ndarray, t: np.ndarray, fs: float, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample signal and time arrays using OPR (Optimal Pacing Resampling) method.
    Performs uniform downsampling of a signal and its corresponding time array by a
    constant integer factor. The sampling factor is calculated as the ratio between
    the original sampling frequency and the target resampling frequency.
    Args:
        y (np.ndarray): Signal array to be downsampled.
        t (np.ndarray): Time array corresponding to the signal, to be downsampled in sync.
        fs (float): Original sampling frequency in Hz.
        fr (float): Target resampling frequency in Hz.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Downsampled signal array
            - Downsampled time array
    Raises:
        ValueError: If fs/fr is not an integer (within tolerance of 1e-9),
                   as exact OPR sampling requires an integer downsampling factor.
    """

    ratio = fs / fr
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("fs/fr must be an integer for exact OPR sampling.")
    step = int(round(ratio))
    return y[::step], t[::step]

def segment_opr(opr: np.ndarray, opr_t: np.ndarray, N_seg: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Segment OPR (Operating Point Representation) data and corresponding time arrays into fixed-size chunks.
    Parameters
    ----------
    opr : np.ndarray
        Array containing OPR (Operating Point Representation) values to be segmented.
    opr_t : np.ndarray
        Array containing time values corresponding to the OPR data points.
    N_seg : int
        Size of each segment. The total number of segments will be len(opr) // N_seg.
    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        A tuple containing:
        - segments: List of OPR data segments, each of size N_seg
        - segments_t: List of corresponding time segments, each of size N_seg
    Notes
    -----
    If the total length of opr is not evenly divisible by N_seg, the remaining data points
    will be discarded.
    Examples
    --------
    >>> opr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> opr_t = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> segments, segments_t = segment_opr(opr, opr_t, N_seg=3)
    >>> len(segments)
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
