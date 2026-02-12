
from __future__ import annotations
from typing import Tuple
import numpy as np
from ..utils.opr import segment_opr
from .entropy import entropy_from_segments, EntropyEstimator
from ..models.maxent import fit_maxent_gaussians, MaxEntModels

def offline_train_maxent_sprt(
    opr_free: np.ndarray,
    opr_chat: np.ndarray,
    opr_t_free: np.ndarray,
    opr_t_chat: np.ndarray,
    N_seg: int,
    estimator: EntropyEstimator | None = None,
) -> Tuple[MaxEntModels, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train Maximum Entropy SPRT models on offline OPR (Operating Point Range) data.
    This function segments operational data into windows, computes entropy metrics for each segment,
    and fits maximum entropy Gaussian models to distinguish between free and chatter states.
    Parameters
    ----------
    opr_free : np.ndarray
        Operating Point Range data for free (non-chatter) condition.
    opr_chat : np.ndarray
        Operating Point Range data for chatter condition.
    opr_t_free : np.ndarray
        Time indices or timestamps corresponding to opr_free data.
    opr_t_chat : np.ndarray
        Time indices or timestamps corresponding to opr_chat data.
    N_seg : int
        Number of segments to partition the OPR data into.
    estimator : EntropyEstimator | None, optional
        Entropy estimation method. If None, uses default estimator.
        Default is None.
    Returns
    -------
    Tuple[MaxEntModels, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - models : MaxEntModels
            Fitted maximum entropy Gaussian models for free (p0) and chatter (p1) conditions.
        - H_free : np.ndarray
            Computed entropy values for free condition segments.
        - H_chat : np.ndarray
            Computed entropy values for chatter condition segments.
        - t_mid_free : np.ndarray
            Midpoint timestamps for free condition segments.
        - t_mid_chat : np.ndarray
            Midpoint timestamps for chatter condition segments.
    Raises
    ------
    ValueError
        If insufficient segments are generated from the input data. Check N_seg and OPR data length.
    """


    # 1) OPR Segmentation
    segments_free, segments_t_free = segment_opr(opr_free, opr_t_free, N_seg=N_seg)
    segments_chat, segments_t_chat = segment_opr(opr_chat, opr_t_chat, N_seg=N_seg)

    if len(segments_free) == 0 or len(segments_chat) == 0:
        raise ValueError("Insufficient segments generated for training. Check N_seg and OPR data length.")

    # 2) Calculation of the indicator (entropy) per segment
    H_free = entropy_from_segments(segments_free, estimator=estimator)
    H_chat = entropy_from_segments(segments_chat, estimator=estimator)

    t_mid_free = np.array([np.mean(seg_t) for seg_t in segments_t_free])
    t_mid_chat = np.array([np.mean(seg_t) for seg_t in segments_t_chat])

    # 3) Fitting of pdfs p0(H) and p1(H)
    models = fit_maxent_gaussians(H_free, H_chat)

    return models, H_free, H_chat, t_mid_free, t_mid_chat
