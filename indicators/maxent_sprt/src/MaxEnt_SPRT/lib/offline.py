
from __future__ import annotations
from typing import Tuple
import numpy as np
from ..utils.opr import segment_opr, segment_signal_raw
from .entropy import entropy_from_segments, EntropyEstimator
from ..models.maxent import fit_maxent_gaussians, MaxEntModels

def offline_train_maxent_sprt(
    opr_free: np.ndarray,
    opr_chat: np.ndarray,
    opr_t_free: np.ndarray,
    opr_t_chat: np.ndarray,
    N_seg: int,
    estimator: EntropyEstimator | None = None,
    step: int | None = None,
    segmentation: str = "opr",
    N_samples_per_seg: int | None = None,
) -> Tuple[MaxEntModels, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train Maximum Entropy SPRT models on offline OPR (Operating Point Range) data.

    This function segments operational data into windows, computes entropy metrics for each segment,
    and fits maximum entropy Gaussian models to distinguish between free and chatter states.

    :param opr_free: OPR signal representing the stable reference condition.
    :param opr_chat: OPR signal representing the chatter reference condition.
    :param opr_t_free: Time vector aligned with ``opr_free``.
    :param opr_t_chat: Time vector aligned with ``opr_chat``.
    :param N_seg: Number of OPR samples grouped into each training segment (``segmentation="opr"``).
    :param estimator: Segment-to-entropy estimator. If ``None``, the default entropy estimator is used.
    :param step: Hop size in domain samples between consecutive segment starts.
        ``None`` (default) is equivalent to ``step = N_seg`` (no overlap).
    :param segmentation: ``"opr"`` (default) – use OPR-decimated arrays with
        :func:`segment_opr`; ``"raw"`` – use raw signal arrays directly with
        :func:`segment_signal_raw` (no prior OPR decimation needed).
    :param N_samples_per_seg: Length of each raw block in samples.  Required
        when ``segmentation="raw"``; ignored otherwise.

    Returns:
        Tuple[MaxEntModels, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Trained Gaussian models, entropy sequence for stable segments, entropy
        sequence for chatter segments, and the midpoint timestamps for both
        segment sets.

    Raises
    ------
    ValueError
        If insufficient segments are generated from the input data. Check N_seg and OPR data length.
    """


    # 1) Segmentation  (OPR or raw)
    if segmentation == "raw":
        if N_samples_per_seg is None:
            raise ValueError(
                "N_samples_per_seg must be provided when segmentation='raw'."
            )
        segments_free, segments_t_free = segment_signal_raw(
            opr_free, opr_t_free, N_samples_per_seg=N_samples_per_seg, step=step
        )
        segments_chat, segments_t_chat = segment_signal_raw(
            opr_chat, opr_t_chat, N_samples_per_seg=N_samples_per_seg, step=step
        )
    else:  # "opr" (default)
        segments_free, segments_t_free = segment_opr(opr_free, opr_t_free, N_seg=N_seg, step=step)
        segments_chat, segments_t_chat = segment_opr(opr_chat, opr_t_chat, N_seg=N_seg, step=step)

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
