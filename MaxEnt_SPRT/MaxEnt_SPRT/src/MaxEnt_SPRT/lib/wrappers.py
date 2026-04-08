
from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
from ..models.maxent import MaxEntModels
from .llr import GaussianIndicatorLLR
from .sprt import SPRTConfig
from .detector import SequentialProbabilityRatioTest
from ..utils.validation import validate_alpha_beta
from ..utils.opr import sample_opr, segment_opr
from .entropy import entropy_from_segments, EntropyEstimator

@validate_alpha_beta
def sprt_detect_sequence(
    H_seq: Iterable[float],
    models: MaxEntModels,
    alpha: float = 0.01,
    beta: float = 0.01,
    reset_on_H0: bool = True,
) -> Tuple[str, int, np.ndarray, float, float]:
    """
    Functional convenience wrapper around the object-oriented SPRT engine.

    This helper is preserved for quick scripts and notebook use, but internally
    it delegates to ``SequentialProbabilityRatioTest`` and
    ``GaussianIndicatorLLR``.

    :param H_seq: Ordered indicator sequence, typically one entropy value per segment.
    :param models: Pair of Gaussian models used to score the sequence.
    :param alpha: Target false-alarm probability.
    :param beta: Target missed-detection probability.
    :param reset_on_H0: Whether to reset the cumulative SPRT statistic after a lower-threshold crossing.

    Returns:
        Tuple[str, int, np.ndarray, float, float]: Final state, decision index,
        cumulative statistic history, lower threshold, and upper threshold.
    """
    config = SPRTConfig(alpha=alpha, beta=beta, reset_on_H0=reset_on_H0)
    llr_model = GaussianIndicatorLLR(models=models)
    sprt = SequentialProbabilityRatioTest(llr_model=llr_model, config=config)
    result = sprt.run(H_seq=H_seq)

    return (
        result.final_state,
        result.decision_index,
        result.S_history,
        result.a,
        result.b,
    )

@validate_alpha_beta
def online_maxent_sprt_from_signal(
    y_online: np.ndarray,
    t_online: np.ndarray,
    rpm: float,
    ratio_sampling: float,
    N_seg: int,
    models: MaxEntModels,
    alpha: float = 0.01,
    beta: float = 0.01,
    reset_on_H0: bool = True,
    estimator: EntropyEstimator | None = None,
) -> Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Complete functional online MaxEnt-SPRT pipeline for a new signal.

    The routine performs OPR resampling, fixed-length segmentation, entropy
    extraction, and finally sequential decision-making on the resulting feature
    sequence.

    :param y_online: Raw signal samples to classify.
    :param t_online: Time vector aligned with ``y_online``.
    :param rpm: Spindle speed in revolutions per minute.
    :param ratio_sampling: Number of OPR samples per revolution.
    :param N_seg: Number of OPR samples grouped into each entropy segment.
    :param models: Previously trained Gaussian pair used by the LLR and SPRT stages.
    :param alpha: Target false-alarm probability.
    :param beta: Target missed-detection probability.
    :param reset_on_H0: Whether to reset the SPRT statistic after a stable decision.
    :param estimator: Optional custom estimator used to convert segments into entropy values.

    Returns:
        Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, float, float]: Final
        state, decision index, entropy sequence, cumulative SPRT history,
        segment midpoint times, lower threshold, and upper threshold.
    """
    fr = rpm / 60.0      # Hz, frecuencia de rotación
    fs = ratio_sampling * fr  # Hz, frecuencia de muestreo

    # 1) OPR
    opr_online, opr_t_online = sample_opr(y_online, t_online, fs=fs, fr=fr)

    # 2) Segmentación
    segments_online, segments_t_online = segment_opr(opr_online, opr_t_online, N_seg=N_seg)
    if len(segments_online) == 0:
        raise ValueError("No hay segmentos suficientes en la señal online.")

    # 3) Entropía por segmento
    H_seq = entropy_from_segments(segments_online, estimator=estimator)
    t_mid_segments = np.array([np.mean(seg_t) for seg_t in segments_t_online])

    # 4) SPRT (usando wrapper funcional)
    final_state, decision_index, S_hist, a, b = sprt_detect_sequence(
        H_seq=H_seq,
        models=models,
        alpha=alpha,
        beta=beta,
        reset_on_H0=reset_on_H0,
    )

    return final_state, decision_index, H_seq, S_hist, t_mid_segments, a, b