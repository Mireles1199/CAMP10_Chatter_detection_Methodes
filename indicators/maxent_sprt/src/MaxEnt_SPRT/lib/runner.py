from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Sequence, Optional, Tuple

from collections import defaultdict
from ..utils.types import SignalData, IndicatorResult
from ..lib.detector import MaxEntSPRTConfig, MaxEntSPRTDetector
from ..lib.entropy import GaussianMaxEntEstimator, EmpiricalHistogramEntropyEstimator, entropy_from_segments
from ..utils.opr import sample_opr

import numpy as np

IndicatorFunc = Callable[..., IndicatorResult]
logger = logging.getLogger(__name__)

def run_maxent_sprt(signal: SignalData, INDICATOR_CONFIG: dict ) -> IndicatorResult:
    """
    Execute the Maximum Entropy Sequential Probability Ratio Test (MaxEnt SPRT) indicator.

    This function serves as a wrapper that retrieves the appropriate analysis function
    from the configuration and executes it with the provided signal data and parameters.

    :param signal: Input signal bundle containing the analysis array, aligned time axis, sampling frequency, and optional metadata.
    :param INDICATOR_CONFIG: Dispatcher dictionary containing the selected function under ``func`` and its keyword arguments under ``params``.

    Returns:
        IndicatorResult: Result object returned by the selected indicator
        function.

    Raises
    ------
    KeyError
        If required keys are missing from ``INDICATOR_CONFIG``.
    TypeError
        If the selected function is not callable or the signal is invalid.
    """

    results: IndicatorResult = None

    func: IndicatorFunc = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = _maxent_sprt_pipeline

    params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})

    results = func(signal, **params)

    return results

def _cut_signal( t,x , time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restrict a time series to a closed time interval.

    :param t: Time vector of the signal.
    :param x: Signal values aligned with ``t``.
    :param time_range: Start and end times delimiting the interval to keep.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Time and signal arrays restricted to the
        requested interval.
    """
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]

def _maxent_sprt_pipeline(
    signal: SignalData,
    rpm: float,
    N_seg: int,
    t_stable_total: float,
    alpha: float,
    beta: float,
    reset_on_H0: bool,
    ratio_sampling: Optional[float] = None,
    cut_start_time: Optional[float] = None,
    cut_end_time: Optional[float] = None,

    ) -> IndicatorResult:
    """
    Execute the Maximum Entropy Sequential Probability Ratio Test (MaxEnt SPRT) pipeline for chatter detection.

    This function performs a complete chatter detection workflow consisting of offline training and online detection phases.
    It processes a signal into stable (chatter-free) and chatter-included segments, trains a Gaussian maximum entropy
    estimator on sampled Operating Point Response (OPR) data, and applies SPRT for online chatter detection.

    :param signal: Full input bundle containing the time axis, analysis signal, sampling frequency, and any source metadata.
    :param rpm: Spindle speed in revolutions per minute.
    :param N_seg: Number of revolutions or OPR samples grouped into one analysis segment.
    :param t_stable_total: Duration in seconds from the beginning of the record assumed to be chatter-free and used for stable-state training.
    :param alpha: Target false-alarm probability used to derive SPRT thresholds.
    :param beta: Target missed-detection probability used to derive SPRT thresholds.
    :param reset_on_H0: Whether the cumulative SPRT statistic is reset after accepting the stable hypothesis.
    :param ratio_sampling: Optional OPR sampling ratio used during online detection.
    :param cut_start_time: Optional lower time bound applied before splitting the signal into stable and chatter portions.
    :param cut_end_time: Optional upper time bound applied before splitting the signal into stable and chatter portions.

    Returns:
        IndicatorResult: Result object with segment timestamps, SPRT statistic
        history, detected chatter times, and rich metadata containing the
        intermediate models, signals, detector state, and derived quantities.

    Notes
    -----
    The pipeline consists of three main phases:

    1. Signal Preparation: Splits the input signal into stable and chatter-included portions.
    2. Offline Training: Samples OPR from both signal portions and trains a Gaussian MaxEnt estimator.
    3. Online Detection: Applies SPRT on the entire signal in segments and identifies chatter points.

    Chatter points are identified where the SPRT statistic S exceeds the detection threshold (b).
    """

    t_analysis = signal.t_analysis
    signal_analysis = signal.signal_analysis
    fs = signal.fs

    fr: float = rpm / 60.0       # Hz, frequency of rotation
    t_total = t_analysis[-1]-t_analysis[0]

    t_stable_total = t_stable_total  # seconds to consider stable
    t_chatter_total = t_analysis[-1] - t_stable_total

    t_stable, signal_analysis_stable = _cut_signal( t_analysis, signal_analysis , (cut_start_time, t_stable_total) )
    t_chatter, signal_analysis_chatter = _cut_signal( t_analysis, signal_analysis , (t_stable_total, cut_end_time) )

    logger.info_plus("Signal loaded:")
    logger.info_plus(f" - Samples: {signal_analysis.size}")
    logger.info_plus(f" - Duration: {t_analysis[-1]-t_analysis[0]:.2f} s")
    logger.info_plus(f" - Sampling freq.: {fs:.1f} Hz")
    logger.info_plus(f" - Rotation freq.: {fr:.1f} Hz")
    logger.info_plus(f" - Segments of {N_seg} revolutions: {N_seg/fr:.2f} s each")
    logger.info_plus(f" - Total segments available: {int(t_total*fr/N_seg)}")

    logger.info_plus("Generated chatter-free and chatter-included signals.")
    logger.info_plus(f"Size of signal free: {signal_analysis_stable.size} samples.")
    logger.info_plus(f"Size of signal chatter: {signal_analysis_chatter.size} samples.")


    # =========== Fase Ofline : OPR Training ==========
    opr_free, t_opr_free = sample_opr(signal_analysis_stable, t_stable, fs=fs, fr=fr)
    opr_chat, t_opr_chat = sample_opr(signal_analysis_chatter, t_chatter, fs=fs, fr=fr)
    logger.info_plus(f"\n Sampled OPR: {opr_free.size} samples free, {opr_chat.size} samples chatter.")

    # ============ Offline Phase:END-TO-END GAUSSIAN ===========
    detector_cfg = MaxEntSPRTConfig(alpha=alpha, beta=beta, reset_on_H0=reset_on_H0)
    gaussian_estimator = GaussianMaxEntEstimator()
    detector = MaxEntSPRTDetector(config=detector_cfg, estimator=gaussian_estimator)

    # Offline phase: OPR Training
    detector.fit_offline_from_opr(
        opr_free=opr_free,
        opr_t_free=t_opr_free,
        opr_chat=opr_chat,
        opr_t_chat=t_opr_chat,
        N_seg=N_seg,
    )

    models_trained = detector._check_models()
    logger.info_plus("\n OFFLINE MODEL (Gaussian MaxEnt):")
    logger.info_plus(f"  FREE:  mu0={models_trained.p0.mu:.5f}, sigma0={models_trained.p0.sigma:.5f}")
    logger.info_plus(f"  CHAT:  mu1={models_trained.p1.mu:.5f}, sigma1={models_trained.p1.sigma:.5f}")

    sprt_result, H_seq_online, t_mid_segments = detector.detect_online_from_signal(
        y_online=signal_analysis,
        t_online=t_analysis,
        rpm=rpm,
        ratio_sampling=ratio_sampling,
        N_seg=N_seg,
        fs=fs,
    )

    #%%
    # ============ Online Phase: Results visualization ===========

    logger.info_plus(f"ONLINE FINAL STATE: {sprt_result.final_state}, decision at segment {sprt_result.decision_index}")

    # =========== Early chatter Results - Points Chatter ==========
    mask = np.where(sprt_result.S_history >= sprt_result.b)[0]
    chatter_points_time = t_mid_segments[mask] if mask.size > 0 else np.array([])
    chatter_points_values = sprt_result.S_history[mask] if mask.size > 0 else np.array([])

    result = IndicatorResult(
        name="MaxEnt_SPRT",
        t=t_mid_segments,
        I_t=sprt_result.S_history,
        t_d=chatter_points_time,
        meta={
            "Samples": signal_analysis.size,
            "Duration": t_total,
            "fs": fs,
            "Rotational_Frequency_Hz": fr,
            "N_seg": N_seg,
            "alpha": alpha,
            "beta": beta,
            "rpm": rpm,
            "ratio_sampling": ratio_sampling,
            "Total_segments": int(t_total*fr/N_seg),
            "Size_signal_free": signal_analysis_stable.size,
            "Size_signal_chatter": signal_analysis_chatter.size,
            "Sampled OPR free": opr_free.size,
            "Sampled OPR chatter": opr_chat.size,
            "P0_mu": models_trained.p0.mu,
            "P0_sigma": models_trained.p0.sigma,
            "P1_mu": models_trained.p1.mu,
            "P1_sigma": models_trained.p1.sigma,
            "detector": detector,
            "gaussian_estimator": gaussian_estimator,
            "sprt_result": sprt_result,
            "models_trained": models_trained,
            "H_seq_online": H_seq_online,
            "chatter_points_values": chatter_points_values,
            "t_stable": t_stable,
            "signal_analysis_stable": signal_analysis_stable,
            "t_chatter": t_chatter,
            "signal_analysis_chatter": signal_analysis_chatter,
            "t_opr_free": t_opr_free,
            "opr_free": opr_free,
            "t_opr_chat": t_opr_chat,
            "opr_chat": opr_chat,

        },
    )

    return result
