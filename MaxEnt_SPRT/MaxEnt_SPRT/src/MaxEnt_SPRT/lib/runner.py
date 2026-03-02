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
    Args:
        signal (SignalData): The input signal data to be analyzed.
        INDICATOR_CONFIG (dict): Configuration dictionary containing:
            - "func" (str or callable): The function to execute. If "Default", uses
                _maxent_sprt_pipeline. Can also be a custom callable function.
            - "params" (dict, optional): Additional keyword arguments to pass to the
                function. Defaults to an empty dict if not provided.
    Returns:
        IndicatorResult: The result object containing the analysis output from the
            executed indicator function.
    Raises:
        KeyError: If required keys are missing from INDICATOR_CONFIG.
        TypeError: If the specified function is not callable or signal is invalid.
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
    Cuts the signal to the specified time range.
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
    Parameters
    ----------
    signal : SignalData
        Input signal data object containing time array (t_analysis), signal array (signal_analysis), and sampling frequency (fs).
    rpm : float
        Rotational speed in revolutions per minute.
    ratio_sampling : float
        Sampling ratio for the online detection phase.
    N_seg : int
        Number of revolutions per segment for signal segmentation.
    t_stable_total : float
        Duration (in seconds) of the stable (chatter-free) signal portion from the start.
    alpha : float
        Type I error probability (false positive rate) for the SPRT detector.
    beta : float
        Type II error probability (false negative rate) for the SPRT detector.
    reset_on_H0 : bool
        If True, reset the SPRT test statistic when accepting H0 (no chatter hypothesis).
    cut_start_time : Optional[float], optional
        Start time for cutting the signal. If None, uses the signal's start time. Default is None.
    cut_end_time : Optional[float], optional
        End time for cutting the signal. If None, uses the signal's end time. Default is None.
    Returns
    -------
    IndicatorResult
        An IndicatorResult object containing:
        - name: Indicator name ("MaxEnt_SPRT")
        - t: Time array of segment midpoints
        - I_t: SPRT test statistic history (S_history)
        - t_d: Time points where chatter was detected
        - meta: Dictionary with comprehensive metadata including signal parameters, trained model statistics,
                detector configuration, intermediate signals, and all intermediate processing results.
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
