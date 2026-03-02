from __future__ import annotations
from typing import Any, Callable, Dict, List, Sequence, Optional

from collections import defaultdict

from ..utils.types import SignalData, IndicatorResult
from ..lib.pipeline_chatter import ChatterPipeline, PipelineConfig
from ..lib.tf_transformers import SSQ_STFT, STFT
from ..lib.detection_strategies import ThreeSigmaWithLilliefors

import numpy as np

IndicatorFunc = Callable[..., IndicatorResult]

def run_sst_svd(signal: SignalData, INDICATOR_CONFIG: dict ) -> IndicatorResult:
    """
    Execute a Synchrosqueezing Transform (SST) with Singular Value Decomposition (SVD) analysis on signal data.
    Args:
        signal (SignalData): The input signal data to be analyzed.
        INDICATOR_CONFIG (dict): Configuration dictionary containing:
            - "func" (str or callable): The function to execute. If "Default", uses _sst_svd_pipeline.
            - "params" (dict, optional): Additional keyword arguments to pass to the function.
    Returns:
        IndicatorResult: The result of the SST-SVD analysis containing indicator metrics and computed values.
    Raises:
        KeyError: If required keys are missing from INDICATOR_CONFIG.
        TypeError: If the specified function is not callable or signal is not SignalData type.
    """

    results: IndicatorResult = None

    func: IndicatorFunc = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = _sst_svd_pipeline

    params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})

    # ── VALIDACIÓN hop vs ventana ─────────────────────────────
    win_length_ms = params.get("win_length_ms")
    hop_ms = params.get("hop_ms")

    if win_length_ms is None or hop_ms is None:
        raise KeyError("Both 'win_length_ms' and 'hop_ms' must be provided in params")

    hop_min = 0.25 * win_length_ms
    hop_max = 0.50 * win_length_ms

    if not (hop_min <= hop_ms <= hop_max):
        raise ValueError(
            f"hop_ms must be between 25% and 50% of win_length_ms.")
    # ───────────────────────────────────────────────────────────

    results = func(signal, **params)

    return results


def _sst_svd_pipeline(
    signal: SignalData,
    n_fft_power: int,
    win_length_ms: float,
    hop_ms: float,
    Ai_length: int,
    mode: str,
    sigma: float,
    frac_stable: float,
    alpha: float,
    z: float,
    fallback_mad: bool,
) -> IndicatorResult:
    """
    Execute Synchrosqueezing Transform (SST) with SVD-based chatter detection pipeline.
    This function performs time-frequency analysis of a milling signal using Synchrosqueezing
    STFT (SSQ-STFT) and applies a statistical detection rule to identify chatter events.
    The pipeline combines signal transformation and anomaly detection strategies following
    a dependency injection pattern.
    Parameters
    ----------
    signal : SignalData
        Input signal object containing raw signal data, analysis time vector, and sampling frequency.
    n_fft_power : int
        Power of 2 for FFT size calculation. Actual n_fft = 1024 * (2**n_fft_power).
    win_length_ms : float
        Window length in milliseconds for STFT analysis.
    hop_ms : float
        Hop length in milliseconds between successive frames.
    Ai_length : int
        Length parameter for amplitude analysis window.
    mode : str
        Mode parameter for the processing pipeline configuration.
    sigma : float
        Smoothing parameter for Synchrosqueezing Transform.
    frac_stable : float
        Fraction parameter for Lilliefors normality test stability threshold.
    alpha : float
        Significance level for statistical hypothesis testing.
    z : float
        Z-score threshold multiplier for outlier detection.
    fallback_mad : bool
        If True, use Median Absolute Deviation (MAD) as fallback for standard deviation estimation
        when normality test fails.
    Returns
    -------
    IndicatorResult
        Result object containing:
        - name: Indicator name ("SST_SVD")
        - t: Time vector of analysis
        - I_t: Indicator values (d1 statistic) over time
        - t_d: Time points where chatter was detected
        - meta: Dictionary with all processing parameters, intermediate results (Jsx, Sx, W, etc.),
                and detection thresholds (lim_inf, lim_sup, p_value, chatter percentage)
    Notes
    -----
    - Uses Three-Sigma rule with Lilliefors normality test for chatter detection
    - Synchrosqueezing STFT provides improved time-frequency resolution
    - Detection thresholds are automatically computed based on signal statistics
    """

    t = signal.t_analysis
    signal_analysis = signal.signal_analysis
    fs = signal.fs

    # ========= Configuration SSTFT + SSQ ============
    n_fft_power = n_fft_power
    n_fft = 1024*(2**n_fft_power)
    cfg: PipelineConfig = PipelineConfig(
        fs=fs,
        win_length_ms=win_length_ms,
        hop_ms=hop_ms,
        n_fft=n_fft,
        Ai_length=Ai_length,
        mode = mode,
    )

    # SSQ-STFT (Strategy)
    hop_length = int(cfg.hop_ms * 1e-3 * cfg.fs)
    tf_strategy = SSQ_STFT(
        win_length=int(cfg.win_length_ms * 1e-3 * cfg.fs),
        hop_length=int(cfg.hop_ms * 1e-3 * cfg.fs),
        n_fft=cfg.n_fft,
        sigma=sigma,
    )

    # detection rule (Strategy)
    detect_rule = ThreeSigmaWithLilliefors(frac_stable=frac_stable ,
                                        alpha=alpha, z=z,
                                        fallback_mad=fallback_mad,)

    # Pipeline (Context)
    pipe = ChatterPipeline(transformer=tf_strategy, detector=detect_rule, config=cfg)

    Tsx: np.ndarray
    Sx: np.ndarray
    fs_out: float
    tt: np.ndarray
    A_i: np.ndarray
    t_i: np.ndarray
    D: np.ndarray
    d1: np.ndarray
    res: Dict[str, Any]
    w: np.ndarray
    dWx: np.ndarray

#%%
    # ========= Run pipeline ============
    Tsx, Sx, fs_out, tt, A_i, t_i, D, d1, res, w, dWx = pipe.run(signal_analysis)

    chatter_points_mask = np.where(d1 > res['lim_sup'])[0]
    chatter_points_time = t_i[chatter_points_mask] if chatter_points_mask.size > 0 else np.array([])
    chatter_points_values = d1[chatter_points_mask] if chatter_points_mask.size > 0 else np.array([])

    result = IndicatorResult(
        name="SST_SVD",
        t=t_i,
        I_t=d1,
        t_d=chatter_points_time,
        meta={
            "fs_out": fs_out,
            "n_fft_power": n_fft_power,
            "win_length_ms": win_length_ms,
            "hop_ms": hop_ms,
            "Ai_length": Ai_length,
            "mode": mode,
            "sigma": sigma,
            "frac_stable": frac_stable,
            "alpha": alpha,
            "z": z,
            "fallback_mad": fallback_mad,
            "W": w,
            "tt": tt,
            "dWx": dWx,
            "Tsx": Tsx,
            "Sx": Sx,
            "A_i": A_i,
            "D": D,
            "lim_inf": res["lim_inf"],
            "lim_sup": res["lim_sup"],
            "metodo_umbral": res["metodo_umbral"],
            "normal_ok": res["normal_ok"],
            "p_value": res["p_value"],
            "mu": res["mu"],
            "sigma": res["sigma"],
            "chatter": f"{100*res['mask'].mean():.2f}%",
        },
    )

    return result
