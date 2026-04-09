from __future__ import annotations
from typing import Any, Callable, Dict, List, Sequence, Optional

from collections import defaultdict

import numpy as np

from rms_cv.utils.types import SignalData, IndicatorResult
from rms_cv import rms_sequence, CVOnlineConfig, CVOnlineMonitor


IndicatorFunc = Callable[..., IndicatorResult]

def run_rms_cv(signal: SignalData, INDICATOR_CONFIG: dict ) -> IndicatorResult:
    """
    Execute the RMS-CV chatter detection pipeline.
    This function orchestrates the chatter detection analysis by executing the specified
    indicator function with the provided configuration parameters. It extracts the function
    reference and parameters from the configuration dictionary and applies them to the input signal.
    Parameters
    ----------
    signal : SignalData
        The input signal data object containing time series and metadata for analysis.
        Expected to include time arrays, velocity, displacement, and force measurements.
    INDICATOR_CONFIG : dict
        Configuration dictionary containing:
        - "func" (str or callable): The indicator function name or callable. If "Default",
          uses the rms_cv_pipeline function.
        - "params" (dict, optional): Dictionary of keyword arguments to pass to the
          indicator function. If not provided, defaults to an empty dictionary.
    Returns
    -------
    IndicatorResult
        The result object containing chatter detection indicators, thresholds, and
        analysis metrics produced by the executed indicator function.
    Examples
    --------
    >>> config = {
    ...     "func": "Default",
    ...     "params": {"cv_threshold": 1.05, "rms_threshold": 0.9}
    ... }
    >>> result = run_rms_cv(signal_data, config)
    """


    results: IndicatorResult = None

    func: IndicatorFunc = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = rms_cv_pipeline

    params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})

    results = func(signal, **params)

    return results

def rms_cv_pipeline(
    signal: SignalData,
    n_max: int,
    samples_per_window: int,
    use_unbiased_std: bool = True,
    eps: float = 1e-12,

    overlap_pct: Optional[float] = 0.0,
    detrend: Optional[bool] = None,
    pad_mode: Optional[str] = None,

    # ==========CV Online Config=============
    cv_threshold: Optional[float] = None,
    rms_threshold: Optional[float] = None,

    n_min_cv: int = 5,
    warmup_ignore_alerts: bool = False,

    fs_rms: Optional[float] = None,

    start_time: float = 0.0,

) -> IndicatorResult:
    """
    Detect chatter in a signal using RMS and Coefficient of Variation (CV) analysis.
    This pipeline computes RMS (Root Mean Square) values over sliding windows of the input signal,
    then applies online CV monitoring to detect anomalies/chatter based on statistical thresholds.
    Args:
        signal (SignalData):
            Input signal data object containing time vector, analysis signal, and sampling frequency.
        n_max (int):
            Maximum number of samples to use in the online CV calculation window.
        samples_per_window (int):
            Number of samples per RMS computation window.
        use_unbiased_std (bool, optional):
            Whether to use unbiased standard deviation. Defaults to True.
        eps (float, optional):
            Small value to avoid division by zero. Defaults to 1e-12.
        overlap_pct (Optional[float], optional):
            Overlap percentage between consecutive windows (0.0 to 1.0). Defaults to 0.0.
        detrend (Optional[bool], optional):
            Whether to detrend the signal before RMS computation. Defaults to None.
        pad_mode (Optional[str], optional):
            Padding mode for windowing. Defaults to None.
        cv_threshold (Optional[float], optional):
            Threshold for CV-based alert detection. Defaults to None.
        rms_threshold (Optional[float], optional):
            Threshold for RMS-based alert detection. Defaults to None.
        n_min_cv (int, optional):
            Minimum number of samples before CV monitoring is enabled. Defaults to 5.
        warmup_ignore_alerts (bool, optional):
            Whether to ignore alerts during warmup phase. Defaults to False.
        fs_rms (Optional[float], optional):
            Reserved parameter for RMS sampling frequency. Defaults to None.
        start_time (float, optional):
            Start time offset for monitoring. Defaults to 0.0.
    Returns:
        IndicatorResult:
            Result object containing:
            - name: "RMS_CV"
            - t: Time vector of CV computation
            - I_t: CV values over time
            - t_d: Time points where chatter was detected (CV >= threshold)
            - meta: Dictionary with detailed computation parameters and intermediate results
    """

    t = signal.t_analysis
    signal_analysis = signal.signal_analysis
    fs = signal.fs

    window_sec: float = samples_per_window / fs

    dt_rms: float = window_sec * (1.0 - overlap_pct)

    out = rms_sequence(signal_analysis, fs, window_sec=window_sec,
                          overlap_pct=overlap_pct, detrend=detrend,
                          pad_mode=pad_mode,
                          return_indices=True)
    rms_vals = out["rms"]
    times = out["times"]
    # plot_rms(times, rms_vals, title="RM S sequence")
    # plt.show()

    # ======== CV Online Monitoring ========
    cfg = CVOnlineConfig(
        n_max=n_max
        , use_unbiased_std=use_unbiased_std, eps=eps,
        cv_threshold=cv_threshold, rms_threshold=rms_threshold,
        n_min_cv=n_min_cv, warmup_ignore_alerts=warmup_ignore_alerts,
        dt_rms=dt_rms, start_time=start_time
    )
    mon = CVOnlineMonitor(cfg)

    results = defaultdict(list)
    for r in rms_vals:
        res = mon.update(float(r))
        for k, v in res.items():
            results[k].append(v)

    # plot_cv(results["time"], results["cv"], cfg.cv_threshold,  title="CV over time")
    # plt.show()
    mask = np.where(np.asarray(results["cv"]) >= cfg.cv_threshold)[0]
    chatter_points_time = np.array(results["time"])[mask]
    chatter_points_cv = np.array(results["cv"])[mask]


    # ======================
    # Package the result
    # ======================
    result = IndicatorResult(
        name="RMS_CV",  # nombre del indicador (pon el que quieras)
        t=results["time"],
        I_t=results["cv"],
        # t_d=results["idx"],
        t_d=chatter_points_time,
        meta={
            "n" : results["n"],
            "mu" : results["mu"],
            "sigma" : results["sigma"],
            "alert" : results["alert"],
            "reason" : results["reason"],
            "cv_threshold": cv_threshold,
            "rms_threshold": rms_threshold,
            "n_max": n_max,
            "use_unbiased_std": use_unbiased_std,
            "eps": eps,
            "n_min_cv": n_min_cv,
            "warmup_ignore_alerts": warmup_ignore_alerts,
            "fs_rms": fs_rms,
            "dt_rms": dt_rms,
            "start_time": start_time,
            "samples_per_window": samples_per_window,
            "t_rms": times,
            "rms_values": rms_vals,
            "cv_time": results["time"],
            "cv_values": results["cv"],
            "window_sec": window_sec,
            "idx_rms_windows": out["indices"],
        },
    )

    return result

# if __name__ == "__main__":
#     print("Indicator RMS-CV Initialized")
