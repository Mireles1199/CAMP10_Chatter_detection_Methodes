"""Top-level pipeline runner for the RMS-CV chatter detection indicator.

Exposes two public callables:

* :func:`run_rms_cv` — thin dispatcher that resolves the indicator function
  from ``INDICATOR_CONFIG`` and invokes it with the supplied parameters.
* :func:`rms_cv_pipeline` — default end-to-end implementation: windowed RMS
  via :func:`~rms_cv.utils.rms.rms_sequence` followed by online CV
  monitoring via :class:`~rms_cv.lib.cv_monitor.CVOnlineMonitor`.

Typical usage::

    from rms_cv import run_rms_cv, SignalData

    config = {
        "func": "Default",
        "params": {
            "n_max": 20,
            "samples_per_window": 512,
            "cv_threshold": 1.05,
        },
    }
    result = run_rms_cv(signal_data, config)
"""

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
    """Run the default RMS-CV chatter detection pipeline on a signal.

    The pipeline executes two stages:

    1. **RMS windowing** — :func:`~rms_cv.utils.rms.rms_sequence` segments
       ``signal.signal_analysis`` into overlapping frames of length
       *samples_per_window* and computes one RMS value per frame.
    2. **Online CV monitoring** — :class:`~rms_cv.lib.cv_monitor.CVOnlineMonitor`
       ingests the RMS sequence frame by frame, maintaining a sliding window
       of size *n_max* and triggering a chatter alert when the Coefficient of
       Variation (CV) reaches or exceeds *cv_threshold*.

    Args:
        signal (SignalData): Input signal container.  The pipeline uses
            ``signal.signal_analysis``, ``signal.t_analysis``, and
            ``signal.fs``.
        n_max (int): Maximum window size for the online CV computation
            [frames].  Only the last *n_max* RMS values are used to compute
            \u03bc and \u03c3 at each step.
        samples_per_window (int): RMS frame length [samples].  Together with
            ``signal.fs`` this determines ``window_sec``.
        use_unbiased_std (bool, optional): Use the Bessel-corrected (n-1)
            standard deviation estimator.  Defaults to ``True``.
        eps (float, optional): Small constant added to |\u03bc| before division
            when computing CV to avoid division by zero.  Defaults to
            ``1e-12``.
        overlap_pct (Optional[float], optional): Fractional overlap between
            consecutive RMS frames in ``[0, 1)``.  Defaults to ``0.0``
            (non-overlapping).
        detrend (Optional[bool], optional): Remove the per-frame mean before
            computing RMS.  Passed directly to
            :func:`~rms_cv.utils.rms.rms_sequence`.  Defaults to ``None``.
        pad_mode (Optional[str], optional): Edge-padding strategy forwarded
            to :func:`~rms_cv.utils.rms.rms_sequence`.  Defaults to
            ``None`` (no padding).
        cv_threshold (Optional[float], optional): CV value at or above which
            a chatter detection event is recorded.  Set to ``None`` to
            disable CV-triggered alerts.
        rms_threshold (Optional[float], optional): Raw RMS value above which
            a warmup-phase alert is raised (before *n_min_cv* frames are
            accumulated).  ``None`` disables warmup alerts.
        n_min_cv (int, optional): Minimum number of frames in the CV window
            before CV-based alerting is enabled.  Defaults to ``5``.
        warmup_ignore_alerts (bool, optional): If ``True``, suppress all
            alerts (both RMS and CV) during the warmup phase.  Defaults to
            ``False``.
        fs_rms (Optional[float], optional): Reserved parameter; not used in
            the current implementation.  Defaults to ``None``.
        start_time (float, optional): Time offset [s] added to each frame
            timestamp.  Useful when the analysis window does not start at
            ``t=0``.  Defaults to ``0.0``.

    Returns:
        IndicatorResult: Detection result with fields:

        * ``name`` — ``"RMS_CV"``.
        * ``t`` — list of frame timestamps [s].
        * ``I_t`` — CV values for each frame.
        * ``t_d`` — array of timestamps [s] where CV \u2265 *cv_threshold*
          (empty array if no detection).
        * ``meta`` — dictionary echoing all computation parameters and
          intermediate results (RMS values, CV values, window indices, etc.)
          for diagnostics and plotting.
    """

    t = signal.t_analysis
    signal_analysis = signal.signal_analysis
    fs = signal.fs

    # ── Derive temporal parameters from sample counts ────────────────────────
    # Each RMS frame spans samples_per_window samples at the acquisition rate
    window_sec: float = samples_per_window / fs

    # Effective time step between frame centres (0 overlap → dt_rms == window_sec)
    dt_rms: float = window_sec * (1.0 - overlap_pct)

    # ── Stage 1: compute the windowed RMS sequence ───────────────────────────
    # Returns a dict with 'rms', 'times', and 'indices' arrays
    out = rms_sequence(signal_analysis, fs, window_sec=window_sec,
                          overlap_pct=overlap_pct, detrend=detrend,
                          pad_mode=pad_mode,
                          return_indices=True)
    rms_vals = out["rms"]
    times = out["times"]

    # ── Stage 2: online CV monitoring, one RMS frame at a time ───────────────
    # CVOnlineConfig bundles all monitor hyper-parameters; dt_rms lets
    # the monitor compute the correct wall-clock timestamp for each frame
    cfg = CVOnlineConfig(
        n_max=n_max
        , use_unbiased_std=use_unbiased_std, eps=eps,
        cv_threshold=cv_threshold, rms_threshold=rms_threshold,
        n_min_cv=n_min_cv, warmup_ignore_alerts=warmup_ignore_alerts,
        dt_rms=dt_rms, start_time=start_time
    )
    mon = CVOnlineMonitor(cfg)

    # Iterate frame-by-frame, collecting all monitor outputs in a dict-of-lists
    results = defaultdict(list)
    for r in rms_vals:
        res = mon.update(float(r))
        for k, v in res.items():
            results[k].append(v)

    # ── Stage 3: extract detection instants (frames where CV ≥ threshold) ───
    # mask selects the frames that crossed the CV threshold
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
