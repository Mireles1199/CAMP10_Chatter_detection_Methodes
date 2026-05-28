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
from typing import Any, Callable, Dict, List, Sequence, Optional, Tuple

from collections import defaultdict
import math
import logging

import numpy as np

from rms_cv.utils.types import SignalData, IndicatorResult
from rms_cv import rms_sequence, CVOnlineConfig, CVOnlineMonitor
from rms_cv.lib.cv_monitor import CVStableRegionDetector


IndicatorFunc = Callable[..., IndicatorResult]
logger = logging.getLogger(__name__)

# ── keys forwarded unchanged to rms_cv_pipeline ────────────────────────────
_RMSCV_PASS_THROUGH_PARAMS: frozenset = frozenset({
    "cv_threshold", "rms_threshold", "n_min_cv", "warmup_ignore_alerts",
    "use_unbiased_std", "eps", "detrend", "pad_mode", "start_time", "fs_rms",
    # stable-region adaptive threshold
    "stable_time", "stable_index", "frac_stable", "z", "alpha", "fallback_mad",
})


def _resolve_physical_params_rmscv(
    param_mode: str,
    params_physical: Dict[str, Any],
    fs: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate a physical parameter specification into native RMS-CV parameters.

    Two physical modes are supported:

    * ``by_revolution`` – all temporal parameters expressed in spindle-revolution
      units via ``T_rev``.
    * ``by_modal`` – all temporal parameters expressed in modal-period units
      via ``T_modal``.

    Within each mode, ``n_max`` can be specified in two sub-modes controlled
    by the ``n_max_mode`` key inside ``params_physical``:

    * ``"frames"`` – pass ``n_max`` directly as an integer number of RMS frames.
    * ``"total_window"`` – pass the desired total CV-window span in physical
      units (revolutions or modal periods); ``n_max`` is derived via ``ceil``.

    ``samples_per_window`` is always derived via ``ceil`` to guarantee that the
    RMS window covers **at least** the requested number of physical periods.

    Args:
        param_mode: ``"by_revolution"`` or ``"by_modal"``.
        params_physical: Physical parameter dictionary.
        fs: Signal sampling frequency [Hz] (from ``SignalData.fs``).

    Returns:
        Tuple[Dict, Dict]: *native_params* ready for ``rms_cv_pipeline``;
        *trace* containing the full traceability record.

    Raises:
        ValueError: On missing keys or physically inadmissible values.
    """
    # pass-through params forwarded unchanged
    native_params: Dict[str, Any] = {
        k: v for k, v in params_physical.items()
        if k in _RMSCV_PASS_THROUGH_PARAMS
    }
    quant_notes: List[str] = []

    # ── resolve physical unit ───────────────────────────────────────────────
    if param_mode == "by_revolution":
        for key in ("T_rev", "N_rev_window", "step_rev"):
            if key not in params_physical:
                raise ValueError(
                    f"by_revolution requires '{key}' in params_physical."
                )
        T_unit    = float(params_physical["T_rev"])
        N_win     = float(params_physical["N_rev_window"])
        step      = float(params_physical["step_rev"])
        unit_name = "rev"
        K_key     = "K_rev_cv"
        nmax_key  = "n_max_rev"

    elif param_mode == "by_modal":
        for key in ("T_modal", "N_modal_window", "step_modal"):
            if key not in params_physical:
                raise ValueError(
                    f"by_modal requires '{key}' in params_physical."
                )
        T_unit    = float(params_physical["T_modal"])
        N_win     = float(params_physical["N_modal_window"])
        step      = float(params_physical["step_modal"])
        unit_name = "modal"
        K_key     = "K_modal_cv"
        nmax_key  = "n_max_modal"

    else:
        raise ValueError(
            f"Unknown param_mode '{param_mode}'. "
            "Valid options: 'native', 'by_revolution', 'by_modal'."
        )

    if T_unit <= 0:
        raise ValueError(f"T_unit must be > 0, got {T_unit}.")
    if N_win < 1:
        raise ValueError(f"N_win must be >= 1, got {N_win}.")
    if not (0 < step <= N_win):
        raise ValueError(
            f"step must be in (0, N_win], got step={step}, N_win={N_win}."
        )

    # ── samples_per_window (ceil guarantees >= N_win periods) ──────────────
    N_exact            = N_win * T_unit * fs
    samples_per_window = math.ceil(N_exact)
    t_win_exact        = N_win * T_unit
    t_win_real         = samples_per_window / fs
    quant_notes.append(
        f"samples_per_window: ceil({N_win} x {T_unit:.6f} x {fs:.0f})"
        f" = ceil({N_exact:.4f}) -> {samples_per_window} samples"
    )
    quant_notes.append(
        f"t_win: exact={t_win_exact:.6f} s | real={t_win_real:.6f} s"
        f" | delta=+{(t_win_real - t_win_exact):.2e} s"
    )

    # ── overlap_pct ────────────────────────────────────────────────────────
    overlap_pct  = 1.0 - step / N_win
    dt_rms_real  = t_win_real * (1.0 - overlap_pct)
    quant_notes.append(
        f"overlap_pct: 1 - {step}/{N_win} = {overlap_pct:.6f}"
    )
    quant_notes.append(
        f"dt_rms: step_exact={step * T_unit:.6f} s | real={dt_rms_real:.6f} s"
    )

    # ── n_max ───────────────────────────────────────────────────────────────
    n_max_mode = params_physical.get("n_max_mode", "frames")

    if n_max_mode == "frames":
        if nmax_key not in params_physical:
            raise ValueError(
                f"n_max_mode='frames' requires '{nmax_key}' in params_physical."
            )
        n_max = int(params_physical[nmax_key])
        if n_max < 1:
            raise ValueError(f"n_max must be >= 1, got {n_max}.")
        quant_notes.append(f"n_max = {nmax_key} = {n_max} (direct)")

    elif n_max_mode == "total_window":
        if K_key not in params_physical:
            raise ValueError(
                f"n_max_mode='total_window' requires '{K_key}' in params_physical."
            )
        K_desired   = float(params_physical[K_key])
        if K_desired <= N_win:
            raise ValueError(
                f"{K_key}={K_desired} must be > N_win={N_win}."
            )
        n_max_exact = (K_desired - N_win) / step + 1.0
        n_max       = math.ceil(n_max_exact)
        K_real      = N_win + (n_max - 1) * step
        quant_notes.append(
            f"n_max: ceil(({K_desired} - {N_win}) / {step} + 1)"
            f" = ceil({n_max_exact:.4f}) -> {n_max} frames"
        )
        quant_notes.append(
            f"K_cv: desired={K_desired} {unit_name}s | real={K_real:.4f} {unit_name}s"
            f" | delta=+{K_real - K_desired:.4f}"
        )

    else:
        raise ValueError(
            f"Unknown n_max_mode '{n_max_mode}'. Valid: 'frames', 'total_window'."
        )

    # ── derived t_cv_total ──────────────────────────────────────────────────
    t_cv_total = t_win_real + (n_max - 1) * dt_rms_real
    K_cv_units = t_cv_total / T_unit
    quant_notes.append(
        f"t_cv_total = {t_cv_total:.5f} s = {K_cv_units:.3f} {unit_name}s"
    )

    # ── derived exact (deseado) t_cv_total ─────────────────────────────────
    dt_rms_exact        = step * T_unit          # step in seconds (not quantised)
    t_cv_total_exact    = t_win_exact + (n_max - 1) * dt_rms_exact
    K_cv_units_exact    = t_cv_total_exact / T_unit

    # ── assemble native params ──────────────────────────────────────────────
    native_params["samples_per_window"] = samples_per_window
    native_params["overlap_pct"]        = round(overlap_pct, 10)
    native_params["n_max"]              = n_max

    trace: Dict[str, Any] = {
        "physical_params_input":  dict(params_physical),
        "native_params_resolved": {
            "samples_per_window": samples_per_window,
            "overlap_pct":        round(overlap_pct, 10),
            "n_max":              n_max,
        },
        "quantization_notes":      "; ".join(quant_notes),
        # -- window (deseado / efectivo) --
        "t_win_exact_ms":          t_win_exact  * 1.0e3,
        "t_win_real_ms":           t_win_real   * 1.0e3,
        # -- step between RMS frames (deseado / efectivo) --
        "dt_rms_exact_ms":         dt_rms_exact * 1.0e3,
        "dt_rms_real_ms":          dt_rms_real  * 1.0e3,
        # -- total CV span (deseado / efectivo) --
        "t_cv_total_exact_s":      t_cv_total_exact,
        "K_cv_total_exact_units":  K_cv_units_exact,
        "t_cv_total_s":            t_cv_total,
        "K_cv_total_units":        K_cv_units,
        # -- bookkeeping --
        "unit_name":  unit_name,
        "T_unit":     T_unit,
        "N_win":      N_win,
        "step":       step,
        "dt_rms_real":dt_rms_real,
    }
    return native_params, trace


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


    param_mode: str = INDICATOR_CONFIG.get("param_mode", "native")

    func: IndicatorFunc = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = rms_cv_pipeline

    trace: Optional[Dict[str, Any]] = None

    if param_mode == "native":
        params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})
    else:
        params_physical: Dict[str, Any] = INDICATOR_CONFIG["params_physical"]
        params, trace = _resolve_physical_params_rmscv(
            param_mode, params_physical, signal.fs
        )
        _phys_display = {
            k: v for k, v in trace["physical_params_input"].items()
            if k not in _RMSCV_PASS_THROUGH_PARAMS
        }
        logger.info_plus(
            "Physical parametrization [%s] -> native:\n"
            "  Physical input : %s\n"
            "  Native resolved: %s\n"
            "  Quantization   : %s",
            param_mode,
            _phys_display,
            trace["native_params_resolved"],
            trace["quantization_notes"],
        )

    result: IndicatorResult = func(signal, **params)

    # ── traceability in meta ────────────────────────────────────────────────
    result.meta["param_mode"] = param_mode
    if trace is not None:
        result.meta["physical_params_input"]  = trace["physical_params_input"]
        result.meta["native_params_resolved"] = trace["native_params_resolved"]
        result.meta["quantization_notes"]      = trace["quantization_notes"]
        result.meta["t_win_exact_ms"]           = trace["t_win_exact_ms"]
        result.meta["t_win_real_ms"]            = trace["t_win_real_ms"]
        result.meta["dt_rms_exact_ms"]          = trace["dt_rms_exact_ms"]
        result.meta["dt_rms_real_ms"]           = trace["dt_rms_real_ms"]
        result.meta["t_cv_total_exact_s"]       = trace["t_cv_total_exact_s"]
        result.meta["K_cv_total_exact_units"]   = trace["K_cv_total_exact_units"]
        result.meta["t_cv_total_s"]             = trace["t_cv_total_s"]
        result.meta["K_cv_total_units"]         = trace["K_cv_total_units"]
        result.meta["unit_name"]                = trace["unit_name"]
        result.meta["T_unit"]                   = trace["T_unit"]

    return result

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

    # ── stable-region adaptive threshold (replaces fixed cv_threshold) ──
    stable_time: Optional[Any] = None,
    stable_index: Optional[Any] = None,
    frac_stable: float = 0.30,
    z: float = 3.0,
    alpha: float = 0.05,
    fallback_mad: bool = True,

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
        dt_rms=dt_rms, start_time=window_sec + t[0]
    )
    mon = CVOnlineMonitor(cfg)

    # Iterate frame-by-frame, collecting all monitor outputs in a dict-of-lists
    results = defaultdict(list)
    for r in rms_vals:
        res = mon.update(float(r))
        for k, v in res.items():
            results[k].append(v)

    # ── Stage 3: threshold — adaptive (stable region) or fixed ─────────────
    _use_stable = (stable_time is not None) or (stable_index is not None) or (frac_stable > 0)
    cv_array    = np.asarray(results["cv"])
    t_array     = np.array(results["time"], dtype=float)

    stable_det_meta: dict = {}
    if _use_stable:
        det = CVStableRegionDetector(
            frac_stable=frac_stable,
            z=z,
            alpha=alpha,
            fallback_mad=fallback_mad,
            stable_time=stable_time,
            stable_index=stable_index,
        )
        det_res            = det.detect(cv_array, t=t_array)
        cv_threshold_used  = float(det_res["threshold"])
        cv_threshold_method = "stable_region"
        stable_det_meta = {
            "mu_stable":          det_res["mu"],
            "sigma_stable":       det_res["sigma"],
            "normal_ok":          det_res["normal_ok"],
            "p_value":            det_res["p_value"],
            "metodo_umbral":      det_res["metodo_umbral"],
            "idx_estable_usados": det_res["idx_estable_usados"],
        }
    else:
        cv_threshold_used   = float(cv_threshold) if cv_threshold is not None else 0.0
        cv_threshold_method = "fixed"

    mask                = np.where(cv_array > cv_threshold_used)[0]
    chatter_points_time = t_array[mask]
    chatter_points_cv   = cv_array[mask]

    # ======================
    # Package the result
    # ======================
    result = IndicatorResult(
        name="RMS_CV",
        t=results["time"],
        I_t=results["cv"],
        t_d=chatter_points_time,
        meta={
            "n" : results["n"],
            "mu" : results["mu"],
            "sigma" : results["sigma"],
            "alert" : results["alert"],
            "reason" : results["reason"],
            "cv_threshold":        cv_threshold,
            "cv_threshold_used":   cv_threshold_used,
            "cv_threshold_method": cv_threshold_method,
            "rms_threshold": rms_threshold,
            "n_max": n_max,
            "use_unbiased_std": use_unbiased_std,
            "eps": eps,
            "n_min_cv": n_min_cv,
            "warmup_ignore_alerts": warmup_ignore_alerts,
            "fs_rms": fs_rms,
            "dt_rms": dt_rms,
            "samples_per_window": samples_per_window,
            "t_rms": times,
            "rms_values": rms_vals,
            "cv_time": results["time"],
            "cv_values": results["cv"],
            "window_sec": window_sec,
            "idx_rms_windows": out["indices"],
            **stable_det_meta,
        },
    )

    return result

# if __name__ == "__main__":
#     print("Indicator RMS-CV Initialized")
