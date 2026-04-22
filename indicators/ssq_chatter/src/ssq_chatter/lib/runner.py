from __future__ import annotations
from typing import Any, Callable, Dict, List, Sequence, Optional, Tuple

from collections import defaultdict
import math
import logging

from ..utils.types import SignalData, IndicatorResult
from ..lib.pipeline_chatter import ChatterPipeline, PipelineConfig
from ..lib.tf_transformers import SSQ_STFT, STFT
from ..lib.detection_strategies import ThreeSigmaWithLilliefors

import numpy as np

IndicatorFunc = Callable[..., IndicatorResult]
logger = logging.getLogger(__name__)

# ── keys forwarded unchanged to _sst_svd_pipeline ──────────────────────────
_SSQ_PASS_THROUGH_PARAMS: frozenset = frozenset({
    "n_fft_power", "mode", "sigma", "frac_stable", "alpha", "z", "fallback_mad",
})


def _resolve_physical_params_ssq(
    param_mode: str,
    params_physical: Dict[str, Any],
    fs: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate a physical parameter specification into native SSQ-STFT parameters.

    Two physical modes:

    * ``by_revolution`` – temporal parameters in spindle-revolution units via ``T_rev``.
    * ``by_modal``      – temporal parameters in modal-period units via ``T_modal``.

    ``Ai_length`` (the SVD window in STFT frames) is controlled by ``Ai_length_mode``:

    * ``"frames"``       – pass directly as integer.
    * ``"total_window"`` – derive from total desired span in physical units via ``ceil``.

    ``win_length_ms`` and ``hop_ms`` are floats derived by direct unit conversion
    (no rounding). Quantization to integer sample counts happens inside the
    pipeline; the resolver reports both exact and effective values for traceability.

    Args:
        param_mode: ``"by_revolution"`` or ``"by_modal"``.
        params_physical: Physical parameter dictionary.
        fs: Signal sampling frequency [Hz] (from ``SignalData.fs``).

    Returns:
        Tuple[Dict, Dict]: *native_params* ready for ``_sst_svd_pipeline``;
        *trace* with full traceability record.

    Raises:
        ValueError: On missing keys or physically inadmissible values.
    """
    # pass-through params forwarded unchanged
    native_params: Dict[str, Any] = {
        k: v for k, v in params_physical.items()
        if k in _SSQ_PASS_THROUGH_PARAMS
    }
    # quant_notes: List[str] = []

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
        K_key     = "K_rev_svd"
        ai_key    = "Ai_length_rev"

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
        K_key     = "K_modal_svd"
        ai_key    = "Ai_length_modal"

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

    # ── win_length_ms (float, direct conversion) ────────────────────────────
    t_win_exact    = N_win * T_unit * 1.0e3
    win_length_ms  = t_win_exact
    win_samples    = int(math.ceil(win_length_ms * 1e-3 * fs))   # mirrors pipeline truncation
    t_win_efectivo = win_samples / fs * 1.0e3
    # quant_notes.append(
    #     f"win_length_ms: {N_win} x {T_unit * 1e3:.4f} ms = {win_length_ms:.4f} ms"
    # )
    # quant_notes.append(
    #     f"win_samples = int({win_length_ms:.4f} x 1e-3 x {fs:.0f}) = {win_samples}"
    # )
    # quant_notes.append(
    #     f"t_win: exact={t_win_exact:.4f} ms | real={t_win_efectivo:.4f} ms"
    #     f" | delta={abs(t_win_efectivo - t_win_exact) * 1e3:.3f} µs"
    # )

    # ── hop_ms (float, direct conversion) ──────────────────────────────────
    t_hop_exact = step * T_unit * 1.0e3
    hop_ms      = t_hop_exact
    hop_samples = int(math.ceil(hop_ms * 1e-3 * fs))
    t_hop_efectivo  = hop_samples / fs * 1.0e3
    # quant_notes.append(
    #     f"hop_ms: {step} x {T_unit * 1e3:.4f} ms = {hop_ms:.4f} ms"
    # )
    # quant_notes.append(
    #     f"hop_samples = int({hop_ms:.4f} x 1e-3 x {fs:.0f}) = {hop_samples}"
    # )
    # quant_notes.append(
    #     f"t_hop: exact={t_hop_exact:.4f} ms | real={t_hop_efectivo:.4f} ms"
    #     f" | delta={abs(t_hop_efectivo - t_hop_exact) * 1e3:.3f} µs"
    # )

    # ── Ai_length ───────────────────────────────────────────────────────────
    ai_mode = params_physical.get("Ai_length_mode", "frames")

    if ai_mode == "frames":
        if ai_key not in params_physical:
            raise ValueError(
                f"Ai_length_mode='frames' requires '{ai_key}' in params_physical."
            )
        Ai_length = int(params_physical[ai_key])
        if Ai_length < 1:
            raise ValueError(f"Ai_length must be >= 1, got {Ai_length}.")
        # quant_notes.append(f"Ai_length = {ai_key} = {Ai_length} (direct)")

    elif ai_mode == "total_window":
        if K_key not in params_physical:
            raise ValueError(
                f"Ai_length_mode='total_window' requires '{K_key}' in params_physical."
            )
        K_desired    = float(params_physical[K_key])
        if K_desired <= N_win:
            raise ValueError(
                f"{K_key}={K_desired} must be > N_win={N_win}."
            )
        ai_exact  = (K_desired - N_win) / step + 1.0
        Ai_length = math.ceil(ai_exact)
        # K_real    = N_win + (Ai_length - 1) * step
        # quant_notes.append(
        #     f"Ai_length: ceil(({K_desired} - {N_win}) / {step} + 1)"
        #     f" = ceil({ai_exact:.4f}) -> {Ai_length} frames"
        # )
        # quant_notes.append(
        #     f"K_svd: desired={K_desired} {unit_name}s | real={K_real:.4f} {unit_name}s"
        #     f" | delta=+{K_real - K_desired:.4f}"
        # )

    else:
        raise ValueError(
            f"Unknown Ai_length_mode '{ai_mode}'. Valid: 'frames', 'total_window'."
        )

    # ── t_svd_total ─────────────────────────────────────────────────────────
    t_svd_total_exact    = t_win_exact + (Ai_length - 1) * t_hop_exact
    K_svd_units_exact    = t_svd_total_exact / T_unit
    t_svd_total_efectivo  = t_win_efectivo + (Ai_length - 1) * t_hop_efectivo
    K_svd_units_efectivo  = t_svd_total_efectivo / T_unit
    # quant_notes.append(
    #     f"t_svd_total_real = {t_svd_total_real:.4f} ms = {K_svd_units:.3f} {unit_name}s"
    # )

    # ── assemble native params ──────────────────────────────────────────────
    native_params["win_length_ms"] = t_win_efectivo
    native_params["hop_ms"]        = t_hop_efectivo
    native_params["Ai_length"]     = Ai_length

    trace: Dict[str, Any] = {
        "physical_params_input":  dict(params_physical),
        "native_params_resolved": {
            "win_length_ms": win_length_ms,
            "hop_ms":        hop_ms,
            "Ai_length":     Ai_length,
        },
        # "quantization_notes": "; ".join(quant_notes),
        "t_svd_total_exact_s":          t_svd_total_exact,
        "K_svd_total_exact_units":      K_svd_units_exact,
        "t_svd_total_efectivo_s":      t_svd_total_efectivo,
        "K_svd_total_efectivo_units":  K_svd_units_efectivo,
        "unit_name":          unit_name,
        "T_unit":             T_unit,
        "N_win":              N_win,
        "step":               step,
        "win_length_ms":      win_length_ms,
        "hop_ms":             hop_ms,
        "win_samples":        win_samples,
        "hop_samples":        hop_samples,
        "t_win_exact_ms":     t_win_exact,
        "t_win_efectivo_ms":      t_win_efectivo,
        "t_hop_exact_ms":     t_hop_exact,
        "t_hop_efectivo_ms":      t_hop_efectivo,
    }
    return native_params, trace


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

    param_mode: str = INDICATOR_CONFIG.get("param_mode", "native")

    func: IndicatorFunc = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = _sst_svd_pipeline

    trace: Optional[Dict[str, Any]] = None

    if param_mode == "native":
        params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})
    else:
        params_physical: Dict[str, Any] = INDICATOR_CONFIG["params_physical"]
        params, trace = _resolve_physical_params_ssq(
            param_mode, params_physical, signal.fs
        )
        _phys_display = {
            k: v for k, v in trace["physical_params_input"].items()
            if k not in _SSQ_PASS_THROUGH_PARAMS
        }

    # ── VALIDACIÓN hop vs ventana ─────────────────────────────
    win_length_ms = params.get("win_length_ms")
    hop_ms = params.get("hop_ms")

    if win_length_ms is None or hop_ms is None:
        raise KeyError("Both 'win_length_ms' and 'hop_ms' must be provided in params")

    hop_min = 0.0 * win_length_ms
    hop_max = 1.0 * win_length_ms

    if not (hop_min <= hop_ms <= hop_max):
        raise ValueError(
            f"hop_ms must be between 0% and 100% of win_length_ms.")
    # ───────────────────────────────────────────────────────────

    result: IndicatorResult = func(signal, **params)

    # ── traceability in meta ────────────────────────────────────────────────
    result.meta["param_mode"] = param_mode
    if trace is not None:
        result.meta["physical_params_input"]  = trace["physical_params_input"]
        result.meta["native_params_resolved"] = trace["native_params_resolved"]
        # result.meta["quantization_notes"]     = trace["quantization_notes"]
        result.meta["t_svd_total_exact_s"]          = trace["t_svd_total_exact_s"]
        result.meta["K_svd_total_exact_units"]      = trace["K_svd_total_exact_units"]
        result.meta["t_svd_total_efectivo_s"]          = trace["t_svd_total_efectivo_s"]
        result.meta["K_svd_total_efectivo_units"]      = trace["K_svd_total_efectivo_units"]
        result.meta["unit_name"]              = trace["unit_name"]
        result.meta["T_unit"]                 = trace["T_unit"]
        result.meta["t_win_exact_ms"]         = trace["t_win_exact_ms"]
        result.meta["t_win_efectivo_ms"]          = trace["t_win_efectivo_ms"]
        result.meta["t_hop_exact_ms"]         = trace["t_hop_exact_ms"]
        result.meta["t_hop_efectivo_ms"]          = trace["t_hop_efectivo_ms"]

    return result


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

    signal_time = signal.t_analysis
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
        win_length=round(cfg.win_length_ms * 1e-3 * cfg.fs),
        hop_length=round(cfg.hop_ms * 1e-3 * cfg.fs),
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
    Tsx, Sx, fs_out, tt, A_i, t_i, D, d1, res, w, dWx = pipe.run(signal_analysis, signal_time)

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
