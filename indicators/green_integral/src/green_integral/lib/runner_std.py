"""Standard interface for the green_integral indicator.

Exposes ``run_green_std`` which mirrors the calling convention of other
CAMP10 indicators (maxent_sprt, rms_cv, ssq_chatter):

* **Input**: :class:`~green_integral.utils.types.StdSignalData` with fields
  ``t_analysis``, ``signal_analysis``, ``path``, ``fs``, ``meta``.
  ``signal_analysis`` is treated as displacement.  Velocity is taken from
  ``meta["velocity"]`` when provided, otherwise estimated by central-difference
  differentiation.

* **Config** (same shape as MaxEnt / RMS-CV / SSQ)::

    {
        "id": "green_fixed_4cyc_1step",     # optional
        "func": "Default" | "FixedWindow",  # which green variant
        "params_physical": {
            "f_modal":          150.0,  # Hz — bandpass filter frequency  (required)
            "f_cycle":          150.0,  # Hz — cycle frequency for window/step  (required)
                                        #   f_cycle = f_modal     → window per modal period
                                        #   f_cycle = 1 / T_rev   → window per revolution
            "N_cycles_per_seg": 4,      # cycles per window  (required)
            "step_cycles":      1.0,    # step in cycles  (default 1.0)

            # pass-through to the internal config:
            "data_filtrated": True,
            "use_area_threshold": False,
            "training_intervals": None,
            "z_sigma": 3.0,
            # ... any other GreenIntegralConfig / FixedWindowConfig field
        },
    }

* **Output**: :class:`~green_integral.utils.types.IndicatorResult`.

  For ``func="Default"``:
    - ``t``   = per-window representative time [s] (``indicadores["t_n"]``)
    - ``I_t`` = per-window ``delta_n`` values

  For ``func="FixedWindow"``:
    - ``t``   = window start times (``result.t_wins``)
    - ``I_t`` = instantaneous Lyapunov exponent σ̂ (``result.sigma_ewma``)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.types import (
    SignalData as _GreenSignalData,
    StdSignalData,
    IndicatorResult,
)
from .runner import run_green_integral
from .runner_fixed import run_fixed_window
from ..logging_setup import _section

logger = logging.getLogger(__name__)

# ── Keys that are passed directly to the internal config dicts ────────────────
_GREEN_PASS_THROUGH: frozenset = frozenset({
    "data_filtrated", "hilbert", "while_loop_extend", "cycles_cluster_points",
    "thein_sen", "use_area_threshold", "training_intervals", "frac_stable",
    "stable_time", "z_sigma", "debug_level", "debug_window_range",
    "save_figures_windows", "work_space", "t_theorical",  # for debug/plots, not used in detection
    # FixedWindow extras
    "lambda_ewma", "accumulate", "G_memory", "sigma_method", "sigma_local_n",
    "area_noise_eps"," z_sigma",
    "use_beta_from_cycles", "use_zero_crossing_cycles", "zc_detrend", "v_cycle_mode",
    "cycle_area_norm", "center_win"
})

# ── Keys specific to the physical resolver (never forwarded to config) ────────
_RESOLVER_KEYS: frozenset = frozenset({
    "f_cycle", "N_cycles_per_seg", "step_cycles",
})


def _resolve_physical_params_green(
    params_physical: Dict[str, Any],
) -> Tuple[float, int, float, Dict[str, Any], Dict[str, Any]]:
    """Convert physical parameters to green_integral native parameters.

    Parameters
    ----------
    params_physical : Physical parameter dict with keys:

        ``f_modal``          – bandpass filter frequency [Hz] (required).
        ``f_cycle``          – cycle frequency for window/step sizing [Hz] (required).
                               Set equal to ``f_modal`` for modal-period windows,
                               or to ``1/T_rev`` for revolution-period windows.
        ``N_cycles_per_seg`` – number of cycles per window (required).
        ``step_cycles``      – window step in cycles (default 1.0).

    Returns
    -------
    f_modal_physical : float
        True modal frequency [Hz] — stored in trace only (NOT passed to core).
    f_cycle : float
        Cycle frequency [Hz] — passed to the core as its ``f_modal`` so that
        ``T_window = num_T / f_cycle = N_cycles * T_cycle`` exactly.
    num_T : int
        Exactly ``N_cycles_per_seg`` — no conversion, no rounding.
    dt : float
        Window step [s].
    extra : dict
        Pass-through keys forwarded to the internal config.
    trace : dict
        Traceability record with input values and resolved native params.
    """
    for key in ("f_modal", "f_cycle", "N_cycles_per_seg"):
        if key not in params_physical:
            raise ValueError(f"params_physical must contain '{key}'.")

    f_modal  = float(params_physical["f_modal"])
    f_cycle  = float(params_physical["f_cycle"])
    if f_modal <= 0.0:
        raise ValueError(f"f_modal must be > 0, got {f_modal}.")
    if f_cycle <= 0.0:
        raise ValueError(f"f_cycle must be > 0, got {f_cycle}.")
    T_modal  = 1.0 / f_modal
    T_cycle  = 1.0 / f_cycle

    N_cycles = int(params_physical["N_cycles_per_seg"])
    if N_cycles < 1:
        raise ValueError(f"N_cycles_per_seg must be >= 1, got {N_cycles}.")
    step_cycles = float(params_physical.get("step_cycles", 1.0))

    extra: Dict[str, Any] = {
        k: v for k, v in params_physical.items()
        if k in _GREEN_PASS_THROUGH
    }

    # Physical window duration and step in seconds
    T_window = N_cycles * T_cycle               # [s]
    dt       = step_cycles * T_cycle            # [s]

    # num_T is passed directly as N_cycles_per_seg.
    # The core receives f_cycle as its "f_modal" so that:
    #   T_window_core = num_T / f_cycle = N_cycles * T_cycle  (exact, no rounding)
    # The true physical f_modal is stored in the trace for reference only.
    num_T = N_cycles

    trace = {
        "f_modal":          f_modal,      # physical (stored for reference)
        "T_modal":          T_modal,
        "f_cycle":          f_cycle,      # defines the cycle / window size
        "T_cycle":          T_cycle,
        "N_cycles_per_seg": N_cycles,
        "step_cycles":      step_cycles,
        "T_window_s":       T_window,
        "resolved_num_T":   num_T,        # = N_cycles_per_seg exactly
        "resolved_dt":      dt,
    }

    return f_modal, f_cycle, num_T, dt, extra, trace


def run_green_std(
    signal_data: StdSignalData,
    config: Dict[str, Any],
) -> IndicatorResult:
    """Run green_integral using the standard CAMP10 interface.

    Parameters
    ----------
    signal_data : :class:`~green_integral.utils.types.StdSignalData`
        Standard signal container.  ``signal_analysis`` = displacement;
        velocity taken from ``meta["velocity"]`` or computed numerically.
    config : dict
        Standard config dict with keys ``func``,
        ``params_physical`` (and optional ``id``).  See module docstring.

    Returns
    -------
    :class:`~green_integral.utils.types.IndicatorResult`
    """
    # ── Validate config keys ─────────────────────────────────────────────────
    func = config.get("func", "Default")
    if func not in ("Default", "FixedWindow"):
        raise ValueError(f"config['func'] must be 'Default' or 'FixedWindow', got '{func}'.")

    params_physical = config.get("params_physical", {})
    name            = "Green_Integral"

    # ── Resolve physical → native params ────────────────────────────────────
    f_modal, f_cycle, num_T, dt, extra_params, trace = _resolve_physical_params_green(
        params_physical
    )



    # ── Build displacement & velocity arrays ─────────────────────────────────
    t_arr  = np.asarray(signal_data.t_analysis,    dtype=float)
    x_arr  = np.asarray(signal_data.signal_analysis, dtype=float)

    if "velocity" in signal_data.meta and signal_data.meta["velocity"] is not None:
        v_arr = np.asarray(signal_data.meta["velocity"], dtype=float)
        vel_source = "meta['velocity']"
    else:
        logger.warning("Warning: velocity not found in meta; using np.gradient for estimation.")
        # Central-difference estimate (same length as x_arr)
        v_arr = np.gradient(x_arr, t_arr)
        vel_source = "np.gradient (estimated)"

    logger.debug("run_green_std | velocity source: %s", vel_source)

    # ── Build internal GreenSignalData ───────────────────────────────────────
    sig_name = (
        signal_data.meta.get("signal", None)
        or signal_data.meta.get("name", None)
        or signal_data.path
        or "signal"
    )
    internal_sig = _GreenSignalData(
        t=t_arr,
        displacement=x_arr,
        velocity=v_arr,
        name=str(sig_name),
    )

    # ── Build internal config dict ────────────────────────────────────────────
    # The core uses T_window = num_T / f_modal internally.
    # We pass f_cycle as the core's "f_modal" so that:
    #   T_window = num_T / f_cycle = N_cycles_per_seg * T_cycle  (exact)
    # The true physical f_modal is stored in the trace only.
    internal_params: Dict[str, Any] = {
        "f_modal": f_cycle,   # ← f_cycle, NOT f_modal, so T_window = N_cycles/f_cycle
        "num_T": num_T,       # = N_cycles_per_seg exactly
        "dt": dt,
        **extra_params,
    }

    # ── Run indicator ─────────────────────────────────────────────────────────
    use_area_threshold = bool(extra_params.get("use_area_threshold", False))

    if func == "Default":
        green_cfg = {"func": "Default", "params": internal_params}
        raw_result = run_green_integral(internal_sig, green_cfg)

        # Extract time-series from per-window results
        t_out = np.array(
            [dw["indicadores"]["t_n"] for dw in raw_result.data_window],
            dtype=float,
        )
        if use_area_threshold:
            # I_t = área Ak (lo que se umbraliza para calcular t_d)
            I_t_out = np.array(
                [dw.get("center_area_value") or dw.get("median_area") or np.nan
                 for dw in raw_result.data_window],
                dtype=float,
            )
        else:
            # I_t = delta_n (log-ratio de áreas; negativo → chatter)
            I_t_out = np.array(
                [dw["indicadores"]["delta_n"] for dw in raw_result.data_window],
                dtype=float,
            )
        t_d = raw_result.t_d

    else:  # FixedWindow
        fixed_cfg = {"func": "FixedWindow", "params": internal_params}
        raw_result = run_fixed_window(internal_sig, fixed_cfg)

        t_out = np.asarray(raw_result.t_wins, dtype=float)
        if use_area_threshold:
            # I_t = área Ak (lo que se umbraliza para calcular t_d)
            I_t_out = np.asarray(raw_result.areas, dtype=float)
        else:
            # I_t = σ̂_ewma (exponente de Lyapunov; positivo → chatter)
            I_t_out = np.asarray(raw_result.sigma_ewma, dtype=float)
            
        t_d = raw_result.t_d
        t_d_no_FAR = raw_result.t_d_no_FAR



    if t_d is not None and np.asarray(t_d).size > 0:
        logger.info(_section("CHATTER INDICATOR - Green Area"))
        logger.info("  %-24s %s",     "Indicador:",         name)
        logger.info("  %-24s %s",    "Modo config",         config.get("param_mode", "N/A"))
        logger.info("  %-24s %s",    "Función ",            func)
        logger.info("  %-24s %.3f Hz",   "Frecuency Cycle:", params_physical.get("f_cycle", "n/a"))

        logger.info("  %-24s %d",     "Area Windows:",      params_physical.get("N_cycles_per_seg", "n/a"))
        logger.info("  %-24s %d",      "Total Windows",     params_physical.get("N_cycles_per_seg", "n/a"))

        logger.info("  %-24s %.3f ",   "Step:",             params_physical.get("step_cycles", "n/a"))
        logger.info("  %-24s %s",     "Area Threshold:",    params_physical.get("use_area_threshold", "n/a"))

        if params_physical["use_area_threshold"] == False:
            logger.info("  %-24s %.3f",  "lambda_ewma:",    params_physical.get("lambda_ewma", "n/a"))
            logger.info("  %-24s %s",    "acumulate:",      params_physical.get("accumulate", "n/a"))
            logger.info("  %-24s %s",    "G_memory:",       params_physical.get("G_memory", "n/a"))
            logger.info("  %-24s %s",    "sigma_method:",   params_physical.get("sigma_method", "n/a"))
            logger.info("  %-24s %s",    "sigma_local_n:",  params_physical.get("sigma_local_n", "n/a"))

        logger.info(
            "  %-24s mu: %.10f, sigma: %.10f",
            "Training Area:",
            raw_result.mu_log if hasattr(raw_result, "mu_log") else np.nan,
            raw_result.sigma_log if hasattr(raw_result, "sigma_log") else np.nan,
        )
        logger.info("  %-24s %.10f", "Upper Limit:",     raw_result.upper_log if hasattr(raw_result, "upper_log") else np.nan)
        logger.info("  %-24s %.10f", "Lower Limit:",      raw_result.lower_log if hasattr(raw_result, "lower_log") else np.nan)
        logger.info("  %-24s %.3f s", "First Detection:", raw_result.t_d[0] if hasattr(raw_result, "t_d") and raw_result.t_d is not None else np.nan)
        logger.info("  %-24s %.3f s", "First Detection Non FAR:",  raw_result.t_d_no_FAR[0] if hasattr(raw_result, "t_d_no_FAR") and raw_result.t_d_no_FAR is not None else np.nan)
        logger.info("  %-24s %d",     "Total Detections:", raw_result.t_d.size if hasattr(raw_result, "t_d") and raw_result.t_d is not None else 0)
        logger.info("  %-24s %.4f, %.4f ms", "Tiempo I[0], I[1]:", raw_result.t_wins[0]*1000, raw_result.t_wins[1]*1000)
        logger.info("  %-24s %.4f, %.4f ms", "Area[0], Area[1] ", raw_result.t_wins[1]*1000 - raw_result.t_wins[0]*1000, raw_result.t_wins[2]*1000 - raw_result.t_wins[1]*1000 )





    run_name = config.get("id", f"green_{func.lower()}")


    return IndicatorResult(
        name=name,
        t=t_out,
        I_t=I_t_out,
        t_d=t_d,
        t_d_no_FAR = t_d_no_FAR,
        meta={
            "func": func,
            "use_area_threshold": use_area_threshold,
            "I_t_meaning": "areas_Ak" if use_area_threshold else ("delta_n" if func == "Default" else "sigma_ewma"),
            "resolver_trace": trace,
            "vel_source": vel_source,
            "raw_result": raw_result,
            "signal": internal_sig,
            "signal_path": signal_data.path,
            "f_cycle": f_cycle,
            "N_cycles_per_seg": params_physical.get("N_cycles_per_seg", "n/a"),
            "Total_window": params_physical.get("N_cycles_per_seg", "n/a"),
            "step_cycles": params_physical.get("step_cycles", "n/a"),
            "data_filtrated":  params_physical.get("data_filtrated", None),
            "hilbert":        params_physical.get("hilbert", None),   
            "lambda_ewma":     params_physical.get("lambda_ewma", None),
            "while_loop_extend": params_physical.get("while_loop_extend", None),
            "accumulate":        params_physical.get("accumulate", None),
            "G_memory":          params_physical.get("G_memory", None),
            "cycles_cluster_points": params_physical.get("cycles_cluster_points", None),
            "sigma_method":         params_physical.get("sigma_method", None),
            "sigma_local_n":        params_physical.get("sigma_local_n", None),
            "thein_sen":           params_physical.get("thein_sen", None),
            "area_noise_eps":       params_physical.get("area_noise_eps", 1e-25),
            "use_area_threshold":   params_physical.get("use_area_threshold", None),
            "training_intervals":   params_physical.get("training_intervals", None),
            "z_sigma":             params_physical.get("z_sigma", None),
        },
    )
