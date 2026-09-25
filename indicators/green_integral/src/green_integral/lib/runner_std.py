"""Standard interface for the green_integral indicator.

Exposes ``run_green_std`` which mirrors the calling convention of other
CAMP10 indicators (maxent_sprt, rms_cv, ssq_chatter) per
``indicators/COMMON_TEMPLATE.md``:

* **Input**: :class:`~green_integral.utils.types.StdSignalData` with fields
  ``t_analysis``, ``signal_analysis``, ``path``, ``fs``, ``meta``.
  ``signal_analysis`` is treated as displacement.  Velocity is taken from
  ``meta["velocity"]`` when provided, otherwise estimated by central-difference
  differentiation.

* **Config** (same shape as MaxEnt / RMS-CV / SSQ, see COMMON_TEMPLATE.md §3)::

    {
        "id": "green_lyapunov_4cyc_1step",     # optional
        "func": "Default" | "Lyapunov",  # which green variant
        "param_mode": "native" | "by_revolution" | "by_modal",

        # param_mode == "native": native kwargs, forwarded as-is.
        "params": {
            "f_modal": 150.0, "num_T": 6, "dt": 0.005,
            "data_filtrated": True, ...
        },

        # param_mode == "by_revolution" | "by_modal": physical parameters.
        "params_physical": {
            "T_rev":        0.005,  # s — spindle revolution period (by_revolution, required)
            "N_rev_window": 4,      # revolutions per window (by_revolution, required)
            "step_rev":     1.0,    # hop in revolutions (by_revolution, required)

            "T_modal":        1/150.0,  # s — modal period (by_modal, required)
            "N_modal_window": 4,        # modal periods per window (by_modal, required)
            "step_modal":     1.0,      # hop in modal periods (by_modal, required)

            # pass-through to the internal config (both modes):
            "data_filtrated": True,
            "use_area_threshold": False,
            "training_intervals": None,
            "z_sigma": 3.0,
            # ... any other GreenIntegralConfig / LyapunovConfig field
        },

        # optional, top-level (any param_mode): external reference signal
        # already labeled "stable" (see the DOE reference-dataset pipeline).
        # When set, it is windowed with the same f_modal/num_T/dt and used in
        # full as the training population for the mu +- z*sigma area
        # threshold, replacing training_intervals/stable_time/frac_stable.
        # Omit it (default) for 100% unchanged behavior. See
        # result.meta["training_source"] ("external_reference" | "internal").
        "reference_signal": None,  # Optional[StdSignalData]
    }

  Unlike MaxEnt/RMS-CV/SSQ, Green does not adopt the ``segmentation``
  (``"opr"``/``"raw"``) key: its windows are sized directly in seconds
  (``T_window = N_window * T_unit``), not decimated to per-revolution
  samples, so there is no raw-sample-count variant to switch to.
  ``N_rev_window``/``N_modal_window`` must always be an exact integer cycle
  count (``ValueError`` otherwise, never silently truncated); ``step_rev``/
  ``step_modal`` may be fractional since it only sets a continuous-time hop.

* **Output**: :class:`~green_integral.utils.types.IndicatorResult`.

  For ``func="Default"``:
    - ``t``   = per-window representative time [s] (``indicadores["t_n"]``)
    - ``I_t`` = per-window ``delta_n`` values

  For ``func="Lyapunov"``:
    - ``t``   = window start times (``result.t_wins``)
    - ``I_t`` = instantaneous Lyapunov exponent σ̂ (``result.sigma_ewma``)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.types import (
    SignalData as _GreenSignalData,
    StdSignalData,
    IndicatorResult,
)
from .runner import run_green_integral
from .runner_lyapunov import run_lyapunov
from ..logging_setup import _section

logger = logging.getLogger(__name__)

# ── Keys that are passed directly to the internal config dicts ────────────────
_GREEN_PASS_THROUGH: frozenset = frozenset({
    "data_filtrated", "hilbert", "while_loop_extend", "cycles_cluster_points",
    "thein_sen", "use_area_threshold", "training_intervals", "frac_stable",
    "stable_time", "z_sigma", "debug_level", "debug_window_range",
    "save_figures_windows", "work_space", "t_theorical",  # for debug/plots, not used in detection
    # Lyapunov extras
    "lambda_ewma", "accumulate", "G_memory", "sigma_method", "sigma_local_n",
    "area_noise_eps",
    "use_beta_from_cycles", "use_zero_crossing_cycles", "zc_detrend", "v_cycle_mode",
    "cycle_area_norm", "center_win"
})


def _resolve_physical_params_green(
    param_mode: str,
    params_physical: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate physical parameters into native green_integral kwargs.

    See ``indicators/COMMON_TEMPLATE.md`` §3-4 for the contract shared with
    maxent_sprt/rms_cv/ssq_chatter. ``f_cycle`` (renamed ``f_modal`` for the
    internal pipeline, which uses it purely as the cycle-duration reference
    for windowing/clustering — see ``lib/window_processor.py``) is derived
    strictly from ``param_mode``, never from which keys happen to be present.

    Parameters
    ----------
    param_mode : ``"by_revolution"`` or ``"by_modal"``.
    params_physical : Mode-specific keys (see module docstring) plus optional
        pass-through keys (``_GREEN_PASS_THROUGH``).

    Returns
    -------
    native_params : kwargs ready for ``run_green_integral``/``run_lyapunov``
        (``f_modal`` = cycle frequency, ``num_T``, ``dt``, plus pass-through).
    trace : traceability record (physical inputs, resolved native values).

    Raises
    ------
    ValueError
        If required keys are missing or have inadmissible values.
    """
    if param_mode == "by_revolution":
        for key in ("T_rev", "N_rev_window", "step_rev"):
            if key not in params_physical:
                raise ValueError(f"by_revolution mode requires '{key}' in params_physical.")
        T_unit = float(params_physical["T_rev"])
        if T_unit <= 0.0:
            raise ValueError(f"T_rev must be > 0, got {T_unit}.")
        N_win_key, step_key = "N_rev_window", "step_rev"
        unit_name = "rev"

    elif param_mode == "by_modal":
        for key in ("T_modal", "N_modal_window", "step_modal"):
            if key not in params_physical:
                raise ValueError(f"by_modal mode requires '{key}' in params_physical.")
        T_unit = float(params_physical["T_modal"])
        if T_unit <= 0.0:
            raise ValueError(f"T_modal must be > 0, got {T_unit}.")
        T_rev_raw = params_physical.get("T_rev")  # informational only, not used for windowing
        if T_rev_raw is not None and float(T_rev_raw) <= 0.0:
            raise ValueError(f"T_rev must be > 0, got {T_rev_raw}.")
        N_win_key, step_key = "N_modal_window", "step_modal"
        unit_name = "modal"

    else:
        raise ValueError(
            f"Unknown param_mode '{param_mode}'. "
            "Valid options: 'native', 'by_revolution', 'by_modal'."
        )

    N_win_raw = float(params_physical[N_win_key])
    if not N_win_raw.is_integer():
        raise ValueError(
            f"{N_win_key} must be an exact integer cycle count, got {N_win_raw}."
        )
    N_win = int(N_win_raw)
    if N_win < 1:
        raise ValueError(f"{N_win_key} must be >= 1, got {N_win}.")

    step = float(params_physical[step_key])
    if not (0 < step <= N_win):
        raise ValueError(
            f"{step_key} must satisfy 0 < {step_key} <= {N_win_key}={N_win}, got {step}."
        )

    f_cycle = 1.0 / T_unit
    T_window = N_win * T_unit  # [s]
    dt = step * T_unit         # [s]

    extra: Dict[str, Any] = {
        k: v for k, v in params_physical.items() if k in _GREEN_PASS_THROUGH
    }

    native_params: Dict[str, Any] = {
        "f_modal": f_cycle,   # cycle frequency, drives internal windowing only
        "num_T": N_win,
        "dt": dt,
        **extra,
    }

    trace: Dict[str, Any] = {
        "physical_params_input": dict(params_physical),
        "native_params_resolved": {"num_T": N_win, "dt": dt, "f_cycle": f_cycle},
        "unit_name": unit_name,
        "T_unit": T_unit,
        "N_win": N_win,
        "step": step,
        "T_window_s": T_window,
    }
    return native_params, trace


def _to_internal_signal(
    signal_data: StdSignalData,
    label: str,
) -> Tuple[_GreenSignalData, str]:
    """Convert a :class:`StdSignalData` into Green's internal ``SignalData``.

    Velocity is taken from ``meta["velocity"]`` when provided, otherwise
    estimated via ``np.gradient``. Shared by the analyzed signal and an
    optional ``reference_signal`` so both are built identically.
    """
    t_arr = np.asarray(signal_data.t_analysis, dtype=float)
    x_arr = np.asarray(signal_data.signal_analysis, dtype=float)

    if "velocity" in signal_data.meta and signal_data.meta["velocity"] is not None:
        v_arr = np.asarray(signal_data.meta["velocity"], dtype=float)
        vel_source = "meta['velocity']"
    else:
        logger.warning(
            "Warning: velocity not found in meta for %s; using np.gradient for estimation.",
            label,
        )
        v_arr = np.gradient(x_arr, t_arr)
        vel_source = "np.gradient (estimated)"

    sig_name = (
        signal_data.meta.get("signal", None)
        or signal_data.meta.get("name", None)
        or signal_data.path
        or label
    )
    internal_sig = _GreenSignalData(
        t=t_arr, displacement=x_arr, velocity=v_arr, name=str(sig_name),
    )
    return internal_sig, vel_source


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
        Standard config dict with keys ``func``, ``param_mode``, and
        ``params`` (native mode) or ``params_physical`` (physical modes).
        See module docstring.

    Returns
    -------
    :class:`~green_integral.utils.types.IndicatorResult`
    """
    func = config.get("func", "Default")
    if func not in ("Default", "Lyapunov"):
        raise ValueError(f"config['func'] must be 'Default' or 'Lyapunov', got '{func}'.")

    param_mode: str = config.get("param_mode", "native")
    name = "Green_Integral"

    trace: Optional[Dict[str, Any]] = None
    if param_mode == "native":
        native_params: Dict[str, Any] = config.get("params", {})
        unit_name, T_unit, N_win, step = "native", float("nan"), None, None
        f_cycle = native_params.get("f_modal", float("nan"))
    else:
        params_physical = config["params_physical"]
        native_params, trace = _resolve_physical_params_green(param_mode, params_physical)
        unit_name, T_unit  = trace["unit_name"], trace["T_unit"]
        N_win, step        = trace["N_win"], trace["step"]
        f_cycle            = trace["native_params_resolved"]["f_cycle"]

    use_area_threshold = bool(native_params.get("use_area_threshold", False))

    # ── Build internal GreenSignalData (analyzed signal + optional reference) ──
    internal_sig, vel_source = _to_internal_signal(signal_data, "signal")
    logger.debug("run_green_std | velocity source: %s", vel_source)

    reference_signal_std = config.get("reference_signal")
    training_source = "external_reference" if reference_signal_std is not None else "internal"
    if reference_signal_std is not None:
        # A single StdSignalData is one piece; a list is windowed piece by
        # piece downstream (see COMMON_TEMPLATE.md / runner.py|runner_lyapunov.py)
        # — never concatenated into one raw signal first.
        ref_pieces_std = (
            reference_signal_std if isinstance(reference_signal_std, list)
            else [reference_signal_std]
        )
        internal_ref_pieces = []
        for i, piece_std in enumerate(ref_pieces_std):
            internal_piece, ref_vel_source = _to_internal_signal(piece_std, f"reference_signal[{i}]")
            logger.debug("run_green_std | reference_signal[%d] velocity source: %s", i, ref_vel_source)
            internal_ref_pieces.append(internal_piece)
        native_params = {**native_params, "reference_signal": internal_ref_pieces}

    # ── Run indicator ─────────────────────────────────────────────────────────
    if func == "Default":
        green_cfg = {"func": "Default", "params": native_params}
        raw_result = run_green_integral(internal_sig, green_cfg)

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
        t_d_raw = raw_result.t_d
        t_d_no_FAR_raw = None  # not tracked by the Default/clustering variant

    else:  # Lyapunov (constant-duration window, exponential-growth estimate)
        lyapunov_cfg = {"func": "Lyapunov", "params": native_params}
        raw_result = run_lyapunov(internal_sig, lyapunov_cfg)

        t_out = np.asarray(raw_result.t_wins, dtype=float)
        if use_area_threshold:
            I_t_out = np.asarray(raw_result.areas, dtype=float)
        else:
            # I_t = σ̂_ewma (exponente de Lyapunov; positivo → chatter)
            I_t_out = np.asarray(raw_result.sigma_ewma, dtype=float)

        t_d_raw = raw_result.t_d
        t_d_no_FAR_raw = raw_result.t_d_no_FAR

    # ── Standardize detection timestamps: always np.ndarray, never None ──────
    t_d = np.atleast_1d(t_d_raw).astype(float) if t_d_raw is not None else np.array([])
    t_d_no_FAR = (
        np.atleast_1d(t_d_no_FAR_raw).astype(float)
        if t_d_no_FAR_raw is not None else np.array([])
    )

    if t_d.size > 0:
        logger.info(_section("CHATTER INDICATOR - Green Area"))
        logger.info("  %-24s %s",     "Indicador:",         name)
        logger.info("  %-24s %s",     "Modo config:",       param_mode)
        logger.info("  %-24s %s",     "Función:",           func)
        logger.info("  %-24s %.3f Hz","Frecuency Cycle:",   f_cycle)
        logger.info("  %-24s %s",     "Cycles per window:", N_win if N_win is not None else "n/a")
        logger.info("  %-24s %s",     "Step (cycles):",     step if step is not None else "n/a")
        logger.info("  %-24s %s",     "Area Threshold:",    use_area_threshold)

        if not use_area_threshold:
            logger.info("  %-24s %s",  "lambda_ewma:",   native_params.get("lambda_ewma", "n/a"))
            logger.info("  %-24s %s",  "acumulate:",     native_params.get("accumulate", "n/a"))
            logger.info("  %-24s %s",  "G_memory:",      native_params.get("G_memory", "n/a"))
            logger.info("  %-24s %s",  "sigma_method:",  native_params.get("sigma_method", "n/a"))
            logger.info("  %-24s %s",  "sigma_local_n:", native_params.get("sigma_local_n", "n/a"))

        def _attr_or_nan(obj: Any, name: str) -> float:
            value = getattr(obj, name, None)
            return np.nan if value is None else value

        logger.info(
            "  %-24s mu: %.10f, sigma: %.10f",
            "Training Area:",
            _attr_or_nan(raw_result, "mu_log"),
            _attr_or_nan(raw_result, "sigma_log"),
        )
        logger.info("  %-24s %.10f", "Upper Limit:", _attr_or_nan(raw_result, "upper_log"))
        logger.info("  %-24s %.10f", "Lower Limit:", _attr_or_nan(raw_result, "lower_log"))
        logger.info("  %-24s %.3f s", "First Detection:", t_d[0])
        if t_d_no_FAR.size > 0:
            logger.info("  %-24s %.3f s", "First Detection Non FAR:", t_d_no_FAR[0])
        else:
            logger.info("  %-24s %s", "First Detection Non FAR:", "n/a")
        logger.info("  %-24s %d", "Total Detections:", t_d.size)
        if t_out.size > 1:
            logger.info("  %-24s %.4f, %.4f ms", "Tiempo I[0], I[1]:", t_out[0]*1000, t_out[1]*1000)

    run_name = config.get("id", f"green_{func.lower()}")

    meta: Dict[str, Any] = {
        "id": run_name,
        "func": func,
        "param_mode": param_mode,
        "unit_name": unit_name,
        "T_unit": T_unit,
        "f_cycle": f_cycle,
        "N_cycles": N_win,
        "step_cycles": step,
        "Total_window": N_win,
        "use_area_threshold": use_area_threshold,
        "training_source": training_source,
        "I_t_meaning": "areas_Ak" if use_area_threshold else ("delta_n" if func == "Default" else "sigma_ewma"),
        "vel_source": vel_source,
        "raw_result": raw_result,
        "signal": internal_sig,
        "signal_path": signal_data.path,
        "native_params_resolved": native_params,
    }
    if trace is not None:
        meta["physical_params_input"]  = trace["physical_params_input"]
        meta["resolver_trace"] = trace

    return IndicatorResult(
        name=name,
        t=t_out,
        I_t=I_t_out,
        t_d=t_d,
        t_d_no_FAR=t_d_no_FAR,
        meta=meta,
    )
