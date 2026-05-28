"""
sweep/run_one.py
================
Safe execution of one parameter combo and extraction of all result fields.

``run_combo`` dispatches to the appropriate indicator runner
(``run_rms_cv``, ``run_sst_svd``, or ``run_maxent_sprt``), computes
performance metrics, and returns a flat ``RunResult`` dataclass.

All exceptions from the indicator runner are caught and surfaced as
``run_ok=False`` with ``error_str`` set; this allows the sweep loop to
continue even when specific combos fail (e.g. insufficient n_fft for SST-SVD
at very small N_win).

T_total_actual_s extraction
----------------------------
- RMS-CV  : ``result.meta["t_cv_total_s"]``
- SST-SVD : ``result.meta["t_svd_total_efectivo_s"]``
- MaxEnt  : ``result.meta["native_params_resolved"]["N_seg"] * T_unit``
  (T_total for MaxEnt is the window size; N_seg is stored in the trace)

Usage
-----
    from sweep.run_one import run_combo
    result = run_combo(signal, config, "rms_cv", t_gt=5.365,
                       T_unit=1/150.0, K_total=8, lam=1.0)
"""

from __future__ import annotations

import math
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .metrics import MetricResult, evaluate

__all__ = ["RunResult", "run_combo"]

# ── indicator id normalisation ────────────────────────────────────────────────
_RMS_CV_IDS  = {"rms_cv"}
_SST_SVD_IDS = {"sst_svd"}
_MAXENT_IDS  = {"maxent", "maxent_sprt"}


@dataclass
class RunResult:
    """All outputs from a single parameter-combo run.

    Scalar fields are stored here directly; large arrays are stored in the
    ``arrays`` sub-dict to make DataFrame construction lightweight.

    Attributes
    ----------
    run_id : str
        Unique identifier (UUID4 hex prefix) for this run.
    indicator : str
        Normalised indicator name: ``"rms_cv"``, ``"sst_svd"``, ``"maxent"``.
    basis_mode : str
        Basis mode: ``"by_modal"`` or ``"by_revolution"``.
    K_total : int
        Total physical cycles for this combo.
    N_win : Optional[int]
        Window size in physical cycles (None for MaxEnt).
    step : int
        Hop size in physical cycles.
    n_accum : Optional[int]
        Number of accumulated frames (None for MaxEnt).
    overlap_frac : Optional[float]
        Fractional overlap = 1 - step/N_win (None for MaxEnt).
    T_des_s : float
        Desired T_total [s] = K_total * T_unit.
    T_win_s : float
        Desired window duration [s] = N_win * T_unit (NaN for MaxEnt).
    T_hop_s : float
        Desired hop duration [s] = step * T_unit.
    T_total_actual_s : float
        Actual T_total extracted from indicator result meta [s].
        NaN if run failed or key not found.
    K_total_actual : float
        T_total_actual_s / T_unit.
    delta_K : float
        K_total_actual - K_total.
    delta_T_total_vs_des : float
        T_total_actual_s - T_des_s [s].
    t_d_first : float
        First detection timestamp [s] (NaN if no detection or run failed).
    delta_t_d : float
        First-detection latency [s] = t_d_first - t_gt (NaN if not detected).
    N_fa : int
        False-alarm count.
    P_det : int
        1 if any detection, else 0.
    score : float
        Composite score (NaN if not detected).
    lower_bound_delta_td : float
        Theoretical minimum latency = K_total * T_unit.
    score_lb : float
        Lower bound on score.
    n_pts_indicator : int
        Length of the indicator time series (0 if failed).
    run_ok : bool
        True if the indicator ran without exception.
    error_str : str
        Exception traceback (empty string if run_ok).
    n_combos_valid : int
        Total valid combos for this (indicator, K_total) pair.
    arrays : dict
        Sub-dict with large array outputs:
        ``{"t_indicator": ..., "I_t": ..., "t_d_array": ...}``.
    """

    run_id:               str
    indicator:            str
    basis_mode:           str
    K_total:              int
    N_win:                Optional[int]
    step:                 int
    n_accum:              Optional[int]
    overlap_frac:         Optional[float]
    T_des_s:              float
    T_win_s:              float
    T_hop_s:              float
    T_total_actual_s:     float
    K_total_actual:       float
    delta_K:              float
    delta_T_total_vs_des: float
    t_d_first:            float
    t_d_first_true:       float
    delta_t_d:            float
    N_fa:                 int
    P_det:                int
    score:                float
    lower_bound_delta_td: float
    score_lb:             float
    n_pts_indicator:      int
    run_ok:               bool
    error_str:            str
    n_combos_valid:       int
    arrays:               Dict[str, Any] = field(default_factory=dict)
    meta:                 Dict[str, Any] = field(default_factory=dict)


def run_combo(
    signal: Any,
    indicator_config: Dict[str, Any],
    indicator_id: str,
    t_gt: float,
    T_unit: float,
    K_total: int,
    lam: float,
    combo: Optional[Dict[str, Any]] = None,
    n_combos_valid: int = 0,
    basis_mode: str = "by_modal",
    run_id = None,
) -> RunResult:
    """
    Execute one indicator combo and return a :class:`RunResult`.

    Parameters
    ----------
    signal : SignalData
        Input signal bundle.
    indicator_config : dict
        ``INDICATOR_CONFIG`` dict built by :func:`~sweep.config_builder.build_indicator_config`.
    indicator_id : str
        Indicator name (``"rms_cv"``, ``"sst_svd"``, ``"maxent"``).
    t_gt : float
        Ground-truth chatter onset time [s].
    T_unit : float
        Physical time unit for the basis [s].
    K_total : int
        Total physical cycles for this combo.
    lam : float
        False-alarm penalty coefficient.
    combo : dict, optional
        The combo dict from :func:`~sweep.enumerator.enumerate_feasible`.
        Used to extract N_win, step, n_accum, overlap_frac, n_combos_valid.
    n_combos_valid : int
        Total number of valid combos for (indicator, K_total).  Overridden by
        ``combo["n_combos_valid"]`` when ``combo`` is provided.
    basis_mode : str
        Basis mode string for recording.

    Returns
    -------
    RunResult
    """
    ind = indicator_id.lower()

    # ── extract combo fields ─────────────────────────────────────────────────
    if combo is not None:
        N_win        = combo.get("N_win")
        step         = int(combo["step"])
        n_accum      = combo.get("n_accum")
        overlap_frac = combo.get("overlap_frac")
        n_combos_valid = int(combo.get("n_combos_valid", n_combos_valid))
    else:
        # Fallback: extract from config
        phys   = indicator_config.get("params_physical", {})
        N_win  = None
        step   = int(
            phys.get("step_modal", phys.get("step_rev", phys.get("step_seg", 1)))
        )
        n_accum      = None
        overlap_frac = None

    # ── derived timing fields ────────────────────────────────────────────────
    T_des_s = K_total * T_unit
    T_win_s = (N_win * T_unit) if N_win is not None else math.nan
    T_hop_s = step * T_unit



    # ── run indicator ────────────────────────────────────────────────────────
    run_ok    = False
    error_str = ""
    result    = None

    try:
        result = _dispatch(signal, indicator_config, ind)
        run_ok = True
    except Exception:
        error_str = traceback.format_exc()
        print(f"Exception in run_id={run_id}:\n{error_str}")

    # ── extract arrays and meta ──────────────────────────────────────────────
    if run_ok and result is not None:
        t_indicator = np.asarray(result.t,   dtype=float)
        I_t         = np.asarray(result.I_t, dtype=float)
        t_d_arr     = _normalise_t_d(result.t_d)
        n_pts       = len(t_indicator)

        T_total_actual_s = _extract_T_total(result, ind, T_unit)

        # For MaxEnt: inject a training note in meta for traceability
        if ind in _MAXENT_IDS:
            result.meta["training_note"] = (
                f"t_stable_total = t_gt = {t_gt:.6f} s "
                f"(known onset — optimistic alpha estimate)"
            )

        result_meta = result.meta if hasattr(result, "meta") else {}
    else:
        t_indicator      = np.array([], dtype=float)
        I_t              = np.array([], dtype=float)
        t_d_arr          = np.array([], dtype=float)
        n_pts            = 0
        T_total_actual_s = math.nan
        result_meta      = {}

    # ── metrics ──────────────────────────────────────────────────────────────
    if run_ok:
        m: MetricResult = evaluate(t_d_arr, t_gt, T_unit, K_total, lam)
    else:
        m = MetricResult(
            delta_t_d=math.nan,
            N_fa=0,
            P_det=0,
            score=math.nan,
            lower_bound_delta_td=K_total * T_unit,
            score_lb=K_total * T_unit,
        )

    # ── derived T_total comparison fields ────────────────────────────────────
    if math.isfinite(T_total_actual_s):
        K_total_actual       = T_total_actual_s / T_unit
        delta_K              = K_total_actual - K_total
        delta_T_total_vs_des = T_total_actual_s - T_des_s
    else:
        K_total_actual       = math.nan
        delta_K              = math.nan
        delta_T_total_vs_des = math.nan

    t_d_first      = float(t_d_arr[0])       if t_d_arr.size > 0    else math.nan
    _nfa           = m.N_fa
    t_d_first_true = float(t_d_arr[_nfa])  if t_d_arr.size > _nfa else math.nan

    return RunResult(
        run_id               = run_id,
        indicator            = ind,
        basis_mode           = basis_mode,
        K_total              = K_total,
        N_win                = N_win,
        step                 = step,
        n_accum              = n_accum,
        overlap_frac         = overlap_frac,
        T_des_s              = T_des_s,
        T_win_s              = T_win_s,
        T_hop_s              = T_hop_s,
        T_total_actual_s     = T_total_actual_s,
        K_total_actual       = K_total_actual,
        delta_K              = delta_K,
        delta_T_total_vs_des = delta_T_total_vs_des,
        t_d_first            = t_d_first,
        t_d_first_true       = t_d_first_true,
        delta_t_d            = m.delta_t_d,
        N_fa                 = m.N_fa,
        P_det                = m.P_det,
        score                = m.score,
        lower_bound_delta_td = m.lower_bound_delta_td,
        score_lb             = m.score_lb,
        n_pts_indicator      = n_pts,
        run_ok               = run_ok,
        error_str            = error_str,
        n_combos_valid       = n_combos_valid,
        arrays               = {
            "t_indicator": t_indicator,
            "I_t":         I_t,
            "t_d_array":   t_d_arr,
        },
        meta                 = result_meta,
    )


# ── internal helpers ──────────────────────────────────────────────────────────

def _dispatch(signal: Any, config: Dict[str, Any], ind: str) -> Any:
    """Dispatch to the correct indicator runner."""
    if ind in _RMS_CV_IDS:
        from rms_cv import run_rms_cv
        return run_rms_cv(signal, config)
    elif ind in _SST_SVD_IDS:
        from ssq_chatter import run_sst_svd
        return run_sst_svd(signal, config)
    elif ind in _MAXENT_IDS:
        from MaxEnt_SPRT import run_maxent_sprt
        return run_maxent_sprt(signal, config)
    else:
        raise ValueError(f"Unknown indicator: {ind!r}")


def _normalise_t_d(t_d: Any) -> np.ndarray:
    """Convert any t_d form to a 1-D float array."""
    if t_d is None:
        return np.array([], dtype=float)
    if np.isscalar(t_d):
        return np.atleast_1d(float(t_d))
    arr = np.asarray(t_d, dtype=float).ravel()
    return arr


def _extract_T_total(result: Any, ind: str, T_unit: float) -> float:
    """
    Extract actual T_total [s] from the indicator result meta.

    - RMS-CV  : ``result.meta["t_cv_total_s"]``
    - SST-SVD : ``result.meta["t_svd_total_efectivo_s"]``
    - MaxEnt  : ``result.meta["native_params_resolved"]["N_seg"] * T_unit``
    """
    meta = result.meta if hasattr(result, "meta") else {}

    if ind in _RMS_CV_IDS:
        return float(meta.get("t_cv_total_s", math.nan))

    elif ind in _SST_SVD_IDS:
        return float(meta.get("t_svd_total_efectivo_s", math.nan))

    elif ind in _MAXENT_IDS:
        nat = meta.get("native_params_resolved", {})
        N_seg = nat.get("N_seg")
        if N_seg is not None:
            return float(N_seg) * T_unit
        return math.nan

    return math.nan
