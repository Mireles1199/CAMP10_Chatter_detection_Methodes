"""
Optimizacion/study_phase1/study_phase1.py
==========================================
Phase 1 — Modal vs Revolution Mode Comparison Study.

**Objective**
    Compare ``by_modal`` and ``by_revolution`` segmentation for each
    indicator on an equal-information basis: both modes analyse *the same
    total duration* of signal (T_window [s]) for each grid point.

**Grid  (neutral — not biased towards either mode)**
    T_lcm = lcm(T_modal, T_rev) = 1 / gcd(f_modal_int, f_rev_int)

    For f_modal=150 Hz, RPM=12000 (f_rev=200 Hz):
        T_lcm = 1/gcd(150,200) = 1/50 = 20 ms

    K_LCM_GRID = [1, 2, …, N_LCM_MAX]  →  T_des = K × T_lcm

    Each mode independently derives N_win from T_des:
        N_win_modal = max(2, round(T_des / T_modal))
        N_win_rev   = max(2, round(T_des / T_rev  ))

    Quantisation error  δT = |T_win_ef - T_des|  is recorded for both.

**Step (hop)**
    Each mode uses its own natural step of 1 unit:
      step_modal = 1   (1 × T_modal)
      step_rev   = 1   (1 × T_rev  )

**Result**
    One row per (indicator, K_lcm):
      - N_win and effective duration for each mode
      - Δt_d, N_fa, score, detected flag for each mode
    Saved to ``sweep_output_p1/phase1_results.csv``
    and      ``sweep_output_p1/phase1_result.pkl``.

═══════════════════════════════════════════════════════════════════════════════
USER-CONFIGURABLE CONSTANTS  (edit the block below)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_OPT    = os.path.dirname(_HERE)                    # Optimizacion/
_CAMP10 = os.path.dirname(_OPT)                     # CAMP10_Chatter_detection_Methodes/
_SWEEP  = os.path.join(_OPT, "study_phase3", "sweep")  # reuse sweep package from phase3

# Indicator source directories
for _pkg in ("maxent_sprt/src", "rms_cv/src", "ssq_chatter/src"):
    _p = os.path.join(_CAMP10, "indicators", _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# sweep package (from study_phase3)
if _SWEEP not in sys.path:
    sys.path.insert(0, os.path.dirname(_SWEEP))   # add study_phase3/ so "sweep" is importable

# ── sweep imports ────────────────────────────────────────────────────────────
from sweep import (
    StudyBasis,
    build_indicator_config,
    run_combo,
)
from sweep.run_one import RunResult

# ── HDF5 reader & SignalData ─────────────────────────────────────────────────
from MaxEnt_SPRT import HDF5Reader, run_maxent_sprt
from rms_cv.utils.types import SignalData
from scipy.stats import norm as _norm


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  USER-CONFIGURABLE CONSTANTS                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Physical parameters ──────────────────────────────────────────────────────
F_MODAL = 150.0       # [Hz]  dominant chatter frequency
RPM     = 12_000.0    # [rpm] spindle speed

# ── K grid (integer number of modal periods → T_window) ─────────────────────
# Multipliers of T_lcm — fully neutral (not derived from either mode).
# T_lcm is computed automatically from F_MODAL and RPM in run_phase1_sweep().
N_LCM_MAX    = 10       # upper bound of grid: K_lcm = K_LCM_MIN … N_LCM_MAX
# K_LCM_MIN is computed automatically (see below, after n_accum constants).
# Do NOT set K_LCM_GRID here — it is built at the bottom of this block.

# ── RMS/SST accumulation policy ─────────────────────────────────────────────
# K_LCM_GRID is the grid of EFFECTIVE SPANS (T_eff), not sub-window sizes.
# T_eff = K_eff * T_unit is equal across modes by LCM construction.
#
# With step=1:  K_eff = N_win + (n_accum - 1)
#               N_win = K_eff - (n_accum - 1)   ← derived, NOT fixed
#
# The clamp N_win >= N_WIN_MIN is only inactive when:
#   K_eff_modal = (f_modal/f_lcm) * K_lcm  >=  n_accum + N_WIN_MIN - 1
# → K_LCM_MIN is computed automatically so the clamp never fires.
N_ACCUM_FIXED_RMS_SST  = 1  # accumulation depth (RMS-CV needs high values)
N_WIN_MIN_RMS_SST      = 1    # safety floor for N_win

# Auto K_LCM_MIN: smallest K_lcm such that K_eff_modal >= n_accum + N_WIN_MIN - 1
# K_eff_modal = (f_modal_int / gcd(f_modal_int, f_rev_int)) * K_lcm
# For F_MODAL=150, RPM=12000: f_modal_int/gcd = 150/50 = 3  → K_eff_modal = 3*K_lcm
# Condition: 3*K >= n_accum + N_WIN_MIN - 1  → K >= ceil((n_accum + N_WIN_MIN - 1) / 3)
import math as _math
_f_modal_int = round(F_MODAL)
_f_rev_int   = round(RPM / 60.0)
_g           = _math.gcd(_f_modal_int, _f_rev_int)
_k_modal     = _f_modal_int // _g          # = 3  (units of K_lcm per K_eff_modal)
_K_LCM_MIN   = _math.ceil((N_ACCUM_FIXED_RMS_SST + N_WIN_MIN_RMS_SST - 1) / _k_modal)
K_LCM_GRID   = list(range(_K_LCM_MIN, N_LCM_MAX + 1))

# ── Ground-truth chatter onset time [s] ─────────────────────────────────────
T_GT = 5.365770208787228   # from 1DOF_150Hz/out.hdf5

# ── False-alarm penalty coefficient ─────────────────────────────────────────
LAMBDA = 1.0

# ── Signal channel per indicator ─────────────────────────────────────────────
#   "velocity"     → tool_dyn_o col 1
#   "displacement" → tool_dyn col 1
SIGNAL_CHANNEL = {
    "rms_cv":  "displacement",
    "sst_svd": "velocity",
    "maxent":  "velocity",
}

# ── Indicators to compare ────────────────────────────────────────────────────
INDICATORS = ["rms_cv", "sst_svd", "maxent"]


# ── MaxEnt β mode ───────────────────────────────────────────────────────────
# "symmetric" → α = β = 0.00135  (SPRT diseño equilibrado)
# "classical"  → α = 0.00135, β = P(H < μ₀+3σ₀ | P₁)  (umbral fijo clásico) No implmentado
MAXENT_BETA_MODE = "symmetric"   # <-- cambia aquí

# ── Debug level  (0=off, 1=info) ─────────────────────────────────────────────
DEBUG_LEVEL = 1

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(_HERE, "sweep_output_p1")

# ── Data location ────────────────────────────────────────────────────────────
DIR_CONO  = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"
CUT_RANGE = (0.05, 16.0)   # [s] analysis window  — idem MaxEnt_Detection_NEW.py

# ── Indicator base parameters (same as study_phase3) ─────────────────────────

_BASE_MAXENT: dict = {
    "t_stable_total":     T_GT,             # legacy fallback (used if training_intervals=None)
    "training_intervals": [
        (CUT_RANGE[0], T_GT, "stable"),     # chatter-free training region  (0.05 … T_GT)
        (T_GT,         10,   "chatter"),    # chatter training region        (T_GT … 10 s)
    ],
    "alpha":             0.00135,   # norm.sf(3.0) = z=3 sigma
    "beta":              0.00135,
    "reset_on_H0":       True,
    "cut_start_time":    CUT_RANGE[0],
    "cut_end_time":      10,
    "segmentation":    "raw"
}

_BASE_RMS_CV: dict = {
    "detrend":              False,
    "pad_mode":             "none",
    "use_unbiased_std":     True,
    "eps":                  1e-12,
    "cv_threshold":         None,
    "rms_threshold":        None,
    "n_min_cv":             2,
    "warmup_ignore_alerts": False,
    "stable_time":          (0.0, T_GT),  # [s] required stable duration before T_GT
    # "frac_stable":          0.30,
    "z":                    3.0,
    "alpha":                0.05,   # Lilliefors test
    "fallback_mad":         True,
}

_BASE_SST_SVD: dict = {
    "n_fft_power":  4,
    "mode":         "causal_inclusive",
    "sigma":        6.0,
    "frac_stable":  0.36052,
    "alpha":        0.05,           # Lilliefors test
    "z":            3.0,
    "fallback_mad": False,
}

_BASE_PARAMS: dict = {
    "rms_cv":      _BASE_RMS_CV,
    "sst_svd":     _BASE_SST_SVD,
    "maxent":      _BASE_MAXENT,
    "maxent_sprt": _BASE_MAXENT,
}

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("study_phase1")


# ═════════════════════════════════════════════════════════════════════════════
# Physical helpers
# ═════════════════════════════════════════════════════════════════════════════

def _compute_T_lcm(f_modal: float, rpm: float) -> float:
    """
    Compute the LCM of T_modal and T_rev using integer GCD.

    T_lcm = 1 / gcd(round(f_modal), round(rpm/60))

    This is the smallest window duration that is an **exact** integer
    multiple of both T_modal and T_rev, making the grid neutral.
    """
    f_modal_int = round(f_modal)
    f_rev_int   = round(rpm / 60.0)
    g = math.gcd(f_modal_int, f_rev_int)
    return 1.0 / g


# def _n_win_from_T(T_des: float, T_unit: float, n_min: int = 2) -> int:
#     """Independently derive N_win for one mode from a target duration."""
#     return max(n_min, round(T_des / T_unit))


# def _n_accum_for_target_span(K_target: float, N_win: int, step: int) -> int:
#     """
#     Choose n_accum so effective span matches target as closely as possible.

#     RMS/SST effective span in physical units follows:
#         K_eff = N_win + (n_accum - 1) * step
#     This helper inverts that relation with ceil so K_eff >= K_target.
#     """
#     if step <= 0:
#         raise ValueError(f"step must be > 0, got {step}.")
#     n = math.ceil((K_target - N_win) / step + 1.0)
#     return max(1, int(n))


# def _resolve_n_accum_rms_sst(K_target: float, N_win: int, step: int) -> int:
#     """Resolve n_accum for RMS/SST according to selected policy."""
#     policy = str(N_ACCUM_POLICY_RMS_SST).strip().lower()
#     if policy == "fixed":
#         return max(2, int(N_ACCUM_FIXED_RMS_SST))
#     if policy == "target_span":
#         # Keep >=2 so accumulation is always active for RMS/SST in this study.
#         return max(2, _n_accum_for_target_span(K_target, N_win, step))
#     raise ValueError(
#         f"Unknown N_ACCUM_POLICY_RMS_SST={N_ACCUM_POLICY_RMS_SST!r}. "
#         "Use 'fixed' or 'target_span'."
#     )


# ═════════════════════════════════════════════════════════════════════════════
# Signal loader
# ═════════════════════════════════════════════════════════════════════════════

def _load_signal(channel: str) -> SignalData:
    """Load and cut the HDF5 signal for the requested channel."""
    hdf5_path = os.path.join(DIR_CONO, "out.hdf5")
    reader    = HDF5Reader(hdf5_path)

    tool_dyn = reader.get_element("tool_dyn/data")
    raw_t    = tool_dyn[:, 0]

    if channel == "velocity":
        raw_y = reader.get_element("tool_dyn_o/data")[:, 1]
    elif channel == "displacement":
        raw_y = tool_dyn[:, 1]
    else:
        raise ValueError(f"Unknown channel {channel!r}. Use 'velocity' or 'displacement'.")

    fs = float(1.0 / (raw_t[1] - raw_t[0]))

    # Cut to analysis window
    mask  = (raw_t >= CUT_RANGE[0]) & (raw_t <= CUT_RANGE[1])
    y     = raw_y[mask]
    t     = raw_t[mask]

    return SignalData(
        t_analysis=t,
        signal_analysis=y,
        path=hdf5_path,
        fs=fs,
        meta={"channel": channel},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Classical β resolver (pre-run MaxEnt to extract P₀/P₁)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_classical_beta(alpha: float, base_cfg: dict) -> float:
    """
    Estimate β_cl = P(H < H_thr | P₁)  via a single pre-run.

    H_thr = μ₀ + n·σ₀   with  n = norm.isf(alpha)  (≈ 3 for alpha=0.00135)
    β_cl  = norm.cdf(H_thr, μ₁, σ₁)

    This is the probability that a classical fixed-threshold test at μ₀+nσ₀
    *misses* a chatter event — i.e. the Type-II error implied by the geometry
    of P₀ and P₁.  SPRT uses it as the \u03b2 design parameter so that the
    sequential boundaries are consistent with that classical threshold.
    """
    sig_pre = _load_signal(SIGNAL_CHANNEL["maxent"])
    cfg_pre = {**base_cfg, "alpha": alpha, "beta": alpha}   # any β for pre-run
    res     = run_maxent_sprt(sig_pre, cfg_pre)
    meta    = res.meta or {}

    P0_mu  = meta["P0_mu"]
    P0_sig = meta["P0_sigma"]
    P1_mu  = meta["P1_mu"]
    P1_sig = meta["P1_sigma"]

    n_sig   = _norm.isf(alpha)                          # ≈ 3.0
    H_thr   = P0_mu + n_sig * P0_sig                    # classical threshold
    beta_cl = float(_norm.cdf(H_thr, P1_mu, P1_sig))   # P(H < H_thr | P₁)

    logger.info(
        "Classical β pre-run:\n"
        "  P0: μ=%.4f  σ=%.4f\n"
        "  P1: μ=%.4f  σ=%.4f\n"
        "  H_thr = %.4f   (μ₀ + %.2f·σ₀)\n"
        "  α          = %.6f\n"
        "  β_cl       = %.6f  ← classical (from distributions)\n"
        "  β_symmetric= %.6f  ← reference (α=β design)\n"
        "  Δβ         = %+.6f",
        P0_mu, P0_sig, P1_mu, P1_sig,
        H_thr, n_sig,
        alpha, beta_cl, alpha, beta_cl - alpha,
    )
    return beta_cl


# ── Resolve MaxEnt β according to MAXENT_BETA_MODE ──────────────────────────
if MAXENT_BETA_MODE == "classical":
    _beta_resolved = _compute_classical_beta(
        alpha    = _BASE_MAXENT["alpha"],
        base_cfg = _BASE_MAXENT,
    )
    _BASE_MAXENT = {**_BASE_MAXENT, "beta": _beta_resolved}
    _BASE_PARAMS["maxent"]       = _BASE_MAXENT
    _BASE_PARAMS["maxent_sprt"]  = _BASE_MAXENT
    logger.info("MAXENT_BETA_MODE='classical': β set to %.6f", _beta_resolved)
elif MAXENT_BETA_MODE == "symmetric":
    logger.info("MAXENT_BETA_MODE='symmetric': α=β=%.6f", _BASE_MAXENT["alpha"])
else:
    raise ValueError(
        f"Unknown MAXENT_BETA_MODE={MAXENT_BETA_MODE!r}. "
        "Use 'symmetric' or 'classical'."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Paired combo runner
# ═════════════════════════════════════════════════════════════════════════════

def _make_combo(N_win: int, step: int, K_total: int, n_accum: int) -> dict:
    """Build a minimal combo dict compatible with run_combo / build_indicator_config."""
    overlap = 1.0 - step / N_win if N_win > step else 0.0
    return {
        "K_total":      K_total,
        "N_win":        N_win,
        "step":         step,
        "n_accum":      n_accum,
        "overlap_frac": overlap,
    }


def _run_paired(
    indicator: str,
    signal: SignalData,
    K_lcm: int,
    T_lcm: float,
    basis_modal: StudyBasis,
    basis_rev: StudyBasis,
) -> dict:
    """
    Run one (indicator, K_lcm) pair for both modes.

    T_eff = K_lcm × T_lcm  is the **effective span** shared by both modes.
    Because T_lcm = lcm(T_modal, T_rev), T_eff is an exact integer multiple
    of both T_modal and T_rev — no rounding error.

    For RMS/SST (step=1):
        K_eff  = T_eff / T_unit          (exact integer)
        N_win  = K_eff - (n_accum - 1)   (derived, NOT freely chosen)
        → T_eff is identical for both modes by construction.

    For MaxEnt:
        K_total = K_eff (N_seg = K_eff, no n_accum concept).

    Returns a flat dict with results from both modes, ready for a DataFrame row.
    """
    T_modal    = basis_modal.T_modal
    T_rev      = basis_rev.T_rev

    # ── Target effective span (equal for both modes, exact) ──────────────────
    T_eff_des = K_lcm * T_lcm   # [s]

    step_modal = 1
    step_rev   = 1

    is_rms_sst = indicator.lower() in {"rms_cv", "sst_svd"}

    # ── K_eff per mode (exact because T_lcm is LCM-based) ───────────────────
    K_eff_modal = round(T_eff_des / T_modal)  # = 3 * K_lcm  (for f=150, rpm=12000)
    K_eff_rev   = round(T_eff_des / T_rev)    # = 4 * K_lcm

    if is_rms_sst:
        # N_win derived so that effective span = K_eff exactly
        n_accum_modal = max(2, int(N_ACCUM_FIXED_RMS_SST))
        n_accum_rev   = max(2, int(N_ACCUM_FIXED_RMS_SST))
        N_win_modal   = max(N_WIN_MIN_RMS_SST, K_eff_modal - (n_accum_modal - 1) * step_modal)
        N_win_rev     = max(N_WIN_MIN_RMS_SST, K_eff_rev   - (n_accum_rev   - 1) * step_rev)
        K_total_modal = K_eff_modal
        K_total_rev   = K_eff_rev
    else:
        # MaxEnt: no n_accum concept, K_total = N_seg = K_eff
        n_accum_modal = 1
        n_accum_rev   = 1
        N_win_modal   = K_eff_modal
        N_win_rev     = K_eff_rev
        K_total_modal = K_eff_modal
        K_total_rev   = K_eff_rev

    # Effective durations (exact by construction for LCM grid)
    T_win_des   = T_eff_des                     # alias for row reporting
    T_win_modal = N_win_modal * T_modal
    T_win_rev   = N_win_rev   * T_rev
    T_eff_modal = K_eff_modal * T_modal         # = T_eff_des (exact)
    T_eff_rev   = K_eff_rev   * T_rev           # = T_eff_des (exact)

    combo_modal = _make_combo(
        N_win_modal,
        step_modal,
        K_total=K_total_modal,
        n_accum=n_accum_modal,
    )
    combo_rev = _make_combo(
        N_win_rev,
        step_rev,
        K_total=K_total_rev,
        n_accum=n_accum_rev,
    )

    base = _BASE_PARAMS.get(indicator, _BASE_PARAMS.get("maxent", {}))

    config_modal = build_indicator_config(indicator, basis_modal, combo_modal, base)
    config_rev   = build_indicator_config(indicator, basis_rev,   combo_rev,   base)

    rr_modal: RunResult = run_combo(
        signal=signal,
        indicator_config=config_modal,
        indicator_id=indicator,
        t_gt=T_GT,
        T_unit=basis_modal.T_unit,
        K_total=K_total_modal,
        lam=LAMBDA,
        combo=combo_modal,
        basis_mode="by_modal",
    )
    rr_rev: RunResult = run_combo(
        signal=signal,
        indicator_config=config_rev,
        indicator_id=indicator,
        t_gt=T_GT,
        T_unit=basis_rev.T_unit,
        K_total=K_total_rev,
        lam=LAMBDA,
        combo=combo_rev,
        basis_mode="by_revolution",
    )

    row = {
        # ── Grid coordinates ────────────────────────────────────────────────
        "indicator":      indicator,
        "K_lcm":          K_lcm,
        "T_lcm_ms":       round(T_lcm * 1e3, 4),
        # ── Window sizes (units) ─────────────────────────────────────────────
        "N_win_modal":    N_win_modal,
        "step_modal":     step_modal,
        "n_accum_modal":  n_accum_modal,
        "K_total_modal":  K_total_modal,
        "N_win_rev":      N_win_rev,
        "step_rev":       step_rev,
        "n_accum_rev":    n_accum_rev,
        "K_total_rev":    K_total_rev,
        # ── Window durations [ms] ─────────────────────────────────────────────
        # T_eff_des = T_eff_modal = T_eff_rev  (exact by LCM grid)
        # T_win_*   = sub-window (N_win × T_unit, smaller than T_eff)
        "T_eff_des_ms":   round(T_win_des  * 1e3, 4),  # = T_eff_modal = T_eff_rev
        "T_eff_modal_ms": round(T_eff_modal * 1e3, 4),
        "T_eff_rev_ms":   round(T_eff_rev   * 1e3, 4),
        "T_win_modal_ms": round(T_win_modal * 1e3, 4),
        "T_win_rev_ms":   round(T_win_rev   * 1e3, 4),
        # ── by_modal results ─────────────────────────────────────────────────
        # t_d_first_modal / delta_td use t_d_first_true (first detection
        # AFTER T_GT, skipping false alarms) so Δt_d is a true latency.
        "detected_modal":   int(not np.isnan(rr_modal.t_d_first_true)),
        "t_d_first_modal":  round(rr_modal.t_d_first_true, 6) if not np.isnan(rr_modal.t_d_first_true) else None,
        "delta_td_modal_ms":round((rr_modal.t_d_first_true - T_GT) * 1e3, 2) if not np.isnan(rr_modal.t_d_first_true) else None,
        "N_fa_modal":       rr_modal.N_fa,
        "score_modal":      round(rr_modal.score, 5)       if not np.isnan(rr_modal.score) else None,
        "run_ok_modal":     int(rr_modal.run_ok),
        # ── by_revolution results ─────────────────────────────────────────────
        "detected_rev":     int(not np.isnan(rr_rev.t_d_first_true)),
        "t_d_first_rev":    round(rr_rev.t_d_first_true, 6)   if not np.isnan(rr_rev.t_d_first_true) else None,
        "delta_td_rev_ms":  round((rr_rev.t_d_first_true - T_GT) * 1e3, 2) if not np.isnan(rr_rev.t_d_first_true) else None,
        "N_fa_rev":         rr_rev.N_fa,
        "score_rev":        round(rr_rev.score, 5)         if not np.isnan(rr_rev.score) else None,
        "run_ok_rev":       int(rr_rev.run_ok),
    }

    traces_modal = {
        "t":            rr_modal.arrays.get("t_indicator", np.array([])),
        "I_t":          rr_modal.arrays.get("I_t",         np.array([])),
        "t_d_array":    rr_modal.arrays.get("t_d_array",   np.array([])),
        "t_d_true":     rr_modal.t_d_first_true  if not np.isnan(rr_modal.t_d_first_true)  else None,
        "K_lcm":        K_lcm,
        "T_eff_ms":     round(T_eff_des * 1e3, 2),
        "N_win":        N_win_modal,
        "step":         step_modal,
        "n_accum":      n_accum_modal,
        "mode":         "by_modal",
        "indicator":    indicator,
        "meta":         dict(rr_modal.meta),
    }
    traces_rev = {
        "t":            rr_rev.arrays.get("t_indicator", np.array([])),
        "I_t":          rr_rev.arrays.get("I_t",         np.array([])),
        "t_d_array":    rr_rev.arrays.get("t_d_array",   np.array([])),
        "t_d_true":     rr_rev.t_d_first_true    if not np.isnan(rr_rev.t_d_first_true)    else None,
        "K_lcm":        K_lcm,
        "T_eff_ms":     round(T_eff_des * 1e3, 2),
        "N_win":        N_win_rev,
        "step":         step_rev,
        "n_accum":      n_accum_rev,
        "mode":         "by_revolution",
        "indicator":    indicator,
        "meta":         dict(rr_rev.meta),
    }
    return row, traces_modal, traces_rev


# ═════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═════════════════════════════════════════════════════════════════════════════

def run_phase1_sweep() -> pd.DataFrame:
    """
    Execute the full Phase 1 paired sweep.

    Returns
    -------
    pd.DataFrame
        One row per (indicator, K_lcm) with side-by-side results.
    """
    T_modal = 1.0 / F_MODAL
    T_rev   = 60.0 / RPM
    T_lcm   = _compute_T_lcm(F_MODAL, RPM)

    basis_modal = StudyBasis("by_modal",     f_modal=F_MODAL, rpm=RPM, maxent_opr_valid=True)
    basis_rev   = StudyBasis("by_revolution", f_modal=F_MODAL, rpm=RPM, maxent_opr_valid=True)

    logger.info("=" * 70)
    logger.info("  PHASE 1 — Modal vs Revolution Comparison  (neutral LCM grid)")
    logger.info("  F_modal = %.1f Hz  |  T_modal = %.4f ms", F_MODAL, T_modal * 1e3)
    logger.info("  RPM     = %.0f     |  T_rev   = %.4f ms", RPM,     T_rev   * 1e3)
    logger.info("  T_lcm   = %.4f ms  (gcd-based, neutral grid unit)", T_lcm * 1e3)
    logger.info("  K_LCM grid: %s  →  T_des: %.0f … %.0f ms",
                K_LCM_GRID, K_LCM_GRID[0] * T_lcm * 1e3, K_LCM_GRID[-1] * T_lcm * 1e3)
    logger.info("  Indicators: %s", INDICATORS)
    logger.info("=" * 70)

    rows   = []
    traces = []   # list of (traces_modal, traces_rev) per run
    total = len(INDICATORS) * len(K_LCM_GRID)
    done  = 0

    for ind in INDICATORS:
        chan_key = ind if ind in SIGNAL_CHANNEL else "maxent"
        channel  = SIGNAL_CHANNEL.get(chan_key, "velocity")
        signal   = _load_signal(channel)

        for K in K_LCM_GRID:
            done += 1
            logger.info("[%d/%d]  %-10s  K_lcm=%2d  T_des=%.1f ms",
                        done, total, ind.upper(), K, K * T_lcm * 1e3)
            try:
                row, tr_m, tr_r = _run_paired(
                    indicator=ind,
                    signal=signal,
                    K_lcm=K,
                    T_lcm=T_lcm,
                    basis_modal=basis_modal,
                    basis_rev=basis_rev,
                )
                rows.append(row)
                traces.append((tr_m, tr_r))

                if DEBUG_LEVEL >= 1:
                    _log_row(row)

            except Exception as exc:
                logger.error("  ERROR  %s K_lcm=%d: %s", ind, K, exc)
                rows.append({
                    "indicator": ind, "K_lcm": K,
                    "run_ok_modal": 0, "run_ok_rev": 0,
                })
                traces.append((None, None))

    df = pd.DataFrame(rows)
    return df, traces


def _log_row(row: dict) -> None:
    """Print a one-line summary for one paired run."""
    ind = row.get("indicator", "?").upper()
    K   = row.get("K_lcm",     "?")
    T   = row.get("T_eff_des_ms", float("nan"))
    d_m = row.get("delta_td_modal_ms")
    d_r = row.get("delta_td_rev_ms")
    nfa_m = row.get("N_fa_modal", "?")
    nfa_r = row.get("N_fa_rev",   "?")

    def _fmt(val):
        return f"{val:+.1f} ms" if val is not None else "  --   "

    logger.info(
        "  %-10s K_lcm=%2d  T_eff=%7.3f ms  |  "
        "modal: Δtd=%s Nfa=%s  |  rev: Δtd=%s Nfa=%s",
        ind, K, T, _fmt(d_m), nfa_m, _fmt(d_r), nfa_r,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Summary & save
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    """Print a comparison table: modal vs revolution side-by-side."""
    sep = "=" * 100
    print(f"\n{sep}")
    print("  PHASE 1 — MODAL vs REVOLUTION  (neutral LCM grid, side-by-side)")
    print(sep)

    for ind in df["indicator"].unique():
        sub = df[df["indicator"] == ind].copy()
        print(f"\n{'─' * 100}")
        print(f"  Indicator: {ind.upper()}")
        print(f"{'─' * 100}")
        dtd_mod_hdr = "\u0394td_mod ms"
        dtd_rev_hdr = "\u0394td_rev ms"
        header = (
            f"  {'K_lcm':>5}  {'T_eff ms':>9}  "
            f"{'Nwin_m':>6}  {'Twin_m ms':>9}  {dtd_mod_hdr:>11}  {'Nfa_mod':>7}  "
            f"{'Nwin_r':>6}  {'Twin_r ms':>9}  {dtd_rev_hdr:>11}  {'Nfa_rev':>7}  "
            f"{'Better':>8}"
        )
        print(header)
        print(f"  {'-' * 96}")

        for _, r in sub.iterrows():
            if pd.isna(r.get("T_eff_des_ms", np.nan)):
                continue

            dtd_m = r.get("delta_td_modal_ms")
            dtd_r = r.get("delta_td_rev_ms")
            s_m   = r.get("score_modal")
            s_r   = r.get("score_rev")
            det_m = r.get("detected_modal", 0)
            det_r = r.get("detected_rev",   0)

            def _fv(v):
                return f"{v:+8.1f}" if v is not None else "      --"

            better = "="
            if s_m is not None and s_r is not None:
                if s_m < s_r - 1e-9:
                    better = "MODAL"
                elif s_r < s_m - 1e-9:
                    better = "REV"
            elif det_m and not det_r:
                better = "MODAL"
            elif det_r and not det_m:
                better = "REV"
            elif not det_m and not det_r:
                better = "NONE"

            print(
                f"  {int(r['K_lcm']):>5}  {r['T_eff_des_ms']:>9.3f}  "
                f"{int(r.get('N_win_modal', 0)):>6}  {r.get('T_win_modal_ms', 0):>9.3f}  "
                f"{_fv(dtd_m):>11}  {int(r.get('N_fa_modal', 0)):>7}  "
                f"{int(r.get('N_win_rev', 0)):>6}  {r.get('T_win_rev_ms', 0):>9.3f}  "
                f"{_fv(dtd_r):>11}  {int(r.get('N_fa_rev', 0)):>7}  "
                f"{better:>8}"
            )

    print(f"\n{sep}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

_COL_MODAL = "#1f77b4"   # blue
_COL_REV   = "#ff7f0e"   # orange
_COL_GT    = "#d62728"   # red   — T_GT line
_COL_TD    = "#2ca02c"   # green — detection line
_COL_FA    = "#9467bd"   # purple — false alarms

# ── Publication style (mirrors configurar_estilo_global in rms_cv_plots) ────
_PLOT_STYLE: dict = {
    # Tipografía general
        'font.family': 'serif',
        'font.size': 9,

        # Tamaños de títulos y etiquetas
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,

        # Estética de líneas
        'lines.linewidth': 1.25,
        'lines.markersize': 6,

        # Bordes y ejes
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,

        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        # Texto matemático
        'mathtext.fontset': 'stix',
        'axes.formatter.use_mathtext': True,

        # Leyenda
        'legend.frameon': False,
        'legend.loc': 'best',
        'legend.handlelength': 2.0,
        'legend.borderaxespad': 0.5,

        # Exportación
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.transparent': True,

        # Fondo
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
}

def _fig_size(ncols: int = 2, scale: float = 1.0, base_width: float = 3.4) -> tuple:
    """IEEE/Elsevier compatible figure size (mirrors fig_size in rms_cv_plots)."""
    w = base_width * ncols * scale
    return (w, w * 0.65)


def _get_threshold(tr: dict) -> float | None:
    """Extract detection threshold from trace meta (indicator-agnostic).

    Key mapping per indicator:
      rms_cv   → meta["cv_threshold_used"]          (float)
      maxent   → meta["sprt_result"].b              (attribute of SPRTResult object)
      sst_svd  → meta["lim_sup"]                    (float, upper sigma bound)
      fallback → meta["threshold"]
    """
    meta = tr.get("meta", {})

    # rms_cv
    v = meta.get("cv_threshold_used", None)
    if v is not None:
        try:
            val = float(v)
            if np.isfinite(val):
                return val
        except (TypeError, ValueError):
            pass

    # maxent — threshold stored as attribute .b on the SPRTResult object
    sprt = meta.get("sprt_result", None)
    if sprt is not None:
        b = getattr(sprt, "b", None)
        if b is not None:
            try:
                val = float(b)
                if np.isfinite(val):
                    return val
            except (TypeError, ValueError):
                pass

    # sst_svd — upper sigma bound
    v = meta.get("lim_sup", None)
    if v is not None:
        try:
            val = float(v)
            if np.isfinite(val):
                return val
        except (TypeError, ValueError):
            pass

    # generic fallback
    v = meta.get("threshold", None)
    if v is not None:
        try:
            val = float(v)
            if np.isfinite(val):
                return val
        except (TypeError, ValueError):
            pass

    if not meta:
        logger.debug("_get_threshold: meta empty for mode=%s K_lcm=%s",
                     tr.get("mode"), tr.get("K_lcm"))
    else:
        logger.debug("_get_threshold: no threshold key found. Indicator=%s mode=%s Keys=%s",
                     tr.get("indicator"), tr.get("mode"),
                     [k for k in meta if not isinstance(meta[k], np.ndarray)])
    return None


def _draw_run(
    ax: "plt.Axes",
    tr_m: dict,
    tr_r: dict,
    t_lim: tuple,
) -> None:
    """Draw one K_lcm run onto ax (modal + rev overlaid, thresholds, markers)."""
    ax.cla()

    K   = tr_m["K_lcm"]
    Tms = tr_m["T_eff_ms"]
    ind = tr_m["indicator"]
    Nref_m = tr_m["N_win"];  dtp_m = tr_m.get("step", 1);  Nfen_m = tr_m["n_accum"]
    Nref_r = tr_r["N_win"];  dtp_r = tr_r.get("step", 1);  Nfen_r = tr_r["n_accum"]

    ax.set_title(
        f"{ind.upper()}   $N_{{\\mathrm{{cyc,LCM}}}}={K}$   "
        f"$T_{{\\mathrm{{eff}}}}={Tms:.0f}$ ms\n"
        f"modal:  $N_{{\\mathrm{{win}}}}={Nref_m}$  "
        f"$\\Delta T_{{\\mathrm{{pas}}}}={dtp_m}$  "
        f"$N_{{\\mathrm{{fen}}}}={Nfen_m}$"
        f"          "
        f"rev:  $N_{{\\mathrm{{win}}}}={Nref_r}$  "
        f"$\\Delta T_{{\\mathrm{{pas}}}}={dtp_r}$  "
        f"$N_{{\\mathrm{{fen}}}}={Nfen_r}$",
        pad=5,
    )

    # ── collect y-values to compute explicit ylim ────────────────────────────
    all_y: list[float] = []

    for tr, color, lbl in [
        (tr_m, _COL_MODAL, "modal"),
        (tr_r, _COL_REV,   "rev"),
    ]:
        t_vec = tr["t"]
        I_vec = tr["I_t"]
        t_d   = tr["t_d_array"]

        if len(t_vec) == 0 or len(I_vec) == 0:
            ax.text(0.5, 0.5, f"NO DATA ({lbl})",
                    transform=ax.transAxes, ha="center", va="center",
                    color=color)
            continue

        finite_I = I_vec[np.isfinite(I_vec)]
        if len(finite_I):
            all_y.extend([float(finite_I.min()), float(finite_I.max())])

        # markers every ~15 visible points
        _n_marks = max(1, len(t_vec) // 100)
        ax.plot(
            t_vec, I_vec,
            color=color, alpha=0.88,
            marker="o", markersize=3.5,
            # markevery=_n_marks,
            markeredgewidth=0.3, markeredgecolor="black",
            zorder=3, label=lbl,
        )

        # ── threshold line (always draw — one per mode) ──────────────────────
        thresh = _get_threshold(tr)
        if thresh is not None:
            all_y.append(thresh)
            ax.axhline(
                thresh, color=color, lw=1.2, ls="--", alpha=0.85, zorder=4,
                label=f"$\\hat{{h}}$ ({lbl}) $= {thresh:.4g}$",
            )
        else:
            logger.warning("No threshold found for %s K_lcm=%s", lbl, K)

        # ── detection markers ────────────────────────────────────────────────
        if len(t_d) > 0:
            fa = t_d[t_d < T_GT - 1e-6]
            tp = t_d[t_d >= T_GT - 1e-6]
            if len(fa) > 0:
                yfa = np.interp(fa, t_vec, I_vec)
                ax.scatter(fa, yfa, marker="x", s=45, color=color,
                           linewidths=1.8, zorder=6,
                           label=f"FA ({lbl})")
            if len(tp) > 0:
                ax.axvline(tp[0], color=color, lw=1.0, ls=":",
                           alpha=0.9, zorder=5)
                ytd = np.interp(tp[0], t_vec, I_vec)
                ax.scatter([tp[0]], [ytd], marker="^", s=55,
                           color=color, zorder=7,
                           edgecolors="white", linewidths=0.4,
                           label=f"det ({lbl})  $t_d={tp[0]:.3f}$ s")

    # T_GT vertical line
    ax.axvline(T_GT, color=_COL_GT, lw=1.4, ls="--", zorder=4,
               label=f"$T_{{\\mathrm{{GT}}}}={T_GT:.3f}$ s")

    # ── explicit ylim that INCLUDES threshold lines ──────────────────────────
    if all_y:
        ylo_d, yhi_d = min(all_y), max(all_y)
        margin = 0.10 * (yhi_d - ylo_d) if yhi_d > ylo_d else max(0.1 * abs(yhi_d), 1e-9)
        ax.set_ylim(ylo_d - margin, yhi_d + margin)

    if t_lim[1] > t_lim[0]:
        ax.set_xlim(t_lim)

    ax.set_xlabel("$t$  [s]")
    ax.set_ylabel(ind.upper())
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0),
                        useMathText=True)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(axis="both", which="major", direction="in",
                   length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in",
                   length=2.5, width=0.6)
    ax.minorticks_on()

    leg = ax.legend(loc="upper left", handlelength=2.0,
                    borderaxespad=0.4, labelspacing=0.35)
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_alpha(0.85)

def fig_size(scale=1.0, ncols=1, base_width=3.4):
    """Return a Matplotlib-compatible figure size tuple.

    Computes width and height so that figures fit the standard column widths
    used by IEEE and Elsevier journals.  The height is always 70 % of the
    computed width.

    Args:
        scale (float, optional): Global scaling factor applied to both
            dimensions.  ``1.0`` gives the nominal journal column width.
            Defaults to ``1.0``.
        ncols (int, optional): Number of journal columns the figure should
            span (``1`` = single-column, ``2`` = double-column).  Defaults
            to ``1``.
        base_width (float, optional): Width [inches] of a single journal
            column.  Defaults to ``3.4`` (IEEE single-column).

    Returns:
        tuple[float, float]: ``(width, height)`` in inches.

    Example:
        >>> fig_size(scale=1.5, ncols=2)
        (10.2, 7.140000000000001)
    """
    width = base_width * ncols * scale
    height = width * 0.7   # relación agradable
    return (width, height)

def plot_traces(
    df: pd.DataFrame,
    traces: list,
    t_lim: tuple = (0.0, -1.0),
    zoom_x: tuple | None = None,
    zoom_y: tuple | None = None,
    zoom_map: dict | None = None,
) -> None:
    """
    Interactive navigator — one figure per indicator.
    A single axes shows one K_lcm run at a time.
    Use the Prev / Next buttons (or ← → keys) to navigate between runs.
    Detection thresholds are drawn as dashed horizontal lines.

    zoom_x / zoom_y : tuple (lo, hi) applied when the "Zoom" button is
    pressed.  R resets to data-driven defaults.  If None the button is hidden.
    """
    from matplotlib.widgets import Button

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ind_groups: dict[str, list] = {}
    for tr_m, tr_r in traces:
        if tr_m is None:
            continue
        ind_groups.setdefault(tr_m["indicator"], []).append((tr_m, tr_r))

    for ind, pairs in ind_groups.items():
        # resolve per-indicator zoom (zoom_map overrides global zoom_x/zoom_y)
        if zoom_map and ind in zoom_map:
            cur_zoom_x, cur_zoom_y = zoom_map[ind]
        else:
            cur_zoom_x, cur_zoom_y = zoom_x, zoom_y

        n_runs  = len(pairs)
        state   = {"idx": 0}
        has_btn = cur_zoom_x is not None or cur_zoom_y is not None
        _zoom: dict = {"xlim": None, "ylim": None}

        plt.rcParams.update(_PLOT_STYLE)
        fig = plt.figure(figsize=fig_size(scale=4.0, ncols=1))

        # main axes (leave room at bottom for buttons)
        ax = fig.add_axes([0.07, 0.18, 0.90, 0.72])

        # ── button axes ──────────────────────────────────────────────────────
        ax_prev = fig.add_axes([0.30, 0.04, 0.12, 0.06])
        ax_next = fig.add_axes([0.58, 0.04, 0.12, 0.06])
        ax_info = fig.add_axes([0.44, 0.04, 0.12, 0.06])
        ax_info.axis("off")
        counter_text = ax_info.text(
            0.5, 0.5, f"1 / {n_runs}",
            ha="center", va="center", transform=ax_info.transAxes,
        )
        btn_prev = Button(ax_prev, "\u25c4  Prev",  color="#e8e8e8", hovercolor="#c8c8c8")
        btn_next = Button(ax_next, "Next  \u25ba",  color="#e8e8e8", hovercolor="#c8c8c8")

        def _refresh() -> None:
            i = state["idx"]
            tr_m, tr_r = pairs[i]
            _draw_run(ax, tr_m, tr_r, t_lim)      # sets data-driven limits
            if _zoom["xlim"] is not None:
                ax.set_xlim(_zoom["xlim"])
            if _zoom["ylim"] is not None:
                ax.set_ylim(_zoom["ylim"])
            counter_text.set_text(f"{i + 1} / {n_runs}")
            fig.canvas.draw()                      # synchronous

        def _on_prev(_event) -> None:
            state["idx"] = (state["idx"] - 1) % n_runs
            _refresh()

        def _on_next(_event) -> None:
            state["idx"] = (state["idx"] + 1) % n_runs
            _refresh()

        def _on_key(event) -> None:
            if event.key in ("right", "n"):
                _on_next(None)
            elif event.key in ("left", "p"):
                _on_prev(None)
            elif event.key in ("r", "R"):
                _on_reset_btn(None)

        btn_prev.on_clicked(_on_prev)
        btn_next.on_clicked(_on_next)
        fig.canvas.mpl_connect("key_press_event", _on_key)
        fig._btn_prev_ref = btn_prev   # type: ignore[attr-defined]
        fig._btn_next_ref = btn_next   # type: ignore[attr-defined]

        # ── Zoom + Reset zoom buttons ────────────────────────────────
        if has_btn:
            ax_zbtn = fig.add_axes([0.15, 0.04, 0.13, 0.06])
            btn_zoom = Button(ax_zbtn, "Zoom", hovercolor="#c6dbef")
            btn_zoom.label.set_fontsize(8)

            def _on_zoom_btn(_event, zx=cur_zoom_x, zy=cur_zoom_y) -> None:
                if zx is not None:
                    _zoom["xlim"] = list(zx)
                if zy is not None:
                    _zoom["ylim"] = list(zy)
                if _zoom["xlim"] is not None:
                    ax.set_xlim(_zoom["xlim"])
                if _zoom["ylim"] is not None:
                    ax.set_ylim(_zoom["ylim"])
                fig.canvas.draw()

            btn_zoom.on_clicked(_on_zoom_btn)
            fig._btn_zoom_ref = btn_zoom  # type: ignore[attr-defined]

        ax_rbtn = fig.add_axes([0.72, 0.04, 0.13, 0.06])
        btn_reset = Button(ax_rbtn, "Reset zoom", hovercolor="#fde0c8")
        btn_reset.label.set_fontsize(8)

        def _on_reset_btn(_event) -> None:
            _zoom["xlim"] = None
            _zoom["ylim"] = None
            _refresh()

        btn_reset.on_clicked(_on_reset_btn)
        fig._btn_reset_ref = btn_reset  # type: ignore[attr-defined]

        _refresh()   # draw first run
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"fig_traces_{ind}_K{pairs[0][0]['K_lcm']}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Traces snapshot saved: %s", out_path)

        plt.show(block=False)   # non-blocking — stays open alongside other figures



def plot_metrics(df: pd.DataFrame, show_score: bool = False) -> None:
    """
    Summary metrics vs K_lcm — publication style.

    Subplots (stacked, shared X):
      1. Δt_d [ms]    — detection latency  (modal vs rev)
      2. N_fa         — false-alarm count
      3. score        — only shown when show_score=True

    X ticks: K_lcm values, secondary labels show T_eff [ms].
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ind in df["indicator"].unique():
        sub   = df[df["indicator"] == ind].sort_values("K_lcm")
        K     = sub["K_lcm"].values
        T_eff = pd.to_numeric(sub["T_eff_des_ms"], errors="coerce").values

        dtd_m = pd.to_numeric(sub["delta_td_modal_ms"], errors="coerce").values
        dtd_r = pd.to_numeric(sub["delta_td_rev_ms"],   errors="coerce").values
        nfa_m = pd.to_numeric(sub["N_fa_modal"],         errors="coerce").values
        nfa_r = pd.to_numeric(sub["N_fa_rev"],           errors="coerce").values
        sc_m  = pd.to_numeric(sub["score_modal"],        errors="coerce").values
        sc_r  = pd.to_numeric(sub["score_rev"],          errors="coerce").values

        n_rows = 3 if show_score else 2
        plt.rcParams.update(_PLOT_STYLE)
        fig, axs = plt.subplots(
            n_rows, 1,
            figsize=_fig_size(ncols=1, scale=3.0),
            sharex=True,
            constrained_layout=True,
        )
        if n_rows == 2:
            axs = list(axs)

        fig.suptitle(
            f"{ind.upper()} — $\\Delta t_d$,  $N_{{FA}}$"
            + ("  ,  score" if show_score else "")
            + "  vs  $K_{{\\mathrm{{LCM}}}}$",
             fontweight="bold",
        )

        # ── subplot 1: Δt_d ───────────────────────────────────────────────
        ax = axs[0]
        ax.axhline(0, color="gray", lw=0.7, ls="--", zorder=1)
        ax.plot(K, dtd_m, "o-", color=_COL_MODAL, lw=1.4, ms=5,
                markeredgecolor="white", markeredgewidth=0.4,
                label="modal", zorder=3)
        ax.plot(K, dtd_r, "s-", color=_COL_REV,   lw=1.4, ms=5,
                markeredgecolor="white", markeredgewidth=0.4,
                label="revolución", zorder=3)
        ax.set_ylabel("$\\Delta t_d$  [ms]")
        ax.legend(loc="best", handlelength=1.8, labelspacing=0.3,
                  frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
        ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.6)
        ax.minorticks_on()

        # ── subplot 2: N_fa ───────────────────────────────────────────────
        ax = axs[1]
        w = 0.32
        ax.bar(K - w/2, nfa_m, width=w, color=_COL_MODAL, alpha=0.80,
               label="modal",      zorder=3)
        ax.bar(K + w/2, nfa_r, width=w, color=_COL_REV,   alpha=0.80,
               label="revolución", zorder=3)
        ax.set_ylabel("$N_{FA}$  [—]")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.legend(loc="best", handlelength=1.4, labelspacing=0.3,
                  frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
        ax.tick_params(axis="y", which="minor", direction="in", length=2.5, width=0.6)
        ax.minorticks_on()

        # ── subplot 3: score (optional) ───────────────────────────────────
        if show_score:
            ax = axs[2]
            ax.plot(K, sc_m, "o-", color=_COL_MODAL, lw=1.4, ms=5,
                    markeredgecolor="white", markeredgewidth=0.4,
                    label="modal", zorder=3)
            ax.plot(K, sc_r, "s-", color=_COL_REV,   lw=1.4, ms=5,
                    markeredgecolor="white", markeredgewidth=0.4,
                    label="revolución", zorder=3)
            ax.set_ylabel("score  [s]")
            ax.legend(loc="best", handlelength=1.8, labelspacing=0.3,
                      frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", direction="in",
                           length=4, width=0.8)
            ax.tick_params(axis="both", which="minor", direction="in",
                           length=2.5, width=0.6)
            ax.minorticks_on()

        # ── shared X axis ─────────────────────────────────────────────────
        axs[-1].set_xlabel("$K_{\\mathrm{LCM}}$")
        axs[-1].set_xticks(K)
        axs[-1].set_xticklabels([str(k) for k in K])

        # secondary X: T_eff labels on top of first subplot
        ax0_twin = axs[0].twiny()
        ax0_twin.set_xlim(axs[0].get_xlim())
        ax0_twin.set_xticks(K)
        ax0_twin.set_xticklabels(
            [f"{t:.0f}" for t in T_eff], rotation=45, ha="left"
        )
        ax0_twin.set_xlabel("$T_{\\mathrm{eff}}$  [ms]")
        ax0_twin.tick_params(axis="x", direction="in", length=3, width=0.6)

        out_path = os.path.join(OUTPUT_DIR, f"fig_metrics_{ind}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Metrics figure saved → %s", out_path)
        plt.show()
        plt.close(fig)


def plot_latency_scatter(df: pd.DataFrame) -> None:
    """
    Figure 1 — Scatter: detection latency Δt_d vs T_eff [ms].

    One marker per (indicator, K_lcm, mode).  Modal = circle, rev = square.
    Horizontal dashed line at Δt_d = 0.  Markers annotated with K value.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ind in df["indicator"].unique():
        sub   = df[df["indicator"] == ind].sort_values("K_lcm")
        T_eff = pd.to_numeric(sub["T_eff_des_ms"],      errors="coerce").values
        dtd_m = pd.to_numeric(sub["delta_td_modal_ms"], errors="coerce").values
        dtd_r = pd.to_numeric(sub["delta_td_rev_ms"],   errors="coerce").values
        K     = sub["K_lcm"].values

        plt.rcParams.update(_PLOT_STYLE)
        fig, ax = plt.subplots(figsize=_fig_size(ncols=2, scale=1.0))

        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1,
                   label="$\\Delta t_d = 0$")

        ax.scatter(T_eff, dtd_m, marker="o", s=52, color=_COL_MODAL, zorder=4,
                   edgecolors="white", linewidths=0.5, label="modal")
        ax.scatter(T_eff, dtd_r, marker="s", s=52, color=_COL_REV,   zorder=4,
                   edgecolors="white", linewidths=0.5, label="revolución")

        # connect paired points with a thin gray line
        for t, ym, yr in zip(T_eff, dtd_m, dtd_r):
            if np.isfinite(ym) and np.isfinite(yr):
                ax.plot([t, t], [ym, yr], color="gray", lw=0.6,
                        ls=":", zorder=2, alpha=0.6)

        # annotate K value next to each modal marker
        for t, ym, k in zip(T_eff, dtd_m, K):
            if np.isfinite(ym):
                ax.annotate(
                    f"$K={k}$",
                    xy=(t, ym), xytext=(3, 4),
                    textcoords="offset points",
                     color=_COL_MODAL, va="bottom",
                )

        ax.set_xlabel("$T_{\\mathrm{eff}}$  [ms]")
        ax.set_ylabel("$\\Delta t_d$  [ms]")
        ax.set_title(
            f"{ind.upper()} — latencia de detección vs ventana efectiva",
            pad=5,
        )
        ax.legend(loc="best", handlelength=1.8, labelspacing=0.3,
                   frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", direction="in",
                       length=4, width=0.8)
        ax.tick_params(axis="both", which="minor", direction="in",
                       length=2.5, width=0.6)
        ax.minorticks_on()

        out_path = os.path.join(OUTPUT_DIR, f"fig_latency_scatter_{ind}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Latency scatter saved → %s", out_path)
        plt.show()
        plt.close(fig)


def plot_detection_timeline(df: pd.DataFrame) -> None:
    """
    Figure 3 — Horizontal detection timeline.

    One row per K_lcm.  T_GT is a vertical red line.
    Modal t_d = blue circle, rev t_d = orange square.
    Rows where detection was missed are marked with ✕.

    Right-hand panel: vertical "colorbar legend" showing N_win per mode for
    each K_lcm row (modal | revolución, coloured accordingly).
    """
    import matplotlib.patches as mpatches
    import matplotlib.colors as _mcolors
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # soft background tints for the legend panel cells
    _alpha = 0.22
    _white = np.array([1.0, 1.0, 1.0])
    _bg_m  = _alpha * np.array(_mcolors.to_rgb(_COL_MODAL)) + (1 - _alpha) * _white
    _bg_r  = _alpha * np.array(_mcolors.to_rgb(_COL_REV))   + (1 - _alpha) * _white

    for ind in df["indicator"].unique():
        sub = df[df["indicator"] == ind].sort_values("K_lcm", ascending=False)
        n   = len(sub)

        plt.rcParams.update(_PLOT_STYLE)
        fig = plt.figure(figsize=_fig_size(ncols=2, scale=1.0 + 0.15 * n))
        gs  = GridSpec(
            1, 2, figure=fig,
            width_ratios=[5, 0.9],
            wspace=0.04,
        )
        ax     = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1], sharey=ax)

        y_pos = np.arange(n)

        # ── main timeline ─────────────────────────────────────────────────
        for yi, (_, row) in enumerate(sub.iterrows()):
            td_m  = row.get("t_d_first_modal", None)
            td_r  = row.get("t_d_first_rev",   None)
            det_m = bool(row.get("detected_modal", 0))
            det_r = bool(row.get("detected_rev",   0))

            ax.axhline(yi, color="#dddddd", lw=0.7, zorder=1)

            if det_m and td_m is not None and np.isfinite(float(td_m)):
                ax.scatter(float(td_m), yi, marker="o", s=55,
                           color=_COL_MODAL, zorder=4,
                           edgecolors="white", linewidths=0.4)
            else:
                ax.scatter([-0.3], yi, marker="x", s=45,
                           color=_COL_MODAL, zorder=4, linewidths=1.4)

            if det_r and td_r is not None and np.isfinite(float(td_r)):
                ax.scatter(float(td_r), yi, marker="s", s=45,
                           color=_COL_REV, zorder=4,
                           edgecolors="white", linewidths=0.4)
            else:
                ax.scatter([-0.3], yi, marker="x", s=35,
                           color=_COL_REV, zorder=4, linewidths=1.2)

        ax.axvline(T_GT, color=_COL_GT, lw=1.4, ls="--", zorder=5)
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor=_COL_MODAL,
                       markersize=6, label="modal"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor=_COL_REV,
                       markersize=6, label="revolución"),
                Line2D([0], [0], color=_COL_GT, lw=1.4, ls="--",
                       label=f"$T_{{\\mathrm{{GT}}}}$"),
            ],
            loc="lower right", frameon=False, handlelength=1.6,
        )

        K_vals    = sub["K_lcm"].values
        Teff_vals = pd.to_numeric(sub["T_eff_des_ms"], errors="coerce").values
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"$K={k}$  ({t:.0f} ms)" for k, t in zip(K_vals, Teff_vals)],
            
        )
        ax.set_xlabel("$t$  [s]")
        ax.set_title(
            f"{ind.upper()} — instante de primera detección por configuración",
            pad=5,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", which="major", direction="in",
                       length=4, width=0.8)
        ax.tick_params(axis="x", which="minor", direction="in",
                       length=2.5, width=0.6)
        ax.minorticks_on()

        # ── N_win legend panel ────────────────────────────────────────────
        # Two virtual columns inside ax_leg: x ∈ [0,1) modal, x ∈ [1,2) rev
        cell_h = 0.44   # half-height of each cell

        for yi, (_, row) in enumerate(sub.iterrows()):
            nwm = int(row.get("N_win_modal", 0))
            nwr = int(row.get("N_win_rev",   0))

            # modal cell
            ax_leg.add_patch(mpatches.FancyBboxPatch(
                (0.02, yi - cell_h), 0.94, cell_h * 2,
                boxstyle="square,pad=0",
                facecolor=_bg_m, edgecolor="none", zorder=1,
            ))
            ax_leg.text(0.49, yi, str(nwm),
                        ha="center", va="center",
                        color=_COL_MODAL, fontweight="bold", zorder=2)

            # rev cell
            ax_leg.add_patch(mpatches.FancyBboxPatch(
                (1.04, yi - cell_h), 0.94, cell_h * 2,
                boxstyle="square,pad=0",
                facecolor=_bg_r, edgecolor="none", zorder=1,
            ))
            ax_leg.text(1.51, yi, str(nwr),
                        ha="center", va="center",
                        color=_COL_REV, fontweight="bold", zorder=2)

        # column headers at the top (just above the last row)
        y_top = n - 0.5 + 0.05
        ax_leg.text(0.49, y_top, "M", ha="center", va="bottom",
                    color=_COL_MODAL, fontweight="bold")
        ax_leg.text(1.51, y_top, "R", ha="center", va="bottom",
                    color=_COL_REV,   fontweight="bold")

        ax_leg.set_xlim(0, 2.06)
        ax_leg.set_ylim(ax.get_ylim())   # already shared, but be explicit
        ax_leg.set_xlabel(
            "$N_{\\mathrm{win}}$", labelpad=3,
        )
        ax_leg.set_yticks([])
        ax_leg.set_xticks([])
        for sp in ax_leg.spines.values():
            sp.set_visible(False)
        # thin separator line between the two axes
        ax_leg.axvline(1.03, color="#bbbbbb", lw=0.5, zorder=0)

        out_path = os.path.join(OUTPUT_DIR, f"fig_timeline_{ind}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Timeline figure saved → %s", out_path)
        plt.show()
        plt.close(fig)


def plot_score_heatmap(df: pd.DataFrame) -> None:
    """
    Figure 4 — Heatmap score: rows = K_lcm, columns = {modal, rev}.

    Colour = score value (lower = better).  Cells where detection failed
    are shown in light grey with a '—' annotation.
    """
    import matplotlib.colors as _mcolors

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ind in df["indicator"].unique():
        sub = df[df["indicator"] == ind].sort_values("K_lcm")
        K   = sub["K_lcm"].values
        n   = len(K)

        sc_m = pd.to_numeric(sub["score_modal"], errors="coerce").values
        sc_r = pd.to_numeric(sub["score_rev"],   errors="coerce").values
        det_m = sub["detected_modal"].astype(int).values
        det_r = sub["detected_rev"].astype(int).values

        # build 2-column matrix; NaN where not detected
        mat = np.column_stack([
            np.where(det_m, sc_m, np.nan),
            np.where(det_r, sc_r, np.nan),
        ])

        # shared scale across both columns
        valid = mat[np.isfinite(mat)]
        vmin  = float(valid.min()) if len(valid) else 0.0
        vmax  = float(valid.max()) if len(valid) else 1.0

        plt.rcParams.update(_PLOT_STYLE)
        fig, ax = plt.subplots(
            figsize=_fig_size(ncols=1, scale=1.0 + 0.06 * n),
        )

        cmap_hm = plt.get_cmap("RdYlGn_r")   # green=low score (good), red=high
        cmap_hm.set_bad(color="#e8e8e8")       # NaN cells in light grey

        im = ax.imshow(
            mat, aspect="auto", cmap=cmap_hm,
            vmin=vmin, vmax=vmax, origin="upper",
        )

        # cell annotations
        for row_i in range(n):
            for col_i, val in enumerate(mat[row_i]):
                if np.isfinite(val):
                    txt   = f"{val:.3g}"
                    tcolor = "white" if (val - vmin) / max(vmax - vmin, 1e-9) > 0.6 else "black"
                else:
                    txt, tcolor = "—", "#888888"
                ax.text(col_i, row_i, txt, ha="center", va="center",
                         color=tcolor)

        # colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label("score  [s]", fontsize=_PLOT_STYLE["axes.labelsize"])
        cbar.ax.tick_params(length=3, width=0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["modal", "revolución"])
        ax.set_yticks(np.arange(n))
        T_eff = pd.to_numeric(sub["T_eff_des_ms"], errors="coerce").values
        ax.set_yticklabels(
            [f"$K={k}$  ({t:.0f} ms)" for k, t in zip(K, T_eff)],
        )
        ax.set_title(
            f"{ind.upper()} — score  (verde = mejor)",
            pad=5,
        )
        ax.tick_params(axis="both", which="both", length=0)

        out_path = os.path.join(OUTPUT_DIR, f"fig_heatmap_{ind}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Heatmap figure saved → %s", out_path)
        plt.show()
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# Per-mode overview — one figure for modal, one for revolution
# All K_lcm curves overlaid, coloured by K value
# ═════════════════════════════════════════════════════════════════════════════

def plot_per_mode(
    df: pd.DataFrame,
    traces: list,
    t_lim: tuple = (0.0, -1.0),
    show_slider: bool = False,
    show_overview: bool = False,
    zoom_x: tuple | None = None,
    zoom_y: tuple | None = None,
    zoom_map: dict | None = None,
) -> None:
    """
    For each indicator produce two figures:
      • Figure A: all modal   traces overlaid (one curve per K_lcm)
      • Figure B: all rev     traces overlaid (one curve per K_lcm)

    Curves are coloured by K_lcm with a viridis colormap.
    A colorbar on the right replaces the legend — ticks show both K and N_win.
    Title replicates the detailed format of _draw_run (N_win, ΔT_pas, N_fen).

    If show_slider=True, an additional interactive slider figure is opened per
    (indicator, mode) where a horizontal slider lets the user scrub through
    each K_lcm run individually.  Style matches the overlaid figures.

    If show_overview=True, a 2-panel overview figure is opened: left panel
    shows all runs (current at alpha=1, others dimmed), right panel shows only
    the current run zoomed to zoom_x/zoom_y.

    zoom_x / zoom_y : tuple (lo, hi) shown when the "Zoom fix" button is
    pressed.  If None the button is not shown.
    """
    import matplotlib.cm as _cm
    import matplotlib.colors as _mcolors
    import matplotlib.colorbar as _mcbar

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ind_groups: dict = {}
    for tr_m, tr_r in traces:
        if tr_m is None:
            continue
        ind_groups.setdefault(tr_m["indicator"], []).append((tr_m, tr_r))

    for ind, pairs in ind_groups.items():
        # resolve per-indicator zoom (zoom_map overrides global zoom_x/zoom_y)
        if zoom_map and ind in zoom_map:
            cur_zoom_x, cur_zoom_y = zoom_map[ind]
        else:
            cur_zoom_x, cur_zoom_y = zoom_x, zoom_y

        n_runs  = len(pairs)
        K_vals  = [p[0]["K_lcm"] for p in pairs]
        K_min, K_max = K_vals[0], K_vals[-1]

        # discrete normalisation: each K gets a distinct solid-colour band
        boundaries = np.arange(K_min - 0.5, K_max + 1.5, 1.0)
        norm = _mcolors.BoundaryNorm(boundaries, ncolors=n_runs)
        cmap = _cm.get_cmap("viridis", n_runs)

        for mode_key, fig_label, tr_idx in [
            ("modal", "modal",      0),
            ("rev",   "revolución", 1),
        ]:
            plt.rcParams.update(_PLOT_STYLE)

            # wider figure to give room for the colorbar
            fig, ax = plt.subplots(figsize=_fig_size(scale=3, ncols=1))

            all_y: list[float] = []

            for i, pair in enumerate(pairs):
                tr    = pair[tr_idx]
                color = cmap(norm(tr["K_lcm"]))

                t_vec = tr["t"]
                I_vec = tr["I_t"]

                if len(t_vec) == 0 or len(I_vec) == 0:
                    continue

                finite_I = I_vec[np.isfinite(I_vec)]
                if len(finite_I):
                    all_y.extend([float(finite_I.min()), float(finite_I.max())])

                _n_marks = max(1, len(t_vec) // 100)
                ax.plot(
                    t_vec, I_vec,
                    color=color, alpha=0.82, lw=1.1,
                    marker="o", markersize=3.5,
                    markevery=_n_marks,
                    markeredgewidth=0.3, markeredgecolor="black",
                    zorder=3,
                )

                thresh = _get_threshold(tr)
                if thresh is not None:
                    all_y.append(thresh)
                    ax.axhline(thresh, color=color, lw=0.9, ls="--",
                               alpha=0.70, zorder=2)

            # T_GT vertical line
            ax.axvline(T_GT, color=_COL_GT, lw=1.4, ls="--", zorder=4,
                       label=f"$T_{{\\mathrm{{GT}}}}={T_GT:.3f}$ s")
            ax.legend(loc="upper left", handlelength=1.6,
                      borderaxespad=0.4, labelspacing=0.30, frameon=False)

            # ── colorbar replacing the per-curve legend ──────────────────────
            sm = _cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=28, shrink=0.92)
            cbar.set_label(
                f"$K_{{\\mathrm{{LCM}}}}$",
                fontsize=_PLOT_STYLE["axes.labelsize"],
            )
            # ticks at each K value — labels show K and N_win
            cbar.set_ticks(K_vals)
            cbar.set_ticklabels(
                [f"$K={tr['K_lcm']}$  ($N_{{\\mathrm{{win}}}}={tr['N_win']}$)"
                 for tr in [p[tr_idx] for p in pairs]],
                # fontsize=7,
            )
            cbar.ax.tick_params(length=3, width=0.6)

            # ── explicit ylim ────────────────────────────────────────────────
            if all_y:
                ylo_d, yhi_d = min(all_y), max(all_y)
                margin = 0.10 * (yhi_d - ylo_d) if yhi_d > ylo_d else max(0.1 * abs(yhi_d), 1e-9)
                ax.set_ylim(ylo_d - margin, yhi_d + margin)

            if t_lim[1] > t_lim[0]:
                ax.set_xlim(t_lim)

            # ── detailed title (same style as _draw_run) ─────────────────────
            # use the first trace to get representative step / n_accum
            tr0   = pairs[0][tr_idx]
            dtp0  = tr0.get("step", 1)
            Nfen0 = tr0["n_accum"]
            # check if all runs share the same step/n_accum
            same_step  = all(p[tr_idx].get("step", 1) == dtp0  for p in pairs)
            same_nfen  = all(p[tr_idx]["n_accum"]      == Nfen0 for p in pairs)
            step_str  = (f"$\\Delta T_{{\\mathrm{{pas}}}}={dtp0}$"
                         if same_step else "$\\Delta T_{{\\mathrm{{pas}}}}$: var.")
            nfen_str  = (f"$N_{{\\mathrm{{fen}}}}={Nfen0}$"
                         if same_nfen else "$N_{{\\mathrm{{fen}}}}$: var.")
            ax.set_title(
                f"{ind.upper()} — {fig_label}   "
                f"$K_{{\\mathrm{{LCM}}}}={K_min}\\ldots{K_max}$\n"
                f"{step_str}    {nfen_str}    "
                f"$N_{{\\mathrm{{win}}}}={pairs[0][tr_idx]['N_win']}\\ldots"
                f"{pairs[-1][tr_idx]['N_win']}$",
                pad=5,
            )

            ax.set_xlabel("$t$  [s]")
            ax.set_ylabel(ind.upper())
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0),
                                useMathText=True)

            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_linewidth(0.9)
            ax.spines["bottom"].set_linewidth(0.9)
            ax.tick_params(axis="both", which="major", direction="in",
                           length=4, width=0.8)
            ax.tick_params(axis="both", which="minor", direction="in",
                           length=2.5, width=0.6)
            ax.minorticks_on()

            plt.tight_layout()

            out_path = os.path.join(
                OUTPUT_DIR,
                f"fig_{mode_key}_{ind}_K{K_min}-{K_max}.png",
            )
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            logger.info("Per-mode figure saved → %s", out_path)
            plt.show(block=False)

            # ── optional slider figure ────────────────────────────────────
            if show_slider:
                _plot_mode_slider(
                    pairs=pairs,
                    tr_idx=tr_idx,
                    ind=ind,
                    mode_label=fig_label,
                    K_vals=K_vals,
                    t_lim=t_lim,
                    cmap=cmap,
                    norm=norm,
                    zoom_x=cur_zoom_x,
                    zoom_y=cur_zoom_y,
                )

            # ── optional overview figure ─────────────────────────────────
            if show_overview:
                _plot_mode_overview(
                    pairs=pairs,
                    tr_idx=tr_idx,
                    ind=ind,
                    mode_label=fig_label,
                    K_vals=K_vals,
                    t_lim=t_lim,
                    cmap=cmap,
                    norm=norm,
                    zoom_x=cur_zoom_x,
                    zoom_y=cur_zoom_y,
                )

            


def _plot_mode_slider(
    pairs: list,
    tr_idx: int,
    ind: str,
    mode_label: str,
    K_vals: list,
    t_lim: tuple,
    cmap,
    norm,
    zoom_x: tuple | None = None,
    zoom_y: tuple | None = None,
    show_neighbours: bool = True,
) -> None:
    """
    Interactive slider figure.

    Fixes vs. previous version:
      • Uses fig.canvas.draw() (synchronous) so ax.get_xlim/ylim always reflect
        the committed state — zoom is preserved going both forward AND backward.
      • ax.set_autoscale_on(False) after ax.cla() prevents matplotlib from
        overriding our saved limits when drawing new artists.
      • _zoom is initialised with the default limits, never None.
      • Optional "Zoom fix" button (shown only when zoom_x or zoom_y is given)
        jumps to user-specified window (parameters zoom_x, zoom_y).
      • show_neighbours: if True, previous and next runs are drawn faintly.
    """
    from matplotlib.widgets import Slider, Button

    n_runs = len(pairs)

    # ── compute global limits ─────────────────────────────────────────────
    all_y_global: list[float] = []
    all_x_global: list[float] = []
    for pair in pairs:
        tr = pair[tr_idx]
        t  = tr["t"]
        I  = tr["I_t"]
        finite_I = I[np.isfinite(I)] if len(I) else np.array([])
        if len(finite_I):
            all_y_global.extend([float(finite_I.min()), float(finite_I.max())])
        if len(t):
            all_x_global.extend([float(t.min()), float(t.max())])
        thresh = _get_threshold(tr)
        if thresh is not None:
            all_y_global.append(thresh)

    if all_y_global:
        _ylo = min(all_y_global)
        _yhi = max(all_y_global)
        _ym  = 0.10 * (_yhi - _ylo) if _yhi > _ylo else max(0.1 * abs(_yhi), 1e-9)
        y_lim_default = (_ylo - _ym, _yhi + _ym)
    else:
        y_lim_default = (0.0, 1.0)

    if t_lim[1] > t_lim[0]:
        x_lim_default = t_lim
    elif all_x_global:
        _xlo, _xhi = min(all_x_global), max(all_x_global)
        _xm = 0.02 * (_xhi - _xlo)
        x_lim_default = (_xlo - _xm, _xhi + _xm)
    else:
        x_lim_default = (0.0, 1.0)

    # zoom state — always valid tuples, never None
    _zoom = {"xlim": list(x_lim_default), "ylim": list(y_lim_default)}

    plt.rcParams.update(_PLOT_STYLE)

    has_btn = zoom_x is not None or zoom_y is not None
    fig_w, fig_h = _fig_size(scale=1.5, ncols=2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.9))
    fig.subplots_adjust(bottom=0.30, top=0.88, left=0.12, right=0.95)

    # ── render ────────────────────────────────────────────────────────────
    def _render(idx: int) -> None:
        # 1. Read current committed limits (from previous draw() or init)
        _zoom["xlim"] = list(ax.get_xlim())
        _zoom["ylim"] = list(ax.get_ylim())

        # 2. Clear and immediately disable autoscale so no drawing
        #    operation can override our saved limits.
        ax.cla()
        ax.set_autoscale_on(False)

        # 3. Apply limits BEFORE drawing — autoscale=False keeps them.
        ax.set_xlim(_zoom["xlim"])
        ax.set_ylim(_zoom["ylim"])

        tr    = pairs[idx][tr_idx]
        color = cmap(norm(tr["K_lcm"]))

        # ── neighbour runs (prev / next) drawn first, in the background ──
        _neighbours = [
            (idx - 1, (1.8, ":", 0.30)),   # prev: dashed,  low alpha
            (idx + 1, (1.8, "--", 0.25)),  # next: dashed,  lower alpha
        ]
        for nb_idx, (nb_lw, nb_ls, nb_alpha) in _neighbours:
            if nb_idx < 0 or nb_idx >= n_runs:
                continue
            nb_tr    = pairs[nb_idx][tr_idx]
            nb_color = cmap(norm(nb_tr["K_lcm"]))
            nb_t     = nb_tr["t"]
            nb_I     = nb_tr["I_t"]
            if len(nb_t) and len(nb_I):
                ax.plot(nb_t, nb_I, color=nb_color,
                        lw=nb_lw, ls=nb_ls, alpha=nb_alpha,
                        zorder=1)

        # ── current run (main, foreground) ───────────────────────────────
        t_vec = tr["t"]
        I_vec = tr["I_t"]
        if len(t_vec) and len(I_vec):
            _n_marks = max(1, len(t_vec) // 100)
            ax.plot(
                t_vec, I_vec,
                color=color, lw=1.4,
                marker="o", markersize=3.5,
                markevery=_n_marks,
                markeredgewidth=0.3, markeredgecolor="black",
                zorder=3,
            )

        thresh = _get_threshold(tr)
        if thresh is not None:
            ax.axhline(thresh, color=color, lw=1.2, ls="--",
                       alpha=0.85, zorder=2,
                       label=f"Limite = {thresh:.4g}")

        t_det = tr.get("t_d_true", None)
        if t_det is not None:
            ax.axvline(
                t_det, color="#2ca02c", lw=1.6, ls="-.", zorder=5,
                label=(
                    f"$t_d={t_det:.3f}$ s  "
                    f"($\\Delta t_d={(t_det - T_GT)*1e3:.1f}$ ms)"
                ),
            )

        ax.axvline(T_GT, color=_COL_GT, lw=1.4, ls="--", zorder=4,
                   label=f"$T_{{\\mathrm{{GT}}}}={T_GT:.3f}$ s")

        if thresh is not None or t_det is not None:
            ax.legend(loc="upper left", handlelength=1.6,
                      borderaxespad=0.4, labelspacing=0.3, frameon=False)

        ax.set_xlabel("$t$  [s]")
        ax.set_ylabel(ind.upper())
        ax.ticklabel_format(style="sci", axis="y",
                            scilimits=(0, 0), useMathText=True)

        dtp  = tr.get("step", 1)
        nfen = tr["n_accum"]
        nwin = tr["N_win"]
        K    = tr["K_lcm"]
        Teff = tr.get("T_eff_ms", "?")
        ax.set_title(
            f"{ind.upper()} — {mode_label}   "
            f"$K_{{\\mathrm{{LCM}}}}={K}$   "
            f"$N_{{\\mathrm{{win}}}}={nwin}$\n"
            f"$\\Delta T_{{\\mathrm{{pas}}}}={dtp}$    "
            f"$N_{{\\mathrm{{fen}}}}={nfen}$    "
            f"$T_{{\\mathrm{{eff}}}}={Teff}$ ms"
            f"   [{idx + 1}/{n_runs}]",
            pad=5,
        )

        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(axis="both", which="major", direction="in",
                       length=4, width=0.8)
        ax.tick_params(axis="both", which="minor", direction="in",
                       length=2.5, width=0.6)
        ax.minorticks_on()

        # 4. Synchronous draw — limits are committed before next event fires
        fig.canvas.draw()

    # First render with default limits explicitly set
    ax.set_xlim(x_lim_default)
    ax.set_ylim(y_lim_default)
    _render(0)
    # After first render, override saved zoom with true defaults
    # (first _render reads limits before any draw → may get 0..1 defaults)
    _zoom["xlim"] = list(x_lim_default)
    _zoom["ylim"] = list(y_lim_default)
    ax.set_xlim(_zoom["xlim"])
    ax.set_ylim(_zoom["ylim"])
    fig.canvas.draw()

    # ── Slider ────────────────────────────────────────────────────────────
    ax_slider = fig.add_axes([0.12, 0.09, 0.76, 0.09])
    slider = Slider(
        ax=ax_slider,
        label="",
        valmin=0,
        valmax=n_runs - 1,
        valinit=0,
        valstep=1,
        color=cmap(norm(K_vals[n_runs // 2])),
    )
    ax_slider.set_xticks(np.arange(n_runs))
    ax_slider.set_xticklabels([str(k) for k in K_vals], fontsize=8.5)
    ax_slider.tick_params(axis="x", length=5, direction="out", pad=2)
    ax_slider.tick_params(axis="y", which="both", length=0, labelleft=False)
    fig.text(0.50, 0.015, "$K_{\\mathrm{LCM}}$",
             ha="center", va="bottom",
             fontsize=_PLOT_STYLE["axes.labelsize"])

    def _fmt(i: int) -> str:
        return f"$K={K_vals[i]}$  [{i+1}/{n_runs}]"

    slider.valtext.set_text(_fmt(0))
    slider.valtext.set_fontsize(9)

    def _on_slider(val: float) -> None:
        idx = int(round(val))
        idx = max(0, min(n_runs - 1, idx))
        slider.valtext.set_text(_fmt(idx))
        _render(idx)

    slider.on_changed(_on_slider)

    # ── Zoom + Reset zoom buttons ───────────────────────────────────
    if has_btn:
        ax_btn = fig.add_axes([0.63, 0.2, 0.14, 0.055])
        btn = Button(ax_btn, "Zoom", hovercolor="#c6dbef")
        btn.label.set_fontsize(8)

        def _on_zoom_btn(_event) -> None:
            if zoom_x is not None:
                _zoom["xlim"] = list(zoom_x)
            if zoom_y is not None:
                _zoom["ylim"] = list(zoom_y)
            ax.set_xlim(_zoom["xlim"])
            ax.set_ylim(_zoom["ylim"])
            fig.canvas.draw()

        btn.on_clicked(_on_zoom_btn)
        fig._btn_zoom_ref = btn          # type: ignore[attr-defined]

    ax_breset = fig.add_axes([0.79, 0.2, 0.14, 0.055])
    btn_reset = Button(ax_breset, "Reset zoom", hovercolor="#fde0c8")
    btn_reset.label.set_fontsize(8)

    def _on_reset_btn(_event) -> None:
        _zoom["xlim"] = list(x_lim_default)
        _zoom["ylim"] = list(y_lim_default)
        ax.set_xlim(_zoom["xlim"])
        ax.set_ylim(_zoom["ylim"])
        fig.canvas.draw()

    btn_reset.on_clicked(_on_reset_btn)
    fig._btn_reset_ref = btn_reset       # type: ignore[attr-defined]

    # ── Keyboard: ← / → step; R reset zoom ───────────────────────────────
    def _on_key(event) -> None:
        cur = int(round(slider.val))
        if event.key == "right":
            slider.set_val(min(n_runs - 1, cur + 1))
        elif event.key == "left":
            slider.set_val(max(0, cur - 1))
        elif event.key in ("r", "R"):
            _on_reset_btn(None)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig._slider_ref = slider             # type: ignore[attr-defined]
    plt.show(block=False)


def _plot_mode_overview(
    pairs: list,
    tr_idx: int,
    ind: str,
    mode_label: str,
    K_vals: list,
    t_lim: tuple,
    cmap,
    norm,
    zoom_x: tuple | None = None,
    zoom_y: tuple | None = None,
) -> None:
    """
    2-panel overview figure (1 row × 2 columns):
      Left  — all runs overlaid at low alpha; current run at alpha=1 (full colour).
      Right — only the current run, zoomed to zoom_x / zoom_y.

    A horizontal slider at the bottom (same style as _plot_mode_slider) selects
    which run is «current».  Keyboard ← / → also work.
    """
    from matplotlib.widgets import Slider

    n_runs = len(pairs)

    # ── global limits (for left panel / default zoom) ─────────────────────
    all_y_global: list[float] = []
    all_x_global: list[float] = []
    for pair in pairs:
        tr = pair[tr_idx]
        t  = tr["t"]; I = tr["I_t"]
        finite_I = I[np.isfinite(I)] if len(I) else np.array([])
        if len(finite_I):
            all_y_global.extend([float(finite_I.min()), float(finite_I.max())])
        if len(t):
            all_x_global.extend([float(t.min()), float(t.max())])
        thresh = _get_threshold(tr)
        if thresh is not None:
            all_y_global.append(thresh)

    if all_y_global:
        _ylo = min(all_y_global); _yhi = max(all_y_global)
        _ym  = 0.10*(_yhi-_ylo) if _yhi>_ylo else max(0.1*abs(_yhi), 1e-9)
        y_full = (_ylo-_ym, _yhi+_ym)
    else:
        y_full = (0.0, 1.0)

    if t_lim[1] > t_lim[0]:
        x_full = t_lim
    elif all_x_global:
        _xlo, _xhi = min(all_x_global), max(all_x_global)
        x_full = (_xlo - 0.02*(_xhi-_xlo), _xhi + 0.02*(_xhi-_xlo))
    else:
        x_full = (0.0, 1.0)

    x_zoom_default = list(zoom_x) if zoom_x is not None else list(x_full)
    y_zoom_default = list(zoom_y) if zoom_y is not None else list(y_full)

    # mutable zoom state for the right panel
    _zoom_r = {"xlim": list(x_zoom_default), "ylim": list(y_zoom_default)}

    plt.rcParams.update(_PLOT_STYLE)
    fig_w, fig_h = _fig_size(scale=1.5, ncols=2)
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(fig_w * 2.0, fig_h + 0.9),
        gridspec_kw={"wspace": 0.18},
    )
    fig.subplots_adjust(bottom=0.30, top=0.88, left=0.07, right=0.97)

    # ── render ────────────────────────────────────────────────────────────
    def _render(idx: int) -> None:
        ax_l.cla(); ax_r.cla()

        tr    = pairs[idx][tr_idx]
        color = cmap(norm(tr["K_lcm"]))

        # LEFT — all runs, current highlighted ─────────────────────────────
        ax_l.set_autoscale_on(False)
        ax_l.set_xlim(x_full); ax_l.set_ylim(y_full)

        for i, pair in enumerate(pairs):
            nb_tr = pair[tr_idx]
            nb_c  = cmap(norm(nb_tr["K_lcm"]))
            nb_t  = nb_tr["t"]; nb_I = nb_tr["I_t"]
            if not (len(nb_t) and len(nb_I)):
                continue
            if i == idx:
                ax_l.plot(nb_t, nb_I, color=nb_c, lw=1.6, alpha=1.0,
                          zorder=4)
            else:
                ax_l.plot(nb_t, nb_I, color=nb_c, lw=0.9, alpha=0.18,
                          zorder=1)

        ax_l.axvline(T_GT, color=_COL_GT, lw=1.2, ls="--", zorder=3)
        t_det_l = tr.get("t_d_true", None)
        if t_det_l is not None:
            ax_l.axvline(t_det_l, color="#2ca02c", lw=1.4, ls="-.", zorder=5)

        ax_l.set_xlabel("$t$  [s]"); ax_l.set_ylabel(ind.upper())
        ax_l.ticklabel_format(style="sci", axis="y", scilimits=(0,0),
                              useMathText=True)
        K = tr["K_lcm"]
        ax_l.set_title(
            f"{ind.upper()} — {mode_label}\n"
            f"todas las runs  (resaltada $K={K}$  [{idx+1}/{n_runs}])",
            pad=4,
        )
        for sp in ax_l.spines.values():
            sp.set_linewidth(0.9)
        ax_l.tick_params(axis="both", which="major", direction="in",
                         length=4, width=0.8)
        ax_l.tick_params(axis="both", which="minor", direction="in",
                         length=2.5, width=0.6)
        ax_l.minorticks_on()

        # RIGHT — current run zoomed ────────────────────────────────────────
        ax_r.set_autoscale_on(False)
        ax_r.set_xlim(_zoom_r["xlim"]); ax_r.set_ylim(_zoom_r["ylim"])

        t_vec = tr["t"]; I_vec = tr["I_t"]
        if len(t_vec) and len(I_vec):
            _nm = max(1, len(t_vec)//100)
            ax_r.plot(t_vec, I_vec, color=color, lw=1.4,
                      marker="o", markersize=3.5, markevery=_nm,
                      markeredgewidth=0.3, markeredgecolor="black", zorder=3)

        thresh = _get_threshold(tr)
        if thresh is not None:
            ax_r.axhline(thresh, color=color, lw=1.2, ls="--",
                         alpha=0.85, zorder=2,
                         label=f"Limite = {thresh:.4g}")

        t_det = tr.get("t_d_true", None)
        if t_det is not None:
            ax_r.axvline(t_det, color="#2ca02c", lw=1.6, ls="-.", zorder=5,
                         label=(f"$t_d={t_det:.3f}$ s  "
                                f"($\\Delta t_d={(t_det-T_GT)*1e3:.1f}$ ms)"))

        ax_r.axvline(T_GT, color=_COL_GT, lw=1.4, ls="--", zorder=4,
                     label=f"$T_{{\\mathrm{{GT}}}}={T_GT:.3f}$ s")

        if thresh is not None or t_det is not None:
            ax_r.legend(loc="upper left", handlelength=1.6,
                        borderaxespad=0.4, labelspacing=0.3, frameon=False)

        ax_r.set_xlabel("$t$  [s]"); ax_r.set_ylabel(ind.upper())
        ax_r.ticklabel_format(style="sci", axis="y", scilimits=(0,0),
                              useMathText=True)
        nwin = tr["N_win"]; dtp = tr.get("step",1)
        nfen = tr["n_accum"]; Teff = tr.get("T_eff_ms","?")
        ax_r.set_title(
            f"zoom   $K={K}$   $N_{{\\mathrm{{win}}}}={nwin}$\n"
            f"$\\Delta T_{{\\mathrm{{pas}}}}={dtp}$    "
            f"$N_{{\\mathrm{{fen}}}}={nfen}$    "
            f"$T_{{\\mathrm{{eff}}}}={Teff}$ ms",
            pad=4,
        )
        for sp in ax_r.spines.values():
            sp.set_linewidth(0.9)
        ax_r.tick_params(axis="both", which="major", direction="in",
                         length=4, width=0.8)
        ax_r.tick_params(axis="both", which="minor", direction="in",
                         length=2.5, width=0.6)
        ax_r.minorticks_on()

        fig.canvas.draw()

    # ── first render ──────────────────────────────────────────────────────
    _render(0)

    # ── Slider ────────────────────────────────────────────────────────────
    ax_slider = fig.add_axes([0.10, 0.09, 0.82, 0.09])
    slider = Slider(
        ax=ax_slider, label="", valmin=0, valmax=n_runs-1,
        valinit=0, valstep=1,
        color=cmap(norm(K_vals[n_runs//2])),
    )
    ax_slider.set_xticks(np.arange(n_runs))
    ax_slider.set_xticklabels([str(k) for k in K_vals], fontsize=8.5)
    ax_slider.tick_params(axis="x", length=5, direction="out", pad=2)
    ax_slider.tick_params(axis="y", which="both", length=0, labelleft=False)
    fig.text(0.51, 0.015, "$K_{\\mathrm{LCM}}$",
             ha="center", va="bottom",
             fontsize=_PLOT_STYLE["axes.labelsize"])

    def _fmt(i: int) -> str:
        return f"$K={K_vals[i]}$  [{i+1}/{n_runs}]"

    slider.valtext.set_text(_fmt(0)); slider.valtext.set_fontsize(9)

    def _on_slider(val: float) -> None:
        idx = max(0, min(n_runs-1, int(round(val))))
        slider.valtext.set_text(_fmt(idx))
        _render(idx)

    slider.on_changed(_on_slider)

    # ── Buttons: Zoom + Reset zoom ──────────────────────────────────
    from matplotlib.widgets import Button as _Btn
    ax_bzoom  = fig.add_axes([0.62, 0.205, 0.12, 0.06])
    ax_breset = fig.add_axes([0.76, 0.205, 0.12, 0.06])
    btn_zoom  = _Btn(ax_bzoom,  "Zoom",   hovercolor="#c6dbef")
    btn_reset = _Btn(ax_breset, "Reset zoom",  hovercolor="#fde0c8")
    btn_zoom.label.set_fontsize(8); btn_reset.label.set_fontsize(8)

    def _on_zoom_btn(_event) -> None:
        if zoom_x is not None:
            _zoom_r["xlim"] = list(zoom_x)
        if zoom_y is not None:
            _zoom_r["ylim"] = list(zoom_y)
        ax_r.set_xlim(_zoom_r["xlim"]); ax_r.set_ylim(_zoom_r["ylim"])
        fig.canvas.draw()

    def _on_reset_btn(_event) -> None:
        _zoom_r["xlim"] = list(x_full)
        _zoom_r["ylim"] = list(y_full)
        ax_r.set_xlim(_zoom_r["xlim"]); ax_r.set_ylim(_zoom_r["ylim"])
        fig.canvas.draw()

    btn_zoom.on_clicked(_on_zoom_btn)
    btn_reset.on_clicked(_on_reset_btn)
    fig._btn_zoom_ref  = btn_zoom   # type: ignore[attr-defined]
    fig._btn_reset_ref = btn_reset  # type: ignore[attr-defined]

    def _on_key(event) -> None:
        cur = int(round(slider.val))
        if event.key == "right":
            slider.set_val(min(n_runs-1, cur+1))
        elif event.key == "left":
            slider.set_val(max(0, cur-1))
        elif event.key in ("r", "R"):
            _on_reset_btn(None)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig._slider_ref = slider   # type: ignore[attr-defined]
    plt.show(block=False)


def _ind_pkl_paths(ind: str) -> tuple[str, str]:
    """Return (df_path, traces_path) for a single indicator."""
    return (
        os.path.join(OUTPUT_DIR, f"phase1_df_{ind}.pkl"),
        os.path.join(OUTPUT_DIR, f"phase1_traces_{ind}.pkl"),
    )


def save_results(df: pd.DataFrame, traces: list) -> None:
    """Save results split by indicator (one pair of pkl files each)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # also keep a combined CSV for quick inspection
    csv_path = os.path.join(OUTPUT_DIR, "phase1_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("CSV saved: %s", csv_path)

    # group traces by indicator (same order as df rows)
    trace_map: dict[str, list] = {}
    for (tr_m, tr_r) in traces:
        ind = (tr_m or tr_r or {}).get("indicator", "unknown")
        trace_map.setdefault(ind, []).append((tr_m, tr_r))

    for ind in df["indicator"].unique():
        df_path, traces_path = _ind_pkl_paths(ind)
        df_ind = df[df["indicator"] == ind].copy()
        with open(df_path, "wb") as f:
            pickle.dump(df_ind, f)
        with open(traces_path, "wb") as f:
            pickle.dump(trace_map.get(ind, []), f)
        logger.info("  Saved %-10s → %s", ind, df_path)


def load_results(indicators: list[str] | None = None) -> tuple[pd.DataFrame, list]:
    """
    Load per-indicator results and merge.

    Parameters
    ----------
    indicators : list[str] | None
        Which indicators to load.  None → all found on disk.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if indicators is None:
        # auto-discover saved indicators
        import glob
        found = glob.glob(os.path.join(OUTPUT_DIR, "phase1_df_*.pkl"))
        indicators = [os.path.basename(p)[len("phase1_df_"):-len(".pkl")] for p in found]
        if not indicators:
            raise FileNotFoundError(f"No saved results in {OUTPUT_DIR}")

    dfs: list[pd.DataFrame] = []
    traces: list = []
    for ind in indicators:
        df_path, traces_path = _ind_pkl_paths(ind)
        if not os.path.exists(df_path) or not os.path.exists(traces_path):
            raise FileNotFoundError(
                f"Missing saved results for '{ind}'. "
                f"Run with RUN_SWEEP=['{ind}'] first."
            )
        with open(df_path, "rb") as f:
            dfs.append(pickle.load(f))
        with open(traces_path, "rb") as f:
            traces.extend(pickle.load(f))
        logger.info("  Loaded %-10s ← %s", ind, df_path)

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Results loaded: %d indicators, %d rows", len(indicators), len(df))
    return df, traces


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ══════════════════════════════════════════════════════════════════════
    # CONTROLES PRINCIPALES — edita solo este bloque
    # ══════════════════════════════════════════════════════════════════════

    # ── ¿Qué indicadores recalcular y guardar?
    #    []                         → no recalcular nada
    #    ["maxent"]                 → solo maxent
    #    ["rms_cv", "maxent"]       → esos dos
    #    True                       → todos los de INDICATORS
    RUN_SWEEP: list[str] | bool = ["maxent"]

    # ── ¿Qué indicadores graficar?
    #    None / []                  → todos los disponibles en disco
    #    ["maxent"]                 → solo maxent
    #    ["rms_cv", "sst_svd"]      → esos dos
    PLOT_INDICATORS: list[str] | None = ["maxent"]

    # ── ¿Qué figuras mostrar?
    SHOW_TRACES   = True    # figura de navegación Prev/Next por run
    SHOW_OVERLAY  = True    # figura overlay (todos los runs superpuestos)
    SHOW_SLIDER   = True    # slider interactivo por modo
    SHOW_OVERVIEW = True    # overview 2-panel (izq=todos, der=zoom)
    SHOW_METRICS   = True   # figuras de métricas agregadas (score, latencias, etc.)

    # ══════════════════════════════════════════════════════════════════════

    # Normalise RUN_SWEEP
    if RUN_SWEEP is True:
        to_run = set(INDICATORS)
    elif RUN_SWEEP is False or RUN_SWEEP == []:
        to_run = set()
    else:
        to_run = set(RUN_SWEEP)

    # Run sweep only for the requested indicators
    if to_run:
        _orig = list(INDICATORS)
        INDICATORS[:] = [i for i in _orig if i in to_run]
        df_new, traces_new = run_phase1_sweep()
        INDICATORS[:] = _orig
        save_results(df_new, traces_new)

    # Load from disk — only the indicators requested for plotting
    load_inds = list(PLOT_INDICATORS) if PLOT_INDICATORS else None
    df, traces = load_results(load_inds)
    print_summary(df)

    zoom_map = {
        "maxent":  ((-0.1, 8.1),   (-15,    50.0)),
        "rms_cv":  ((-0.82, 10.7), (-0.0013, 0.02)),
        "sst_svd": ((-0.3, 8.2),   (-0.3,     2.3)),
    }

    if SHOW_TRACES:
        plot_traces(df, traces, zoom_map=zoom_map)

    if SHOW_OVERLAY or SHOW_SLIDER or SHOW_OVERVIEW:
        plot_per_mode(
            df, traces,
            show_slider=SHOW_SLIDER,
            show_overview=SHOW_OVERVIEW,
            zoom_map=zoom_map,
        )

    if SHOW_METRICS:
        plot_metrics(df, show_score=False)
        # plot_latency_scatter(df)
        # plot_detection_timeline(df)
        # plot_score_heatmap(df)

    plt.show()

if __name__ == "__main__":
    main()
