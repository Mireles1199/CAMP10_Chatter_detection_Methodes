"""
Optimizacion/optimizacion.py
=============================
Entry point for the Phase 1 — Common Observation Window Framework.

What this script does
---------------------
1. Loads real machining data from the 1DOF_150Hz cono dataset (out.hdf5).
2. Constructs an effective_window.SignalData from the velocity channel.
3. Builds a RunnerConfig that imposes the same T_des on all three indicators
   (RMS-CV, MaxEnt-SPRT, SST-SVD) and algebraically resolves each indicator's
   internal parameter from T_des.
4. Calls WindowRunner().run() to execute the full Phase 1 pipeline.
5. Prints result.summary() and (optionally) shows plots.

Optional YAML override
----------------------
If a file ``config.yaml`` is found in the same directory as this script,
its content is merged into the indicator configs.  Format::

    window_basis: REVOLUTION   # or MODAL
    n_cycles: 5
    show_plots: false
    debug_level: 1
    indicators:
      rms_cv:
        solved_var: N
        fixed_vars:
          rho: 0.50
          n_max: 20
        rounding: FLOOR
      maxent:
        solved_var: N_seg
        fixed_vars:
          rpm: 12000.0
        rounding: ROUND
      sst_svd:
        solved_var: n_A
        fixed_vars:
          w: 50.0
          h_ratio: 0.60
        rounding: CEIL

Run
---
    cd CAMP10_Chatter_detection_Methodes
    python Optimizacion/optimizacion.py
"""
from __future__ import annotations

import os
import sys
import logging

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_CAMP10    = os.path.dirname(_HERE)            # CAMP10_Chatter_detection_Methodes/
if _CAMP10 not in sys.path:
    sys.path.insert(0, _CAMP10)

# ── effective_window imports ─────────────────────────────────────────────────
from effective_window import (
    SignalData,
    WindowBasis,
    RoundingPolicy,
    WindowSpec,
    ParameterResolutionConfig,
    IndicatorWindowConfig,
    RunnerConfig,
    WindowRunner,
)
from effective_window.plotting import configure_global_style

# ── External data loader (from MaxEnt_SPRT library) ─────────────────────────
from MaxEnt_SPRT import HDF5Reader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("optimizacion")


# ============================================================================
# Hardcoded defaults  (overridable via config.yaml)
# ============================================================================

# -- Data location -----------------------------------------------------------
DIR_CONO   = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"
CUT_RANGE  = (0.05, 14.0)   # [s]  analysis window

# -- Process parameters ------------------------------------------------------
F_MODAL    = 150.0           # [Hz]  dominant chatter frequency
RPM        = 12_000.0        # [rpm]

# -- Framework parameters ----------------------------------------------------
WINDOW_BASIS = WindowBasis.REVOLUTION
N_CYCLES     = 5             # number of revolutions (or modal periods)
SHOW_PLOTS   = True
DEBUG_LEVEL  = 2             # 0=off, 1=info, 2=verbose, 3=debug+plots

# -- Indicator base params (non-resolved parameters keep their baseline value)
_BASE_MAXENT = {
    "rpm":                  RPM,
    "t_stable_total":        5.365770208787228,
    "alpha":                 0.05,
    "beta":                  0.05,
    "reset_on_H0":           True,
    "cut_start_time":        0.05,
    "cut_end_time":          14.0,
}

_BASE_RMS_CV = {
    "detrend":               False,
    "pad_mode":              "none",
    "use_unbiased_std":      True,
    "eps":                   1e-12,
    "cv_threshold":          1.05,
    "rms_threshold":         0.9,
    "n_min_cv":              2,
    "warmup_ignore_alerts":  False,
    "start_time":            0.05,
}

_BASE_SST_SVD = {
    "n_fft_power":   3,
    "mode":          "causal_inclusive",
    "sigma":         6.0,
    "frac_stable":   0.36052,
    "alpha":         0.05,
    "z":             3.0,
    "fallback_mad":  False,
}


# ============================================================================
# Helpers
# ============================================================================

def _cut_signal(
    t: np.ndarray,
    x: np.ndarray,
    time_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    start, end = time_range
    mask = (t >= start) & (t <= end)
    return t[mask], x[mask]


def _load_yaml_if_present(yaml_path: str) -> dict:
    """Load and return YAML config; return empty dict if not found or PyYAML missing."""
    if not os.path.isfile(yaml_path):
        return {}
    try:
        import yaml  # type: ignore
        with open(yaml_path, "r", encoding="utf-8") as fh:
            content = yaml.safe_load(fh) or {}
        logger.info("YAML config loaded from %s", yaml_path)
        return content
    except ImportError:
        logger.warning("PyYAML not installed — skipping config.yaml")
        return {}
    except Exception as exc:
        logger.warning("Failed to parse config.yaml: %s", exc)
        return {}


def _apply_yaml_overrides(yaml: dict) -> tuple:
    """Return (window_basis, n_cycles, show_plots, debug_level, ind_overrides)."""
    basis_str = yaml.get("window_basis", WINDOW_BASIS.name)
    basis     = WindowBasis[basis_str.upper()]
    n_cycles  = int(yaml.get("n_cycles",   N_CYCLES))
    show_pl   = bool(yaml.get("show_plots", SHOW_PLOTS))
    dbg_level = int(yaml.get("debug_level", DEBUG_LEVEL))
    overrides  = yaml.get("indicators", {})
    return basis, n_cycles, show_pl, dbg_level, overrides


def _make_resolution(
    override: dict,
    default_solved_var: str,
    default_fixed_vars: dict,
    default_rounding: RoundingPolicy,
) -> ParameterResolutionConfig:
    solved_var  = override.get("solved_var",  default_solved_var)
    fixed_vars  = override.get("fixed_vars",  default_fixed_vars)
    rounding_s  = override.get("rounding",    default_rounding.name)
    rounding    = (
        rounding_s if isinstance(rounding_s, RoundingPolicy)
        else RoundingPolicy[rounding_s.upper()]
    )
    return ParameterResolutionConfig(
        solved_var=solved_var,
        fixed_vars=dict(fixed_vars),
        rounding=rounding,
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    configure_global_style()

    # ── Load YAML overrides (optional) ──────────────────────────────────────
    yaml_path = os.path.join(_HERE, "config.yaml")
    yaml_cfg  = _load_yaml_if_present(yaml_path)
    (
        window_basis,
        n_cycles,
        show_plots,
        debug_level,
        ind_overrides,
    ) = _apply_yaml_overrides(yaml_cfg)

    # ── Load HDF5 data ──────────────────────────────────────────────────────
    data_path = os.path.abspath(os.path.join(DIR_CONO, "out.hdf5"))
    logger.info("Loading HDF5: %s", data_path)

    data      = HDF5Reader(data_path)
    tool_dyn  = data.get_element("tool_dyn/data")
    t_raw     = tool_dyn[:, 0]
    x_raw     = tool_dyn[:, 1]
    v_raw     = data.get_element("tool_dyn_o/data")[:, 1]
    force_raw = data.get_element("res_R_p/data")[:, 1]

    fs = 1.0 / (t_raw[1] - t_raw[0])
    logger.info("fs = %.2f Hz  |  total duration = %.3f s", fs, t_raw[-1])

    t_cut,     v_cut     = _cut_signal(t_raw,   v_raw,     CUT_RANGE)
    _,         x_cut     = _cut_signal(t_raw,   x_raw,     CUT_RANGE)
    _,         force_cut = _cut_signal(t_raw,   force_raw, CUT_RANGE)

    # ── Build effective_window.SignalData ────────────────────────────────────
    signal = SignalData(
        t_analysis     = t_cut,
        signal_analysis= v_cut,
        fs             = fs,
        path           = data_path,
        # t_cut          = t_cut,
        # v_cut          = v_cut,
        # x_cut          = x_cut,
        # force_cut      = force_cut,
        # t_original     = t_raw,
        # x_original     = x_raw,
        # v_original     = v_raw,
        # force_original = force_raw,
        meta           = {
            "AP":  "5mm-15mm",
            "RPM": int(RPM),
        },
    )

    # ── WindowSpec ──────────────────────────────────────────────────────────
    spec = WindowSpec(
        basis    = window_basis,
        n_cycles = n_cycles,
        f_modal  = F_MODAL,
        rpm      = RPM,
    )
    logger.info(
        "WindowSpec: basis=%s  n_cycles=%d  T_des=%.3f ms",
        spec.basis.name, spec.n_cycles, spec.compute_T_des() * 1000,
    )

    # ── Resolution configs ──────────────────────────────────────────────────
    rms_res = _make_resolution(
        override          = ind_overrides.get("rms_cv", {}),
        default_solved_var= "N",
        default_fixed_vars= {"rho": 0.0, "n_max": 20},
        default_rounding  = RoundingPolicy.ROUND,
    )

    maxent_res = _make_resolution(
        override          = ind_overrides.get("maxent", {}),
        default_solved_var= "N_seg",
        default_fixed_vars= {"rpm": RPM},
        default_rounding  = RoundingPolicy.ROUND,
    )

    sst_res = _make_resolution(
        override          = ind_overrides.get("sst_svd", {}),
        default_solved_var= "n_A",
        default_fixed_vars= {"w": 12.0, "h_ratio": 0.80},
        default_rounding  = RoundingPolicy.ROUND,
    )

    # ── IndicatorWindowConfig list ──────────────────────────────────────────
    indicators = [
        IndicatorWindowConfig(
            indicator_id      = "rms_cv",
            base_params       = dict(_BASE_RMS_CV),
            resolution        = rms_res,
            strict_constraints= True,
        ),
        IndicatorWindowConfig(
            indicator_id      = "maxent_sprt",
            base_params       = dict(_BASE_MAXENT),
            resolution        = maxent_res,
            strict_constraints= True,
        ),
        IndicatorWindowConfig(
            indicator_id      = "sst_svd",
            base_params       = dict(_BASE_SST_SVD),
            resolution        = sst_res,
            strict_constraints= True,
        ),
    ]

    # ── RunnerConfig ─────────────────────────────────────────────────────────
    config = RunnerConfig(
        window_spec = spec,
        indicators  = indicators,
        show_plots  = show_plots,
        debug_level = debug_level,
    )

    # ── Execute Phase 1 pipeline ─────────────────────────────────────────────
    logger.info("Running WindowRunner …")
    runner = WindowRunner()
    result = runner.run(signal, config)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(result.summary())
    print("=" * 70 + "\n")

    # ── Optional extra plots ──────────────────────────────────────────────────
    if show_plots:
        from effective_window.plotting import (
            plot_delta_Tw_comparison,
            plot_feasibility_summary,
        )
        plot_delta_Tw_comparison(result)
        plot_feasibility_summary(result)


if __name__ == "__main__":
    main()
