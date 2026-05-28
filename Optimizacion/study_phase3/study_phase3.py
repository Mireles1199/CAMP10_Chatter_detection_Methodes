"""
Optimizacion/study_phase3.py
=============================
Phase 2 — Discrete-Parameter Sweep Study.

For a fixed grid of ``K_total`` values (integer number of physical cycles),
enumerate ALL feasible integer parameter combinations ``(N_win, step)`` for
each indicator (RMS-CV, SST-SVD, MaxEnt-SPRT), run each configuration via
the indicators' physical-parameter APIs, compute performance metrics
(detection latency Δt_d and false-alarm count N_fa), and collect the results
into a :class:`~sweep.SweepResult`.

═══════════════════════════════════════════════════════════════════════════════
USER-CONFIGURABLE CONSTANTS  (edit the block below)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sys
import logging
import pickle
import uuid

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_CAMP10 = os.path.dirname(os.path.dirname(_HERE))   # CAMP10_Chatter_detection_Methodes/
_SWEEP  = os.path.join(_HERE, "sweep")

# Add indicator source directories to sys.path
for _pkg in ("maxent_sprt/src", "rms_cv/src", "ssq_chatter/src"):
    _p = os.path.join(_CAMP10, "indicators", _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Add sweep package directory
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── sweep imports ────────────────────────────────────────────────────────────
from sweep import (
    StudyBasis,
    SweepMode,
    SweepResult,
    DebugManager,
    enumerate_feasible,
    build_indicator_config,
    run_combo,
)
from sweep.run_one import RunResult

# ── HDF5 reader (from MaxEnt_SPRT package) ───────────────────────────────────
from MaxEnt_SPRT import HDF5Reader
from rms_cv.utils.types import SignalData


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  USER-CONFIGURABLE CONSTANTS                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Physical basis ───────────────────────────────────────────────────────────
BASIS = StudyBasis("by_revolution", f_modal=150.0, rpm=12_000.0, 
                   maxent_opr_valid=True
                   )
#   Alternatives:
#     StudyBasis("by_revolution", f_modal=150.0, rpm=12_000.0)
#   Note: by_revolution + f_modal > rpm/120 → segmentation="raw" injected automatically.

# ── K_total grid (integer number of physical cycles) ────────────────────────
# K_CYCLES_GRID = [2, 3, 5, 8]
K_CYCLES_GRID = [2,3,4,5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15]
K_CYCLES_GRID = np.linspace(50, 250, 3, dtype=int).tolist()   # 50 points from 50 to 250 inclusive

# ── Ground-truth chatter onset time [s] ─────────────────────────────────────
T_GT = 5.365770208787228   # from 1DOF_150Hz/out.hdf5 scenario metadata

# ── False-alarm penalty coefficient ─────────────────────────────────────────
LAMBDA = 1.0

# ── Signal channel per indicator (str = same for all, dict = per-indicator) ───
#   str  → mismo canal para todos los indicadores
#          e.g.  SIGNAL_CHANNEL = "velocity"
#   dict → cada indicador usa su propio canal
#          e.g.  SIGNAL_CHANNEL = {
#                    "rms_cv":  "velocity",
#                    "sst_svd": "displacement",
#                    "maxent":  "velocity",
#                }
#   Canales soportados: "velocity" (tool_dyn_o col 1), "displacement" (tool_dyn col 1)
# SIGNAL_CHANNEL = "velocity"
SIGNAL_CHANNEL = {
                   "rms_cv":  "velocity",
                   "sst_svd": "velocity",
                   "maxent":  "velocity",
               }

# ── Sweep mode ───────────────────────────────────────────────────────────────
SWEEP_MODE = SweepMode.FREE_ALL

# ── Indicators to sweep ──────────────────────────────────────────────────────
# INDICATORS = ["rms_cv", "sst_svd", "maxent"]
INDICATORS = [ "rms_cv"]


# ── Debug level  (0=off, 1=info, 2=verbose, 3=debug+plots) ──────────────────
DEBUG_LEVEL = 1

# ── Print RESULTADO + CONFIGURACION for every run (same style as examples) ──
#   True  → full output per run (useful for small K_CYCLES_GRID)
#   False → only final summary table printed
PRINT_EACH_RUN = True

# ── Output directory (relative to this script) ──────────────────────────────
OUTPUT_DIR = os.path.join(_HERE, "sweep_output")

# ── Data location ────────────────────────────────────────────────────────────
DIR_CONO   = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"
CUT_RANGE  = (0.05, 16.0)   # [s] analysis window applied to raw signal

# ── Indicator base parameters (non-swept — thresholds, flags, etc.) ─────────

_BASE_MAXENT: dict = {
    # Training / test split: use known onset (optimistic alpha estimate)
    # alpha = beta = norm.sf(3.0) ≈ 0.00135  →  equivalent to z=3 sigma (same FAR as RMS-CV and SSQ)
    "t_stable_total":    T_GT,
    "alpha":             0.00135,
    "beta":              0.00135,
    "reset_on_H0":       True,
    "cut_start_time":    CUT_RANGE[0],
    "cut_end_time":      10,
    # "segmentation" is intentionally OMITTED here — injected by config_builder
    # when basis.maxent_opr_valid is False (by_revolution + aliasing case).
}

_BASE_RMS_CV: dict = {
    "detrend":              False,
    "pad_mode":             "none",
    "use_unbiased_std":     True,
    "eps":                  1e-12,
    # Fixed threshold (fallback, ignored when stable_time is set)
    "cv_threshold":         None,
    "rms_threshold":        None,
    "n_min_cv":             2,
    "warmup_ignore_alerts": False,
    # ── Adaptive threshold from stable region (3-sigma on CV) ──────────────
    "stable_time":   (0.0,T_GT ),   # seconds: region known to be stable
    "frac_stable":   0.30,         # fallback if stable_time yields no frames
    "z":             3.0,
    "alpha":         0.05,
    "fallback_mad":  True,
}

_BASE_SST_SVD: dict = {
    "n_fft_power":  3,
    "mode":         "causal_inclusive",
    "sigma":        6.0,
    "frac_stable":  0.36052,
    "alpha":        0.05,
    "z":            3.0,
    "fallback_mad": False,
}

_BASE_PARAMS: dict = {
    "rms_cv":  _BASE_RMS_CV,
    "sst_svd": _BASE_SST_SVD,
    "maxent":  _BASE_MAXENT,
    "maxent_sprt": _BASE_MAXENT,
}

# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("study_phase3")


# ═════════════════════════════════════════════════════════════════════════════
# Console printing helpers  (same style as the indicator example scripts)
# ═════════════════════════════════════════════════════════════════════════════

def _section(title: str, width: int = 54) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"


def _print_run_result(rr: "RunResult", indicator_config: dict) -> None:
    """
    Imprime RESULTADO + CONFIGURACION DEL INDICADOR en el mismo estilo
    que los scripts de ejemplo, usando rr.meta para los valores resueltos
    reales de cada run (tiempos efectivos, cuantizaciones, etc.).
    """
    ind  = rr.indicator
    meta = rr.meta   # dict con metadata resuelta devuelta por el indicador

    # ── arrays ───────────────────────────────────────────────────────────────
    t_d_arr  = rr.arrays.get("t_d_array", np.array([]))
    t_i      = rr.arrays.get("t_indicator", np.array([]))
    n_frames = len(t_i)

    # ── RESULTADO ─────────────────────────────────────────────────────────────
    if not rr.run_ok:
        logger.warning(_section("RESULTADO  --  ERROR EN EJECUCION"))
        logger.warning("  Indicador: %s  |  error: %s",
                       ind.upper(), rr.error_str.splitlines()[-1] if rr.error_str else "?")
        return

    chatter_pct = (
        f"{len(t_d_arr) / n_frames * 100.0:.2f} %%" if n_frames > 0 and len(t_d_arr) > 0
        else "0.00 %%"
    )

    if len(t_d_arr) > 0:
        logger.info(_section("RESULTADO  --  CHATTER DETECTADO"))
        logger.info("  %-24s %s",       "Indicador:",         ind.upper())
        logger.info("  %-24s %s",       "Modo config:",       rr.basis_mode)
        logger.info("  %-24s %.5f s",   "Primera deteccion:", float(t_d_arr[0]))
        logger.info("  %-24s %+.1f ms", "Delta t_d:",         rr.delta_t_d * 1e3)
        logger.info("  %-24s %d",       "Total detecciones:", len(t_d_arr))
        logger.info("  %-24s %d",       "Falsas alarmas:",    rr.N_fa)
        logger.info("  %-24s %s",       "%% chatter:",        chatter_pct)
        if len(t_i) >= 2:
            logger.info("  %-24s %.4f, %.4f ms", "Tiempo I[0], I[1]:",
                        t_i[0] * 1e3, t_i[1] * 1e3)
            logger.info("  %-24s %.4f ms",       "Hop frames:",
                        (t_i[1] - t_i[0]) * 1e3)
        logger.info("  ")
    else:
        logger.warning(_section("RESULTADO  --  sin deteccion de chatter"))
        logger.warning("  Indicador: %s  |  Modo: %s", ind.upper(), rr.basis_mode)

    # ── CONFIGURACION ─────────────────────────────────────────────────────────
    _KW = 26

    def _kv(key: str, val: str = "", indent: int = 0) -> str:
        pad = "  " * indent
        return f"{pad}{key:<{_KW - 2 * indent}}  {val}"

    def _sep(label: str = "") -> str:
        dash = "\u2500" * 20
        return f"  {dash}  {label}" if label else f"  {dash}"

    param_mode = indicator_config.get("param_mode", "native")
    pp         = indicator_config.get("params_physical", indicator_config.get("params", {}))
    nat        = meta.get("native_params_resolved", {})   # parametros nativos resueltos

    lines = [
        _kv("Indicador", ind.upper()),
        _kv("Modo",      param_mode),
        _sep(),
    ]

    # ── RMS-CV ────────────────────────────────────────────────────────────────
    if ind == "rms_cv":
        if param_mode == "by_modal":
            lines += [
                _kv("T_modal",        f"{pp['T_modal'] * 1e3:.4f} ms"
                                      f"  (f = {1 / pp['T_modal']:.1f} Hz)"),
                _kv("N_modal_window", f"{pp['N_modal_window']} periodos"),
                _kv("step_modal",     f"{pp['step_modal']} periodo(s)"),
                _kv("n_max_mode",     str(pp.get("n_max_mode", "frames"))),
                _kv("n_max_modal",    str(pp.get("n_max_modal", "-"))),
            ]
        elif param_mode == "by_revolution":
            lines += [
                _kv("T_rev",        f"{pp['T_rev'] * 1e3:.3f} ms"
                                    f"  (rpm = {60 / pp['T_rev']:.1f})"),
                _kv("N_rev_window", f"{pp['N_rev_window']} rev"),
                _kv("step_rev",     f"{pp['step_rev']} rev"),
                _kv("n_max_mode",   str(pp.get("n_max_mode", "frames"))),
                _kv("n_max_rev",    str(pp.get("n_max_rev", "-"))),
            ]
        if meta:   # valores resueltos reales desde rr.meta
            lines += [
                _sep("Ventana"),
                _kv("t_win deseado",      f"{meta.get('t_win_exact_ms',  0):.4f} ms",      indent=1),
                _kv("t_win efectivo",     f"{meta.get('t_win_real_ms',   0):.4f} ms",       indent=1),
                _kv("delta t_win",
                    f"+{abs(meta.get('t_win_real_ms', 0) - meta.get('t_win_exact_ms', 0)) * 1e3:.3f} us",
                    indent=1),
                _kv("samples_per_window", f"{nat.get('samples_per_window', '-')} samples",  indent=1),
                _sep("Paso RMS"),
                _kv("step deseado",  f"{meta.get('dt_rms_exact_ms', 0):.4f} ms", indent=1),
                _kv("step efectivo", f"{meta.get('dt_rms_real_ms',  0):.4f} ms", indent=1),
                _kv("delta step",
                    f"+{abs(meta.get('dt_rms_real_ms', 0) - meta.get('dt_rms_exact_ms', 0)) * 1e3:.3f} us",
                    indent=1),
                _kv("overlap_pct",   f"{nat.get('overlap_pct', 0):.4f}", indent=1),
                _sep("Ventana CV"),
                _kv("n_max", str(nat.get("n_max", "-")), indent=1),
                _kv("t_cv_total des",
                    f"{meta.get('t_cv_total_exact_s', 0) * 1e3:.3f} ms"
                    f"  ({meta.get('K_cv_total_exact_units', 0):.2f} periodos)",
                    indent=1),
                _kv("t_cv_total ef",
                    f"{meta.get('t_cv_total_s', 0) * 1e3:.3f} ms"
                    f"  ({meta.get('K_cv_total_units', 0):.2f} periodos)",
                    indent=1),
            ]
        lines += [
            _sep("Thresholds"),
            _kv("cv_threshold",  str(pp.get("cv_threshold",  "-")), indent=1),
            _kv("rms_threshold", str(pp.get("rms_threshold", "-")), indent=1),
            _kv("n_min_cv",      str(pp.get("n_min_cv",      "-")), indent=1),
        ]

    # ── SST-SVD ───────────────────────────────────────────────────────────────
    elif ind == "sst_svd":
        if param_mode == "by_modal":
            lines += [
                _kv("T_modal",         f"{pp['T_modal'] * 1e3:.4f} ms"
                                       f"  (f = {1 / pp['T_modal']:.1f} Hz)"),
                _kv("N_modal_window",  f"{pp['N_modal_window']} periodos"),
                _kv("step_modal",      f"{pp['step_modal']} periodo(s)"),
                _kv("Ai_length_mode",  str(pp.get("Ai_length_mode", "frames"))),
                _kv("Ai_length_modal", str(pp.get("Ai_length_modal", "-"))),
            ]
        elif param_mode == "by_revolution":
            lines += [
                _kv("T_rev",          f"{pp['T_rev'] * 1e3:.3f} ms"
                                      f"  (rpm = {60 / pp['T_rev']:.1f})"),
                _kv("N_rev_window",   f"{pp['N_rev_window']} rev"),
                _kv("step_rev",       f"{pp['step_rev']} rev"),
                _kv("Ai_length_mode", str(pp.get("Ai_length_mode", "frames"))),
                _kv("Ai_length_rev",  str(pp.get("Ai_length_rev",  "-"))),
            ]
        if meta:   # valores resueltos reales
            quant = meta.get("quantization_notes", "")
            lines += [
                _sep("Resultado"),
                _kv("t_win_deseado",  f"{meta.get('t_win_exact_ms',    0):.4f} ms", indent=1),
                _kv("t_win_efectivo", f"{meta.get('t_win_efectivo_ms', 0):.4f} ms", indent=1),
                _kv("delta_t_win",
                    f"+{abs(meta.get('t_win_efectivo_ms', 0) - meta.get('t_win_exact_ms', 0)) * 1e3:.3f} us",
                    indent=1),
                _kv("Hop Deseado",    f"{meta.get('t_hop_exact_ms',    0):.4f} ms", indent=1),
                _kv("Hop Efectivo",   f"{meta.get('t_hop_efectivo_ms', 0):.4f} ms", indent=1),
                _kv("delta_hop",
                    f"+{abs(meta.get('t_hop_efectivo_ms', 0) - meta.get('t_hop_exact_ms', 0)) * 1e3:.3f} us",
                    indent=1),
                _kv("Ai_length",      str(nat.get("Ai_length", "-")), indent=1),
                _kv("t_svd_total des",
                    f"{meta.get('t_svd_total_exact_s', 0):.4f} s"
                    f"  ({meta.get('K_svd_total_exact_units', 0):.4f} periodos)",
                    indent=1),
                _kv("t_svd_total ef",
                    f"{meta.get('t_svd_total_efectivo_s', 0):.4f} s"
                    f"  ({meta.get('K_svd_total_efectivo_units', 0):.4f} periodos)",
                    indent=1),
            ]
            for part in quant.replace("|", ";").split(";"):
                if part.strip():
                    lines.append(f"    {part.strip()}")
        lines += [
            _sep("SSQ / Deteccion"),
            _kv("n_fft_power",
                f"{pp.get('n_fft_power', '?')}  "
                f"(n_fft = {1024 * 2**pp['n_fft_power'] if 'n_fft_power' in pp else '?'})",
                indent=1),
            _kv("sigma",       str(pp.get("sigma",       "-")), indent=1),
            _kv("frac_stable", str(pp.get("frac_stable", "-")), indent=1),
            _kv("alpha",       str(pp.get("alpha",       "-")), indent=1),
            _kv("z",           str(pp.get("z",           "-")), indent=1),
        ]

    # ── MaxEnt-SPRT ───────────────────────────────────────────────────────────
    elif ind in ("maxent", "maxent_sprt"):
        seg_mode = meta.get("segmentation", "opr") if meta else "opr"
        if param_mode == "by_modal":
            step_s     = pp.get("step_modal", pp.get("N_modal_per_seg", 1))
            N_seg_phys = pp.get("N_modal_per_seg", "-")
            overlap_p  = (
                (1.0 - step_s / N_seg_phys)
                if isinstance(N_seg_phys, (int, float)) and N_seg_phys > 0
                else 0.0
            )
            T_rev_v = pp.get("T_rev", 0)
            lines += [
                _kv("T_rev",
                    f"{T_rev_v * 1e3:.3f} ms  (rpm = {60.0 / T_rev_v:.1f})" if T_rev_v else "N/A"),
                _kv("T_modal",         f"{pp['T_modal'] * 1e3:.3f} ms  (f = {1/pp['T_modal']:.1f} Hz)"),
                _kv("N_modal_per_seg", f"{N_seg_phys} periodos modales/seg"),
                _kv("step_modal",      f"{step_s} periodos    (overlap = {overlap_p:.1%})"),
            ]
        elif param_mode == "by_revolution":
            step_s     = pp.get("step_rev", pp.get("N_rev_per_seg", 1))
            N_seg_phys = pp.get("N_rev_per_seg", "-")
            overlap_p  = (
                (1.0 - step_s / N_seg_phys)
                if isinstance(N_seg_phys, (int, float)) and N_seg_phys > 0
                else 0.0
            )
            T_rev_v = pp.get("T_rev", 0)
            lines += [
                _kv("T_rev",
                    f"{T_rev_v * 1e3:.3f} ms  (rpm = {60 / T_rev_v:.1f})" if T_rev_v else "N/A"),
                _kv("N_rev_per_seg", f"{N_seg_phys} rev/seg"),
                _kv("step_rev",      f"{step_s} rev    (overlap = {overlap_p:.1%})"),
            ]
            if seg_mode == "raw":
                lines.append(_kv("segmentation", "raw  (OPR aliasing — raw mode)"))
        if meta:   # valores resueltos reales
            fr    = meta.get("Rotational_Frequency_Hz", 1.0)
            quant = meta.get("quantization_notes", "")
            lines += [
                _sep("Resultado"),
                _kv("N_seg",    str(nat.get("N_seg",    "-")), indent=1),
                _kv("step_seg",
                    str(nat.get("step_seg", nat.get("N_seg", "-"))),
                    indent=1),
                _kv("t_seg",
                    f"{nat.get('N_seg', 0) / fr * 1e3:.2f} ms" if nat.get("N_seg") else "-",
                    indent=1),
            ]
            if seg_mode == "raw":
                nsamp = meta.get("N_samples_per_seg") or nat.get("N_samples_per_seg", "?")
                lines.append(_kv("N_samples_per_seg", f"{nsamp} muestras raw", indent=1))
            for part in quant.replace("|", ";").split(";"):
                if part.strip():
                    lines.append(f"    {part.strip()}")
        lines += [
            _sep("SPRT"),
            _kv("t_stable_total", f"{pp.get('t_stable_total', '?'):.4f} s", indent=1),
            _kv("alpha / beta",   f"{pp.get('alpha', '?')} / {pp.get('beta', '?')}", indent=1),
            _kv("reset_on_H0",    str(pp.get("reset_on_H0", "-")), indent=1),
        ]

    logger.info("%s\n%s", _section("CONFIGURACION DEL INDICADOR"), "\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cut_signal(
    t: np.ndarray,
    x: np.ndarray,
    time_range: tuple,
) -> tuple:
    start, end = time_range
    mask = (t >= start) & (t <= end)
    return t[mask], x[mask]


def load_signal(data_path: str, cut_range: tuple, channel: str) -> "SignalData":
    """Load the HDF5 machining signal and return a :class:`SignalData`."""
    data      = HDF5Reader(data_path)
    tool_dyn  = data.get_element("tool_dyn/data")
    t_raw     = tool_dyn[:, 0]
    x_raw     = tool_dyn[:, 1]
    v_raw     = data.get_element("tool_dyn_o/data")[:, 1]

    fs = 1.0 / (t_raw[1] - t_raw[0])

    if channel == "velocity":
        t_cut, sig_cut = _cut_signal(t_raw, v_raw, cut_range)
    elif channel == "displacement":
        t_cut, sig_cut = _cut_signal(t_raw, x_raw, cut_range)
    else:
        raise ValueError(f"Unknown channel {channel!r}. Supported: 'velocity', 'displacement'.")

    logger.info(
        "Signal loaded: channel=%s  fs=%.1f Hz  duration=%.3f s  samples=%d",
        channel, fs, t_cut[-1] - t_cut[0], len(t_cut),
    )

    return SignalData(
        t_analysis      = t_cut,
        signal_analysis = sig_cut,
        fs              = fs,
        path            = data_path,
        meta            = {"channel": channel, "cut_range": cut_range},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main sweep loop
# ═════════════════════════════════════════════════════════════════════════════

def run_sweep(
    signals:         "dict[str, SignalData]",
    basis:           StudyBasis,
    k_grid:          list,
    indicators:      list,
    sweep_mode:      SweepMode,
    t_gt:            float,
    lam:             float,
    debug_level:     int = 1,
    print_each_run:  bool = True,
) -> SweepResult:
    """
    Execute the full discrete-parameter sweep and return a :class:`SweepResult`.

    Parameters
    ----------
    signals     : dict {indicator_id -> SignalData}  — un canal por indicador
    basis       : StudyBasis
    k_grid      : list of int — K_total grid points
    indicators  : list of str — indicator ids to sweep
    sweep_mode  : SweepMode
    t_gt        : float — ground-truth onset time [s]
    lam         : float — false-alarm penalty coefficient
    debug_level : int  — verbosity 0–3

    Returns
    -------
    SweepResult
    """
    dbg       = DebugManager(level=debug_level, name="study_phase3")
    T_unit    = basis.T_unit
    all_runs: list = []

    total_configs = 0

    for ind in indicators:
        signal = signals.get(ind, next(iter(signals.values())))
        for K_total in k_grid:
            combos = enumerate_feasible(K_total, ind, sweep_mode)
            n_combos = len(combos)
            total_configs += n_combos
            dbg.log_k_step(K_total, n_combos)

            if n_combos == 0:
                dbg.log_warning(
                    f"No feasible combos for indicator={ind} K_total={K_total} "
                    f"(prime or too small). Skipping."
                )
                continue

            base_params = _BASE_PARAMS.get(ind, {})

            for combo in combos:
                config = build_indicator_config(ind, basis, combo, base_params)
                step   = combo["step"]
                N_win  = combo.get("N_win")
                n_acc  = combo.get("n_accum")

                # ── unique run identifier ────────────────────────────────────────────────
                run_id = uuid.uuid4().hex[:12]

                dbg.log_combo(
                    run_id    = run_id,
                    indicator = ind,
                    K_total   = K_total,
                    N_win     = N_win,
                    step      = step,
                    n_accum   = n_acc,
                )

                rr: RunResult = run_combo(
                    signal          = signal,
                    indicator_config= config,
                    indicator_id    = ind,
                    t_gt            = t_gt,
                    T_unit          = T_unit,
                    K_total         = K_total,
                    lam             = lam,
                    combo           = combo,
                    basis_mode      = basis.mode,
                    run_id          = run_id,
                )

                if rr.run_ok:
                    dbg.log_run_ok(rr.run_id, rr.t_d_first, rr.t_d_first_true, rr.delta_t_d, rr.N_fa)
                else:
                    dbg.log_run_fail(rr.run_id, rr.error_str.splitlines()[-1])

                if print_each_run:
                    if DEBUG_LEVEL >= 3:
                        _print_run_result(rr, config)

                all_runs.append(rr)

    logger.info(
        "Sweep complete: %d total configs | %d runs completed",
        total_configs, len(all_runs),
    )

    return SweepResult(all_runs)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load signal ──────────────────────────────────────────────────────────
    data_path = os.path.abspath(os.path.join(DIR_CONO, "out.hdf5"))
    logger.info("Loading HDF5: %s", data_path)

    # Construir dict signals: soporta SIGNAL_CHANNEL como str o dict
    if isinstance(SIGNAL_CHANNEL, str):
        _sig = load_signal(data_path, CUT_RANGE, SIGNAL_CHANNEL)
        signals = {ind: _sig for ind in INDICATORS}
    else:
        # dict mode: cargar cada canal único una sola vez
        _loaded: dict = {}
        for _ch in set(SIGNAL_CHANNEL.values()):
            _loaded[_ch] = load_signal(data_path, CUT_RANGE, _ch)
        signals = {ind: _loaded[SIGNAL_CHANNEL[ind]] for ind in INDICATORS}
        for _ind in INDICATORS:
            logger.info("  %-12s -> channel: %s", _ind, SIGNAL_CHANNEL[_ind])

    logger.info("Basis: %s", BASIS)
    logger.info(
        "K_total grid: %s  (T_unit=%.4f ms)",
        K_CYCLES_GRID,
        BASIS.T_unit * 1e3,
    )
    logger.info(
        "T_total range: [%.2f ms … %.2f ms]",
        min(K_CYCLES_GRID) * BASIS.T_unit * 1e3,
        max(K_CYCLES_GRID) * BASIS.T_unit * 1e3,
    )
    logger.info("Indicators: %s", INDICATORS)
    logger.info("MaxEnt OPR valid: %s", BASIS.maxent_opr_valid)

    # ── Run sweep ────────────────────────────────────────────────────────────
    sweep = run_sweep(
        signals         = signals,
        basis           = BASIS,
        k_grid          = K_CYCLES_GRID,
        indicators      = INDICATORS,
        sweep_mode      = SWEEP_MODE,
        t_gt            = T_GT,
        lam             = LAMBDA,
        debug_level     = DEBUG_LEVEL,
        print_each_run  = PRINT_EACH_RUN,
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(sweep.summary())
    print("=" * 72)

    # ── Sensitivity curves ───────────────────────────────────────────────────
    print("\n── Sensitivity (mean delta_t_d vs K_total) ──")
    try:
        sens = sweep.sensitivity()
        print(sens.to_string())
    except Exception as exc:
        logger.warning("sensitivity() failed: %s", exc)

    # ── Best config per (indicator, K_total) ─────────────────────────────────
    print("\n── Best config per (indicator, K_total) ──")
    print(sweep.best_table().to_string())

    # ── Convergencia del mejor score vs K_total ───────────────────────────────
    print("\n-- Convergencia del mejor score vs K_total --")
    try:
        print(sweep.convergence_vs_k().to_string())
    except Exception as exc:
        logger.warning("convergence_vs_k() failed: %s", exc)

    # ── Ranking de importancia de parametros ──────────────────────────────────
    print("\n-- Ranking de importancia de parametros (todos los indicadores) --")
    try:
        print(sweep.importance_ranking().to_string())
    except Exception as exc:
        logger.warning("importance_ranking() failed: %s", exc)

    # ── Sensibilidad por parametro por indicador ──────────────────────────────
    for _ind in INDICATORS:
        print(f"\n-- Sensibilidad por parametro: {_ind} --")
        try:
            ps = sweep.param_sensitivity(indicator=_ind)
            for _p, _df in ps.items():
                print(f"  -> {_p}")
                print(_df.to_string())
        except Exception as exc:
            logger.warning("param_sensitivity(%s) failed: %s", _ind, exc)

    # ── Trade-off Dtd vs N_fa por step ────────────────────────────────────────
    print("\n-- Trade-off Dtd vs N_fa por step --")
    try:
        print(sweep.tradeoff_table(param="step").to_string())
    except Exception as exc:
        logger.warning("tradeoff_table(step) failed: %s", exc)

    # ── Calidad del espacio factible (score < 0.05) ───────────────────────────
    print("\n-- Calidad del espacio factible (score < 0.05) --")
    try:
        print(sweep.feasible_space_quality(score_threshold=1.4).to_string())
    except Exception as exc:
        logger.warning("feasible_space_quality() failed: %s", exc)

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    sweep.df.to_csv(csv_path, index=False, float_format="%.8g")
    logger.info("Results saved → %s", csv_path)

    pkl_path = os.path.join(OUTPUT_DIR, "sweep_result.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(sweep, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Full sweep object saved → %s", pkl_path)

    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
