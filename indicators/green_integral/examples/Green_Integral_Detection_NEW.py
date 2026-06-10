"""Green Integral Method — example using real HDF5 data.

Usage
-----
    cd indicators/green_integral
    pip install -e .
    python examples/Green_Integral_Detection_NEW.py

Toggle
------
    Set USE_FIXED_WINDOW = True  to run the no-clustering fixed-window variant.
    Set USE_FIXED_WINDOW = False to run the original clustering-based indicator.

Case selector
-------------
    Set _ACTIVE_CASE to one of the keys in _CASES to switch between signals.
"""

from typing import Tuple
import logging
import os
import sys
import pathlib
import numpy as np


# Allow running directly without installing (adds src/ to path)
_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

# ── Logging ────────────────────────────────────────────────────────────────
from green_integral.logging_setup import configure_logging, LOGGING_LEVELS

configure_logging(level=LOGGING_LEVELS["info_plus"])
logger = logging.getLogger(__name__)

# ── Public API ─────────────────────────────────────────────────────────────
from green_integral import (
    HDF5Reader,
    StdSignalData,           # ← standard input  (same shape as MaxEnt / RMS-CV)
    IndicatorResult,         # ← standard output (same shape as MaxEnt / RMS-CV)
    run_green_std,           # ← standard runner  (f_cycle / N_cycles_per_seg / step_cycles)
    plots_green_integral,
    plots_fixed_window,
    plots_signal_diagnostics,
    INFO_PLUS_LEVEL,
)
from green_integral.utils.debug import DebugManager

# ── Toggle ─────────────────────────────────────────────────────────────────
USE_FIXED_WINDOW: bool = True   # True → fixed-window (no clustering)
                                  # False → original clustering indicator

# ═══════════════════════════════════════════════════════════════════════════════
# CASE SELECTOR — change only this line to switch between signals
# ═══════════════════════════════════════════════════════════════════════════════
_ACTIVE_CASE = "cono"   # "cono" | "stable_5mm" | "chatter_15mm" | "custom_case"
_RPM     = 12_000.0
_RPM_MODAL = 150*60.0  # RPM equivalente a f_modal = 150 Hz
_F_MODAL = 150.0
_T_REV   = 60.0 / _RPM   # 0.005 s -- periodo de una revolucion
_F_REV   = 1 / _T_REV  # Hz    -- frecuencia de revoluciones
_T_MODAL = 1.0 / _F_MODAL  # s       -- periodo del modo de chatter (f_modal ~ 150 Hz)

# alpha = beta = norm.sf(3.0) ≈ 0.00135  →  equivalent to z=3 sigma (same FAR as RMS-CV and SSQ)
_Z3_ALPHA = 0.00135   # scipy.stats.y si estanorm.sf(3.0)
_T_GT = 5.365770208787228   # [s] ground-truth chatter onset
# _T_GT = 1.07  # set to None if no chatter onset is expected (e.g. stable_5mm case)
_CUT_START = 0.1
_CUT_END   = 10

# ── Case registry ───────────────────────────────────────────────────────────
_BASE = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"

cono_doe_control =  (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
    r"\3\1DOF_150Hz\out.hdf5"
)

cono_doe_control_sensor =  (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
    r"\3\1DOF_150Hz\sens_out.hdf5"
)


custom_case = (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_180\7\1DOF_150Hz"
    r"\sens_out.hdf5"
)

chatter_15mm = (r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP8-Ventanna_Glisante"
                r"\Nessy2m_Case_Test_Explicit\1DOF_150Hz_15mm\1DOF_150Hz\out.hdf5")

_CASES = {
    # ── Original 2DOF cone (chatter onset at 5.366 s) ──────────────────────
    "cono": {
        "hdf5":               cono_doe_control_sensor,
        # "signal_source": {"disp_path": "tool_dyn/data", "vel_path": "tool_dyn_o/data"},
        "signal_source": {"disp_path": "Axial_disp/data", "vel_path": "Axial_vel/data"},

        "name":               "cono",
        "t_range":            (0.0, 16.0),
        "t_gt":               _T_GT,
        "f_modal":            _F_MODAL,  # 150 Hz
        "T_REV":              _T_REV,  # example value, adjust as needed
        "F_REV":              _F_REV,  # example value, adjust as needed
        "num_T":              4,
        "use_area_threshold": True,
        "training_intervals": [
            (2.0, _T_GT, "stable_1"),
            # (3.3,  4.46,    "stable_2"),   # tighter stable sub-band
            # (_T_GT, 10, "stable_1"),
        ],
    },
    # ── Stable case — ap = 5 mm (no chatter) ───────────────────────────────
    "stable_5mm": {
        "hdf5":               (rf"{_BASE}\Chatter-Criteria\CAMP8-Ventanna_Glisante"
                               r"\Nessy2m_Case_Test_Explicit\1DOF_150Hz_5mm\1DOF_150Hz\out.hdf5"),
        "signal_source": {"disp_path": "tool_dyn/data", "vel_path": "tool_dyn_o/data"},
        "name":               "5mm_stable",
        "t_range":            (0.05, 16.0),
        "t_gt":               _T_GT,          # no chatter in this case
        "f_modal":            _F_MODAL,
        "T_REV":              _T_REV,      # example value, adjust as needed
        "F_REV":              _F_REV,      # example value, adjust as needed
        "num_T":              4,
        "use_area_threshold": False,           # area threshold is noisy for cono but works well for this case

    },
    # ── Chatter case — ap = 15 mm (chatter from ~0.05 s) ───────────────────
    "chatter_15mm": {
        "hdf5":               chatter_15mm,
        "signal_source": {"disp_path": "tool_dyn/data", "vel_path": "tool_dyn_o/data"},
        "name":               "15mm_chatter",
        "t_range":            (0.05, 16.0),
        "t_gt":               _T_GT,          # chatter after initial transient
        "f_modal":            _F_MODAL,
        "T_REV":              _T_REV,      # example value, adjust as needed
        "F_REV":              _F_REV,      # example value, adjust as needed
        "num_T":              1,
        "use_area_threshold": False,

    },

    "custom_case": {
        # "hdf5":               r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_180\3\1DOF_150Hz\out.hdf5",
        # Para usar la señal de sensor (misma escala que el DOE):
        "hdf5":             custom_case,
        # "signal_source": {"disp_path": "tool_dyn/data", "vel_path": "tool_dyn_o/data"},
        # signal_source para sens_out.hdf5:
        "signal_source": {"disp_path": "Axial_disp/data", "vel_path": "Axial_vel/data"},
        "name":               "custom",
        "t_range":            (0.00, 16.0),    # adjust
        "t_gt":               _T_GT,           # set if known
        "f_modal":            _F_MODAL,          # adjust based on modal analysis  
        "T_REV":              _T_REV,       # example value, adjust as needed
        "F_REV":              _F_REV,       # example value, adjust as needed
        "num_T":              4,             # adjust based on expected cycles in window
        "use_area_threshold": True,          # adjust based on signal characteristics
        "training_intervals": [
            (0.00, _T_GT, "stable"),             # adjust based on expected stable/chatter intervals
        ],
    }
}

# ── Unpack active case ──────────────────────────────────────────────────────
_cfg        = _CASES[_ACTIVE_CASE]
_HDF5       = _cfg["hdf5"]
_SIG_NAME   = _cfg["name"]
_T0, _T1    = _cfg["t_range"]
_T_GT       = _cfg["t_gt"]           # None if no chatter
_F_MODAL    = _cfg["f_modal"]
_NUM_T      = _cfg["num_T"]
_T_REV      = _cfg["T_REV"]
_F_REV      = _cfg["F_REV"]
_USE_THR    = _cfg.get("use_area_threshold", True)  # True only for cono
_TRAIN_IV   = _cfg.get("training_intervals", [])
_CUT_START  = _T0

# -- helpers ------------------------------------------------------------------
def _cut_signal(t, x, time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]

# ── Load signal from HDF5 ──────────────────────────────────────────────────
_src         = _cfg.get("signal_source", {"disp_path": "tool_dyn/data", "vel_path": "tool_dyn_o/data"})
_DISP_PATH   = _src["disp_path"]
_VEL_PATH    = _src["vel_path"]
# col0=time, col1=signal for both out.hdf5 (N,2) and sens_out.hdf5 (N,3)

data         = HDF5Reader(_HDF5)

raw_disp     = data.get_element(_DISP_PATH)
t            = raw_disp[:, 0]
tool_dyn     = raw_disp[:, 1]
try:
    tool_dyn_vel = data.get_element(_VEL_PATH)[:, 1]
except Exception:
    tool_dyn_vel = np.gradient(tool_dyn, t)

# force channel may not exist in all cases
try:
    force_N = data.get_element("res_R_p/data")[:, 1]
except Exception:
    force_N = np.zeros_like(t)

v  = tool_dyn_vel
fs = 1.0 / (t[1] - t[0])

t_cut, v_cut  = _cut_signal(t, v,        (_T0, _T1))
_,     x_cut  = _cut_signal(t, tool_dyn, (_T0, _T1))
_,     f_cut  = _cut_signal(t, force_N,  (_T0, _T1))
# ── Build StdSignalData (interfaz estándar CAMP10) ──────────────────────────
# signal_analysis = desplazamiento; velocidad va en meta["velocity"]
sig_std = StdSignalData(
    t_analysis=t_cut,
    signal_analysis=x_cut,
    path=_HDF5,
    fs=fs,
    meta={"velocity": v_cut, "name": _SIG_NAME},
)

# Ground-truth chatter onset (used for training_intervals and plots)
# _T_GT is None when no chatter is expected (e.g. stable_5mm case)

# ── Indicator configuration — formato estándar CAMP10 ──────────────────────
# Interfaz unificada: f_cycle define el tamaño del ciclo de la ventana.
#   f_cycle = 1/T_rev  → ventana por revolución
#   f_cycle = f_modal  → ventana por periodo modal
#
# Variante Default (clustering, zero-crossings)
config_std = {
    "func":       "Default",
    "params_physical": {
        "f_modal":          _F_MODAL,      # Hz — filtro bandpass
        "f_cycle":          1.0 / _T_REV,  # Hz — ventana por revolución (f_rev)
        # "f_cycle":        _F_MODAL,      # ← si quieres ventana por periodo modal
        "N_cycles_per_seg": _NUM_T,        # ciclos por ventana
        "step_cycles":      1.0,           # step = 1 ciclo

        "data_filtrated":       True,
        "hilbert":              False,
        "while_loop_extend":    False,
        "cycles_cluster_points": 35,
        "thein_sen":            False,
        # --- mu ± 3sigma threshold ---
        "use_area_threshold":   _USE_THR,
        "training_intervals":   _TRAIN_IV,
        "z_sigma":              3.0,
        # --- debug ---
        "debug_level":          2,
        "debug_window_range":   (10, 15),
        "save_figures_windows": False,
        "work_space":           None,
    },
}

# Variante FixedWindow (sin clustering, exponent de Lyapunov σ̂)
config_std_fixed = {
    "func":       "FixedWindow",
    "param_mode": "by_revolution",
    "params_physical": {
        "f_modal":          _F_MODAL,  # Hz — filtro bandpass y ciclo modal
        "f_cycle":          _F_MODAL,  # Hz — ventana por periodo modal
        # "f_cycle":        1.0/_T_REV, # ← si quieres ventana por revolución
        "N_cycles_per_seg": _NUM_T,    # ciclos por ventana
        "step_cycles":      1.0,       # step = 1 ciclo
        "data_filtrated":       True,
        "lambda_ewma":          None, # EWMA para suavizar σ̂ entre ventanas (0 = no suavizado, 1 = suavizado total)
        "accumulate":           True, # acumula áreas de ventanas anteriores para detección (similar a integral acumulada)
        "G_memory":             _T_REV*50, #
        "sigma_method":         "ratio", # frozen_time or ratio
        "sigma_local_n":        5,
        "area_noise_eps":       1e-30,
        "use_area_threshold":   _USE_THR, 
        "training_intervals":   _TRAIN_IV,
        "z_sigma":              3.0,
        "debug_level":          1,
        "debug_window_range":   (10, 15),
        "t_theorical":         _T_GT,  # para plots, no afecta la detección
    },
}



# ── Run indicator — interfaz estándar ──────────────────────────────────────
# choose config and run
config_used = config_std if not USE_FIXED_WINDOW else config_std_fixed
result_std = run_green_std(sig_std, config_used)

meta_r       = result_std.meta
raw          = meta_r["raw_result"]
sig_internal = meta_r["signal"]
t_d          = result_std.t_d
t_d_no_FAR   = result_std.t_d_no_FAR



# ------------------------------------------------------------------
# Debug: create a DebugManager mirroring the internal pipeline settings
# ------------------------------------------------------------------
params_physical = config_used.get("params_physical", {})
dbg_level = int(params_physical.get("debug_level", 0))
dbg_range = params_physical.get("debug_window_range", (0, None))
dbg_save = bool(params_physical.get("save_figures_windows", False))
dbg = DebugManager(debug_level=dbg_level, window_range=dbg_range, save_figures=False)



# =============================================================================
# RESULTADOS -- salida estructurada por nivel de logger
#
#  INFO  : configuracion del indicador y parametros resueltos
# =============================================================================

# ---------- INFO: configuracion del indicador --------------------------------
if logger.isEnabledFor(logging.INFO):
    _KW = 22   # ancho columna clave

    def _kv(key: str, val: str = "", indent: int = 0) -> str:
        pad = "  " * indent
        return f"{pad}{key:<{_KW - 2 * indent}}  {val}"

    def _sep(label: str = "") -> str:
        dash = "\u2500" * 20
        return f"  {dash}  {label}" if label else f"  {dash}"

    trace_r = meta_r.get("resolver_trace", {})

    lines = [
        _kv("Indicador",          result_std.name),
        _kv("Func",               meta_r.get("func", "?")),
        _kv("I_t_meaning",        meta_r.get("I_t_meaning", "?")),
        _kv("vel_source",         meta_r.get("vel_source", "?")),
        _sep(),
        _kv("use_area_threshold", str(meta_r.get("use_area_threshold", False))),
        _sep(),
    ]

    f_cycle_r  = trace_r.get("f_cycle", 0.0)
    T_cycle_r  = trace_r.get("T_cycle", 0.0)
    f_modal_r  = trace_r.get("f_modal", 0.0)
    T_modal_r  = trace_r.get("T_modal", 0.0)
    N_cyc_r    = trace_r.get("N_cycles_per_seg", 0)
    step_c_r   = trace_r.get("step_cycles", 1.0)
    num_T_r    = trace_r.get("resolved_num_T", 0)
    dt_r       = trace_r.get("resolved_dt", 0.0)
    T_win_r    = trace_r.get("T_window_s", 0.0)
    overlap_p  = 1.0 - step_c_r / N_cyc_r if N_cyc_r > 0 else 0.0
    lines += [
        _kv("f_modal",          f"{f_modal_r:.1f} Hz  (filtro bandpass)"),
        _kv("f_cycle",          f"{f_cycle_r:.1f} Hz"
                                f"  (T_cycle = {T_cycle_r*1e3:.3f} ms)"),
        _kv("N_cycles_per_seg", f"{N_cyc_r} ciclos/seg"),
        _kv("step_cycles",      f"{step_c_r}"
                                f"  (overlap = {overlap_p:.1%})"),
        _sep("Resultado"),
        _kv("T_window",  f"{T_win_r*1e3:.3f} ms  = {N_cyc_r} × T_cycle", indent=1),
        _kv("num_T",     f"{num_T_r}  (= ⌈T_window × f_modal⌉)", indent=1),
        _kv("dt",        f"{dt_r*1e3:.3f} ms  = {step_c_r} × T_cycle", indent=1),
        # _kv("Time Theoretical", f"{_T_GT:.4f} s" if _T_GT is not None else "N/A"),
        # _kv(" Time detection ", f"{t_d:.4f} s" if t_d is not None else "N/A"),
    ]

    _bar = "=" * 56
    _hdr = f"\n  {_bar}\n    CONFIGURACION DEL INDICADOR\n  {_bar}"
    logger.info("%s\n%s", _hdr, "\n".join(lines))


# ---------- Imprimir resultados y graficar -----------------------------------
if not USE_FIXED_WINDOW:
    delta_n_median = float(np.nanmedian(result_std.I_t))

    print(f"\nMediana delta_n : {delta_n_median:.4f}")
    print(
        f"Interpretation  : {'UNSTABLE (chatter)' if delta_n_median < 0 else 'STABLE'}"
    )
    print(f"Windows analysed: {len(result_std.t)}")
    if t_d is not None:
        _gt_str = f"{_T_GT:.5f} s" if _T_GT is not None else "N/A"
        print(f"t_d (area thr)  : {t_d:.4f} s  (t_gt = {_gt_str})")
    else:
        print("t_d (area thr)  : not detected (or threshold disabled)")

    plots_green_integral(signal=sig_internal, result=raw)

else:
    sigma_mean = float(np.nanmean(result_std.I_t))

    print(f"\nWindows computed: {len(result_std.t)}")
    print(f"Valid sigma_hat points  : {int(np.sum(np.isfinite(result_std.I_t)))}")
    print(f"Mean sigma_hat          : {sigma_mean:.4f} 1/s")
    if raw.G_hat.size > 0:
        G_final = float(raw.G_hat[-1])
        print(f"Final G_hat             : {G_final:.4f}")
        print(
            f"Interpretation  : {'UNSTABLE (chatter)' if G_final > 0 else 'STABLE'}"
        )
    else:
        print(
            f"Interpretation  : {'UNSTABLE (chatter)' if sigma_mean > 0 else 'STABLE'}"
        )
    if t_d is not None:
        _gt_str = f"{_T_GT:.5f} s" if _T_GT is not None else "N/A"
        print(f"t_d (area thr)  : {t_d[0]:.4f} s  (t_gt = {_gt_str})")
    else:
        print("t_d (area thr)  : not detected (or threshold disabled)")

    plots_fixed_window(
        signal=sig_internal,
        result=raw,
        t_gt=_T_GT,                          # None → no ground-truth line
        training_intervals=_TRAIN_IV,
    )
    # plots_signal_diagnostics(
    #     signal=sig,
    #     result=result_fw,
    #     stable_range=(0.5, 5.0),   # zona estable del cono
    #     zoom_range=(6.6,8),     # zoom 200 ms para ver la señal
    #     eq_smooth_s=0.050,         # 50 ms → moving avg para x_eq
    # )

