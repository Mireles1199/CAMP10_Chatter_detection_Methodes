"""Green Integral Method — example using real HDF5 data.

Usage
-----
    cd indicators/green_integral
    pip install -e .
    python examples/Green_Integral_Detection_NEW.py

Toggle
------
    Set USE_LYAPUNOV = True  to run the no-clustering Lyapunov variant.
    Set USE_LYAPUNOV = False to run the original clustering-based indicator.

    Set USE_EXTERNAL_REFERENCE = True  to train the mu +- z*sigma area
    threshold from the external "stable" signal in `reference_combined.h5`
    (DOE reference-dataset pipeline) instead of `training_intervals`.
    Set USE_EXTERNAL_REFERENCE = False for the original internal-training
    behavior — flip it back and forth to compare both on the same case.

Case selector
-------------
    Set ACTIVE_CASE to one of the keys in CASES to switch signal + config
    together — each entry splits "signal_source" (hdf5_path/case_name/
    disp_name/vel_name/force_name, same shape across the 4 CAMP10
    indicators) from "indicator_config" (everything else, Green-specific).
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
    load_signal,             # ← shared (t, y) loader, raw sim layout or DOE-repackaged layout
    StdSignalData,           # ← standard input  (same shape as MaxEnt / RMS-CV)
    IndicatorResult,         # ← standard output (same shape as MaxEnt / RMS-CV)
    run_green_std,           # ← standard runner  (param_mode + params_physical, see COMMON_TEMPLATE.md)
    plots_green_integral,
    plots_lyapunov,
    plots_signal_diagnostics,
    INFO_PLUS_LEVEL,
)
from green_integral.utils.debug import DebugManager


# -- helpers ------------------------------------------------------------------
def _cut_signal(t, x, time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]


def _load_reference_combined(path: str, channel: str, label: str = "stable") -> StdSignalData:
    """Minimal reader for `reference_combined.h5` (Repo-DOE Fase 1/2 pipeline).

    Layout written by `reference_dataset.py`'s `save_combined` (read here
    directly with h5py instead of importing that Repo-DOE module — same rule
    as the rest of this codebase, see COMMON_TEMPLATE.md §8): one group per
    combined signal named ``f"{label}__{channel}"``, with datasets ``t``/``y``
    (already stitched into one continuous, monotonic time axis) and attrs
    ``fs``/``channel``/``label``/``n_pieces``/``source_ids``. Velocity is
    pulled from the sibling ``f"{label}__Axial_vel"`` group when present,
    otherwise `run_green_std` estimates it via `np.gradient`.
    """
    import h5py

    with h5py.File(path, "r") as f:
        g = f[f"{label}__{channel}"]
        t = g["t"][()]
        y = g["y"][()]
        fs = float(g.attrs["fs"])
        vel_key = f"{label}__Axial_vel"
        velocity = f[vel_key]["y"][()] if vel_key in f else None

    return StdSignalData(
        t_analysis=t,
        signal_analysis=y,
        path=path,
        fs=fs,
        meta={"velocity": velocity, "name": f"reference_{label}_{channel}"},
    )


def main() -> None:
    # ── Toggle ─────────────────────────────────────────────────────────────
    USE_LYAPUNOV: bool = True   # True → Lyapunov (no clustering)
                                      # False → original clustering indicator

    # True  → train mu +- z*sigma threshold from reference_combined.h5's
    #         external "stable" signal (Fase 3, DOE reference dataset).
    # False → original behavior, threshold trained from training_intervals.
    USE_EXTERNAL_REFERENCE: bool = True
    _REFERENCE_COMBINED_H5 = (
        r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
        r"\CAMP10_Chatter_detection_Methodes\Convergency_Simulation"
        r"\4_DOE_Data_Training_Tube\DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0"
        r"\reference_combined.h5"
    )

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

    # ── Case registry ───────────────────────────────────────────────────────
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

    # CASES shape shared across the 4 CAMP10 indicators: each entry splits
    # "signal_source" (hdf5_path/case_name/disp_name/vel_name/force_name —
    # see COMMON_TEMPLATE.md) from "indicator_config" (everything else,
    # content stays Green-specific). Change only ACTIVE_CASE below to pick
    # both the signal and its config together.
    CASES: dict = {
        # ── Original 2DOF cone (chatter onset at 5.366 s) ──────────────────
        "cono": {
            "signal_source": {
                # "hdf5_path": cono_doe_control_sensor,
                # "disp_name": "tool_dyn", "vel_name": "tool_dyn_o", "case_name": None,
                "hdf5_path": cono_doe_control_sensor,
                "disp_name": "Axial_disp", "vel_name": "Axial_vel",
                "force_name": "res_R_p", "case_name": None,
            },
            "indicator_config": {
                "name":               "cono",
                "t_range":            (0.05,16.0),
                "t_gt":               _T_GT,
                "f_modal":            _F_MODAL,  # 150 Hz
                "T_REV":              _T_REV,  # example value, adjust as needed
                "F_REV":              _F_REV,  # example value, adjust as needed
                "num_T":              4,
                "use_area_threshold": True,
                "training_intervals": [
                    (0.05, _T_GT, "stable_1"),
                    # (3.3,  4.46,    "stable_2"),   # tighter stable sub-band
                    # (_T_GT, 10, "stable_1"),
                ],
            },
        },
        # ── Stable case — ap = 5 mm (no chatter) ───────────────────────────
        "stable_5mm": {
            "signal_source": {
                "hdf5_path": (rf"{_BASE}\Chatter-Criteria\CAMP8-Ventanna_Glisante"
                              r"\Nessy2m_Case_Test_Explicit\1DOF_150Hz_5mm\1DOF_150Hz\out.hdf5"),
                "disp_name": "tool_dyn", "vel_name": "tool_dyn_o",
                "force_name": "res_R_p", "case_name": None,
            },
            "indicator_config": {
                "name":               "5mm_stable",
                "t_range":            (0.05, 16.0),
                "t_gt":               _T_GT,          # no chatter in this case
                "f_modal":            _F_MODAL,
                "T_REV":              _T_REV,      # example value, adjust as needed
                "F_REV":              _F_REV,      # example value, adjust as needed
                "num_T":              4,
                "use_area_threshold": False,  # area threshold is noisy for cono but works well for this case
            },
        },
        # ── Chatter case — ap = 15 mm (chatter from ~0.05 s) ───────────────
        "chatter_15mm": {
            "signal_source": {
                "hdf5_path": chatter_15mm,
                "disp_name": "tool_dyn", "vel_name": "tool_dyn_o",
                "force_name": "res_R_p", "case_name": None,
            },
            "indicator_config": {
                "name":               "15mm_chatter",
                "t_range":            (0.05, 16.0),
                "t_gt":               _T_GT,          # chatter after initial transient
                "f_modal":            _F_MODAL,
                "T_REV":              _T_REV,      # example value, adjust as needed
                "F_REV":              _F_REV,      # example value, adjust as needed
                "num_T":              4,
                "use_area_threshold": False,
            },
        },

        "custom_case": {
            "signal_source": {
                # "hdf5_path": r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_180\3\1DOF_150Hz\out.hdf5",
                # "disp_name": "tool_dyn", "vel_name": "tool_dyn_o", "case_name": None,
                # signal_source para sens_out.hdf5 (o "case_name": "case_003" para un doe_results.h5):
                "hdf5_path": custom_case,
                "disp_name": "Axial_disp", "vel_name": "Axial_vel",
                "force_name": "res_R_p", "case_name": None,
            },
            "indicator_config": {
                "name":               "custom",
                "t_range":            (0.05, 16.0),    # adjust
                "t_gt":               _T_GT,           # set if known
                "f_modal":            _F_MODAL,          # adjust based on modal analysis
                "T_REV":              _T_REV,       # example value, adjust as needed
                "F_REV":              _F_REV,       # example value, adjust as needed
                "num_T":              4,             # adjust based on expected cycles in window
                "use_area_threshold": True,          # adjust based on signal characteristics
                "training_intervals": [
                    (0.00, _T_GT, "stable"),             # adjust based on expected stable/chatter intervals
                ],
            },
        },
    }

    # ═══════════════════════════════════════════════════════════════════════
    # CASE SELECTOR — change only this line to switch signal + config together
    # ═══════════════════════════════════════════════════════════════════════
    ACTIVE_CASE      = "cono"   # "cono" | "stable_5mm" | "chatter_15mm" | "custom_case"
    SIGNAL_SOURCE    = CASES[ACTIVE_CASE]["signal_source"]
    INDICATOR_CONFIG = CASES[ACTIVE_CASE]["indicator_config"]

    # ── Unpack active case ────────────────────────────────────────────────
    _cfg        = INDICATOR_CONFIG
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

    # ── Load signal from HDF5 (shared load_signal() — raw sim layout when
    # case_name is None, DOE-repackaged layout when it names a case group) ──
    _src         = SIGNAL_SOURCE
    _HDF5        = _src["hdf5_path"]
    _DISP_NAME   = _src["disp_name"]
    _VEL_NAME    = _src["vel_name"]
    _CASE_NAME   = _src.get("case_name")
    _FORCE_NAME  = _src.get("force_name", "res_R_p")

    data         = HDF5Reader(_HDF5)

    t, tool_dyn  = load_signal(data, _DISP_NAME, _CASE_NAME)
    try:
        _, tool_dyn_vel = load_signal(data, _VEL_NAME, _CASE_NAME)
    except Exception:
        tool_dyn_vel = np.gradient(tool_dyn, t)

    # force channel may not exist in all cases
    try:
        _, force_N = load_signal(data, _FORCE_NAME, _CASE_NAME)
    except Exception:
        force_N = np.zeros_like(t)

    v  = tool_dyn_vel
    fs = 1.0 / (t[1] - t[0])

    t_cut, v_cut  = _cut_signal(t, v,        (_T0, _T1))
    _,     x_cut  = _cut_signal(t, tool_dyn, (_T0, _T1))
    _,     f_cut  = _cut_signal(t, force_N,  (_T0, _T1))
    # ── Build StdSignalData (interfaz estándar CAMP10) ──────────────────────
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

    # ── Optional external reference signal (Fase 3, DOE reference dataset) ──
    # channel="Axial_disp" matches this case's displacement channel above.
    reference_signal_std = (
        _load_reference_combined(_REFERENCE_COMBINED_H5, channel="Axial_disp", label="stable")
        if USE_EXTERNAL_REFERENCE else None
    )

    # ── Indicator configuration — formato estándar CAMP10 ────────────────────
    # param_mode define en qué unidad se mide el ciclo de la ventana:
    #   "by_revolution" → ventana de N_rev_window revoluciones (T_rev)
    #   "by_modal"      → ventana de N_modal_window periodos modales (T_modal = 1/_F_MODAL)
    #
    # Variante Default (clustering, zero-crossings) — ventana por revolución
    config_std = {
        "func":       "Default",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,   # s — periodo de revolución
            "N_rev_window": _NUM_T,   # revoluciones por ventana
            "step_rev":     1.0,      # step = 1 revolución

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
            "debug_window_range":   (8, 15),
            "save_figures_windows": False,
            "work_space":           None,
        },
        # Top-level (sibling to params_physical, not inside it — see
        # COMMON_TEMPLATE.md §3). None (USE_EXTERNAL_REFERENCE=False) =
        # cero impacto, same as before this key existed.
        "reference_signal": reference_signal_std,
    }

    # Variante Lyapunov (sin clustering, exponente de Lyapunov σ̂) — ventana por revolución
    config_std_lyapunov = {
        "func":       "Lyapunov",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,   # s — periodo de revolución
            # Para ventana por periodo modal en vez de por revolución, usar
            # param_mode="by_modal" con "T_modal": 1.0/_F_MODAL en su lugar.
            "N_rev_window": _NUM_T,   # revoluciones por ventana
            "step_rev":     1.0,      # step = 1 revolución
            "data_filtrated":       True,
            "lambda_ewma":          None, # EWMA para suavizar σ̂ entre ventanas (0 = no suavizado, 1 = suavizado total)
            "accumulate":           False, # acumula áreas de ventanas anteriores para detección (similar a integral acumulada)
            "G_memory":             _T_REV*10, #
            "sigma_method":         "ratio", # frozen_time or ratio
            "sigma_local_n":        10,
            "area_noise_eps":       1e-30,
            "use_area_threshold":   _USE_THR,
            "training_intervals":   _TRAIN_IV,
            "z_sigma":              3.0,
            "debug_level":          1,
            "debug_window_range":   (3.92, 16),
            "t_theorical":         _T_GT,  # para plots, no afecta la detección

            "use_zero_crossing_cycles": True, # alpha cycles
            "use_beta_from_cycles": False, # Creat beta from alpha cycles
            "zc_detrend": True,
            "v_cycle_mode": "zero",  # "zero - 1" | "original - dentrend for v=0 , poyecion a Trayectoria Original" |
                                    #  "detrended - detren for v=0 , trayectoria detrend"
            "cycle_area_norm": "none",  # "none" | "mean" | "median"
        },
        # Top-level, same as in config_std — see comment there.
        "reference_signal": reference_signal_std,
    }



    # ── Run indicator — interfaz estándar ────────────────────────────────────
    # choose config and run
    config_used = config_std if not USE_LYAPUNOV else config_std_lyapunov
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

    # ---------- INFO: configuracion del indicador ----------------------------
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


    # ---------- Imprimir resultados y graficar --------------------------------
    if not USE_LYAPUNOV:
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

        plots_lyapunov(
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


if __name__ == "__main__":
    main()
