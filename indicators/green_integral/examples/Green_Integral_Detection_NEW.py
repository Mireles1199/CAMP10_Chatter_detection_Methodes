"""Green Integral Method — example using real HDF5 data.

Usage
-----
    cd indicators/green_integral
    pip install -e .
    python examples/Green_Integral_Detection_NEW.py

Case selector
-------------
    Set ACTIVE_CASE to one of the keys in CASES to switch parametrization
    mode + func together — same CASES/ACTIVE_CASE/signal_source/
    indicator_config skeleton as MaxEnt/RMS-CV/SST (COMMON_TEMPLATE.md):
    one _SIGNAL_SOURCE shared by every case (this example always analyzes
    the same real signal), and INDICATOR_CONFIG_<func>_<param_mode> holding
    the complete run_green_std() config for that combination — no separate
    per-signal case registry, no USE_LYAPUNOV toggle (func is now part of
    ACTIVE_CASE): "std_native" | "std_by_revolution" | "std_by_modal" |
    "lyapunov_native" | "lyapunov_by_revolution" | "lyapunov_by_modal".

    Set USE_EXTERNAL_REFERENCE = True  to train the mu +- z*sigma area
    threshold from the external "stable" signal in `reference_combined.h5`
    (DOE reference-dataset pipeline) instead of `training_intervals`.
    Set USE_EXTERNAL_REFERENCE = False for the original internal-training
    behavior — flip it back and forth to compare both on the same case.
"""

from typing import Tuple
import logging
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
    # -- fuente de la señal ---------------------------------------------------

    # Rutas alternativas de datasets -- cambiar DATA_DIR para usar otra.
    _DATA_DIRS = {
        "control": (
            r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
            r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
            r"\3\1DOF_150Hz\out.hdf5"
        ),
        "control_sensor": (
            r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
            r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
            r"\5\1DOF_150Hz\sens_out.hdf5"
        ),
        "custom": (
            r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
            r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
            r"\0\1DOF_150Hz\sens_out.hdf5"
        ),
        "custom_dir": (
            r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
            r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_180"
            r"\12\1DOF_150Hz\sens_out.hdf5"
        ),
        "cono_dexel_20e_5": (
            r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
            r"\2DOF_Cone_New\Cono_dexel_20e-5_dt_200\0\1DOF_150Hz\sens_out.hdf5"
        ),
    }

    _SIGNAL_SOURCE = {
        "hdf5_path": _DATA_DIRS["cono_dexel_20e_5"],
        "case_name": None,
        "disp_name": "Axial_disp",
        "vel_name": "Axial_vel",
        "force_name": "res_R_p",
    }
    _SIG_NAME = "cono"

    data = HDF5Reader(_SIGNAL_SOURCE["hdf5_path"])
    t, tool_dyn = load_signal(data, _SIGNAL_SOURCE["disp_name"], _SIGNAL_SOURCE["case_name"])
    try:
        _, v = load_signal(data, _SIGNAL_SOURCE["vel_name"], _SIGNAL_SOURCE["case_name"])
    except Exception:
        v = np.gradient(tool_dyn, t)
    try:
        _, force_N = load_signal(data, _SIGNAL_SOURCE["force_name"], _SIGNAL_SOURCE["case_name"])
    except Exception:
        force_N = np.zeros_like(t)

    _CUT_START = 0.05
    t_cut, v_cut  = _cut_signal(t, v,        (_CUT_START, 16))
    _,     x_cut  = _cut_signal(t, tool_dyn, (_CUT_START, 16))
    _,     f_cut  = _cut_signal(t, force_N,  (_CUT_START, 16))

    fs = 1.0 / (t[1] - t[0])

    # =============================================================================
    # INDICATOR_CONFIG -- seis variantes: {std, lyapunov} x {native, by_revolution,
    # by_modal}. Cambiar ACTIVE_CASE (bloque CASES, más abajo) para elegir modo.
    # =============================================================================
    _RPM     = 12_000.0
    _F_MODAL = 150.0
    _T_REV   = 60.0 / _RPM        # 0.005 s -- periodo de una revolucion
    _T_MODAL = 1.0 / _F_MODAL     # s -- periodo del modo de chatter (f_modal ~ 150 Hz)
    _T_GT    = 5.365770208787228  # [s] ground-truth chatter onset

    # -- parámetros compartidos por TODOS los modos y funcs ----------------------
    _COMMON_ALL = {
        "use_area_threshold": True,
        # "training_intervals": [
        #     (_CUT_START, _T_GT, "stable_1"),
        #     # (3.3,  4.46,    "stable_2"),   # tighter stable sub-band
        # ],
        "z_sigma":            3.0,
        "debug_level":        2,
        "debug_window_range": (8, 15),
        "save_figures_windows": False,
        "work_space":          None,
    }
    # -- extras propios de la variante Default (clustering, zero-crossings) ------
    _COMMON_STD = {
        "data_filtrated":        True,
        "hilbert":                False,
        "while_loop_extend":      False,
        "cycles_cluster_points":  35,
        "thein_sen":              False,
    }
    # -- extras propios de la variante Lyapunov (sin clustering, exponente σ̂) ----
    _COMMON_LYAPUNOV = {
        "data_filtrated":       True,
        "lambda_ewma":          None,   # EWMA para suavizar σ̂ (None = sin suavizado)
        "accumulate":           False,  # acumula áreas de ventanas anteriores (integral acumulada)
        "G_memory":             _T_REV * 10,
        "sigma_method":         "ratio",  # "ratio" | "frozen_time"
        "sigma_local_n":        10,
        "area_noise_eps":       1e-30,
        "debug_level":          1,          # overrides _COMMON_ALL's debug_level for this func
        "debug_window_range":   (3.92, 16), # overrides _COMMON_ALL's range for this func
        "t_theorical":          _T_GT,      # para plots, no afecta la detección
        "use_zero_crossing_cycles": True,   # alpha cycles
        "use_beta_from_cycles":     False,  # beta = union de ciclos completos
        "zc_detrend":                True,
        "v_cycle_mode":              "zero",  # "zero" | "original" | "detrended"
        "cycle_area_norm":           "none",  # "none" | "mean" | "median"
    }

    # -- 1. Modo nativo (Default) --------------------------------------------------
    INDICATOR_CONFIG_std_native = {
        "id":   "Green_Integral",
        "func": "Default",
        "params": {
            "f_modal": _F_MODAL,
            "num_T":   4,
            "dt":      _T_REV,   # step entre ventanas [s]
            **_COMMON_ALL, **_COMMON_STD,
        },
    }

    # -- 2. Modo by_revolution (Default) ------------------------------------------
    INDICATOR_CONFIG_std_by_revolution = {
        "id":         "Green_Integral",
        "func":       "Default",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,
            "N_rev_window": 4,
            "step_rev":     1.0,
            **_COMMON_ALL, **_COMMON_STD,
        },
    }

    # -- 3. Modo by_modal (Default) ------------------------------------------------
    INDICATOR_CONFIG_std_by_modal = {
        "id":         "Green_Integral",
        "func":       "Default",
        "param_mode": "by_modal",
        "params_physical": {
            "T_modal":        _T_MODAL,
            "N_modal_window": 4,
            "step_modal":     1.0,
            **_COMMON_ALL, **_COMMON_STD,
        },
    }

    # -- 4. Modo nativo (Lyapunov) --------------------------------------------------
    INDICATOR_CONFIG_lyapunov_native = {
        "id":   "Green_Integral",
        "func": "Lyapunov",
        "params": {
            "f_modal": _F_MODAL,
            "num_T":   4,
            "dt":      _T_REV,
            **_COMMON_ALL, **_COMMON_LYAPUNOV,
        },
    }

    # -- 5. Modo by_revolution (Lyapunov) -------------------------------------------
    INDICATOR_CONFIG_lyapunov_by_revolution = {
        "id":         "Green_Integral",
        "func":       "Lyapunov",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,
            "N_rev_window": 4,
            "step_rev":     1.0,
            **_COMMON_ALL, **_COMMON_LYAPUNOV,
        },
    }

    # -- 6. Modo by_modal (Lyapunov) -------------------------------------------------
    INDICATOR_CONFIG_lyapunov_by_modal = {
        "id":         "Green_Integral",
        "func":       "Lyapunov",
        "param_mode": "by_modal",
        "params_physical": {
            "T_modal":        _T_MODAL,
            "N_modal_window": 4,
            "step_modal":     1.0,
            **_COMMON_ALL, **_COMMON_LYAPUNOV,
        },
    }

    # -- CASES / ACTIVE_CASE -------------------------------------------------------
    # Mismo esqueleto en los 4 indicadores (COMMON_TEMPLATE.md): cada caso
    # empaqueta "signal_source" (misma forma en los 4) + "indicator_config"
    # (propio de Green — se pasa directo a run_green_std). Cambiar solo
    # ACTIVE_CASE para elegir func + modo de parametrización.
    CASES: dict = {
        "std_native":             {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_std_native},
        "std_by_revolution":      {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_std_by_revolution},
        "std_by_modal":           {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_std_by_modal},
        "lyapunov_native":        {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_lyapunov_native},
        "lyapunov_by_revolution": {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_lyapunov_by_revolution},
        "lyapunov_by_modal":      {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_lyapunov_by_modal},
    }
    ACTIVE_CASE = "lyapunov_by_revolution"   # <- cambiar solo esta línea para elegir func + modo

    SIGNAL_SOURCE    = CASES[ACTIVE_CASE]["signal_source"]
    INDICATOR_CONFIG = CASES[ACTIVE_CASE]["indicator_config"]
    is_lyapunov      = INDICATOR_CONFIG["func"] == "Lyapunov"

    # =============================================================================
    # REFERENCE SIGNAL (Fase 3) -- entrenar contra una señal "stable" EXTERNA ya
    # etiquetada (pipeline Repo-DOE, combinada por label) en vez de recortar
    # training_intervals de la propia señal analizada. Tiene prioridad sobre
    # training_intervals cuando ambos están presentes (ver runner.py/runner_lyapunov.py).
    #
    #   USE_EXTERNAL_REFERENCE = True   -> usa reference_signal (este bloque)
    #   USE_EXTERNAL_REFERENCE = False  -> comportamiento normal, sin cambios
    #                                       (training_intervals de _COMMON_ALL)
    # =============================================================================
    USE_EXTERNAL_REFERENCE = True
    _REFERENCE_COMBINED_H5 = (
        r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
        r"\CAMP10_Chatter_detection_Methodes\Convergency_Simulation"
        r"\4_DOE_Data_Training_Tube\DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0"
        r"\reference_combined.h5"
    )
    INDICATOR_CONFIG["reference_signal"] = (
        _load_reference_combined(_REFERENCE_COMBINED_H5, channel="Axial_disp", label="stable")
        if USE_EXTERNAL_REFERENCE else None
    )

    # ── Build StdSignalData (interfaz estándar CAMP10) ──────────────────────
    # signal_analysis = desplazamiento; velocidad va en meta["velocity"]
    sig_std = StdSignalData(
        t_analysis=t_cut,
        signal_analysis=x_cut,
        path=SIGNAL_SOURCE["hdf5_path"],
        fs=fs,
        meta={"velocity": v_cut, "name": _SIG_NAME},
    )

    # ── Run indicator — interfaz estándar ────────────────────────────────────
    result_std = run_green_std(sig_std, INDICATOR_CONFIG)

    meta_r       = result_std.meta
    raw          = meta_r["raw_result"]
    sig_internal = meta_r["signal"]
    t_d          = result_std.t_d
    t_d_no_FAR   = result_std.t_d_no_FAR

    # ------------------------------------------------------------------
    # Debug: create a DebugManager mirroring the internal pipeline settings
    # (params_physical for by_revolution/by_modal, params for native)
    # ------------------------------------------------------------------
    _active_params = INDICATOR_CONFIG.get("params_physical") or INDICATOR_CONFIG.get("params", {})
    dbg_level = int(_active_params.get("debug_level", 0))
    dbg_range = _active_params.get("debug_window_range", (0, None))
    dbg_save = bool(_active_params.get("save_figures_windows", False))
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
        ]

        _bar = "=" * 56
        _hdr = f"\n  {_bar}\n    CONFIGURACION DEL INDICADOR\n  {_bar}"
        logger.info("%s\n%s", _hdr, "\n".join(lines))

    # ---------- Imprimir resultados y graficar --------------------------------
    if not is_lyapunov:
        delta_n_median = float(np.nanmedian(result_std.I_t))

        print(f"\nMediana delta_n : {delta_n_median:.4f}")
        print(
            f"Interpretation  : {'UNSTABLE (chatter)' if delta_n_median < 0 else 'STABLE'}"
        )
        print(f"Windows analysed: {len(result_std.t)}")
        if t_d.size > 0:
            _gt_str = f"{_T_GT:.5f} s" if _T_GT is not None else "N/A"
            print(f"t_d (area thr)  : {t_d[0]:.4f} s  (t_gt = {_gt_str})")
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
        if t_d.size > 0:
            _gt_str = f"{_T_GT:.5f} s" if _T_GT is not None else "N/A"
            print(f"t_d (area thr)  : {t_d[0]:.4f} s  (t_gt = {_gt_str})")
        else:
            print("t_d (area thr)  : not detected (or threshold disabled)")

        plots_lyapunov(
            signal=sig_internal,
            result=raw,
            t_gt=_T_GT,                          # None → no ground-truth line
            training_intervals=_active_params.get("training_intervals", []),
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
