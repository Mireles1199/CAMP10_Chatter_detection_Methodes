import logging
from typing import List, Tuple
import os
import sys
import numpy as np

# Force the local src/ over the fixed-path editable install (see
# indicators/COMMON_TEMPLATE.md §8) so this script always runs against the
# code in this worktree, not whatever checkout `pip install -e` points at.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rms_cv import SignalData
from rms_cv import HDF5Reader
from rms_cv import load_signal
from rms_cv import run_rms_cv
from rms_cv import plots_rms_cv
from rms_cv import INFO_PLUS_LEVEL
from rms_cv.logging_setup import configure_logging, LOGGING_LEVELS

# =============================================================================
# LOGGING -- niveles y contenido mostrado en terminal
#
#   WARNING  ->  solo resultado critico cuando NO hay deteccion
#   INFO     ->  resultado + configuracion del indicador + parametros
#   INFO_PLUS -> + logs internos del pipeline (resoluciones, etc.)  [default]
#   DEBUG    ->  + senal, tabla completa de detecciones
# =============================================================================
# _LOG_LEVEL = LOGGING_LEVELS["warning"]
_LOG_LEVEL = LOGGING_LEVELS["info"]
# _LOG_LEVEL = INFO_PLUS_LEVEL
# _LOG_LEVEL = LOGGING_LEVELS["debug"]

configure_logging(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)

# =============================================================================
# SIGNAL_SOURCE / CASES / ACTIVE_CASE -- declarados dentro de main() (ver mas
# abajo), misma convencion compartida por los 4 indicadores (indicators/
# COMMON_TEMPLATE.md): que senal analizar (signal_source) + con que
# configuracion (indicator_config), uno de los 4 modos propios de RMS-CV
# (native / by_revolution / by_revolution_total / by_modal).
# =============================================================================

# =============================================================================
# FASE 3 -- reference_signal externo (Convergency_Simulation/4_DOE_Data_Training_Tube)
#
#   True  -> arma INDICATOR_CONFIG["reference_signal"] como una lista de
#            tramos (uno por caso) leidos de reference_dataset.h5; cada tramo
#            se ventanea y se monitorea con su propio CVOnlineMonitor (nunca
#            se concatena la senal cruda entre casos -- solo los resultados
#            de CV por tramo se juntan para el pool de mu/sigma). Reemplaza
#            stable_time/frac_stable como region de entrenamiento del CV.
#   False -> comportamiento normal (stable_time/frac_stable de _COMMON, sin tocar).
#
# El combinado reference_combined.h5 NO se toca aqui -- sigue siendo la
# entrada del visor doe_unified_selector.py, solo deja de usarse para
# aprendizaje (mezclaba los tramos en una sola senal, contaminando la costura
# entre casos con el buffer n_max del CV).
#
# Para comparar ambos modos alternar solo esta linea.
# =============================================================================
USE_EXTERNAL_REFERENCE = True

_REFERENCE_H5 = (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
    r"\CAMP10_Chatter_detection_Methodes\Convergency_Simulation"
    r"\4_DOE_Data_Training_Tube\DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0"
    r"\reference_dataset.h5"
)


# -- helpers ------------------------------------------------------------------
def _cut_signal(t, x, time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]


def _load_reference_pieces(path: str, label: str = "stable", channel: str = "Axial_vel") -> List[SignalData]:
    """Read every per-case piece of one label/channel from `reference_dataset.py
    build`'s output (Repo-DOE) directly with h5py -- never import that package
    from here (indicators/COMMON_TEMPLATE.md prohibits cross-indicator imports).

    Layout: /<label>/<case>/<channel>__NNN/{t, y}, attrs channel/fs/signal_id.
    Returns one SignalData per piece (case), kept separate on purpose -- the
    caller windows/monitors each piece independently instead of concatenating
    the raw signals, so the CV pool never mixes the tail of one case with the
    head of the next.
    """
    import h5py

    out: List[SignalData] = []
    with h5py.File(path, "r") as f:
        for case in f[label]:
            for piece in f[label][case].values():
                if piece.attrs["channel"] != channel:
                    continue
                out.append(SignalData(
                    t_analysis=piece["t"][()], signal_analysis=piece["y"][()],
                    path=path, fs=float(piece.attrs["fs"]),
                    meta={"label": label, "channel": channel, "signal_id": piece.attrs["signal_id"]},
                ))
    return out


def _section(title: str, width: int = 54) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"


def main() -> None:
    # -- datos ----------------------------------------------------------------
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

        "tubo_stable_6_88e_5" : (
            r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
            r"\Chatter-Criteria\CAMP10_Chatter_detection_Methodes"
            r"\Convergency_Simulation\1_Detection_Limite_Lobes"
            r"\DOE_Detection_Limite_Lobes_dxl_20e-5_RUN_10"
            r"\4\1DOF_150Hz\sens_out.hdf5"
        ),

    }

    # See COMMON_TEMPLATE.md §11 -- forma estándar de declarar el origen de la señal.
    _SIGNAL_SOURCE = {
        "hdf5_path": _DATA_DIRS["tubo_stable_6_88e_5"],
        "case_name": None,  # None (layout crudo) | "case_003" (layout DOE)
        "disp_name": "Axial_disp",
        "vel_name": "Axial_vel",
        "force_name": "force_N",
    }

    data = HDF5Reader(_SIGNAL_SOURCE["hdf5_path"])

    t, tool_dyn      = load_signal(data, _SIGNAL_SOURCE["disp_name"], _SIGNAL_SOURCE["case_name"])
    _, tool_dyn_vel  = load_signal(data, _SIGNAL_SOURCE["vel_name"],  _SIGNAL_SOURCE["case_name"])

    try:
        _, force_N = load_signal(data, _SIGNAL_SOURCE["force_name"], _SIGNAL_SOURCE["case_name"])
    except KeyError:
        force_N = np.zeros_like(t)



    v  = tool_dyn_vel
    fs = 1.0 / (t[1] - t[0])

    _CUT_START = 0.05
    t_cut, v_cut  = _cut_signal(t, v,        (_CUT_START, 15))
    _,     x_cut  = _cut_signal(t, tool_dyn, (_CUT_START, 15))
    _,     f_cut  = _cut_signal(t, force_N,  (_CUT_START, 15))

    # =============================================================================
    # INDICATOR_CONFIG -- cuatro modos de parametrizacion
    #
    #   native                  -> parametros nativos directos (comportamiento original)
    #   by_revolution / frames  -> ventana y paso en revoluciones, n_max directo
    #   by_revolution / total   -> ventana y paso en revoluciones, K_rev_cv total
    #   by_modal      / frames  -> ventana y paso en periodos modales, n_max directo
    #
    # Cambiar ACTIVE_CASE al final del bloque para elegir modo.
    # =============================================================================
    _RPM     = 12_000.0
    _F_MODAL = 150.0
    _T_REV   = 60.0 / _RPM        # 0.005 s -- periodo de una revolucion
    _T_MODAL = 1.0 / _F_MODAL     # 0.00667 s -- periodo del modo de chatter (150 Hz)
    _T_GT  = 5.365770208787228   # [s] ground-truth chatter onset
    _COMMON = {
        # fixed threshold (ignored when stable_time is set)
        "cv_threshold":         None,
        "rms_threshold":        None,
        "n_min_cv":             2,
        "warmup_ignore_alerts": False,
        "use_unbiased_std":     True,
        "eps":                  1e-12,
        "detrend":              False,
        "pad_mode":             "none",
        # ── adaptive threshold: 3-sigma on CV of stable region ──────────────
        # "stable_time":  (0.0, _T_GT),   # seconds: region known to be stable
        # "frac_stable":  0.3610633440512648,         # fallback if stable_time yields no frames
        "z":            3.0,
        "alpha":        0.05,
        "fallback_mad": True,
        "t_theorical":   _T_GT,  # for debug/plots, not used in detection
    }

    # -- 1. Modo nativo -----------------------------------------------------------
    INDICATOR_CONFIG_native = {
        "id":   "RMS_CV",
        "func": "Default",
        "params": {
            "n_max":              28,
            "samples_per_window": 4000,
            "overlap_pct":        0.0,
            **_COMMON,
        },
    }

    # -- 2. Modo by_revolution / n_max_mode="frames" ------------------------------
    #   N_rev_window=16, step_rev=8  -> overlap_pct = 0.5
    #   samples_per_window = ceil(16 x 0.005 x 50000) = 4000 samples
    #   n_max = n_max_rev = 28 (directo)
    INDICATOR_CONFIG_by_revolution = {
        "id":         "RMS_CV",
        "func":       "Default",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,
            "N_rev_window": 32,
            "step_rev":     4,
            "n_max_mode":   "frames",
            "n_max_rev":    4,
            **_COMMON,
        },
    }

    # -- 3. Modo by_revolution / n_max_mode="total_window" ------------------------
    #   K_rev_cv = N_win + (n_max-1)*step = 16 + 27*16 = 448 revoluciones
    #   -> n_max = ceil((448 - 16) / 16 + 1) = 28
    INDICATOR_CONFIG_by_revolution_total = {
        "id":         "RMS_CV",
        "func":       "Default",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,
            "N_rev_window": 16,
            "step_rev":     16,
            "n_max_mode":   "total_window",
            "K_rev_cv":     448,            # -> n_max = 28
            **_COMMON,
        },
    }

    # -- 4. Modo by_modal / n_max_mode="frames" -----------------------------------
    #   N_modal_window=5, step_modal=0.5  -> overlap_pct = 0.0
    #   samples_per_window = ceil(5 x 0.00667 x 50000) = 1667 samples
    #   n_max = 28 (directo)
    INDICATOR_CONFIG_by_modal = {
        "id":         "RMS_CV",
        "func":       "Default",
        "param_mode": "by_modal",
        "params_physical": {
            "T_modal":        _T_MODAL,
            "N_modal_window": 1,
            "step_modal":     1,
            "n_max_mode":     "frames",
            "n_max_modal":    16,
            **_COMMON,
        },
    }

    CASES: dict = {
        "native":              {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_native},
        "by_revolution":       {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_by_revolution},
        "by_revolution_total": {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_by_revolution_total},
        "by_modal":            {"signal_source": _SIGNAL_SOURCE, "indicator_config": INDICATOR_CONFIG_by_modal},
    }
    ACTIVE_CASE = "by_revolution"   # <- cambiar solo esta linea para elegir senal + config

    SIGNAL_SOURCE    = CASES[ACTIVE_CASE]["signal_source"]
    INDICATOR_CONFIG = CASES[ACTIVE_CASE]["indicator_config"]

    # -- Fase 3: reference_signal externo (opcional) ------------------------------
    # Alternar con USE_EXTERNAL_REFERENCE (arriba del archivo) para comparar con y
    # sin la referencia externa. Comentar esta linea tambien vuelve al modo normal.
    if USE_EXTERNAL_REFERENCE:
        INDICATOR_CONFIG["reference_signal"] = _load_reference_pieces(
            _REFERENCE_H5, label="stable", channel="Axial_vel"
        )

    # -- Senal de entrada ---------------------------------------------------------
    sig = SignalData(
        t_analysis=t_cut,
        signal_analysis=v_cut,
        path=SIGNAL_SOURCE["hdf5_path"],
        fs=fs,
        meta={"AP": "5mm-15mm", "RPM": 12_000},
    )

    # =============================================================================

    # EJECUCION
    # =============================================================================
    resultat_rms = run_rms_cv(sig, INDICATOR_CONFIG)

    # =============================================================================
    # RESULTADOS -- salida estructurada por nivel de logger
    #
    #  WARNING : resultado critico cuando NO hay deteccion
    #  INFO    : resultado + configuracion del indicador y parametros usados
    #  DEBUG   : + senal, tabla completa de detecciones
    # =============================================================================
    meta       = resultat_rms.meta
    param_mode = meta.get("param_mode", "native")
    t_d        = resultat_rms.t_d
    t_rms      = resultat_rms.t        # array de tiempos de los frames RMS

    # -- % chatter (frames marcados / total frames) --------------------------------
    _n_rms_frames = len(t_rms) if t_rms is not None else 0
    chatter_pct   = (len(t_d) / _n_rms_frames * 100.0) if _n_rms_frames > 0 and t_d is not None and len(t_d) > 0 else 0.0




    # ---------- INFO: configuracion del indicador ---------------------------------
    if logger.isEnabledFor(logging.INFO):
        _KW = 28   # ancho columna clave

        def _kv(key: str, val: str = "", indent: int = 0) -> str:
            pad = "  " * indent
            return f"{pad}{key:<{_KW - 2 * indent}}  {val}"

        def _sep(label: str = "") -> str:
            dash = "\u2500" * 22
            return f"  {dash}  {label}" if label else f"  {dash}"

        # -- cabecera comun -------------------------------------------------------
        lines = [
            _kv("Indicador", resultat_rms.name),
            _kv("Modo",      param_mode),
            _sep(),
        ]

        if param_mode == "native":
            # --- parametros nativos directos ------------------------------------
            p     = INDICATOR_CONFIG["params"]
            t_win = p["samples_per_window"] / sig.fs
            lines += [
                _kv("samples_per_window", f"{p['samples_per_window']} samples"
                                          f"  ({t_win * 1e3:.3f} ms)"),
                _kv("overlap_pct",        f"{p['overlap_pct']:.4f}"),
                _kv("n_max",              str(p["n_max"])),
                _sep("Thresholds"),
                _kv("cv_threshold_method", meta.get("cv_threshold_method", "-"),       indent=1),
                _kv("training_source",     meta.get("training_source", "-"),           indent=1),
                _kv("cv_threshold_used",   f"{meta.get('cv_threshold_used', '-'):.6g}", indent=1),
                _kv("metodo_umbral",       str(meta.get("metodo_umbral", "-")),         indent=1),
                _kv("mu_stable",           f"{meta.get('mu_stable',    float('nan')):.6g}", indent=1),
                _kv("sigma_stable",        f"{meta.get('sigma_stable', float('nan')):.6g}", indent=1),
                _kv("normal_ok",           str(meta.get("normal_ok", "-")),             indent=1),
                _kv("n_min_cv",            str(p["n_min_cv"]),                          indent=1),
            ]

        elif param_mode == "by_revolution":
            # --- entrada en revoluciones -> resultado derivado ------------------
            pp  = INDICATOR_CONFIG["params_physical"]
            nat = meta.get("native_params_resolved", {})
            lines += [
                _kv("T_rev",        f"{pp['T_rev'] * 1e3:.3f} ms"
                                    f"  (rpm = {60 / pp['T_rev']:.1f})"),
                _kv("N_rev_window", f"{pp['N_rev_window']} rev"),
                _kv("step_rev",     f"{pp['step_rev']} rev"),
                _kv("n_max_mode",   pp.get("n_max_mode", "frames")),
            ]
            if pp.get("n_max_mode", "frames") == "frames":
                lines.append(_kv("n_max_rev", str(pp.get("n_max_rev", "-"))))
            else:
                lines.append(_kv("K_rev_cv", f"{pp.get('K_rev_cv', '-')} rev"))
            lines += [
                _sep("Ventana"),
                _kv("t_win deseado",  f"{meta.get('t_win_exact_ms', 0):.4f} ms",  indent=1),
                _kv("t_win efectivo", f"{meta.get('t_win_real_ms',  0):.4f} ms",  indent=1),
                _kv("delta t_win",    f"+{abs(meta.get('t_win_real_ms', 0) - meta.get('t_win_exact_ms', 0)) * 1e3:.3f} us",
                                      indent=1),
                _kv("samples_per_window", f"{nat.get('samples_per_window', '-')} samples", indent=1),
                _sep("Paso RMS"),
                _kv("step deseado",   f"{meta.get('dt_rms_exact_ms', 0):.4f} ms", indent=1),
                _kv("step efectivo",  f"{meta.get('dt_rms_real_ms',  0):.4f} ms", indent=1),
                _kv("delta step",     f"+{abs(meta.get('dt_rms_real_ms', 0) - meta.get('dt_rms_exact_ms', 0)) * 1e3:.3f} us",
                                      indent=1),
                _kv("overlap_pct",    f"{nat.get('overlap_pct', 0):.4f}",          indent=1),
                _sep("Ventana CV"),
                _kv("n_max",          str(nat.get("n_max", "-")),                  indent=1),
                _kv("t_cv_total deseado",
                    f"{meta.get('t_cv_total_exact_s', 0) * 1e3:.3f} ms"
                    f"  ({meta.get('K_cv_total_exact_units', 0):.2f} rev)",
                    indent=1),
                _kv("t_cv_total efectivo",
                    f"{meta.get('t_cv_total_s', 0) * 1e3:.3f} ms"
                    f"  ({meta.get('K_cv_total_units', 0):.2f} rev)",
                    indent=1),
                _sep("Thresholds"),
                _kv("cv_threshold_method", meta.get("cv_threshold_method", "-"),       indent=1),
                _kv("training_source",     meta.get("training_source", "-"),           indent=1),
                _kv("cv_threshold_used",   f"{meta.get('cv_threshold_used', '-'):.6g}", indent=1),
                _kv("metodo_umbral",       str(meta.get("metodo_umbral", "-")),         indent=1),
                _kv("mu_stable",           f"{meta.get('mu_stable',    float('nan')):.6g}", indent=1),
                _kv("sigma_stable",        f"{meta.get('sigma_stable', float('nan')):.6g}", indent=1),
                _kv("normal_ok",           str(meta.get("normal_ok", "-")),             indent=1),
                _kv("n_min_cv",            str(pp["n_min_cv"]),                         indent=1),
            ]

        elif param_mode == "by_modal":
            # --- entrada en periodos modales -> resultado derivado ---------------
            pp  = INDICATOR_CONFIG["params_physical"]
            nat = meta.get("native_params_resolved", {})
            lines += [
                _kv("T_modal",        f"{pp['T_modal'] * 1e3:.4f} ms"
                                      f"  (f = {1 / pp['T_modal']:.1f} Hz)"),
                _kv("N_modal_window", f"{pp['N_modal_window']} periodos"),
                _kv("step_modal",     f"{pp['step_modal']} periodo(s)"),
                _kv("n_max_mode",     pp.get("n_max_mode", "frames")),
            ]
            if pp.get("n_max_mode", "frames") == "frames":
                lines.append(_kv("n_max_modal", str(pp.get("n_max_modal", "-"))))
            else:
                lines.append(_kv("K_modal_cv", f"{pp.get('K_modal_cv', '-')} periodos"))
            lines += [
                _sep("Ventana"),
                _kv("t_win deseado",  f"{meta.get('t_win_exact_ms', 0):.4f} ms",  indent=1),
                _kv("t_win efectivo", f"{meta.get('t_win_real_ms',  0):.4f} ms",  indent=1),
                _kv("delta t_win",    f"+{abs(meta.get('t_win_real_ms', 0) - meta.get('t_win_exact_ms', 0)) * 1e3:.3f} us",
                                      indent=1),
                _kv("samples_per_window", f"{nat.get('samples_per_window', '-')} samples", indent=1),
                _sep("Paso RMS"),
                _kv("step deseado",   f"{meta.get('dt_rms_exact_ms', 0):.4f} ms", indent=1),
                _kv("step efectivo",  f"{meta.get('dt_rms_real_ms',  0):.4f} ms", indent=1),
                _kv("delta step",     f"+{abs(meta.get('dt_rms_real_ms', 0) - meta.get('dt_rms_exact_ms', 0)) * 1e3:.3f} us",
                                      indent=1),
                _kv("overlap_pct",    f"{nat.get('overlap_pct', 0):.4f}",          indent=1),
                _sep("Ventana CV"),
                _kv("n_max",          str(nat.get("n_max", "-")),                  indent=1),
                _kv("t_cv_total deseado",
                    f"{meta.get('t_cv_total_exact_s', 0) * 1e3:.3f} ms"
                    f"  ({meta.get('K_cv_total_exact_units', 0):.2f} periodos)",
                    indent=1),
                _kv("t_cv_total efectivo",
                    f"{meta.get('t_cv_total_s', 0) * 1e3:.3f} ms"
                    f"  ({meta.get('K_cv_total_units', 0):.2f} periodos)",
                    indent=1),
                _sep("Thresholds"),
                _kv("cv_threshold_method", meta.get("cv_threshold_method", "-"),       indent=1),
                _kv("training_source",     meta.get("training_source", "-"),           indent=1),
                _kv("cv_threshold_used",   f"{meta.get('cv_threshold_used', '-'):.6g}", indent=1),
                _kv("metodo_umbral",       str(meta.get("metodo_umbral", "-")),         indent=1),
                _kv("mu_stable",           f"{meta.get('mu_stable',    float('nan')):.6g}", indent=1),
                _kv("sigma_stable",        f"{meta.get('sigma_stable', float('nan')):.6g}", indent=1),
                _kv("normal_ok",           str(meta.get("normal_ok", "-")),             indent=1),
                _kv("n_min_cv",            str(pp["n_min_cv"]),                         indent=1),
            ]

        logger.info("%s\n%s", _section("CONFIGURACION DEL INDICADOR"), "\n".join(lines))


    # ---------- DEBUG: senal y tabla de detecciones -------------------------------
    if logger.isEnabledFor(logging.DEBUG):
        import pandas as pd

        _n_frames = len(t_rms) if t_rms is not None else 0
        logger.debug("%s", _section("SENAL"))
        logger.debug("  %-28s %g Hz",   "fs:",            sig.fs)
        logger.debug("  %-28s %.4f s",  "Duracion:",      len(sig.signal_analysis) / sig.fs)
        logger.debug("  %-28s %d",      "Muestras:",      len(sig.signal_analysis))
        logger.debug("  %-28s %d",      "Frames RMS:",    _n_frames)
        if t_rms is not None and _n_frames >= 1:
            logger.debug("  %-28s %.4f ms", "t_rms[0]:",  t_rms[0]  * 1e3)
            logger.debug("  %-28s %.4f ms", "t_rms[-1]:", t_rms[-1] * 1e3)
        if t_rms is not None and _n_frames >= 2:
            logger.debug("  %-28s %.4f ms", "step efectivo frames:", (t_rms[1] - t_rms[0]) * 1e3)

        logger.debug("%s", _section("DETECCIONES"))
        if t_d is not None and len(t_d) > 0:
            # usa resultat_rms.I_t (array del indicador) para la columna CV
            _i_t_all = np.asarray(resultat_rms.I_t)
            _t_all   = np.asarray(t_rms) if t_rms is not None else np.array([])
            if len(_i_t_all) > 0 and len(_t_all) > 0:
                _idx   = np.clip(np.searchsorted(_t_all, t_d), 0, len(_i_t_all) - 1)
                _i_det = _i_t_all[_idx]
            else:
                _i_det = np.full(len(t_d), float("nan"))
            df_det = pd.DataFrame({
                "t [s]": t_d,
                "CV":    _i_det,
            })
            logger.debug("\n%s", df_det.to_string(index=False))
        else:
            logger.debug("  (sin detecciones)")


    # =============================================================================
    # GRAFICA
    # =============================================================================
    plots_rms_cv(
        signal=sig, result=resultat_rms,
        show_signal=True, zoom_x=None, zoom_y=None,
        vlines=None, hlines=None,
        t_gt=_T_GT,
    )


if __name__ == "__main__":
    main()
