import logging
from typing import Tuple
import os
import numpy as np

from rms_cv import SignalData
from rms_cv import HDF5Reader
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


# -- helpers ------------------------------------------------------------------
def _cut_signal(t, x, time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]


def _section(title: str, width: int = 54) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"


# -- datos --------------------------------------------------------------------
dir_cono     = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"
dir_path_use = dir_cono

data_dir = os.path.abspath(os.path.join(dir_path_use, "out.hdf5"))
data     = HDF5Reader(data_dir)

tool_dyn     = data.get_element("tool_dyn/data")
t            = tool_dyn[:, 0]
tool_dyn     = tool_dyn[:, 1]
tool_dyn_vel = data.get_element("tool_dyn_o/data")[:, 1]
force_N      = data.get_element("res_R_p/data")[:, 1]

v  = tool_dyn_vel
fs = 1.0 / (t[1] - t[0])

t_cut, v_cut  = _cut_signal(t, v,        (0.00, 15))
_,     x_cut  = _cut_signal(t, tool_dyn, (0.00, 15))
_,     f_cut  = _cut_signal(t, force_N,  (0.00, 15))

# =============================================================================
# INDICATOR_CONFIG -- cuatro modos de parametrizacion
#
#   native                  -> parametros nativos directos (comportamiento original)
#   by_revolution / frames  -> ventana y paso en revoluciones, n_max directo
#   by_revolution / total   -> ventana y paso en revoluciones, K_rev_cv total
#   by_modal      / frames  -> ventana y paso en periodos modales, n_max directo
#
# Cambiar la linea INDICATOR_CONFIG = ... al final del bloque para elegir modo.
# =============================================================================
_RPM     = 12_000.0
_F_MODAL = 150.0
_T_REV   = 60.0 / _RPM        # 0.005 s -- periodo de una revolucion
_T_MODAL = 1.0 / _F_MODAL     # 0.00667 s -- periodo del modo de chatter (150 Hz)

_COMMON = {
    "cv_threshold":         1.9e-2,
    "rms_threshold":        0.9,
    "n_min_cv":             2,
    "warmup_ignore_alerts": False,
    "use_unbiased_std":     True,
    "eps":                  1e-12,
    "detrend":              False,
    "pad_mode":             "none",
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
        "N_rev_window": 5,
        "step_rev":     1,
        "n_max_mode":   "frames",
        "n_max_rev":    28,
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
        "N_modal_window": 5,
        "step_modal":     1,
        "n_max_mode":     "frames",
        "n_max_modal":    10,
        **_COMMON,
    },
}

# -- Selector (descomentar el modo deseado) -----------------------------------
# INDICATOR_CONFIG = INDICATOR_CONFIG_native
# INDICATOR_CONFIG = INDICATOR_CONFIG_by_revolution
# INDICATOR_CONFIG = INDICATOR_CONFIG_by_revolution_total
INDICATOR_CONFIG = INDICATOR_CONFIG_by_modal

# -- Senal de entrada ---------------------------------------------------------
sig = SignalData(
    t_analysis=t_cut,
    signal_analysis=x_cut,
    path=data_dir,
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


# ---------- resultado critico ------------------------------------------------
if t_d is not None and len(t_d) > 0:
    # Chatter detectado -> INFO (igual que SSQ_STFT, dato util siempre visible)
    logger.info(_section("RESULTADO  --  CHATTER DETECTADO"))
    logger.info("  %-24s %s",      "Indicador:",         resultat_rms.name)
    logger.info("  %-24s %s",      "Modo config:",       param_mode)
    logger.info("  %-24s %.5f s",  "Primera deteccion:", t_d[0])
    logger.info("  %-24s %d",      "Total detecciones:", len(t_d))
    logger.info("  %-24s %.2f %%", "% chatter:",         chatter_pct)
    if t_rms is not None and len(t_rms) >= 2:
        logger.info("  %-24s %.4f ms",  "t_rms[0]:",  t_rms[0]  * 1e3)
        logger.info("  %-24s %.4f ms",  "t_rms[1]:",  t_rms[1]  * 1e3)
        logger.info("  %-24s %.4f ms",  "step frames real:", (t_rms[1] - t_rms[0]) * 1e3)
else:
    logger.warning(_section("RESULTADO  --  sin deteccion de chatter"))
    logger.warning("  Indicador: %s  |  Modo: %s", resultat_rms.name, param_mode)


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
            _kv("cv_threshold",       str(p["cv_threshold"]),      indent=1),
            _kv("rms_threshold",      str(p["rms_threshold"]),     indent=1),
            _kv("n_min_cv",           str(p["n_min_cv"]),          indent=1),
            _kv("start_time",         f"{p['start_time']:.4f} s",  indent=1),
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
            _kv("cv_threshold",   str(pp["cv_threshold"]),     indent=1),
            _kv("rms_threshold",  str(pp["rms_threshold"]),    indent=1),
            _kv("n_min_cv",       str(pp["n_min_cv"]),         indent=1),
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
            _kv("cv_threshold",   str(pp["cv_threshold"]),     indent=1),
            _kv("rms_threshold",  str(pp["rms_threshold"]),    indent=1),
            _kv("n_min_cv",       str(pp["n_min_cv"]),         indent=1),
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
vlines = [5.365770208787228, 7.947208594272872]
plots_rms_cv(
    signal=sig, result=resultat_rms,
    show_signal=True, zoom_x=None, zoom_y=None,
    vlines=vlines, hlines=None,
)
