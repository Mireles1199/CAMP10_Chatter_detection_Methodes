import logging
from typing import Tuple
import os
import numpy as np

from ssq_chatter import SignalData, HDF5Reader
from ssq_chatter import run_sst_svd
from ssq_chatter import plots_sst_svd

# =============================================================================
# LOGGING -- niveles y contenido mostrado en terminal
#
#   WARNING  ->  solo resultado critico (primera deteccion, % chatter)
#   INFO     ->  + configuracion del indicador + parametros
#   INFO_PLUS -> + logs internos del pipeline (OPR, modelos, SPRT...)  [default]
#   DEBUG    ->  + senal, tabla completa de detecciones
# =============================================================================
from ssq_chatter import INFO_PLUS_LEVEL
from ssq_chatter.logging_setup import configure_logging, LOGGING_LEVELS



# _LOG_LEVEL = LOGGING_LEVELS["warning"]
# _LOG_LEVEL = LOGGING_LEVELS["info"]
_LOG_LEVEL =  INFO_PLUS_LEVEL
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

custome =  (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200\doe_noise_results.h5"
)

work_space_5mm   = 'D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Chatter-Criteria/CAMP8-Ventanna_Glisante/Nessy2m_Case_Test_Explicit/1DOF_150Hz_5mm/1DOF_150Hz/out.hdf5'

dir_path_use = custome

# Caso a probar dentro del HDF5.
# None -> usa las rutas globales Axial_disp/data y Axial_vel/data.
# Ejemplo: "case_000" -> busca case_000/Axial_disp/data y case_000/Axial_vel/data.

data     = HDF5Reader(dir_path_use)

# disp_path_hdf5 = "tool_dyn/data"
# vel_path_hdf5  = "tool_dyn_o/data"

# disp_path_hdf5 = "Axial_disp/data"
# vel_path_hdf5  = "Axial_vel/data"



CASE_NAME = "snr_005.00"


if CASE_NAME is not None:


    case_prefix = f"{CASE_NAME}/" if CASE_NAME else ""
    disp_path_hdf5 = f"{case_prefix}Axial_disp/values"
    vel_path_hdf5  = f"{case_prefix}Axial_vel/values"
    time_path_hdf5 = f"{case_prefix}Axial_disp/time"

    tool_dyn     = data.get_element(disp_path_hdf5)
    t            = data.get_element(time_path_hdf5)
    v            = data.get_element(vel_path_hdf5)

else:
    disp_path_hdf5 = f"Axial_disp/data"
    vel_path_hdf5  = f"Axial_vel/data"


    tool_dyn     = data.get_element(disp_path_hdf5)
    t            = tool_dyn[:, 0]
    tool_dyn     = tool_dyn[:, 1]
    v            = data.get_element(vel_path_hdf5)[:, 1]

    try:
        force_N = data.get_element("force_N/data")[:, 1]
    except KeyError:
        force_N = np.zeros_like(t)

    


_CUT_START = 0.1
t_cut, v_cut = _cut_signal(t, v,        (_CUT_START, 16))
_,     x_cut = _cut_signal(t, tool_dyn, (_CUT_START, 16))
# _,     f_cut = _cut_signal(t, force_N,  (_CUT_START, 16))

fs = 1.0 / (t[1] - t[0])
# =============================================================================
# INDICATOR_CONFIG -- cuatro modos de parametrizacion
#
#   native                  -> win_length_ms / hop_ms / Ai_length directos
#   by_revolution / frames  -> ventana y hop en revoluciones, Ai_length directo
#   by_revolution / total   -> ventana y hop en revoluciones, K_rev_svd ²total
#   by_modal      / frames  -> ventana y hop en periodos modales, Ai_length directo
#
# Cambiar la linea INDICATOR_CONFIG = ... al final del bloque para elegir modo.
# =============================================================================
_RPM     = 12_000.0
_F_MODAL = 150.0
_T_REV   = 60.0 / _RPM        # 0.005 s -- periodo de una revolucion
_T_MODAL = 1.0 / _F_MODAL     # 0.00667 s -- periodo modal (150 Hz)
_TGT     = 5.365770208787228   # [s] ground-truth chatter onset



_COMMON = {
    "n_fft_power":  3,
    "mode":         "causal_inclusive",
    "sigma":        6.0,
    "frac_stable":   0.3610633440512648,    # fallback cuando training_intervals=None
    # ── training_intervals: lista de ((t0, t1), "label") ────────────────────
    # Usar "stable" como etiqueta para que el indicador use ese tramo como
    # region de referencia (reemplaza frac_stable cuando está definido).
    # Se pueden añadir varios intervalos con distintas etiquetas.
    "training_intervals": [
        (_CUT_START, _TGT, "stable"),
        # (_TGT,      10.0,  "chatter"),


    ],
    "alpha":        0.05,
    "z":            3.0,
    "fallback_mad": False,
    "t_theorical":  _TGT,
}

# -- 1. Modo nativo -----------------------------------------------------------
INDICATOR_CONFIG_native = {
    "id":   "SST_SVD",
    "func": "Default",
    "params": {
        "win_length_ms": 40.0,
        "hop_ms":        15.0,
        "Ai_length":     3,
        **_COMMON,
    },
}

# -- 2. Modo by_revolution / Ai_length_mode="frames" -------------------------
#   N_rev_window=5, step_rev=5  -> win=5x5=25ms, hop=5x5=25ms
#   hop/win = 5/5 = 100%  (valido: 0-100%)
#   Ai_length = Ai_length_rev = 3 (directo)
INDICATOR_CONFIG_by_revolution = {
    "id":         "SST_SVD",
    "func":       "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev":          _T_REV,
        "N_rev_window":   4,
        "step_rev":       1,
        "Ai_length_mode": "frames",
        "Ai_length_rev":  4,
        **_COMMON,
    },
}

# -- 3. Modo by_revolution / Ai_length_mode="total_window" -------------------
#   K_rev_svd = N_win + (Ai-1)*step = 5 + (4-1)*5 = 20 revoluciones
#   -> Ai_length = ceil((20 - 5) / 5 + 1) = 4
INDICATOR_CONFIG_by_revolution_total = {
    "id":         "SST_SVD",
    "func":       "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev":          _T_REV,
        "N_rev_window":   4,
        "step_rev":       1,
        "Ai_length_mode": "total_window",
        "K_rev_svd":      14,            # -> Ai_length = 3
        **_COMMON,
    },
}

# -- 4. Modo by_modal / Ai_length_mode="frames" ------------------------------
#   N_modal_window=3, step_modal=1
#   win = 3 x 6.667ms = 20.001ms  |  hop = 1 x 6.667ms = 6.667ms
#   hop/win = 1/3 = 33.33%  (valido: 0-100%)
INDICATOR_CONFIG_by_modal = {
    "id":         "SST_SVD",
    "func":       "Default",
    "param_mode": "by_modal",
    "params_physical": {
        "T_modal":        _T_MODAL,
        "N_modal_window": 4,
        "step_modal":     1,
        "Ai_length_mode": "frames",
        "Ai_length_modal":2,
        **_COMMON,
    },
}

# -- Selector (descomentar el modo deseado) -----------------------------------
# INDICATOR_CONFIG = INDICATOR_CONFIG_native
INDICATOR_CONFIG = INDICATOR_CONFIG_by_revolution
# INDICATOR_CONFIG = INDICATOR_CONFIG_by_revolution_total
# INDICATOR_CONFIG = INDICATOR_CONFIG_by_modal

# -- Senal de entrada ---------------------------------------------------------
sig = SignalData(
    t_analysis=t_cut,
    signal_analysis=v_cut,
    path=dir_path_use,
    fs=fs,
    meta={"AP": "5mm-15mm", "RPM": 12_000},
)

# =============================================================================
# EJECUCION
# =============================================================================
results_SST_SVD = run_sst_svd(sig, INDICATOR_CONFIG)

# =============================================================================
# RESULTADOS -- salida estructurada por nivel de logger
#
#  WARNING : resultado critico (primera deteccion, % chatter)
#  INFO    : + configuracion del indicador y parametros usados
#  DEBUG   : + senal, tabla completa de detecciones
# =============================================================================
meta       = results_SST_SVD.meta
param_mode = meta.get("param_mode", "native")
t_i = results_SST_SVD.t
t_d        = results_SST_SVD.t_d
t_d_no_FAR = results_SST_SVD.t_d_no_FAR
chatter_pct = meta.get("chatter", "N/A")



# ---------- INFO: configuracion del indicador ---------------------------------
if logger.isEnabledFor(logging.INFO):
    _KW = 26   # ancho columna clave

    def _kv(key: str, val: str = "", indent: int = 0) -> str:
        pad = "  " * indent
        return f"{pad}{key:<{_KW - 2 * indent}}  {val}"

    def _sep(label: str = "") -> str:
        dash = "\u2500" * 20
        return f"  {dash}  {label}" if label else f"  {dash}"

    # cabecera comun
    lines = [
        _kv("Indicador", results_SST_SVD.name),
        _kv("Modo",      param_mode),
        _sep(),
    ]

    if param_mode == "native":
        # --- parametros nativos directos ------------------------------------
        p = INDICATOR_CONFIG["params"]
        win_samples = int(p["win_length_ms"] * 1e-3 * sig.fs)
        hop_samples = int(p["hop_ms"]        * 1e-3 * sig.fs)
        lines += [
            _kv("win_length_ms", f"{p['win_length_ms']:.2f} ms"
                                 f"  ({win_samples} samples)"),
            _kv("hop_ms",        f"{p['hop_ms']:.2f} ms"
                                 f"  ({hop_samples} samples)"),
            _kv("Ai_length",     str(p["Ai_length"])),
            _kv("n_fft_power",   f"{p['n_fft_power']}  (n_fft = {1024 * 2**p['n_fft_power']})"),
            _kv("sigma",         str(p["sigma"])),
            _sep("Deteccion"),
            _kv("frac_stable",   str(p["frac_stable"]),   indent=1),
            _kv("alpha",         str(p["alpha"]),          indent=1),
            _kv("z",             str(p["z"]),              indent=1),
            _kv("fallback_mad",  str(p["fallback_mad"]),   indent=1),
        ]

    elif param_mode == "by_revolution":
        # --- entrada en revoluciones -> resultado derivado ------------------
        pp    = INDICATOR_CONFIG["params_physical"]
        nat   = meta.get("native_params_resolved", {})
        quant = meta.get("quantization_notes", "")
        lines += [
            _kv("T_rev",          f"{pp['T_rev'] * 1e3:.3f} ms"
                                  f"  (rpm = {60 / pp['T_rev']:.1f})"),
            _kv("N_rev_window",   f"{pp['N_rev_window']} rev"),
            _kv("step_rev",       f"{pp['step_rev']} rev"),
            _kv("Ai_length_mode", pp.get("Ai_length_mode", "frames")),
        ]
        if pp.get("Ai_length_mode", "frames") == "frames":
            lines.append(_kv("Ai_length_rev", str(pp.get("Ai_length_rev", "-"))))
        else:
            lines.append(_kv("K_rev_svd", f"{pp.get('K_rev_svd', '-')} rev"))
        lines += [
            _sep("Resultado"),
            _kv("t_win_deseado",  f"{meta.get('t_win_exact_ms', 0):.4f} ms", indent=1),
            _kv("t_win_efectivo", f"{meta.get('t_win_efectivo_ms', 0):.4f} ms", indent=1),
            _kv("delta_t_win",    f"+{abs(meta.get('t_win_efectivo_ms', 0) - meta.get('t_win_exact_ms', 0)) * 1e3:.3f} µs",
                                  indent=1),
            _kv("Hop Deseado",  f"{meta.get('t_hop_exact_ms', 0):.4f} ms", indent=1),
            _kv("Hop Efectivo", f"{meta.get('t_hop_efectivo_ms', 0):.4f} ms", indent=1),
            _kv("delta_hop",      f"+{abs(meta.get('t_hop_efectivo_ms', 0) - meta.get('t_hop_exact_ms', 0)) * 1e3:.3f} µs",
                                  indent=1),
            _kv("Ai_length",      str(nat.get("Ai_length", "-")), indent=1),
            _kv("t_svd_total Deseado",    f"{meta.get('t_svd_total_exact_s', 0):.4f} ms"
                                  f"  ({meta.get('K_svd_total_exact_units', 0):.4f} periodos)",
                                  indent=1),
            _kv("t_svd_total_efectivo",    f"{meta.get('t_svd_total_efectivo_s', 0) :.4f} ms"
                                  f"  ({meta.get('K_svd_total_efectivo_units', 0):.4f} periodos)",
                                  indent=1),
            # _sep("Cuantificacion"),
        ]
        for part in quant.replace("|", ";").split(";"):
            if part.strip():
                lines.append(f"    {part.strip()}")
        lines += [
            _sep("SSQ / Deteccion"),
            _kv("n_fft_power",  f"{pp['n_fft_power']}  (n_fft = {1024 * 2**pp['n_fft_power']})", indent=1),
            _kv("sigma",        str(pp["sigma"]),        indent=1),
            _kv("frac_stable",  str(pp["frac_stable"]),  indent=1),
            _kv("alpha",        str(pp["alpha"]),         indent=1),
            _kv("z",            str(pp["z"]),             indent=1),
            _kv("fallback_mad", str(pp["fallback_mad"]),  indent=1),
        ]

    elif param_mode == "by_modal":
        # --- entrada en periodos modales -> resultado derivado ---------------
        pp    = INDICATOR_CONFIG["params_physical"]
        nat   = meta.get("native_params_resolved", {})
        quant = meta.get("quantization_notes", "")
        lines += [
            _kv("T_modal",        f"{pp['T_modal'] * 1e3:.4f} ms"
                                  f"  (f = {1 / pp['T_modal']:.1f} Hz)"),
            _kv("N_modal_window", f"{pp['N_modal_window']} periodos"),
            _kv("step_modal",     f"{pp['step_modal']} periodo(s)"),
            _kv("Ai_length_mode", pp.get("Ai_length_mode", "frames")),
        ]
        if pp.get("Ai_length_mode", "frames") == "frames":
            lines.append(_kv("Ai_length_modal", str(pp.get("Ai_length_modal", "-"))))
        else:
            lines.append(_kv("K_modal_svd", f"{pp.get('K_modal_svd', '-')} periodos"))
        lines += [
            _sep("Resultado"),
            _kv("t_win_deseado",  f"{meta.get('t_win_exact_ms', 0):.4f} ms", indent=1),
            _kv("t_win_efectivo", f"{meta.get('t_win_efectivo_ms', 0):.4f} ms", indent=1),
            _kv("delta_t_win",    f"+{abs(meta.get('t_win_efectivo_ms', 0) - meta.get('t_win_exact_ms', 0)) * 1e3:.3f} µs",
                                  indent=1),
            _kv("Hop Deseado",  f"{meta.get('t_hop_exact_ms', 0):.4f} ms", indent=1),
            _kv("Hop Efectivo", f"{meta.get('t_hop_efectivo_ms', 0):.4f} ms", indent=1),
            _kv("delta_hop",      f"+{abs(meta.get('t_hop_efectivo_ms', 0) - meta.get('t_hop_exact_ms', 0)) * 1e3:.3f} µs",
                                  indent=1),
            _kv("Ai_length",      str(nat.get("Ai_length", "-")), indent=1),
            _kv("t_svd_total Deseado",    f"{meta.get('t_svd_total_exact_s', 0):.4f} ms"
                                  f"  ({meta.get('K_svd_total_exact_units', 0):.4f} periodos)",
                                  indent=1),
            _kv("t_svd_total_efectivo",    f"{meta.get('t_svd_total_efectivo_s', 0):.4f} ms"
                                  f"  ({meta.get('K_svd_total_efectivo_units', 0):.4f} periodos)",
                                  indent=1),
            # _sep("Cuantificacion"),
        ]
        for part in quant.replace("|", ";").split(";"):
            if part.strip():
                lines.append(f"    {part.strip()}")
        lines += [
            _sep("SSQ / Deteccion"),
            _kv("n_fft_power",  f"{pp['n_fft_power']}  (n_fft = {1024 * 2**pp['n_fft_power']})", indent=1),
            _kv("sigma",        str(pp["sigma"]),        indent=1),
            _kv("frac_stable",  str(pp["frac_stable"]),  indent=1),
            _kv("alpha",        str(pp["alpha"]),         indent=1),
            _kv("z",            str(pp["z"]),             indent=1),
            _kv("fallback_mad", str(pp["fallback_mad"]),  indent=1),
        ]

    logger.info("%s\n%s", _section("CONFIGURACION DEL INDICADOR"), "\n".join(lines))


# ---------- DEBUG: senal y tabla de detecciones -------------------------------
if logger.isEnabledFor(logging.DEBUG):
    import pandas as pd

    logger.debug("%s", _section("SENAL"))
    logger.debug("  %-28s %g Hz",  "fs:",       sig.fs)
    logger.debug("  %-28s %.4f s", "Duracion:", len(sig.signal_analysis) / sig.fs)
    logger.debug("  %-28s %d",     "Muestras:", len(sig.signal_analysis))

    logger.debug("%s", _section("DETECCIONES"))
    if t_d is not None and len(t_d) > 0:
        _i_t_all = np.asarray(results_SST_SVD.I_t)
        _t_all   = np.asarray(results_SST_SVD.t)
        if len(_i_t_all) > 0 and len(_t_all) > 0:
            _idx    = np.clip(np.searchsorted(_t_all, t_d), 0, len(_i_t_all) - 1)
            _i_det  = _i_t_all[_idx]
        else:
            _i_det = np.full(len(t_d), float("nan"))
        df_det = pd.DataFrame({
            "t [s]":  t_d,
            "I_t":    _i_det,
        })
        logger.debug("\n%s", df_det.to_string(index=False))
    else:
        logger.debug("  (sin detecciones)")


# =============================================================================
# GRAFICA
# =============================================================================
_T_GT = 5.365770208787228   # theoretical chatter onset time [s]
plots_sst_svd(
    signal=sig, result=results_SST_SVD,
    show_signal=True, zoom_x=None, zoom_y=None,
    vlines=None, hlines=None,
    t_gt=_T_GT,
    waterfall_lines="surface",  # "surface" | "time" | "freq" | "both" | "wire"
)
