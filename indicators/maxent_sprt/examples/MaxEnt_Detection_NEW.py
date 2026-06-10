import logging
from typing import Tuple
import os
import numpy as np
import pandas as pd

from MaxEnt_SPRT import SignalData
from MaxEnt_SPRT import HDF5Reader
from MaxEnt_SPRT import run_maxent_sprt
from MaxEnt_SPRT import plots_maxent_sprt

# =============================================================================
# LOGGING -- niveles y contenido mostrado en terminal
#
#   WARNING  ->  solo resultado critico (tiempo de primera deteccion)
#   INFO     ->  configuracion del indicador + resultado              [default]
#   INFO_PLUS ->  todo lo anterior + resultado + tiempos de calculo
#   DEBUG    ->  todo lo anterior + senal, modelos, tabla de eventos
# =============================================================================
from MaxEnt_SPRT import INFO_PLUS_LEVEL
from MaxEnt_SPRT.logging_setup import configure_logging
from MaxEnt_SPRT.logging_setup import _section

_LOG_LEVEL = logging.INFO
# _LOG_LEVEL = logging.WARNING
# _LOG_LEVEL = logging.DEBUG
# _LOG_LEVEL = INFO_PLUS_LEVEL

# Configure logging for this example (application-level)
configure_logging(level=_LOG_LEVEL)

logger = logging.getLogger(__name__)


# -- helpers ------------------------------------------------------------------
def _cut_signal(t, x, time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]




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



dir_custome     = r"D:\\Thesis\\03-Code_Storage\\02-Altintlas_Nessy2m_Storage\\2DOF_Cone_DOE\\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_180\\12\\1DOF_150Hz\\sens_out.hdf5"
data_dir = cono_doe_control_sensor

# data_dir = os.path.abspath(os.path.join(dir_path_use, "out.hdf5"))
data     = HDF5Reader(data_dir)

# disp_path_hdf5 = "tool_dyn/data"
# vel_path_hdf5  = "tool_dyn_o/data"

disp_path_hdf5 = "Axial_disp/data"
vel_path_hdf5  = "Axial_vel/data"


tool_dyn     = data.get_element(disp_path_hdf5)
t            = tool_dyn[:, 0]
tool_dyn     = tool_dyn[:, 1]
tool_dyn_vel = data.get_element(vel_path_hdf5)[:, 1]

try:
    force_N = data.get_element("force_N/data")[:, 1]
except KeyError:
    force_N = np.zeros_like(t)

v  = tool_dyn_vel
fs = 1.0 / (t[1] - t[0])

_CUT_START = 0.0
_CUT_END   = 16

t_cut, v_cut = _cut_signal(t, v,        (_CUT_START, _CUT_END))
_,     x_cut = _cut_signal(t, tool_dyn, (_CUT_START, _CUT_END))
_,     f_cut = _cut_signal(t, force_N,  (_CUT_START, _CUT_END))

# =============================================================================
# INDICATOR_CONFIG -- tres modos de parametrizacion
#
#   native        -> parametros nativos directos (comportamiento original)
#   by_revolution -> N_seg como numero de revoluciones por segmento
#   by_modal      -> N_seg como multiplo del periodo modal
#
# Cambiar la linea INDICATOR_CONFIG = ... al final del bloque para elegir modo.
# =============================================================================

_RPM     = 12_000.0
_RPM_MODAL = 150*60.0  # RPM equivalente a f_modal = 150 Hz
_F_MODAL = 150.0
_T_REV   = 60.0 / _RPM   # 0.005 s -- periodo de una revolucion
_T_MODAL = 1.0 / _F_MODAL  # s       -- periodo del modo de chatter (f_modal ~ 150 Hz)

# alpha = beta = norm.sf(3.0) ≈ 0.00135  →  equivalent to z=3 sigma (same FAR as RMS-CV and SSQ)
_Z3_ALPHA = 0.00135   # scipy.stats.y si estanorm.sf(3.0)
_T_GT = 5.365770208787228   # [s] ground-truth chatter onset
# _T_GT = 1.07  # [s] ground-truth chatter onset



_COMMON = {
    "t_stable_total":     _T_GT,          # legacy fallback (used if training_intervals=None)
    "training_intervals": [
        # (_CUT_START,3.3,    "stable_1"),  # chatter-free training region
        # (3.3, 4.46, "stable_2"), # stable training region
        # (4.46, _T_GT, "stable_1"), # chatter training region
        # # (3.3,        4.4, "chatter"), # chatter training region
        # # (4.5,        _T_GT, "stable"), # transition region
        # (_T_GT,      10.0, "chatter"), # chatter training region
        # # (6.6,        10.0, "chatter_2"), # chatter training region

        (_CUT_START, _T_GT,  "stable"),  # chatter-free training region
        (_T_GT,       10, "chatter"), # chatter training region
    ],
    "alpha":          _Z3_ALPHA,
    "beta":           _Z3_ALPHA,
    "reset_on_H0":    True,
    "cut_start_time": _CUT_START,
    "cut_end_time":   _CUT_END,
    "t_theorical":   _T_GT,  # for debug/plots, not used in detection
}

# -- 1. Modo nativo -----------------------------------------------------------
INDICATOR_CONFIG_native = {
    "id":   "MaxEnt_SPRT",
    "func": "Default",
    "params": {
        "rpm":   _RPM_MODAL,
        "N_seg": 2,           # 1 rev/seg -> t_seg = 0.005 s
        # "ratio_sampling": 50.0,
        **_COMMON,
    },
}

# -- 4. Modo by_revolution con overlap ----------------------------------------
#   step_rev = 2  ->  hop = 2 rev  ->  overlap = 1 - 2/5 = 60 %
INDICADOR_CONFIG_by_revolution_overlap = {
    "id":         "MaxEnt_SPRT",
    "func":       "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev":         _T_REV,
        "N_rev_per_seg": 5,        # -> N_seg = 5
        "step_rev":      1,        # hop de 2 rev  =>  overlap 60 %
        "segmentation":  "opr",
        **_COMMON,
    },
}

# -- 5. Modo by_modal con overlap --------------------------------------------
#   step_modal = 2  ->  hop = 2 periodos modales  ->  overlap = 60 %
INDICADOR_CONFIG_by_modal_overlap = {
    "id":         "MaxEnt_SPRT",
    "func":       "Default",
    "param_mode": "by_modal",
    "params_physical": {
        "T_rev":           _T_REV,
        "T_modal":         _T_MODAL,
        "N_modal_per_seg": 2.0,    # -> N_seg = 5
        "step_modal":      1,      # hop de 2 periodos modales  =>  overlap 60 %
        **_COMMON,
    },
}

# -- 6. Modo by_revolution con segmentation RAW --------------------------------
#   En lugar de decimacion OPR (1 muestra/rev), usa todos los valores raw
#   dentro de cada bloque de N_rev_per_seg revoluciones.
#   N_samples_per_seg = N_rev_per_seg x round(fs / fr)  (calculado por el resolver)
INDICADOR_CONFIG_by_revolution_raw = {
    "id":         "MaxEnt_SPRT",
    "func":       "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev":         _T_REV,
        "N_rev_per_seg": 4,        # -> N_seg = 5 rev  ->  N_samples = 5 x round(fs/fr)
        "segmentation":  "raw",    # <- nueva opcion: usa senal raw sin OPR
        "step_rev":      1,        # hop de 2 rev  =>  overlap 60 %
        "use_sprt":      True,      # ←  SPRT
        **_COMMON,
    },
}

# -- 7. Modo by_modal con segmentation RAW -----------------------------------
INDICADOR_CONFIG_by_modal_raw = {
    "id":         "MaxEnt_SPRT",
    "func":       "Default",
    "param_mode": "by_modal",
    "params_physical": {
        "T_rev":           _T_REV,
        "T_modal":         _T_MODAL,
        "N_modal_per_seg": 4.0,    # -> N_samples = 5 x round(T_modal x fs)
        "segmentation":    "raw",  # <- usa senal raw
        'step_modal':      1.0,    # hop de 1 periodos modales  =>  overlap 60 %
        'use_sprt':      True,    # <- SPRT
        **_COMMON,
    },
}
# -- Selector (descomentar el modo deseado) -----------------------------------
# INDICATOR_CONFIG = INDICATOR_CONFIG_native
# INDICATOR_CONFIG = INDICATOR_CONFIG_by_revolution
# INDICATOR_CONFIG = INDICATOR_CONFIG_by_modal
INDICATOR_CONFIG = INDICADOR_CONFIG_by_revolution_overlap
# INDICATOR_CONFIG = INDICADOR_CONFIG_by_modal_overlap
INDICATOR_CONFIG = INDICADOR_CONFIG_by_revolution_raw
# INDICATOR_CONFIG = INDICADOR_CONFIG_by_modal_raw

# -- Senal de entrada ---------------------------------------------------------
sig = SignalData(
    t_analysis=t_cut,
    signal_analysis=v_cut,
    path=data_dir,
    fs=fs,
    meta={"AP": "5mm-15mm", "RPM": 12_000},
)

# =============================================================================
# EJECUCION
# =============================================================================

resultat_maxent_sprt = run_maxent_sprt(sig, INDICATOR_CONFIG)

# =============================================================================
# RESULTADOS -- salida estructurada por nivel de logger
#
#  WARNING : resultado critico (tiempo de primera deteccion)
#  INFO    : + configuracion del indicador y parametros usados
#  DEBUG   : + senal, modelos entrenados, tabla completa de detecciones
# =============================================================================
meta       = resultat_maxent_sprt.meta
t_i        = np.asarray(resultat_maxent_sprt.t)
fr         = meta["Rotational_Frequency_Hz"]
param_mode = meta.get("param_mode", "native")
t_d        = resultat_maxent_sprt.t_d
S_vals     = meta.get("chatter_points_values", np.array([]))

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 100)



# ---------- INFO: configuracion del indicador --------------------------------
if logger.isEnabledFor(logging.INFO):
    _KW = 22   # ancho columna clave

    def _kv(key: str, val: str = "", indent: int = 0) -> str:
        pad = "  " * indent
        return f"{pad}{key:<{_KW - 2 * indent}}  {val}"

    def _sep(label: str = "") -> str:
        dash = "\u2500" * 20
        return f"  {dash}  {label}" if label else f"  {dash}"

    # cabecera comun
    lines = [
        _kv("Indicador",      resultat_maxent_sprt.name),
        _kv("Modo",           param_mode),
        _kv("Segmentacion",   meta.get("segmentation", "opr")),
        _sep(),
        _kv("t_stable_total", f"{_COMMON['t_stable_total']:.4f} s"),
        _kv("alpha / beta",   f"{meta['alpha']} / {meta['beta']}"),
        _sep(),
    ]

    if param_mode == "native":
        # --- solo parametros nativos directos --------------------------------
        lines += [
            _kv("rpm",   f"{meta['rpm']:.1f} RPM"),
            _kv("N_seg", f"{meta['N_seg']} rev/seg"),
            _kv("t_seg", f"{meta['N_seg'] / fr * 1e3:.2f} ms"),
        ]

    elif param_mode == "by_revolution":
        # --- entrada en revoluciones -> resultado derivado -------------------
        phys  = meta.get("physical_params_input", {})
        nat   = meta.get("native_params_resolved", {})
        quant = meta.get("quantization_notes", "")
        step_s = nat.get("step_seg", nat.get("N_seg", 1))
        overlap_p = 1.0 - step_s / nat.get("N_seg", 1)
        seg_mode  = meta.get("segmentation", "opr")
        lines += [
            _kv("T_rev",         f"{phys.get('T_rev', 0)*1e3:.3f} ms"
                                 f"  (rpm = {nat.get('rpm', 0):.1f})"),
            _kv("N_rev_per_seg", f"{phys.get('N_rev_per_seg', '-')} rev/seg"),
            _kv("step_rev",      f"{phys.get('step_rev', nat.get('N_seg', '-'))} rev  "
                                 f"  (overlap = {overlap_p:.1%})"),
            _sep("Resultado"),
            _kv("N_seg",     str(nat.get("N_seg", "-")), indent=1),
            _kv("step_seg",  str(step_s), indent=1),
            _kv("t_seg",     f"{nat.get('N_seg', 0) / fr * 1e3:.2f} ms", indent=1),
        ]
        if seg_mode == "raw":
            nsamp = meta.get("N_samples_per_seg") or nat.get("N_samples_per_seg", "?")
            lines.append(_kv("N_samples_per_seg", f"{nsamp} muestras raw", indent=1))
        for part in quant.replace("|", ";").split(";"):
            if part.strip():
                lines.append(f"    {part.strip()}")

    elif param_mode == "by_modal":
        # --- entrada en periodos modales -> resultado derivado ---------------
        phys  = meta.get("physical_params_input", {})
        nat   = meta.get("native_params_resolved", {})
        quant = meta.get("quantization_notes", "")
        step_s = nat.get("step_seg", nat.get("N_seg", 1))
        overlap_p = 1.0 - step_s / nat.get("N_seg", 1)
        seg_mode  = meta.get("segmentation", "opr")
        lines += [
            _kv("T_rev",           f"{phys.get('T_rev', 0)*1e3:.3f} ms"
                                   f"  (rpm = {60.0 / phys.get('T_rev', 1):.1f})"),
            _kv("T_modal",         f"{phys.get('T_modal', 0)*1e3:.3f} ms"
                                   f"  (f = {1/phys.get('T_modal', 1):.1f} Hz)"),
            _kv("N_modal_per_seg", f"{phys.get('N_modal_per_seg', '-')} periodos modales/seg"),
            _kv("step_modal",      f"{phys.get('step_modal', nat.get('N_seg', '-'))} periodos  "
                                   f"  (overlap = {overlap_p:.1%})"),
            _sep("Resultado"),
            _kv("N_seg",     str(nat.get("N_seg", "-")), indent=1),
            _kv("step_seg",  str(step_s), indent=1),
            _kv("t_seg",     f"{nat.get('N_seg', 0) / fr * 1e3:.2f} ms", indent=1),
        ]
        if seg_mode == "raw":
            nsamp = meta.get("N_samples_per_seg") or nat.get("N_samples_per_seg", "?")
            lines.append(_kv("N_samples_per_seg", f"{nsamp} muestras raw", indent=1))
        for part in quant.replace("|", ";").split(";"):
            if part.strip():
                lines.append(f"    {part.strip()}")

    logger.info("%s\n%s", _section("CONFIGURACION DEL INDICADOR"), "\n".join(lines))


# ---------- DEBUG: senal, modelos, tabla de eventos --------------------------
if logger.isEnabledFor(logging.DEBUG):

    rows_sig = [
        ("Duracion senal",   f"{meta['Duration']:.3f} s"),
        ("fs",               f"{meta['fs']:.0f} Hz"),
        ("Muestras totales", f"{meta['Samples']:,}"),
        ("Segmentos totales",f"{meta['Total_segments']:,}"),
        ("Muestras libres",  f"{meta['Size_signal_free']:,}"),
        ("Muestras chatter", f"{meta['Size_signal_chatter']:,}"),
        ("OPR libres",       str(meta.get("Sampled OPR free",  "N/A (raw)" ))),
        ("OPR chatter",      str(meta.get("Sampled OPR chatter","N/A (raw)" ))),
    ]
    df_sig = pd.DataFrame(rows_sig, columns=["Magnitud", "Valor"]).set_index("Magnitud")
    logger.debug("%s\n%s", _section("SENAL"), df_sig.to_string(header=False))

    sprt   = meta["sprt_result"]
    rows_m = [
        ("P0  mu  (libre)",   f"{meta['P0_mu']:.6f}"),
        ("P0  sig (libre)",   f"{meta['P0_sigma']:.6f}"),
        ("P1  mu  (chatter)", f"{meta['P1_mu']:.6f}"),
        ("P1  sig (chatter)", f"{meta['P1_sigma']:.6f}"),
        ("Umbral a (H0)",     f"{sprt.a:.4f}"),
        ("Umbral b (H1)",     f"{sprt.b:.4f}"),
    ]
    df_mdl = pd.DataFrame(rows_m, columns=["Parametro", "Valor"]).set_index("Parametro")
    logger.debug("%s\n%s", _section("MODELOS MaxEnt-Gaussiano + SPRT"), df_mdl.to_string(header=False))

    if t_d.size > 0:
        n_ev  = t_d.size
        df_det = pd.DataFrame({
            "t deteccion [s]": np.round(t_d, 5),
            "S (SPRT)":        np.round(S_vals, 4),
            "umbral b":        round(sprt.b, 4),
        })
        df_det.index = df_det.index + 1
        df_det.index.name = "#"
        logger.debug("%s\n%s",
                     _section(f"TABLA DETECCIONES  ({n_ev} evento(s))"),
                     df_det.to_string())


# =============================================================================
# GRAFICA
# =============================================================================
  # theoretical chatter onset time [s]
plots_maxent_sprt(
    signal=sig,
    result=resultat_maxent_sprt,
    show_signal=True,
    zoom_x=None,
    zoom_y=None,
    vlines=None,
    hlines=None,
    t_gt=_T_GT,
)
