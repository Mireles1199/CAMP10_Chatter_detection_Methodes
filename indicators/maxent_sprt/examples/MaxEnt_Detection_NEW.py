"""MaxEnt-SPRT detection example.

Loads a signal from HDF5, runs ``run_maxent_sprt`` with one of five
INDICATOR_CONFIG variants (native / by_revolution / by_modal, with and
without raw segmentation), prints a structured summary, and plots the
result.

Usage:
    python MaxEnt_Detection_NEW.py [--config NAME]

``NAME`` is one of: native, by_revolution_overlap, by_modal_overlap,
by_revolution_raw (default), by_modal_raw.
"""
import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# -- path setup -----------------------------------------------------------
# Prefer the local (worktree) src/ over whatever MaxEnt_SPRT is installed
# editable-mode against, which may point at a different checkout/worktree.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MaxEnt_SPRT import SignalData
from MaxEnt_SPRT import HDF5Reader
from MaxEnt_SPRT import run_maxent_sprt
from MaxEnt_SPRT import plots_maxent_sprt
from MaxEnt_SPRT.logging_setup import configure_logging, _section

logger = logging.getLogger(__name__)

# =============================================================================
# LOGGING -- niveles y contenido mostrado en terminal
#
#   WARNING  ->  solo resultado critico (tiempo de primera deteccion)
#   INFO     ->  configuracion del indicador + resultado              [default]
#   INFO_PLUS ->  todo lo anterior + resultado + tiempos de calculo
#   DEBUG    ->  todo lo anterior + senal, modelos, tabla de eventos
# =============================================================================
_LOG_LEVEL = logging.INFO
# _LOG_LEVEL = logging.WARNING
# _LOG_LEVEL = logging.DEBUG
# _LOG_LEVEL = INFO_PLUS_LEVEL  (from MaxEnt_SPRT import INFO_PLUS_LEVEL)


# -- helpers ------------------------------------------------------------------
def _cut_signal(t, x, time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]


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
        r"\3\1DOF_150Hz\sens_out.hdf5"
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
}
DATA_DIR = _DATA_DIRS["control_sensor"]
CASE_NAME = None

_CUT_START = 0.0
_CUT_END = 16

# =============================================================================
# INDICATOR_CONFIG -- cinco variantes de parametrizacion
#
#   native                 -> parametros nativos directos
#   by_revolution_overlap  -> ventana por revolucion, OPR, con overlap
#   by_modal_overlap       -> ventana por periodo modal, OPR, con overlap
#   by_revolution_raw      -> ventana por revolucion, senal raw
#   by_modal_raw           -> ventana por periodo modal, senal raw
#
# Elegir con --config al correr el script (ver main() mas abajo).
# =============================================================================
_RPM = 12_000.0
_RPM_MODAL = 150 * 60.0  # RPM equivalente a f_modal = 150 Hz
_F_MODAL = 150.0
_T_REV = 60.0 / _RPM  # 0.005 s -- periodo de una revolucion
_T_MODAL = 1.0 / _F_MODAL  # s -- periodo del modo de chatter (f_modal ~ 150 Hz)

# alpha = beta = norm.sf(3.0) ≈ 0.00135  →  equivalent to z=3 sigma (same FAR as RMS-CV and SSQ)
_Z3_ALPHA = 0.00135
_T_GT = 5.365770208787228  # [s] ground-truth chatter onset

_COMMON = {
    "t_stable_total": _T_GT,  # legacy fallback (used if training_intervals=None)
    "training_intervals": [
        (_CUT_START, _T_GT, "stable"),  # chatter-free training region
        (_T_GT, 10, "chatter"),  # chatter training region
    ],
    "alpha": _Z3_ALPHA,
    "beta": _Z3_ALPHA,
    "reset_on_H0": True,
    "cut_start_time": _CUT_START,
    "cut_end_time": _CUT_END,
    "t_theorical": _T_GT,  # for debug/plots, not used in detection
}

INDICATOR_CONFIG_native = {
    "id": "MaxEnt_SPRT",
    "func": "Default",
    "params": {
        "rpm": _RPM_MODAL,
        "N_seg": 2,  # 1 rev/seg -> t_seg = 0.005 s
        **_COMMON,
    },
}

# step_rev = 1  ->  hop = 1 rev  ->  overlap = 1 - 1/5 = 80 %
INDICATOR_CONFIG_by_revolution_overlap = {
    "id": "MaxEnt_SPRT",
    "func": "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev": _T_REV,
        "N_rev_window": 5,  # -> N_seg = 5
        "step_rev": 1,  # hop de 1 rev  =>  overlap 80 %
        "segmentation": "opr",
        **_COMMON,
    },
}

# step_modal = 1  ->  hop = 1 periodo modal  ->  overlap = 50 %
INDICATOR_CONFIG_by_modal_overlap = {
    "id": "MaxEnt_SPRT",
    "func": "Default",
    "param_mode": "by_modal",
    "params_physical": {
        "T_modal": _T_MODAL,
        "N_modal_window": 2.0,  # -> N_seg = 2
        "step_modal": 1,  # hop de 1 periodo modal  =>  overlap 50 %
        **_COMMON,
    },
}

# Usa senal raw (sin decimacion OPR) dentro de cada bloque de N_rev_window revoluciones.
# N_samples_per_seg = N_rev_window x round(fs / fr)  (calculado por el resolver)
INDICATOR_CONFIG_by_revolution_raw = {
    "id": "MaxEnt_SPRT",
    "func": "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev": _T_REV,
        "N_rev_window": 4,  # -> N_seg = 4 rev  ->  N_samples = 4 x round(fs/fr)
        "segmentation": "raw",  # usa senal raw sin OPR, acepta fracciones
        "step_rev": 1,  # hop de 1 rev  =>  overlap 75 %
        "use_sprt": True,
        **_COMMON,
    },
}

INDICATOR_CONFIG_by_modal_raw = {
    "id": "MaxEnt_SPRT",
    "func": "Default",
    "param_mode": "by_modal",
    "params_physical": {
        "T_modal": _T_MODAL,
        "N_modal_window": 4.0,  # -> N_samples = 4 x round(T_modal x fs)
        "segmentation": "raw",  # usa senal raw, acepta fracciones
        "step_modal": 1.0,  # hop de 1 periodo modal  =>  overlap 75 %
        "use_sprt": True,
        **_COMMON,
    },
}

CONFIGS = {
    "native": INDICATOR_CONFIG_native,
    "by_revolution_overlap": INDICATOR_CONFIG_by_revolution_overlap,
    "by_modal_overlap": INDICATOR_CONFIG_by_modal_overlap,
    "by_revolution_raw": INDICATOR_CONFIG_by_revolution_raw,
    "by_modal_raw": INDICATOR_CONFIG_by_modal_raw,
}


def _load_signal(data_dir: str, case_name: str | None, cut_range: Tuple[float, float]) -> SignalData:
    data = HDF5Reader(data_dir)

    if case_name is not None:
        case_prefix = f"{case_name}/"
        disp_path_hdf5 = f"{case_prefix}Axial_disp/values"
        vel_path_hdf5 = f"{case_prefix}Axial_vel/values"
        time_path_hdf5 = f"{case_prefix}Axial_disp/time"

        tool_dyn = data.get_element(disp_path_hdf5)
        t = data.get_element(time_path_hdf5)
        v = data.get_element(vel_path_hdf5)
    else:
        tool_dyn = data.get_element("Axial_disp/data")
        t = tool_dyn[:, 0]
        tool_dyn = tool_dyn[:, 1]
        v = data.get_element("Axial_vel/data")[:, 1]

    try:
        force_n = data.get_element("force_N/data")[:, 1]
    except KeyError:
        force_n = np.zeros_like(t)

    fs = 1.0 / (t[1] - t[0])
    t_cut, v_cut = _cut_signal(t, v, cut_range)
    _, x_cut = _cut_signal(t, tool_dyn, cut_range)
    _, f_cut = _cut_signal(t, force_n, cut_range)

    return SignalData(
        t_analysis=t_cut,
        signal_analysis=v_cut,
        path=data_dir,
        fs=fs,
        meta={"AP": "5mm-15mm", "RPM": 12_000},
    )


def _log_config_summary(result, fr: float) -> None:
    """INFO-level structured summary of the config actually used to run the pipeline."""
    if not logger.isEnabledFor(logging.INFO):
        return

    meta = result.meta
    param_mode = meta.get("param_mode", "native")
    _KW = 22  # ancho columna clave

    def _kv(key: str, val: str = "", indent: int = 0) -> str:
        pad = "  " * indent
        return f"{pad}{key:<{_KW - 2 * indent}}  {val}"

    def _sep(label: str = "") -> str:
        dash = "\u2500" * 20
        return f"  {dash}  {label}" if label else f"  {dash}"

    lines = [
        _kv("Indicador", result.name),
        _kv("Modo", param_mode),
        _kv("Segmentacion", meta.get("segmentation", "opr")),
        _sep(),
        _kv("t_stable_total", f"{_COMMON['t_stable_total']:.4f} s"),
        _kv("alpha / beta", f"{meta['alpha']} / {meta['beta']}"),
        _sep(),
    ]

    if param_mode == "native":
        lines += [
            _kv("rpm", f"{meta['rpm']:.1f} RPM"),
            _kv("N_seg", f"{meta['N_seg']} rev/seg"),
            _kv("t_seg", f"{meta['N_seg'] / fr * 1e3:.2f} ms"),
        ]

    elif param_mode == "by_revolution":
        phys = meta.get("physical_params_input", {})
        nat = meta.get("native_params_resolved", {})
        quant = meta.get("quantization_notes", "")
        step_s = nat.get("step_seg", nat.get("N_seg", 1))
        # overlap must be computed in cycle units (phys), not nat["step_seg"]:
        # in raw mode nat["step_seg"] is a raw-sample count, not revolutions.
        n_win_phys = phys.get("N_rev_window", nat.get("N_seg", 1))
        step_phys = phys.get("step_rev", n_win_phys)
        overlap_p = 1.0 - step_phys / n_win_phys
        seg_mode = meta.get("segmentation", "opr")
        lines += [
            _kv("T_rev", f"{phys.get('T_rev', 0)*1e3:.3f} ms  (rpm = {nat.get('rpm', 0):.1f})"),
            _kv("N_rev_window", f"{phys.get('N_rev_window', '-')} rev/seg"),
            _kv("step_rev", f"{phys.get('step_rev', nat.get('N_seg', '-'))} rev    (overlap = {overlap_p:.1%})"),
            _sep("Resultado"),
            _kv("N_seg", str(nat.get("N_seg", "-")), indent=1),
            _kv("step_seg", str(step_s), indent=1),
            _kv("t_seg", f"{nat.get('N_seg', 0) / fr * 1e3:.2f} ms", indent=1),
        ]
        if seg_mode == "raw":
            nsamp = meta.get("N_samples_per_seg") or nat.get("N_samples_per_seg", "?")
            lines.append(_kv("N_samples_per_seg", f"{nsamp} muestras raw", indent=1))
        for part in quant.replace("|", ";").split(";"):
            if part.strip():
                lines.append(f"    {part.strip()}")

    elif param_mode == "by_modal":
        phys = meta.get("physical_params_input", {})
        nat = meta.get("native_params_resolved", {})
        quant = meta.get("quantization_notes", "")
        step_s = nat.get("step_seg", nat.get("N_seg", 1))
        # overlap must be computed in cycle units (phys), not nat["step_seg"]:
        # in raw mode nat["step_seg"] is a raw-sample count, not modal periods.
        n_win_phys = phys.get("N_modal_window", nat.get("N_seg", 1))
        step_phys = phys.get("step_modal", n_win_phys)
        overlap_p = 1.0 - step_phys / n_win_phys
        seg_mode = meta.get("segmentation", "opr")
        t_rev_phys = phys.get("T_rev")
        lines += [
            _kv("T_rev", (f"{t_rev_phys*1e3:.3f} ms  (rpm = {60.0/t_rev_phys:.1f})"
                           if t_rev_phys else "n/a (informativo, opcional en by_modal)")),
            _kv("T_modal", f"{phys.get('T_modal', 0)*1e3:.3f} ms  (f = {1/phys.get('T_modal', 1):.1f} Hz)"),
            _kv("N_modal_window", f"{phys.get('N_modal_window', '-')} periodos modales/seg"),
            _kv("step_modal", f"{phys.get('step_modal', nat.get('N_seg', '-'))} periodos    (overlap = {overlap_p:.1%})"),
            _sep("Resultado"),
            _kv("N_seg", str(nat.get("N_seg", "-")), indent=1),
            _kv("step_seg", str(step_s), indent=1),
            _kv("t_seg", f"{nat.get('N_seg', 0) / fr * 1e3:.2f} ms", indent=1),
        ]
        if seg_mode == "raw":
            nsamp = meta.get("N_samples_per_seg") or nat.get("N_samples_per_seg", "?")
            lines.append(_kv("N_samples_per_seg", f"{nsamp} muestras raw", indent=1))
        for part in quant.replace("|", ";").split(";"):
            if part.strip():
                lines.append(f"    {part.strip()}")

    logger.info("%s\n%s", _section("CONFIGURACION DEL INDICADOR"), "\n".join(lines))


def _log_debug_tables(result) -> None:
    """DEBUG-level tables: signal stats, trained models, detection events."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    meta = result.meta
    t_d = result.t_d
    S_vals = meta.get("chatter_points_values", np.array([]))

    rows_sig = [
        ("Duracion senal", f"{meta['Duration']:.3f} s"),
        ("fs", f"{meta['fs']:.0f} Hz"),
        ("Muestras totales", f"{meta['Samples']:,}"),
        ("Segmentos totales", f"{meta['Total_segments']:,}"),
        ("Muestras libres", f"{meta['Size_signal_free']:,}"),
        ("Muestras chatter", f"{meta['Size_signal_chatter']:,}"),
        ("OPR libres", str(meta.get("Sampled OPR free", "N/A (raw)"))),
        ("OPR chatter", str(meta.get("Sampled OPR chatter", "N/A (raw)"))),
    ]
    df_sig = pd.DataFrame(rows_sig, columns=["Magnitud", "Valor"]).set_index("Magnitud")
    logger.debug("%s\n%s", _section("SENAL"), df_sig.to_string(header=False))

    sprt = meta["sprt_result"]
    rows_m = [
        ("P0  mu  (libre)", f"{meta['P0_mu']:.6f}"),
        ("P0  sig (libre)", f"{meta['P0_sigma']:.6f}"),
        ("P1  mu  (chatter)", f"{meta['P1_mu']:.6f}"),
        ("P1  sig (chatter)", f"{meta['P1_sigma']:.6f}"),
        ("Umbral a (H0)", f"{sprt.a:.4f}"),
        ("Umbral b (H1)", f"{sprt.b:.4f}"),
    ]
    df_mdl = pd.DataFrame(rows_m, columns=["Parametro", "Valor"]).set_index("Parametro")
    logger.debug("%s\n%s", _section("MODELOS MaxEnt-Gaussiano + SPRT"), df_mdl.to_string(header=False))

    if t_d.size > 0:
        df_det = pd.DataFrame({
            "t deteccion [s]": np.round(t_d, 5),
            "S (SPRT)": np.round(S_vals, 4),
            "umbral b": round(sprt.b, 4),
        })
        df_det.index = df_det.index + 1
        df_det.index.name = "#"
        logger.debug("%s\n%s", _section(f"TABLA DETECCIONES  ({t_d.size} evento(s))"), df_det.to_string())


def main(config_name: str = "by_revolution_raw") -> None:
    if config_name not in CONFIGS:
        raise SystemExit(f"Config desconocida '{config_name}'. Opciones: {sorted(CONFIGS)}")

    configure_logging(level=_LOG_LEVEL)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 100)

    sig = _load_signal(DATA_DIR, CASE_NAME, (_CUT_START, _CUT_END))

    result = run_maxent_sprt(sig, CONFIGS[config_name])

    fr = result.meta["Rotational_Frequency_Hz"]
    _log_config_summary(result, fr)
    _log_debug_tables(result)

    plots_maxent_sprt(
        signal=sig,
        result=result,
        show_signal=True,
        zoom_x=None,
        zoom_y=None,
        vlines=None,
        hlines=None,
        t_gt=_T_GT,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MaxEnt-SPRT detection example.")
    parser.add_argument("--config", default="by_revolution_raw", choices=sorted(CONFIGS),
                         help="INDICATOR_CONFIG variant to run.")
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
