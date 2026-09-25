"""MaxEnt-SPRT detection example.

Loads a signal from HDF5, runs ``run_maxent_sprt`` with one of five
CASES presets (native / by_revolution / by_modal, with and without raw
segmentation) -- each pairing a ``signal_source`` with an
``indicator_config``, see COMMON_TEMPLATE.md §11/§12 -- prints a
structured summary, and plots the result.

Usage:
    python MaxEnt_Detection_NEW.py [--case NAME]

``NAME`` is one of: native, by_revolution_overlap, by_modal_overlap,
by_revolution_raw (default), by_modal_raw. To change the default without
the CLI flag, edit ``ACTIVE_CASE`` below.
"""
import argparse
import logging
import os
import sys
from typing import Tuple

import h5py
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
from MaxEnt_SPRT import HDF5Reader, load_signal
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
    )

}

# See COMMON_TEMPLATE.md §11 -- forma estándar de declarar el origen de la señal.
SIGNAL_SOURCE = {
    "hdf5_path": _DATA_DIRS["cono_dexel_20e_5"],
    "case_name": None,  # None (layout crudo) | "case_003" (layout DOE)
    "disp_name": "Axial_disp",
    "vel_name": "Axial_vel",
    "force_name": "force_N",
}

_CUT_START = 0.05
_CUT_END = 16

# -- external reference (reference_signal / reference_signal_chatter,
#    see COMMON_TEMPLATE.md §10; the chatter side is a MaxEnt-specific mirror
#    of it, not part of the shared contract) --------------------------------
# True  -> calibrate P0 (stable) training against Repo-DOE's externally-labelled
#          "stable" region in reference_combined.h5 instead of the internal
#          training_intervals split below.
# False -> P0 stays internal (comment this flag or flip it to False to
#          compare both modes on the same run).
USE_EXTERNAL_REFERENCE = True
# Same toggle, but for P1 (chatter) against the "unstable" region of the same file.
USE_EXTERNAL_REFERENCE_CHATTER = True
_REFERENCE_H5 = (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
    r"\CAMP10_Chatter_detection_Methodes\Convergency_Simulation"
    r"\4_DOE_Data_Training_Tube\DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0"
    r"\reference_combined.h5"
)
_REFERENCE_CHANNEL = "Axial_vel"

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
    # "training_intervals": [
    #     (_CUT_START, _T_GT, "stable"),  # chatter-free training region
    #     (_T_GT, 10, "chatter"),  # chatter training region
    # ],
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

# CASES: mismo esqueleto signal_source + indicator_config para los 4 indicadores
# (ver COMMON_TEMPLATE.md §11/§12) -- cada entrada pareja la señal a analizar con
# una variante de parametrización. Todas comparten SIGNAL_SOURCE hoy (son distintos
# esquemas de ventaneo sobre la misma señal); cambiar el diccionario de una entrada
# puntual si hace falta analizar una señal distinta con esa variante.
CASES: dict[str, dict] = {
    "native": {
        "signal_source": SIGNAL_SOURCE,
        "indicator_config": INDICATOR_CONFIG_native,
    },
    "by_revolution_overlap": {
        "signal_source": SIGNAL_SOURCE,
        "indicator_config": INDICATOR_CONFIG_by_revolution_overlap,
    },
    "by_modal_overlap": {
        "signal_source": SIGNAL_SOURCE,
        "indicator_config": INDICATOR_CONFIG_by_modal_overlap,
    },
    "by_revolution_raw": {
        "signal_source": SIGNAL_SOURCE,
        "indicator_config": INDICATOR_CONFIG_by_revolution_raw,
    },
    "by_modal_raw": {
        "signal_source": SIGNAL_SOURCE,
        "indicator_config": INDICATOR_CONFIG_by_modal_raw,
    },
}

ACTIVE_CASE = "by_revolution_raw"  # <- cambiar solo esta linea para elegir señal + config
SIGNAL_SOURCE = CASES[ACTIVE_CASE]["signal_source"]
INDICATOR_CONFIG = CASES[ACTIVE_CASE]["indicator_config"]  # se pasa directo a run_maxent_sprt(signal, INDICATOR_CONFIG)


def _load_signal(source: dict, cut_range: Tuple[float, float]) -> SignalData:
    reader = HDF5Reader(source["hdf5_path"])
    case_name = source.get("case_name")

    t, disp = load_signal(reader, source["disp_name"], case_name)
    _, vel = load_signal(reader, source["vel_name"], case_name)
    try:
        _, force_n = load_signal(reader, source["force_name"], case_name)
    except KeyError:
        force_n = np.zeros_like(t)

    fs = 1.0 / (t[1] - t[0])
    t_cut, v_cut = _cut_signal(t, vel, cut_range)
    _, x_cut = _cut_signal(t, disp, cut_range)
    _, f_cut = _cut_signal(t, force_n, cut_range)

    return SignalData(
        t_analysis=t_cut,
        signal_analysis=v_cut,
        path=source["hdf5_path"],
        fs=fs,
        meta={"AP": "5mm-15mm", "RPM": 12_000},
    )


def _load_reference_signal(h5_path: str, channel: str, label: str = "stable") -> SignalData:
    """Build a ``reference_signal`` SignalData from Repo-DOE's reference_combined.h5.

    That file follows ``reference_dataset.py``'s ``save_combined``/``load_combined``
    layout (one group per ``"{label}__{channel}"``, datasets ``t``/``y``, attrs
    ``fs``/``label``/``channel``/``n_pieces``/``source_ids``). Read directly with
    h5py -- never import the Repo-DOE script, to avoid coupling packages (see
    COMMON_TEMPLATE.md §8/§10).
    """
    group_name = f"{label}__{channel}"
    with h5py.File(h5_path, "r") as f:
        if group_name not in f:
            raise KeyError(f"'{group_name}' not found in {h5_path}. Available: {list(f.keys())}")
        g = f[group_name]
        t = g["t"][:]
        y = g["y"][:]
        fs = float(g.attrs["fs"])
        n_pieces = int(g.attrs.get("n_pieces", 0))

    return SignalData(
        t_analysis=t,
        signal_analysis=y,
        path=h5_path,
        fs=fs,
        meta={"label": label, "channel": channel, "n_pieces": n_pieces,
              "source": "Repo-DOE reference_combined.h5"},
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
        _kv("Training source (P0/stable)", meta.get("training_source", "internal")),
        _kv("Chatter source (P1/unstable)", meta.get("chatter_source", "internal")),
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


def main(case_name: str = ACTIVE_CASE) -> None:
    if case_name not in CASES:
        raise SystemExit(f"Case desconocido '{case_name}'. Opciones: {sorted(CASES)}")

    configure_logging(level=_LOG_LEVEL)

    case = CASES[case_name]
    signal_source = case["signal_source"]
    indicator_config = case["indicator_config"]

    # "reference_signal"/"reference_signal_chatter" are top-level INDICATOR_CONFIG
    # keys (sibling of func/params), not part of _COMMON -- inject them here,
    # gated independently by the two USE_EXTERNAL_REFERENCE* flags above.
    if USE_EXTERNAL_REFERENCE:
        indicator_config["reference_signal"] = _load_reference_signal(
            _REFERENCE_H5, _REFERENCE_CHANNEL, label="stable"
        )
    if USE_EXTERNAL_REFERENCE_CHATTER:
        indicator_config["reference_signal_chatter"] = _load_reference_signal(
            _REFERENCE_H5, _REFERENCE_CHANNEL, label="unstable"
        )

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 100)

    sig = _load_signal(signal_source, (_CUT_START, _CUT_END))

    result = run_maxent_sprt(sig, indicator_config)

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
    parser.add_argument("--case", default=ACTIVE_CASE, choices=sorted(CASES),
                         help="CASES preset (signal_source + indicator_config) to run.")
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().case)
