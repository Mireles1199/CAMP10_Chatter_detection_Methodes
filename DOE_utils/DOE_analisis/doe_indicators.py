"""doe_indicators.py — Aplica los indicadores de chatter (MaxEnt-SPRT, RMS-CV,
SST-SVD, Green Integral) a cada caso de un HDF5 del DOE y guarda los
resultados en otro HDF5.

Acepta los dos formatos de entrada (se detecta solo por los nombres de grupo):
    doe_results.h5        grupos case_000 …        -> doe_indicator_results.h5
    doe_noise_results.h5  grupos control / snr_*   -> doe_noise_indicator_results.h5

Todo lo editable está en el bloque CONFIG al inicio de main(); los indicadores
se declaran igual que en indicators/*/examples/*_NEW.py.

Guía completa:  python doe_indicators.py --help
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# -- indicadores: siempre el src/ local de cada paquete (indicators/COMMON_TEMPLATE.md §8) --
_CAMP10 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _pkg in ("maxent_sprt", "rms_cv", "ssq_chatter", "green_integral"):
    _src = os.path.join(_CAMP10, "indicators", _pkg, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from MaxEnt_SPRT import run_maxent_sprt, SignalData as _SignalDataMaxEnt  # noqa: E402
from rms_cv import run_rms_cv, SignalData as _SignalDataRMS  # noqa: E402
from ssq_chatter import run_sst_svd, SignalData as _SignalDataSSQ  # noqa: E402
from green_integral import run_green_std, StdSignalData as _StdSignalDataGreen  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==============================================================================
# MAIN — el bloque CONFIG es lo único que se edita
# ==============================================================================

def main() -> None:
    # ==========================================================================
    # CONFIG
    # ==========================================================================

    # -- entrada (si no se pasa --doe_results) ---------------------------------
    BASE_DIR  = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cone_DOE"
    DOE_NAME  = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200_AP_9mm"
    H5_NAME   = "doe_results.h5"   # "doe_results.h5" (case_*) | "doe_noise_results.h5" (control/snr_*)
    LABEL_KEY = None               # None -> auto-detectar el parámetro que varía | ej. "$dxl_size$"
                                   # (en ruido siempre es "snr_db")

    # -- ejecución --------------------------------------------------------------
    NB_WORKERS       = 6       # 1 = secuencial; >1 = paralelo (un proceso por worker)
    ENABLED_CASES    = "all"   # "all" | ["case_000", "case_003"] | ["control", "snr_040.00"]
    SAVE_META_ARRAYS = False   # True -> guarda también los arrays 1D de meta como datasets

    # -- señal / física ---------------------------------------------------------
    _CUT_START = 0.05                # [s] inicio de la ventana de análisis
    _CUT_END   = 16.0                # [s] fin de la ventana de análisis
    _RPM       = 12_000.0
    _F_MODAL   = 150.0               # [Hz] frecuencia modal del chatter
    _T_REV     = 60.0 / _RPM         # 0.005 s -- periodo de una revolución
    _T_MODAL   = 1.0 / _F_MODAL      # 0.00667 s -- periodo modal
    _T_GT      = 5.365770208787228   # [s] onset de chatter (y región estable si USE_EXTERNAL_REFERENCE = False)

    # -- entrenamiento de umbrales -------------------------------------------------
    #   True  -> tramos "stable" (y "unstable" en MaxEnt) de _REFERENCE_H5, uno por
    #            caso, nunca concatenados (COMMON_TEMPLATE.md §10)
    #   False -> tramo (_CUT_START, _T_GT) de la propia señal de cada caso
    USE_EXTERNAL_REFERENCE = True
    _REFERENCE_H5 = (
        r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
        r"\CAMP10_Chatter_detection_Methodes\Convergency_Simulation"
        r"\4_DOE_Data_Training_Tube\DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0"
        r"\reference_dataset_amp.h5"
    )
    _INTERNAL = not USE_EXTERNAL_REFERENCE

    # alpha = beta = norm.sf(3.0) ≈ 0.00135  ->  mismo FAR que z = 3 sigma
    _Z3_ALPHA = 0.00135

    # -- MaxEnt-SPRT  (indicators/maxent_sprt/examples/MaxEnt_Detection_NEW.py) ----
    _COMMON_MAXENT = {
        "t_stable_total": _T_GT,  # legacy fallback (used if training_intervals=None)
        "training_intervals": [
            (_CUT_START, _T_GT, "stable"),
            (_T_GT, 10.0, "chatter"),
        ] if _INTERNAL else None,
        "alpha": _Z3_ALPHA,
        "beta": _Z3_ALPHA,
        "reset_on_H0": True,
        "cut_start_time": _CUT_START,
        "cut_end_time": _CUT_END,
        "t_theorical": _T_GT,
    }

    # step_rev = 1  ->  hop = 1 rev  ->  overlap = 1 - 1/4 = 75 %
    INDICATOR_CONFIG_maxent_by_revolution = {
        "id": "MaxEnt_SPRT",
        "func": "Default",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev": _T_REV,
            "N_rev_window": 4,
            "step_rev": 1,
            "segmentation": "opr",
            **_COMMON_MAXENT,
        },
    }

    # step_modal = 1  ->  hop = 1 periodo modal  ->  overlap = 50 %
    INDICATOR_CONFIG_maxent_by_modal = {
        "id": "MaxEnt_SPRT",
        "func": "Default",
        "param_mode": "by_modal",
        "params_physical": {
            "T_modal": _T_MODAL,
            "N_modal_window": 2.0,
            "step_modal": 1,
            **_COMMON_MAXENT,
        },
    }

    # -- RMS-CV  (indicators/rms_cv/examples/RMS_CV_Chatter_Detection_NEW.py) ------
    _COMMON_RMS = {
        # fixed threshold (ignored when stable_time / reference_signal is set)
        "cv_threshold":         None,
        "rms_threshold":        None,
        "n_min_cv":             2,
        "warmup_ignore_alerts": False,
        "use_unbiased_std":     True,
        "eps":                  1e-12,
        "detrend":              False,
        "pad_mode":             "none",
        # adaptive threshold: 3-sigma on CV of the stable region
        "stable_time":  (0.0, _T_GT) if _INTERNAL else None,
        "z":            3.0,
        "alpha":        0.05,
        "fallback_mad": True,
        "t_theorical":  _T_GT,
    }

    INDICATOR_CONFIG_rms_cv_by_revolution = {
        "id":         "RMS_CV",
        "func":       "Default",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,
            "N_rev_window": 4,
            "step_rev":     1,
            "n_max_mode":   "frames",
            "n_max_rev":    4,
            **_COMMON_RMS,
        },
    }

    INDICATOR_CONFIG_rms_cv_by_modal = {
        "id":         "RMS_CV",
        "func":       "Default",
        "param_mode": "by_modal",
        "params_physical": {
            "T_modal":        _T_MODAL,
            "N_modal_window": 1,
            "step_modal":     1,
            "n_max_mode":     "frames",
            "n_max_modal":    16,
            **_COMMON_RMS,
        },
    }

    # -- SST-SVD  (indicators/ssq_chatter/examples/SSQ_STFT_Chatter_Detection_NEW.py) --
    _COMMON_SSQ = {
        "n_fft_power":  3,
        "mode":         "causal_inclusive",
        "sigma":        6.0,
        "frac_stable":  0.3610633440512648,  # fallback cuando training_intervals=None
        "training_intervals": [(_CUT_START, _T_GT, "stable")] if _INTERNAL else None,
        "alpha":        0.05,
        "z":            3.0,
        "fallback_mad": False,
        "t_theorical":  _T_GT,
    }

    INDICATOR_CONFIG_ssq_by_revolution = {
        "id":         "SST_SVD",
        "func":       "Default",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":          _T_REV,
            "N_rev_window":   4,
            "step_rev":       1,
            "Ai_length_mode": "frames",
            "Ai_length_rev":  4,
            **_COMMON_SSQ,
        },
    }

    INDICATOR_CONFIG_ssq_by_modal = {
        "id":         "SST_SVD",
        "func":       "Default",
        "param_mode": "by_modal",
        "params_physical": {
            "T_modal":         _T_MODAL,
            "N_modal_window":  4,
            "step_modal":      1,
            "Ai_length_mode":  "frames",
            "Ai_length_modal": 2,
            **_COMMON_SSQ,
        },
    }

    # -- Green Integral  (indicators/green_integral/examples/Green_Integral_Detection_NEW.py) --
    _COMMON_GREEN_ALL = {
        "use_area_threshold": True,
        "training_intervals": [(_CUT_START, _T_GT, "stable")] if _INTERNAL else None,
        "z_sigma":            3.0,
        "debug_level":        0,      # _NEW usa 2/1 (figuras por ventana); en lote siempre 0
        "t_theorical":        _T_GT,
    }
    _COMMON_GREEN_STD = {
        "data_filtrated":        True,
        "hilbert":               False,
        "while_loop_extend":     False,
        "cycles_cluster_points": 35,
        "thein_sen":             False,
    }
    _COMMON_GREEN_LYAPUNOV = {
        "data_filtrated":           True,
        "lambda_ewma":              None,     # EWMA para suavizar σ̂ (None = sin suavizado)
        "accumulate":               False,
        "G_memory":                 _T_REV * 10,
        "sigma_method":             "ratio",  # "ratio" | "frozen_time"
        "sigma_local_n":            10,
        "area_noise_eps":           1e-30,
        "use_zero_crossing_cycles": True,     # alpha cycles
        "use_beta_from_cycles":     False,    # beta = unión de ciclos completos
        "zc_detrend":               True,
        "v_cycle_mode":             "zero",   # "zero" | "original" | "detrended"
        "cycle_area_norm":          "none",   # "none" | "mean" | "median"
    }

    INDICATOR_CONFIG_green_std_by_revolution = {
        "id":         "Green_Integral",
        "func":       "Default",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,
            "N_rev_window": 4,
            "step_rev":     1.0,
            **_COMMON_GREEN_ALL, **_COMMON_GREEN_STD,
        },
    }

    INDICATOR_CONFIG_green_std_by_modal = {
        "id":         "Green_Integral",
        "func":       "Default",
        "param_mode": "by_modal",
        "params_physical": {
            "T_modal":        _T_MODAL,
            "N_modal_window": 4,
            "step_modal":     1.0,
            **_COMMON_GREEN_ALL, **_COMMON_GREEN_STD,
        },
    }

    INDICATOR_CONFIG_green_lyapunov_by_revolution = {
        "id":         "Green_Integral",
        "func":       "Lyapunov",
        "param_mode": "by_revolution",
        "params_physical": {
            "T_rev":        _T_REV,
            "N_rev_window": 4,
            "step_rev":     1.0,
            **_COMMON_GREEN_ALL, **_COMMON_GREEN_LYAPUNOV,
        },
    }

    INDICATOR_CONFIG_green_lyapunov_by_modal = {
        "id":         "Green_Integral",
        "func":       "Lyapunov",
        "param_mode": "by_modal",
        "params_physical": {
            "T_modal":        _T_MODAL,
            "N_modal_window": 4,
            "step_modal":     1.0,
            **_COMMON_GREEN_ALL, **_COMMON_GREEN_LYAPUNOV,
        },
    }

    # -- RUNS: qué se corre y sobre qué canal ---------------------------------------
    # Cada entrada = una config por caso. El grupo HDF5 se nombra solo desde los
    # parámetros (ej. maxent_revo_dec4_1step); "name": "..." lo fuerza a mano.
    RUNS: List[Dict[str, Any]] = [
        {"enabled": True,  "signal": "Axial_vel",  "indicator_config": INDICATOR_CONFIG_maxent_by_revolution},
        {"enabled": False, "signal": "Axial_vel",  "indicator_config": INDICATOR_CONFIG_maxent_by_modal},
        {"enabled": True,  "signal": "Axial_vel",  "indicator_config": INDICATOR_CONFIG_rms_cv_by_revolution},
        {"enabled": False, "signal": "Axial_vel",  "indicator_config": INDICATOR_CONFIG_rms_cv_by_modal},
        {"enabled": True,  "signal": "Axial_vel",  "indicator_config": INDICATOR_CONFIG_ssq_by_revolution},
        {"enabled": False, "signal": "Axial_vel",  "indicator_config": INDICATOR_CONFIG_ssq_by_modal},
        {"enabled": False, "signal": "Axial_disp", "indicator_config": INDICATOR_CONFIG_green_std_by_revolution},
        {"enabled": False, "signal": "Axial_disp", "indicator_config": INDICATOR_CONFIG_green_std_by_modal},
        {"enabled": True,  "signal": "Axial_disp", "indicator_config": INDICATOR_CONFIG_green_lyapunov_by_revolution},
        {"enabled": False, "signal": "Axial_disp", "indicator_config": INDICATOR_CONFIG_green_lyapunov_by_modal},
    ]

    # ==========================================================================
    # FIN CONFIG — de aquí para abajo no hace falta tocar nada
    # ==========================================================================

    args = parse_args({
        "h5": os.path.join(BASE_DIR, DOE_NAME, H5_NAME),
        "workers": NB_WORKERS,
        "cases": ENABLED_CASES,
        "reference": _REFERENCE_H5 if USE_EXTERNAL_REFERENCE else "desactivada (tramo interno)",
        "runs": [_run_name(r) for r in RUNS if r["enabled"]],
    })

    h5_in = os.path.normpath(args.doe_results or os.path.join(BASE_DIR, DOE_NAME, H5_NAME))
    if not os.path.isfile(h5_in):
        log.error("Archivo no encontrado: %s", h5_in)
        log.error("  Editar BASE_DIR / DOE_NAME / H5_NAME en CONFIG o pasar --doe_results.")
        sys.exit(1)

    layout, all_groups = _groups(h5_in)
    label_key = "snr_db" if layout == "noise" else (args.label_key or LABEL_KEY)
    if label_key is None:
        try:
            label_key = _detect_label_key(h5_in)
        except ValueError as exc:
            if not args.list:
                log.error("No se pudo auto-detectar LABEL_KEY:\n  %s", exc)
                sys.exit(1)

    if args.list:
        list_cases(h5_in, all_groups, label_key)
        return

    wanted = args.cases or ENABLED_CASES
    groups = all_groups if wanted == "all" else [g for g in wanted if g in all_groups]
    missing = [] if wanted == "all" else [g for g in wanted if g not in all_groups]
    if missing:
        log.warning("Grupos no encontrados en HDF5: %s", missing)

    reference_h5 = os.path.normpath(_REFERENCE_H5) if USE_EXTERNAL_REFERENCE else None
    if reference_h5 and not os.path.isfile(reference_h5):
        log.error("_REFERENCE_H5 no encontrado: %s", reference_h5)
        log.error("  Corregir la ruta en CONFIG o poner USE_EXTERNAL_REFERENCE = False.")
        sys.exit(1)

    out_path = os.path.normpath(args.out) if args.out else os.path.join(
        os.path.dirname(h5_in), _OUT_NAME[layout])
    workers = args.workers if args.workers is not None else NB_WORKERS
    runs = [r for r in RUNS if r["enabled"]]

    log.info("doe_indicators")
    log.info("  Entrada     : %s  (%s)", h5_in, layout)
    log.info("  Salida      : %s", out_path)
    log.info("  LABEL_KEY   : %s", label_key)
    log.info("  Referencia  : %s", reference_h5 or "desactivada (tramo interno)")
    log.info("  Workers     : %d", workers)
    log.info("  Dry-run     : %s", args.dry_run)
    log.info("  Casos       : %d  %s", len(groups), "(all)" if wanted == "all" else wanted)
    log.info("  Configs activas: %d", len(runs))
    for r in runs:
        log.info("    %-38s  signal=%s", _run_name(r), r["signal"])

    n_done = run_all(
        h5_path=h5_in,
        groups=groups,
        runs=runs,
        settings={
            "cut": (_CUT_START, _CUT_END),
            "label_key": label_key,
            "reference_h5": reference_h5,
        },
        nb_workers=workers,
        dry_run=args.dry_run,
        out_path=None if args.dry_run else out_path,
        save_meta_arrays=SAVE_META_ARRAYS,
    )

    if args.dry_run:
        log.info("[DRY-RUN] %d tareas planificadas — nada escrito.", n_done)
    else:
        log.info("Listo. %d resultados escritos incrementalmente en %s", n_done, out_path)


# ==============================================================================
# CLI / --help
# ==============================================================================

_HELP = r"""
GUÍA RÁPIDA
===========
Qué hace
  Corre MaxEnt-SPRT, RMS-CV, SST-SVD y Green Integral sobre cada caso de un HDF5
  del DOE y guarda t, I_t, t_d por (caso, indicador). El formato de
  entrada se detecta solo:
    doe_results.h5        (case_*)          -> doe_indicator_results.h5
    doe_noise_results.h5  (control, snr_*)  -> doe_noise_indicator_results.h5
  (la salida se escribe junto a la entrada, salvo --out)

Dónde se edita: bloque CONFIG al inicio de main() — nada más
  Entrada ........ BASE_DIR, DOE_NAME, H5_NAME, LABEL_KEY
  Ejecución ...... NB_WORKERS, ENABLED_CASES, SAVE_META_ARRAYS
  Señal/física ... _CUT_START, _CUT_END, _RPM, _F_MODAL, _T_GT
  Entrenamiento .. USE_EXTERNAL_REFERENCE, _REFERENCE_H5
  Indicadores .... _COMMON_* + INDICATOR_CONFIG_*  (mismo formato que examples/*_NEW.py)
  Qué corre ...... RUNS -> "enabled": True / False en cada entrada

Recetas
  Ver los casos del HDF5 (sin correr nada)
    python doe_indicators.py --list
  Ver qué se va a correr (casos x indicadores) sin calcular
    python doe_indicators.py --dry_run
  Correr el DOE del CONFIG  (o F5 en VS Code, sin argumentos)
    python doe_indicators.py
  Correr otro HDF5 (DOE o ruido)
    python doe_indicators.py --doe_results D:\...\doe_noise_results.h5
  Solo algunos casos
    python doe_indicators.py --cases case_000 case_003
  Secuencial, para depurar un error (traceback completo en el log)
    python doe_indicators.py --workers 1 --cases case_000
  Cambiar un parámetro de un indicador
    editar su INDICATOR_CONFIG_* (o el _COMMON_* que comparte)
  Agregar una variante nueva
    copiar un INDICATOR_CONFIG_* de indicators/<indicador>/examples/*_NEW.py
    y agregarlo a RUNS con su "signal" (Axial_vel | Axial_disp)
  Entrenar con la propia señal en vez de la referencia externa
    USE_EXTERNAL_REFERENCE = False   (usa el tramo (_CUT_START, _T_GT))

Salida
  <caso>/<run_name>/{t, I_t, t_d} + attrs (config usada, meta_*)
  run_name se arma solo desde los parámetros, ej. maxent_revo_dec4_1step
  Volver a correr la misma config sobreescribe ese grupo; las demás se conservan.
  Siguiente paso: DOE_plots/doe_indicator_plotter.py | doe_noise_plotter.py
"""


def parse_args(defaults: Dict[str, Any]) -> argparse.Namespace:
    current = (
        "CONFIG actual\n"
        f"  entrada    : {defaults['h5']}\n"
        f"  workers    : {defaults['workers']}   casos: {defaults['cases']}\n"
        f"  referencia : {defaults['reference']}\n"
        f"  runs       : {', '.join(defaults['runs']) or '(ninguno activo)'}\n"
    )
    p = argparse.ArgumentParser(
        description="doe_indicators — Aplica los indicadores de chatter a los casos de un DOE.\n\n" + current,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_HELP,
    )
    p.add_argument("--doe_results", "--doe_noise", default=None, metavar="PATH",
                   help="HDF5 de entrada: doe_results.h5 o doe_noise_results.h5 "
                        "(default: BASE_DIR/DOE_NAME/H5_NAME del CONFIG)")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="HDF5 de salida (default: junto a la entrada, nombre según el formato)")
    p.add_argument("--label_key", default=None, metavar="KEY",
                   help='Parámetro DOE que etiqueta cada caso, ej. "$dxl_size$" (default: auto)')
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help=f"Workers en paralelo (default: NB_WORKERS={defaults['workers']})")
    p.add_argument("--cases", nargs="+", default=None, metavar="GRUPO",
                   help="Solo estos grupos, ej. case_000 case_003 (default: ENABLED_CASES)")
    p.add_argument("--dry_run", action="store_true",
                   help="Imprime el plan de tareas sin correr los indicadores.")
    p.add_argument("--list", action="store_true",
                   help="Imprime la tabla de casos del HDF5 y sale.")
    return p.parse_args()


# ==============================================================================
# INDICADORES — id del INDICATOR_CONFIG -> runner / señal / referencia / nombre
# ==============================================================================

_INDICATORS = {
    "MaxEnt_SPRT":    (run_maxent_sprt, _SignalDataMaxEnt),
    "RMS_CV":         (run_rms_cv, _SignalDataRMS),
    "SST_SVD":        (run_sst_svd, _SignalDataSSQ),
    "Green_Integral": (run_green_std, _StdSignalDataGreen),
}

# clave de referencia externa del config -> label en reference_dataset*.h5
_REFERENCE_KEYS = {
    "MaxEnt_SPRT":    {"reference_signal": "stable", "reference_signal_chatter": "unstable"},
    "RMS_CV":         {"reference_signal": "stable"},
    "SST_SVD":        {"reference_signal": "stable"},
    "Green_Integral": {"reference_signal": "stable"},
}

# prefijo del grupo HDF5 — el mismo de corridas anteriores (los plotters filtran por él)
_PREFIX = {
    ("MaxEnt_SPRT", "Default"):     "maxent",
    ("RMS_CV", "Default"):          "rms_cv",
    ("SST_SVD", "Default"):         "ssq",
    ("Green_Integral", "Default"):  "green_default",
    ("Green_Integral", "Lyapunov"): "green_fixed",
}
_MODE_SHORT = {"by_revolution": "revo", "by_modal": "modal"}
_OUT_NAME = {"doe": "doe_indicator_results.h5", "noise": "doe_noise_indicator_results.h5"}


def _run_name(run: Dict[str, Any]) -> str:
    """Nombre del grupo HDF5 desde los parámetros clave.

    MaxEnt / Green -> {ind}_{mode}_dec{N_win}_{step}step
    RMS / SSQ      -> {ind}_{mode}_aux{N_win}_n_aux{n_aux}_dec{dec}_{step}step
                      con dec = N_win + (n_aux - 1) * step
    """
    if run.get("name"):
        return run["name"]
    cfg = run["indicator_config"]
    ind = _PREFIX[(cfg["id"], cfg.get("func", "Default"))]
    mode = cfg["param_mode"]
    pp = cfg["params_physical"]
    u = "rev" if mode == "by_revolution" else "modal"
    win, step = int(pp[f"N_{u}_window"]), int(pp[f"step_{u}"])
    if ind in ("rms_cv", "ssq"):
        n_aux = int(pp.get(f"n_max_{u}") or pp.get(f"Ai_length_{u}") or 0)
        dec = win + (n_aux - 1) * step
        return f"{ind}_{_MODE_SHORT[mode]}_aux{win}_n_aux{n_aux}_dec{dec}_{step}step"
    return f"{ind}_{_MODE_SHORT[mode]}_dec{win}_{step}step"


# ==============================================================================
# LECTURA: formato, label_key, casos, referencia externa
# ==============================================================================

def _groups(h5_path: str) -> Tuple[str, List[str]]:
    """("doe", [case_*]) o ("noise", [control, snr_*]) según los grupos del HDF5."""
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
    cases = [k for k in keys if k.startswith("case_")]
    return ("doe", cases) if cases else ("noise", keys)


def _detect_label_key(h5_path: str) -> str:
    """Auto-detecta el parámetro DOE que varía entre casos en doe_results.h5.

    Reglas:
      - Solo considera atributos con formato "$...$" (parámetros DOE, excluye wall_time_s etc.)
      - Si exactamente 1 varía -> retorna ese key.
      - Si >1 varían -> lanza ValueError con lista de candidatos y sugerencia --label_key.
      - Si 0 varían -> retorna el primer attr "$...$" disponible (caso degenerado).
    """
    with h5py.File(h5_path, "r") as f:
        case_keys = [k for k in f.keys() if k.startswith("case_")]
        if not case_keys:
            raise ValueError(f"No se encontraron grupos 'case_*' en {h5_path}")

        all_attr_keys = [
            k for k in f[case_keys[0]].attrs.keys()
            if k.startswith("$") and k.endswith("$")
        ]
        if not all_attr_keys:
            raise ValueError(
                f"No se encontraron atributos DOE (formato '$...$') en {h5_path}. "
                "Usa --label_key para especificar una clave manualmente."
            )

        varying = []
        for ak in all_attr_keys:
            vals = {str(float(f[ck].attrs[ak])) for ck in case_keys if ak in f[ck].attrs}
            if len(vals) > 1:
                varying.append(ak)

    if len(varying) == 1:
        log.info("LABEL_KEY auto-detectado: '%s'", varying[0])
        return varying[0]

    if len(varying) > 1:
        raise ValueError(
            f"Múltiples parámetros DOE varían entre casos: {varying}.\n"
            "  Usa --label_key para especificar uno, p.ej.:\n"
            f"    --label_key \"{varying[0]}\""
        )

    log.warning(
        "Ningún parámetro DOE varía entre casos. Usando '%s' como LABEL_KEY.", all_attr_keys[0]
    )
    return all_attr_keys[0]


def _load_case(h5_path: str, grp_name: str, signals: List[str]) -> Dict[str, Any]:
    """Carga un grupo del HDF5 de entrada -> {"attrs": ..., "signals": {nombre: (t, y)}}."""
    with h5py.File(h5_path, "r") as f:
        grp = f[grp_name]
        sigs = {s: (grp[f"{s}/time"][()], grp[f"{s}/values"][()]) for s in signals if s in grp}
        return {"attrs": dict(grp.attrs), "signals": sigs}


def _cut_signal(t: np.ndarray, x: np.ndarray, start: float, end: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (t >= start) & (t <= end)
    return t[mask], x[mask]


# ponytail: caché sin límite por worker (~0.7 GB con vel stable+unstable y disp);
# bajar NB_WORKERS si falta RAM.
@lru_cache(maxsize=None)
def _read_reference(h5_path: str, label: str, channel: str) -> Dict[Tuple[str, str], tuple]:
    """Tramos de reference_dataset*.h5 -> {(caso, NNN): (t, y, fs, signal_id)}.

    Layout /<label>/<caso>/<canal>__NNN/{t, y}. Cada tramo queda separado (nunca
    se concatenan, COMMON_TEMPLATE.md §10). Cada worker lo lee una sola vez.
    """
    out = {}
    with h5py.File(h5_path, "r") as f:
        for case, grp in f[label].items():
            for name, piece in grp.items():
                if piece.attrs["channel"] == channel:
                    out[(case, name.split("__", 1)[-1])] = (
                        piece["t"][()], piece["y"][()],
                        float(piece.attrs["fs"]), str(piece.attrs["signal_id"]),
                    )
    return out


def _reference_pieces(ind_id: str, h5_path: str, label: str, channel: str) -> list:
    """Una SignalData (clase propia del indicador) por tramo; Green lleva además
    la velocidad del tramo hermano Axial_vel__NNN en meta["velocity"]."""
    sig_cls = _INDICATORS[ind_id][1]
    vel = _read_reference(h5_path, label, "Axial_vel") if ind_id == "Green_Integral" else {}
    pieces = []
    for key, (t, y, fs, sid) in _read_reference(h5_path, label, channel).items():
        meta = {"label": label, "channel": channel, "case": key[0], "signal_id": sid, "name": sid}
        if key in vel:
            meta["velocity"] = vel[key][1]
        pieces.append(sig_cls(t_analysis=t, signal_analysis=y, path=h5_path, fs=fs, meta=meta))
    return pieces


# ==============================================================================
# RUNNER POR (caso, config)
# ==============================================================================

def _empty(grp_name: str, run_name: str, label_key: Optional[str],
           label_val: float = float("nan"), **meta) -> Dict[str, Any]:
    return {
        "case": grp_name, "run_name": run_name,
        "t": np.array([]), "I_t": np.array([]),
        "t_d": np.array([]),
        "meta": meta, "attrs": {},
        "label_key": label_key, "label_val": label_val,
    }


def _run_one(
    h5_path: str,
    grp_name: str,
    run: Dict[str, Any],
    settings: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Corre un indicador sobre un caso. Retorna dict con resultados."""
    cfg = run["indicator_config"]
    ind_id = cfg["id"]
    signal = run["signal"]
    run_name = _run_name(run)
    label_key = settings["label_key"]

    if dry_run:
        log.info("  [%s / %s] DRY-RUN — omitido", grp_name, run_name)
        return _empty(grp_name, run_name, label_key, dry_run=True)

    log.info("  [%s / %s] INICIO", grp_name, run_name)

    # Green recibe la velocidad medida en meta["velocity"] (si no, usa np.gradient)
    is_green = ind_id == "Green_Integral"
    case = _load_case(h5_path, grp_name, [signal, "Axial_vel"] if is_green else [signal])
    if signal not in case["signals"]:
        log.warning("  [%s / %s] señal '%s' no encontrada — SKIP", grp_name, run_name, signal)
        return _empty(grp_name, run_name, label_key, skipped=True, reason=f"signal_{signal}_missing")

    try:
        label_val = float(case["attrs"][label_key])
    except (KeyError, TypeError, ValueError):
        label_val = float("nan")

    t_raw, y_raw = case["signals"][signal]
    t_cut, y_cut = _cut_signal(t_raw, y_raw, *settings["cut"])
    sig_meta = {"label_key": label_key, "label_val": label_val, "signal": signal}
    if is_green and "Axial_vel" in case["signals"]:
        sig_meta["velocity"] = _cut_signal(t_raw, case["signals"]["Axial_vel"][1], *settings["cut"])[1]

    runner, sig_cls = _INDICATORS[ind_id]
    sig = sig_cls(t_analysis=t_cut, signal_analysis=y_cut, path=h5_path,
                  fs=1.0 / float(t_raw[1] - t_raw[0]), meta=sig_meta)

    config = dict(cfg)  # copia: la referencia se agrega solo para esta llamada
    if settings["reference_h5"]:
        for key, label in _REFERENCE_KEYS[ind_id].items():
            config[key] = _reference_pieces(ind_id, settings["reference_h5"], label, signal)

    try:
        result = runner(sig, config)
    except Exception as exc:
        log.error("  [%s / %s] ERROR: %s", grp_name, run_name, exc, exc_info=True)
        return _empty(grp_name, run_name, label_key, label_val, error=str(exc))

    t_d = np.asarray(result.t_d if result.t_d is not None else [], dtype=float)
    meta = {
        k: v for k, v in dict(getattr(result, "meta", {})).items()
        if not callable(v) and k not in ("raw_result", "signal")
    }

    if t_d.size > 0:
        log.info("  [%s / %s] t_d = %.4f s", grp_name, run_name, t_d[0])
    else:
        log.info("  [%s / %s] sin detección", grp_name, run_name)
    log.info("  [%s / %s] FIN \n", grp_name, run_name)

    return {
        "case": grp_name,
        "run_name": run_name,
        "t": np.asarray(getattr(result, "t", []), dtype=float),
        "I_t": np.asarray(getattr(result, "I_t", []), dtype=float),
        "t_d": t_d,
        "meta": meta,
        "attrs": {
            "indicator": _PREFIX[(ind_id, cfg.get("func", "Default"))],
            "id": ind_id,
            "func": cfg.get("func", "Default"),
            "mode": cfg["param_mode"],
            "signal": signal,
            "reference_h5": os.path.basename(settings["reference_h5"] or ""),
            **{f"pp_{k}": v for k, v in cfg["params_physical"].items()},
        },
        "label_key": label_key,
        "label_val": label_val,
    }


# ==============================================================================
# ESCRITURA HDF5
# ==============================================================================

def _safe_attr(v):
    """Convierte un valor a un tipo aceptado como attr HDF5 (escalar, str o array 1D numérico)."""
    if isinstance(v, (bool, int, float, str, bytes)):
        return v
    if isinstance(v, np.ndarray):
        if v.ndim == 1 and v.dtype.kind in ("f", "i", "u", "b"):
            return v
        return str(v.tolist())
    if isinstance(v, (list, tuple)):
        try:
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.dtype.kind in ("f", "i", "u"):
                return arr
        except Exception:
            pass
        return str(v)
    return str(v)


def write_results(out_path: str, res: Dict[str, Any], h5_src: str,
                  save_meta_arrays: bool = False) -> None:
    """Escribe un resultado en out_path (modo 'a': se acumula entre corridas).

    La primera vez que aparece un caso copia además sus attrs y las señales
    crudas (Axial_disp, Axial_vel, Axial_acc) desde h5_src.
    """
    with h5py.File(out_path, "a") as out_f:
        case_grp = out_f.require_group(res["case"])
        if res["run_name"] in case_grp:
            del case_grp[res["run_name"]]

        if "signals_written" not in case_grp.attrs:
            try:
                with h5py.File(h5_src, "r") as src_f:
                    src = src_f[res["case"]]
                    for k, v in src.attrs.items():
                        case_grp.attrs[k] = v
                    for sig in ("Axial_disp", "Axial_vel", "Axial_acc"):
                        if sig in src and sig not in case_grp:
                            for ds in ("time", "values"):
                                case_grp.create_dataset(f"{sig}/{ds}", data=src[f"{sig}/{ds}"][()],
                                                        compression="gzip")
                case_grp.attrs["signals_written"] = True
                if res.get("label_key"):
                    case_grp.attrs["label_key"] = res["label_key"]
                if not np.isnan(res["label_val"]):
                    case_grp.attrs["label_val"] = res["label_val"]
            except Exception as exc:
                log.warning("No se pudieron copiar señales del caso '%s': %s", res["case"], exc)

        grp = case_grp.create_group(res["run_name"])
        for ds in ("t", "I_t", "t_d"):
            if res[ds].size > 0:
                grp.create_dataset(ds, data=res[ds], compression="gzip")

        if res.get("label_key"):
            grp.attrs["label_key"] = res["label_key"]
        if not np.isnan(res["label_val"]):
            grp.attrs["label_val"] = res["label_val"]
        for k, v in res["attrs"].items():
            grp.attrs[k] = _safe_attr(v)

        for k, v in res["meta"].items():
            if isinstance(v, (list, np.ndarray)):
                if save_meta_arrays:
                    arr = np.asarray(v)
                    if arr.ndim == 1 and arr.dtype.kind in ("f", "i", "u"):
                        grp.create_dataset(f"meta_{k}", data=arr, compression="gzip")
                continue
            grp.attrs[f"meta_{k}"] = _safe_attr(v)


# ==============================================================================
# EJECUCIÓN PARALELA / SECUENCIAL
# ==============================================================================

def run_all(
    h5_path: str,
    groups: List[str],
    runs: List[Dict[str, Any]],
    settings: Dict[str, Any],
    nb_workers: int,
    dry_run: bool,
    out_path: Optional[str],
    save_meta_arrays: bool,
) -> int:
    """Corre cada config de *runs* sobre cada caso de *groups*.

    Cada resultado se escribe al HDF5 en cuanto llega (no se acumulan en
    memoria). Retorna el número de tareas terminadas.
    """
    tasks = [(grp, run) for grp in groups for run in runs]
    total = len(tasks)
    log.info("Total tareas: %d casos × %d configs = %d", len(groups), len(runs), total)
    n_done = 0

    def _handle(res: Dict[str, Any]) -> None:
        nonlocal n_done
        n_done += 1
        log.info("[%d/%d] completado: %s / %s", n_done, total, res["case"], res["run_name"])
        if out_path:
            write_results(out_path, res, h5_path, save_meta_arrays)

    if nb_workers == 1 or dry_run:
        for grp, run in tasks:
            _handle(_run_one(h5_path, grp, run, settings, dry_run))
    else:
        with ProcessPoolExecutor(max_workers=nb_workers) as executor:
            future_map = {
                executor.submit(_run_one, h5_path, grp, run, settings): (grp, run)
                for grp, run in tasks
            }
            for future in as_completed(future_map):
                grp, run = future_map[future]
                try:
                    _handle(future.result())
                except Exception as exc:
                    log.error("Tarea %s / %s falló: %s", grp, _run_name(run), exc)
    return n_done


# ==============================================================================
# --list
# ==============================================================================

def list_cases(h5_path: str, groups: List[str], label_key: Optional[str]) -> None:
    """Tabla: grupo | valor de label_key | señales."""
    with h5py.File(h5_path, "r") as f:
        rows = [
            (g, f[g].attrs.get(label_key) if label_key else None,
             [k for k in f[g].keys() if isinstance(f[g][k], h5py.Group)])
            for g in groups
        ]
    w = max([len("group")] + [len(r[0]) for r in rows])
    print(f"\n{'group':<{w}}  {str(label_key):>16}  signals")
    print("-" * (w + 50))
    for g, lv, sigs in rows:
        try:
            lv_str = f"{float(lv):g}"
        except (TypeError, ValueError):
            lv_str = "N/A" if lv is None else str(lv)
        print(f"{g:<{w}}  {lv_str:>16}  {', '.join(sigs[:4])}{'...' if len(sigs) > 4 else ''}")
    print(f"\n  Total: {len(rows)} casos  |  label_key: {label_key}\n")


if __name__ == "__main__":
    main()
