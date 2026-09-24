#!/usr/bin/env python
# coding: utf-8
"""Etapa 3 - Sensibilidad a la discretizacion temporal (N_dt).

Mismo flujo que Etapa1/Etapa2 pero variando N_dt (pasos/revolucion) en vez del
tamano de dexel. Los .h5 de 3_Sensitivity_dt todavia no traen la metadata de
clasificacion estable/inestable (attrs "stage_1_*"), asi que este script la
calcula el mismo criterio que Etapa_1.py (RMS movil + pendiente log-RMS +
biseccion), reimplementado localmente para no depender de Etapa_1.py como
modulo -- misma convencion de autocontencion que ya usa Etapa_2.py.
"""

import os
import argparse
from typing import List

import numpy as np
import h5py
import matplotlib.pyplot as plt


# ==============================================================================
# CLASIFICACION ESTABLE/INESTABLE (reimplementacion local de Etapa_1.py)
# ==============================================================================

def _read_case_ap(grp: h5py.Group) -> float:
    """Lee el a_p del caso desde Ap_start o Ap_end."""
    if "$Ap_start$" in grp.attrs:
        return float(grp.attrs["$Ap_start$"])
    if "$Ap_end$" in grp.attrs:
        return float(grp.attrs["$Ap_end$"])
    raise KeyError(f"[{grp.name}] No se encontro Ap_start ni Ap_end en atributos")


def compute_static_deflection(force_ref: float, stiffness: float, alpha_deg: float = 135.0, theta_deg: float = 90.0) -> float:
    """Deflexion estatica proyectada en la direccion modal: q_s = F_c * cos(alpha - theta) / K_sys."""
    modal_force = force_ref * np.cos(np.deg2rad(alpha_deg - theta_deg))
    deflex_static_modal = modal_force / stiffness
    return np.cos(np.deg2rad(alpha_deg - theta_deg)) * deflex_static_modal


def compute_case_force_from_ap(case_ap: float, f_tooth_mm: float, k_cut: float) -> float:
    return k_cut * (case_ap * 1e3) * f_tooth_mm


def read_time_values(case_grp: h5py.Group, signal_name: str):
    """Lee arrays time/values desde un subgrupo de senal dentro de un caso."""
    if signal_name not in case_grp:
        return None
    obj = case_grp[signal_name]
    if not isinstance(obj, h5py.Group):
        return None
    time_ds = obj.get("time")
    values_ds = obj.get("values")
    if time_ds is None or values_ds is None:
        return None
    return time_ds[()], values_ds[()]


def compute_window_samples(time: np.ndarray, T_w: float) -> int:
    """Convierte una ventana temporal T_w en numero de muestras usando la malla de tiempo."""
    time_arr = np.asarray(time, dtype=float)
    if time_arr.size < 2:
        raise ValueError("Se necesitan al menos dos puntos de tiempo para estimar la frecuencia de muestreo")
    dt = float(np.mean(np.diff(time_arr)))
    if dt <= 0:
        raise ValueError("La malla de tiempo debe ser creciente")
    if T_w <= 0:
        raise ValueError("T_w debe ser mayor que cero")
    return max(1, int(round(T_w / dt)))


def compute_moving_rms(time: np.ndarray, values: np.ndarray, window_samples: int, ignore_initial_time_s: float = 0.15):
    """RMS movil sobre la senal (ventana de N muestras), ignorando el transitorio inicial."""
    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if values_arr.ndim == 1:
        values_arr = values_arr[:, np.newaxis]
    if time_arr.size != values_arr.shape[0]:
        raise ValueError("time y values deben tener la misma cantidad de muestras")
    if window_samples < 1:
        raise ValueError("window_samples debe ser >= 1")

    valid_mask = time_arr >= float(ignore_initial_time_s)
    if np.count_nonzero(valid_mask) < window_samples:
        valid_mask = np.ones_like(time_arr, dtype=bool)

    time_arr = time_arr[valid_mask]
    values_arr = values_arr[valid_mask]

    squared_values = values_arr ** 2

    # Media movil via suma acumulada -- O(n) en vez de la convolucion directa
    # O(n*window_samples) de np.convolve. Mismo resultado exacto que convolve
    # con kernel uniforme en modo "valid" (mean(x[i:i+window_samples]) para
    # cada i), pero indispensable para los dt finos de esta etapa: ahi
    # window_samples crece proporcional a nb_dt_rev (mas pasos por T_w fijo),
    # y con convolve directo el costo se volvia minutos por caso.
    rms_columns = []
    for col_idx in range(squared_values.shape[1]):
        cumsum = np.cumsum(squared_values[:, col_idx], dtype=float)
        cumsum = np.insert(cumsum, 0, 0.0)
        moving_sum = cumsum[window_samples:] - cumsum[:-window_samples]
        moving_mean = moving_sum / float(window_samples)
        rms_columns.append(np.sqrt(moving_mean))

    rms_values = np.stack(rms_columns, axis=1)
    rms_time = time_arr[window_samples - 1:]
    return np.asarray(rms_time, dtype=float), np.asarray(rms_values, dtype=float)


def compute_log_rms_trend(time: np.ndarray, rms_values: np.ndarray, ignore_initial_time_s: float = 0.15) -> float:
    """Pendiente de una recta ajustada a log10(RMS); positiva = crece (inestable)."""
    time_arr = np.asarray(time, dtype=float)
    rms_arr = np.asarray(rms_values, dtype=float)
    if rms_arr.ndim > 1:
        rms_arr = rms_arr[:, 0]
    if time_arr.size != rms_arr.size:
        raise ValueError("time y rms_values deben tener la misma cantidad de muestras")

    valid_mask = time_arr >= float(ignore_initial_time_s)
    if np.count_nonzero(valid_mask) < 2:
        valid_mask = np.ones_like(time_arr, dtype=bool)

    positive_mask = (rms_arr > 0) & valid_mask
    if np.count_nonzero(positive_mask) < 2:
        return float("nan")

    log_rms = np.log10(rms_arr[positive_mask])
    time_used = time_arr[positive_mask]
    slope, _intercept = np.polyfit(time_used, log_rms, 1)
    return float(slope)


def classify_trend_from_slope(slope: float, tol: float = 0.0) -> int:
    """0 = estable, 1 = inestable, a partir de la pendiente log-RMS."""
    if not np.isfinite(slope):
        return 1
    return 1 if slope > tol else 0


def compute_numeric_limit_from_h5(case_rows: List[dict], ap_crit_ref: float) -> dict:
    """Primera transicion 0->1 en stable_trend_log (ordenando por eta ascendente).

    lambda_crit_sim = (lambda_minus + lambda_plus) / 2; error respecto a eta=1.
    """
    if not case_rows:
        return {
            "lambda_minus": float("nan"), "lambda_plus": float("nan"),
            "lambda_crit_sim": float("nan"), "ap_crit_sim": float("nan"),
            "ap_minus": float("nan"), "ap_plus": float("nan"),
            "percent_error": float("nan"),
        }

    rows = sorted(case_rows, key=lambda r: float(r["eta"]))

    trans_idx = None
    for i in range(len(rows) - 1):
        if int(rows[i]["stable_trend_log"]) == 0 and int(rows[i + 1]["stable_trend_log"]) == 1:
            trans_idx = i
            break

    if trans_idx is not None:
        lambda_minus = float(rows[trans_idx]["eta"])
        lambda_plus  = float(rows[trans_idx + 1]["eta"])
        ap_minus     = float(rows[trans_idx]["ap"])
        ap_plus      = float(rows[trans_idx + 1]["ap"])
    else:
        stable_rows   = [r for r in rows if int(r["stable_trend_log"]) == 0]
        unstable_rows = [r for r in rows if int(r["stable_trend_log"]) == 1]
        lambda_minus = float(stable_rows[-1]["eta"])  if stable_rows   else float("nan")
        lambda_plus  = float(unstable_rows[0]["eta"]) if unstable_rows else float("nan")
        ap_minus     = float(stable_rows[-1]["ap"])   if stable_rows   else float("nan")
        ap_plus      = float(unstable_rows[0]["ap"])  if unstable_rows else float("nan")

    if np.isfinite(lambda_minus) and np.isfinite(lambda_plus):
        lambda_crit_sim = 0.5 * (lambda_minus + lambda_plus)
    else:
        lambda_crit_sim = float("nan")

    ap_crit_sim   = lambda_crit_sim * ap_crit_ref if np.isfinite(lambda_crit_sim) else float("nan")
    percent_error = abs(lambda_crit_sim - 1.0) * 100.0 if np.isfinite(lambda_crit_sim) else float("nan")

    return {
        "lambda_minus": lambda_minus, "lambda_plus": lambda_plus,
        "ap_minus": ap_minus, "ap_plus": ap_plus,
        "lambda_crit_sim": lambda_crit_sim, "ap_crit_sim": ap_crit_sim,
        "percent_error": percent_error,
    }


def _write_stage1_metadata_dt(h5_path: str, ap_crit: float, T_w: float,
                               f_tooth_mm: float, k_cut: float, k_sys: float,
                               force: bool = False) -> None:
    """Replica por caso la misma estructura que escribe Etapa_1.py
    (_write_stage1_metadata): grupo Out_Deflex (disp corregido por deflexion
    estatica + vel cruda) y grupo RMS_movil (RMS de ambas, con la version O(n)
    de compute_moving_rms de este archivo en vez de la convolucion directa de
    Etapa_1.py), mas los mismos attrs por caso (ap_crit, eta_ap_critic,
    stable_theoric, deflex_theoric_m, force_theoric_N, trend_log_rms_disp/vel/mean,
    stable_trend_log). res_R_p no se toca -- ya viene crudo en el caso y ni
    Etapa_1.py lo modifica en esta etapa de metadata.

    Ademas escribe en la raiz los mismos attrs "stage_1_*" que usa Etapa_2.py
    (para reusar su mismo lector/graficador sin duplicarlo). Idempotente: si
    stage_1_done ya es True y force=False, no reprocesa.
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"doe_results.h5 not found: {h5_path}")

    if not force:
        # Chequeo en modo lectura primero -- no exige el lock exclusivo de
        # escritura para el caso comun (archivo ya procesado en una corrida
        # anterior), asi un .h5 completo pero bloqueado en exclusiva (ej. subida
        # de OneDrive en curso) no impide seguir leyendolo.
        with h5py.File(h5_path, "r") as h5f_read:
            if h5f_read.attrs.get("stage_1_done", False):
                return

    with h5py.File(h5_path, "a") as h5f:
        if h5f.attrs.get("stage_1_done", False) and not force:
            return

        h5f.attrs["stage_1_ap_crit"] = float(ap_crit)

        case_names = sorted(name for name in h5f.keys() if name.startswith("case_"))

        for case_name in case_names:
            grp = h5f[case_name]
            case_ap = _read_case_ap(grp)
            case_eta = (case_ap / ap_crit) if ap_crit != 0 else float("nan")
            case_stable_theoric = 0 if case_ap < ap_crit else 1
            case_force_theoric = compute_case_force_from_ap(case_ap, f_tooth_mm, k_cut)
            case_delta_theoric = compute_static_deflection(case_force_theoric, k_sys)

            disp_data = read_time_values(grp, "Axial_disp")
            vel_data  = read_time_values(grp, "Axial_vel")
            if disp_data is None or vel_data is None:
                raise KeyError(f"[{grp.name}] No se encontro Axial_disp y/o Axial_vel")

            disp_time, disp_values = disp_data
            vel_time, vel_values = vel_data
            if disp_time.shape != vel_time.shape or not np.allclose(disp_time, vel_time):
                raise ValueError(f"[{grp.name}] Axial_disp y Axial_vel no comparten el mismo eje temporal")

            disp_values_corrected = np.asarray(disp_values, dtype=float) - float(case_delta_theoric)

            out_deflex_group = grp.require_group("Out_Deflex")
            disp_out_deflex_group = out_deflex_group.require_group("Axial_disp_out_deflex")
            vel_out_deflex_group  = out_deflex_group.require_group("Axial_vel_out_deflex")

            for sub_group, sub_time, sub_values in (
                (disp_out_deflex_group, disp_time, disp_values_corrected),
                (vel_out_deflex_group, vel_time, vel_values),
            ):
                if "time" in sub_group:
                    del sub_group["time"]
                sub_group.create_dataset("time", data=sub_time)
                if "values" in sub_group:
                    del sub_group["values"]
                sub_group.create_dataset("values", data=sub_values)

            window_samples = compute_window_samples(disp_time, T_w)
            rms_time_disp, rms_values_disp = compute_moving_rms(disp_time, disp_values_corrected, window_samples)
            rms_time_vel, rms_values_vel = compute_moving_rms(vel_time, vel_values, window_samples)

            rms_group = grp.require_group("RMS_movil")
            disp_rms_group = rms_group.require_group("Axial_disp_rms")
            vel_rms_group  = rms_group.require_group("Axial_vel_rms")

            for sub_group, sub_time, sub_values in (
                (disp_rms_group, rms_time_disp, rms_values_disp),
                (vel_rms_group, rms_time_vel, rms_values_vel),
            ):
                if "time" in sub_group:
                    del sub_group["time"]
                sub_group.create_dataset("time", data=sub_time)
                if "values" in sub_group:
                    del sub_group["values"]
                sub_group.create_dataset("values", data=sub_values)

            rms_group.attrs["window_samples"] = int(window_samples)
            rms_group.attrs["T_w_s"] = float(T_w)
            rms_group.attrs["signal_names"] = np.array(["Axial_disp", "Axial_vel"], dtype="S")

            disp_trend_log = compute_log_rms_trend(rms_time_disp, rms_values_disp)
            vel_trend_log  = compute_log_rms_trend(rms_time_vel, rms_values_vel)
            stable_trend_log = classify_trend_from_slope(vel_trend_log)

            grp.attrs["ap_crit"] = float(ap_crit)
            grp.attrs["eta_ap_critic"] = case_eta
            grp.attrs["stable_theoric"] = case_stable_theoric
            grp.attrs["deflex_theoric_m"] = case_delta_theoric
            grp.attrs["force_theoric_N"] = case_force_theoric
            grp.attrs["trend_log_rms_disp"] = disp_trend_log
            grp.attrs["trend_log_rms_vel"] = vel_trend_log
            grp.attrs["trend_log_rms_mean"] = vel_trend_log
            grp.attrs["stable_trend_log"] = stable_trend_log

        case_rows = [
            {
                "eta": float(h5f[name].attrs["eta_ap_critic"]),
                "ap": _read_case_ap(h5f[name]),
                "stable_trend_log": h5f[name].attrs["stable_trend_log"],
            }
            for name in case_names
        ]
        limit_result = compute_numeric_limit_from_h5(case_rows, ap_crit)

        h5f.attrs["stage_1_lambda_minus"]    = float(limit_result["lambda_minus"])
        h5f.attrs["stage_1_lambda_plus"]     = float(limit_result["lambda_plus"])
        h5f.attrs["stage_1_lambda_crit_sim"] = float(limit_result["lambda_crit_sim"])
        h5f.attrs["stage_1_ap_crit_sim"]     = float(limit_result["ap_crit_sim"])
        h5f.attrs["stage_1_percent_error"]   = float(limit_result["percent_error"])
        h5f.attrs["stage_1_ap_minus"]        = float(limit_result["ap_minus"])
        h5f.attrs["stage_1_ap_plus"]         = float(limit_result["ap_plus"])

        # Escrito al final, no antes del loop: si el proceso se corta a mitad
        # de un caso, el archivo queda sin stage_1_done -- una corrida
        # posterior lo reprocesa en vez de darlo por completo con datos a medias.
        h5f.attrs["stage_1_done"] = True


# ==============================================================================
# CARPETAS Y LECTURA DE CONVERGENCIA
# ==============================================================================

def dt_folder_name(dt_value: float, suffix: str = "_MERGED") -> str:
    """Nombre de carpeta DOE para un N_dt dado, ej. 12.5 -> 'DOE_Detection_Limite_Lobes_dt_12.5_MERGED'."""
    label = format(dt_value, "g")
    return f"DOE_Detection_Limite_Lobes_dt_{label}{suffix}"


def build_dt_convergence_folders(cases_dir: str, base_dt: float, factors: List[float],
                                  suffix: str = "_MERGED") -> List[str]:
    """Construye las rutas de las carpetas DOE dentro de cases_dir a partir de base_dt y los factores del barrido."""
    return [
        os.path.join(cases_dir, dt_folder_name(base_dt * factor, suffix))
        for factor in factors
    ]


def read_dt_convergence_data(folders: List[str], ap_crit: float, T_w: float,
                              f_tooth_mm: float, k_cut: float, k_sys: float,
                              force: bool = False) -> List[dict]:
    """Corre la metadata (si falta) y lee nb_dt_rev, ap_crit_sim/theo, ap_minus/plus,
    percent_error y lambda_* de cada HDF5."""
    rows = []
    for folder in folders:
        h5_path = os.path.join(folder, "doe_results.h5")
        if not os.path.isfile(h5_path):
            print(f"[WARN] No encontrado: {h5_path}")
            continue

        try:
            _write_stage1_metadata_dt(h5_path, ap_crit, T_w, f_tooth_mm, k_cut, k_sys, force=force)
        except (OSError, PermissionError) as exc:
            # ponytail: los .h5 viven en una carpeta sincronizada por OneDrive --
            # un archivo recien completado puede quedar bloqueado en exclusiva
            # mientras se sube. Se salta esta corrida (igual que una carpeta
            # faltante); reintentar mas tarde una vez libere el lock.
            print(f"[WARN] No se pudo escribir metadata en {h5_path} (bloqueado?): {exc}")
            continue

        with h5py.File(h5_path, "r") as h5f:
            ap_crit_sim     = float(h5f.attrs.get("stage_1_ap_crit_sim",     float("nan")))
            ap_crit_theo    = float(h5f.attrs.get("stage_1_ap_crit",         float("nan")))
            ap_minus        = float(h5f.attrs.get("stage_1_ap_minus",        float("nan")))
            ap_plus         = float(h5f.attrs.get("stage_1_ap_plus",         float("nan")))
            percent_error   = float(h5f.attrs.get("stage_1_percent_error",   float("nan")))
            lambda_crit_sim = float(h5f.attrs.get("stage_1_lambda_crit_sim", float("nan")))
            lambda_minus    = float(h5f.attrs.get("stage_1_lambda_minus",    float("nan")))
            lambda_plus     = float(h5f.attrs.get("stage_1_lambda_plus",     float("nan")))

            nb_dt_rev = float("nan")
            first = next((n for n in sorted(h5f.keys()) if n.startswith("case_")), None)
            if first:
                nb_dt_rev = float(h5f[first].attrs.get("$nb_dt_rev$", float("nan")))

        rows.append({
            "folder":          folder,
            "nb_dt_rev":       nb_dt_rev,
            "ap_crit_sim":     ap_crit_sim,
            "ap_crit_theo":    ap_crit_theo,
            "ap_minus":        ap_minus,
            "ap_plus":         ap_plus,
            "percent_error":   percent_error,
            "lambda_crit_sim": lambda_crit_sim,
            "lambda_minus":    lambda_minus,
            "lambda_plus":     lambda_plus,
        })
    rows.sort(key=lambda r: r["nb_dt_rev"])
    return rows


# ==============================================================================
# ESTILO DE FIGURA (ver skill article-plot-style) Y RUTA COMPARTIDA DE SALIDA
# ==============================================================================

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Latex", "plots")
FIG_PREFIX = "etapa3_"

ARTICLE_RCPARAMS = {
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 10, "lines.linewidth": 1.2, "lines.markersize": 10,
    "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "legend.frameon": False, "legend.loc": "best",
    "legend.handlelength": 2.0, "legend.borderaxespad": 0.5,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "savefig.transparent": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
}

FIGSIZE_SIMPLE = (3.5, 2.6)
FIGSIZE_WIDE = (7.16, 2.6)
FIGSCALE_SIMPLE = 1.5
PLOT_SHOW = True  # True para mostrar la figura en pantalla, False para solo guardar


def figsize_from_scale(base_figsize: tuple, scale: float) -> tuple:
    """Escala un figsize base preservando su relacion de aspecto."""
    w, h = base_figsize
    return (w * scale, h * scale)


def _lang_text(en: str, fr: str, language: str, sep: str = "\n") -> str:
    """Arma el texto de la figura segun el idioma elegido ("EN" | "FR" | "both")."""
    if language == "EN":
        return en
    if language == "FR":
        return fr
    if language == "both":
        return f"[EN] {en}{sep}[FR] {fr}"
    raise ValueError(f"language debe ser 'EN', 'FR' o 'both', recibido: {language!r}")


def _save_fig(fig, filename: str) -> str:
    """Guarda una figura PNG en PLOTS_DIR con el prefijo de esta etapa (FIG_PREFIX)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{FIG_PREFIX}{filename}")
    fig.savefig(out_path)
    return out_path


def plot_dt_convergence(data: List[dict], language: str = "both",
                         outlier_eps_tol: float = 0.15) -> plt.Figure:
    """Influencia de N_dt (pasos/revolucion) en el limite detectado (eta) vs. el
    teorico -- mismo estilo que plot_epsilon_convergence de Etapa_2.py (banda de
    resolucion de deteccion, broken axis para outliers, notacion eta=a_p/a_p,crit,theo),
    con N_dt en el eje X en vez del tamano de dexel.
    """
    plt.rcParams.update(ARTICLE_RCPARAMS)

    ndt       = np.asarray([r["nb_dt_rev"]       for r in data], dtype=float)
    lam_crit  = np.asarray([r["lambda_crit_sim"] for r in data], dtype=float)
    lam_minus = np.asarray([r["lambda_minus"]    for r in data], dtype=float)
    lam_plus  = np.asarray([r["lambda_plus"]     for r in data], dtype=float)

    yerr_lo = np.where(np.isfinite(lam_crit - lam_minus), lam_crit - lam_minus, 0.0)
    yerr_hi = np.where(np.isfinite(lam_plus - lam_crit), lam_plus - lam_crit, 0.0)

    is_outlier = np.isfinite(lam_crit) & (np.abs(lam_crit - 1.0) > outlier_eps_tol)
    # A diferencia de Etapa_2.py, esta funcion SOLO abre el panel superior
    # (broken axis) si de verdad hay al menos un outlier -- con los N_dt de
    # esta etapa lo habitual es que no haya ninguno, y dibujar los 9 puntos
    # completos en ambos paneles sin filtrar (como hacia la version original)
    # los duplicaba visualmente porque el panel superior nunca quedaba
    # recortado a un rango propio.
    has_outliers = bool(np.any(is_outlier))

    if has_outliers:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True,
            figsize=figsize_from_scale(FIGSIZE_SIMPLE, FIGSCALE_SIMPLE),
            gridspec_kw={"height_ratios": [1.4, 2.2], "hspace": 0.08},
            constrained_layout=True,
        )
    else:
        fig, ax_bot = plt.subplots(
            1, 1, figsize=figsize_from_scale(FIGSIZE_SIMPLE, FIGSCALE_SIMPLE),
            constrained_layout=True,
        )
        ax_top = None

    theo_label = _lang_text(r"Theoretical limit $\eta=1$",
                             r"Limite théorique $\eta=1$", language, sep=" / ")
    band_label = _lang_text(r"Detection resolution [$\eta_-,\eta_+$]",
                             r"Résolution de détection [$\eta_-,\eta_+$]",
                             language, sep=" / ")
    sim_label = _lang_text(r"Simulated $\eta_{crit,sim}$",
                            r"$\eta_{crit,sim}$ simulé",
                            language, sep=" / ")

    def _band(ax, mask):
        order = np.argsort(ndt[mask])
        xs, los, his = ndt[mask][order], lam_minus[mask][order], lam_plus[mask][order]
        valid = np.isfinite(los) & np.isfinite(his)
        if np.count_nonzero(valid) < 2:
            return
        ax.fill_between(xs[valid], los[valid], his[valid],
                         color="steelblue", alpha=0.35, zorder=0, label=band_label)

    def _draw(ax, mask):
        """Dibuja linea teorica, errorbar y topes SOLO para los puntos de `mask`
        -- cada panel muestra su propio subconjunto, no el barrido completo."""
        ax.axhline(1.0, color="crimson", linewidth=1.2, linestyle="--",
                   zorder=1, label=theo_label)
        ax.errorbar(ndt[mask], lam_crit[mask], yerr=[yerr_lo[mask], yerr_hi[mask]],
                    fmt="o", color="steelblue", ecolor="steelblue",
                    capsize=0, elinewidth=1.2, ms=7,
                    zorder=2, label=sim_label)
        ax.scatter(ndt[mask], lam_plus[mask], marker="_", s=90, linewidths=1.6,
                   color="darkorange", zorder=3)
        ax.scatter(ndt[mask], lam_minus[mask], marker="_", s=90, linewidths=1.6,
                   color="green", zorder=3)
        ax.set_xscale("log")

    main_mask = ~is_outlier & np.isfinite(ndt)

    if has_outliers:
        _band(ax_top, is_outlier)
        _draw(ax_top, is_outlier)
    _band(ax_bot, main_mask)
    _draw(ax_bot, main_mask)

    if has_outliers:
        out_vals = np.concatenate([lam_minus[is_outlier], lam_plus[is_outlier], lam_crit[is_outlier]])
        out_vals = out_vals[np.isfinite(out_vals)]
        pad = 0.08 * (np.max(out_vals) - np.min(out_vals) + 1e-9)
        ax_top.set_ylim(np.min(out_vals) - pad, np.max(out_vals) + pad)

    main_vals = np.concatenate([lam_minus[main_mask], lam_plus[main_mask], lam_crit[main_mask], [1.0]])
    main_vals = main_vals[np.isfinite(main_vals)]
    pad = 0.25 * (np.max(main_vals) - np.min(main_vals) + 1e-9)
    ax_bot.set_ylim(np.min(main_vals) - pad, np.max(main_vals) + pad)

    ap_crit_theo_vals = [r["ap_crit_theo"] for r in data if np.isfinite(r["ap_crit_theo"])]
    if ap_crit_theo_vals:
        ap_crit_theo_mm = ap_crit_theo_vals[0] * 1e3
        y0, y1 = ax_bot.get_ylim()
        ax_bot.text(
            0.02, 1.0 + 0.08 * (y1 - 1.0),
            rf"$a_{{p,crit}}^{{theo}} = {ap_crit_theo_mm:.3f}$ mm",
            transform=ax_bot.get_yaxis_transform(), color="crimson",
            fontsize=plt.rcParams["legend.fontsize"], va="bottom", ha="left",
        )

    lambda_def_text = _lang_text(
        r"$\eta_-$: last stable $\eta$  /  $\eta_+$: first unstable $\eta$",
        r"$\eta_-$ : dernier $\eta$ stable  /  $\eta_+$ : premier $\eta$ instable",
        language, sep="\n")
    ax_legend_anchor = ax_top if has_outliers else ax_bot
    ax_legend_anchor.text(
        0.02, 0.98, lambda_def_text,
        transform=ax_legend_anchor.transAxes, color="steelblue",
        fontsize=plt.rcParams["legend.fontsize"] * 0.85, va="top", ha="left",
    )

    if has_outliers:
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)
        ax_bot.xaxis.tick_bottom()

        d = 0.012
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Eje X: escala log estandar. Los N_dt reales (12.5..3200, factor-2, no
    # potencias de 10) se marcan DENTRO del panel con su valor plano, igual
    # convencion que Etapa_2.py con los tamanos de dexel. Se etiquetan TODOS
    # los puntos de main_mask, incluso los que no resolvieron transicion (sin
    # eta_crit_sim, ej. N_dt=3200) -- de lo contrario ese punto no tiene forma
    # de distinguirse de "no hay dato" en el eje.
    def _ndt_label(ndt_val: float) -> str:
        return f"{ndt_val:.4g}"

    y_tick_frac = 0.05
    for x in ndt[main_mask]:
        ax_bot.plot([x, x], [0.0, y_tick_frac], transform=ax_bot.get_xaxis_transform(),
                    color="black", linewidth=0.8, zorder=4)
        ax_bot.text(x, y_tick_frac + 0.015, _ndt_label(x),
                    transform=ax_bot.get_xaxis_transform(),
                    rotation=45, va="bottom", ha="left", color="black",
                    fontsize=plt.rcParams["xtick.labelsize"] * 0.6, zorder=4)

    ax_bot.set_xlabel(_lang_text("$N_{dt}$ [steps/rev]", "$N_{dt}$ [pas/tour]", language))
    fig.supylabel(r"$\eta$")
    ax_title_anchor = ax_top if has_outliers else ax_bot
    ax_title_anchor.set_title(_lang_text(
        "Detected limit vs. $N_{dt}$",
        "Limite détectée vs. $N_{dt}$",
        language))

    if has_outliers:
        bot_handles, bot_labels = ax_bot.get_legend_handles_labels()
        top_handles, top_labels = ax_top.get_legend_handles_labels()
        label_to_handle = dict(zip(bot_labels, bot_handles))
        label_to_handle.update(dict(zip(top_labels, top_handles)))
        ordered_labels = [l for l in (theo_label, band_label, sim_label) if l in label_to_handle]
        ax_top.legend([label_to_handle[l] for l in ordered_labels], ordered_labels, loc="lower left")
    else:
        # Panel unico: la leyenda va afuera del area de datos (a la derecha) en
        # vez de adentro -- el icono del errorbar de sim_label es mas alto que
        # su texto y, puesto adentro, terminaba superpuesto con los puntos.
        bot_handles, bot_labels = ax_bot.get_legend_handles_labels()
        label_to_handle = dict(zip(bot_labels, bot_handles))
        ordered_labels = [l for l in (theo_label, band_label, sim_label) if l in label_to_handle]
        ax_bot.legend([label_to_handle[l] for l in ordered_labels], ordered_labels,
                       loc="center left", bbox_to_anchor=(1.02, 0.5))

    return fig


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Etapa 3 - sensibilidad a la discretizacion temporal N_dt")
    p.add_argument("--dry-run", action="store_true", help="Solo imprime las carpetas resueltas, no procesa ni grafica")
    p.add_argument("--plots", action="store_true", help="Corre la metadata si falta y grafica ap_crit_sim vs N_dt")
    p.add_argument("--force-metadata", action="store_true", help="Reprocesa la metadata aunque stage_1_done ya sea True")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ===========================================================================
    # CONSTANTES DE CORTE  (mismas que Etapa_1.py / Etapa_2.py)
    # ===========================================================================
    ap_crit    = 8.6052e-3    # a_p critico teorico [m]
    T_w        = 0.1          # ventana temporal para RMS movil [s]
    f_tooth_mm = 0.05         # avance por diente [mm/diente]
    k_cut      = 1_000.0      # coeficiente de fuerza especifica de corte [N/mm^2]
    k_sys      = 2.13e8       # rigidez equivalente del sistema [N/m]

    base_dt    = 200
    dt_factors = [1 / 16., 1 / 8., 1 / 4., 1 / 2., 1, 2, 4, 8, 16]

    FIGURE_LANGUAGE = "FR"

    # Los datos pesados viven en el checkout madre (no en este worktree).
    cases_dir = (
        r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
        r"\CAMP10_Chatter_detection_Methodes\Convergency_Simulation\3_Sensitivity_dt"
    )

    convergence_folders = build_dt_convergence_folders(cases_dir, base_dt, dt_factors)

    print("\nEtapa 3 - dt Sensibility")
    print(f"  ap_crit = {ap_crit:.6e} m")
    print("\nCarpetas de convergencia:")
    for folder in convergence_folders:
        status = "OK" if os.path.isfile(os.path.join(folder, "doe_results.h5")) else "MISSING"
        print(f"  [{status}] {folder}")

    if args.dry_run:
        print("[dry-run] No se procesa ni grafica.")
        return

    if args.plots:
        conv_data = read_dt_convergence_data(
            convergence_folders, ap_crit, T_w, f_tooth_mm, k_cut, k_sys,
            force=args.force_metadata,
        )
        if not conv_data:
            print("[WARN] No se encontraron datos validos en las carpetas indicadas.")
        else:
            print(f"\nDatos de convergencia ({len(conv_data)} puntos):")
            print(f"  {'N_dt':>8}  {'eps_crit_sim':>12}  {'eps_minus':>10}  {'eps_plus':>10}")
            for r in conv_data:
                print(f"  {r['nb_dt_rev']:>8.4g}  {r['lambda_crit_sim']:>12.5f}  {r['lambda_minus']:>10.5f}  {r['lambda_plus']:>10.5f}")
            fig = plot_dt_convergence(conv_data, language=FIGURE_LANGUAGE)
            if PLOT_SHOW:
                plt.show()
            out_path = _save_fig(fig, "convergence_epsilon_vs_dt.png")
            print(f"[OK] Figura guardada en: {out_path}")
        return


if __name__ == "__main__":
    main()
