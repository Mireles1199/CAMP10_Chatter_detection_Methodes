#!/usr/bin/env python
# coding: utf-8
"""Etapa 1 - Analisis de a_p del DOE.

Genera valores de profundidad axial a_p a partir de un critero teorico o de
un valor de referencia, y guarda la metainformacion asociada en doe_results.h5.

Uso:
    python Etapa_1.py              # imprime resultados
    python Etapa_1.py --dry-run    # solo imprime, no escribe
"""

import os
import argparse
import numpy as np

from typing import List, Optional

import h5py
import matplotlib.pyplot as plt


def compute_ap_crit_theo(n0: float) -> float:
    """Calculo teorico de a_p critico a partir de n0.

    Implementa aqui la expresion analitica del proyecto si se dispone de ella.
    """
    raise NotImplementedError("Implement compute_ap_crit_theo(n0) or pass ap_crit")


def epsilon(eps_list: List[float] = None) -> List[float]:
    """Devuelve la lista de epsilon a usar en el barrido de a_p."""
    if eps_list is not None:
        return list(eps_list)
    return [0.50, 0.70, 0.85, 0.90, 0.95, 0.98, 0.99, 1.00,
            1.01, 1.02, 1.05, 1.10, 1.15, 1.30, 1.50]


def compute_ap_values(ap_crit: float, eps_list: List[float] = None) -> List[float]:
    """Calcula la lista de a_p a partir de a_p critico y epsilon."""
    return [lam * ap_crit for lam in epsilon(eps_list)]


def generate_ap_values_from_rpm(n0: Optional[float] = None, ap_crit: Optional[float] = None,
                                 eps_list: List[float] = None) -> List[float]:
    """Devuelve la lista de a_p desde n0 o desde a_p critico."""
    if ap_crit is None:
        if n0 is None:
            raise ValueError("Either n0 or ap_crit must be provided")
        ap_crit = compute_ap_crit_theo(n0)
    return compute_ap_values(ap_crit, eps_list=eps_list)


def format_ap_list(ap_list: List[float]) -> List[str]:
    return [f"{v:.6g}" for v in ap_list]


def compute_static_deflection(force_ref: float, stiffness: float, alpha_deg: float = 135.0, theta_deg: float = 90.0) -> float:
    """Calcula la deflexion estatica proyectando la fuerza en la direccion modal.

    La formula usada es q_s = F_c * cos(alpha - theta) / K_sys.
    El resultado queda en metros si K_sys esta en N/m.
    """
    modal_force = force_ref * np.cos(np.deg2rad(alpha_deg - theta_deg))
    deflex_stratic_modal = modal_force / stiffness
    deflex_stratic       = np.cos(np.deg2rad(alpha_deg - theta_deg))*deflex_stratic_modal

    return deflex_stratic


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


def read_time_values(case_grp: h5py.Group, signal_name: str):
    """Lee arrays time/values desde un subgrupo de señal dentro de un caso."""
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


def compute_moving_rms(time: np.ndarray, values: np.ndarray, window_samples: int, ignore_initial_time_s: float = 0.15):
    """Calcula RMS movil sobre la señal usando una ventana de N muestras.

    Por defecto ignora los primeros `ignore_initial_time_s` segundos para evitar
    el transitorio inicial.
    """
    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if values_arr.ndim == 1:
        values_arr = values_arr[:, np.newaxis]
    if time_arr.size != values_arr.shape[0]:
        raise ValueError("time y values deben tener la misma cantidad de muestras")
    if window_samples < 1:
        raise ValueError("window_samples debe ser >= 1")
    if ignore_initial_time_s < 0:
        raise ValueError("ignore_initial_time_s debe ser mayor o igual que cero")

    valid_mask = time_arr >= float(ignore_initial_time_s)
    if np.count_nonzero(valid_mask) < window_samples:
        valid_mask = np.ones_like(time_arr, dtype=bool)

    time_arr = time_arr[valid_mask]
    values_arr = values_arr[valid_mask]

    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    squared_values = values_arr ** 2

    rms_columns = []
    for col_idx in range(squared_values.shape[1]):
        moving_mean = np.convolve(squared_values[:, col_idx], kernel, mode="valid")
        rms_columns.append(np.sqrt(moving_mean))

    rms_values = np.stack(rms_columns, axis=1)
    rms_time = time_arr[window_samples - 1:]
    return np.asarray(rms_time, dtype=float), np.asarray(rms_values, dtype=float)


def compute_log_rms_trend(time: np.ndarray, rms_values: np.ndarray, ignore_initial_time_s: float = 0.15) -> float:
    """Ajusta una recta a log10(RMS) y devuelve su pendiente.

    Una pendiente positiva indica crecimiento de la vibracion; una pendiente
    negativa indica decaimiento. Por defecto ignora los primeros
    `ignore_initial_time_s` segundos para evitar el transitorio inicial.
    """
    time_arr = np.asarray(time, dtype=float)
    rms_arr = np.asarray(rms_values, dtype=float)
    if rms_arr.ndim > 1:
        rms_arr = rms_arr[:, 0]
    if time_arr.size != rms_arr.size:
        raise ValueError("time y rms_values deben tener la misma cantidad de muestras")

    if ignore_initial_time_s < 0:
        raise ValueError("ignore_initial_time_s debe ser mayor o igual que cero")

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
    """Clasifica estabilidad a partir de la pendiente logarítmica del RMS.

    Devuelve 0 para estable, 1 para inestable.
    """
    if not np.isfinite(slope):
        return 1
    return 1 if slope > tol else 0


def _stable_trend_label(value) -> str:
    """Convierte el atributo stable_trend_log a una etiqueta legible."""
    try:
        return "stable" if int(value) == 0 else "unstable"
    except (TypeError, ValueError):
        return "unknown"


def compute_numeric_limit_from_h5(case_rows: List[dict], ap_crit_ref: float) -> dict:
    """Estima el limite numerico buscando la primera transicion 0->1 en stable_trend_log.

    Los casos se ordenan internamente por epsilon ascendente para ser robustos
    a datos mezclados (union de varios HDF5). Se busca la primera pareja
    consecutiva (i, i+1) tal que i es estable (0) e i+1 es inestable (1).
    lambda_crit_sim = (lambda_minus + lambda_plus) / 2
    El error porcentual se mide respecto a 1.0.
    """
    if not case_rows:
        return {
            "lambda_minus": float("nan"), "lambda_plus": float("nan"),
            "lambda_crit_sim": float("nan"), "ap_crit_sim": float("nan"),
            "percent_error": float("nan"),
        }

    # ordenar siempre por epsilon para que la busqueda de transicion sea fiable
    rows = sorted(case_rows, key=lambda r: float(r["epsilon"]))

    # buscar primera transicion consecutiva 0 -> 1
    trans_idx = None
    for i in range(len(rows) - 1):
        if int(rows[i]["stable_trend_log"]) == 0 and int(rows[i + 1]["stable_trend_log"]) == 1:
            trans_idx = i
            break

    if trans_idx is not None:
        lambda_minus = float(rows[trans_idx]["epsilon"])
        lambda_plus  = float(rows[trans_idx + 1]["epsilon"])
        ap_minus     = float(rows[trans_idx]["ap"])
        ap_plus      = float(rows[trans_idx + 1]["ap"])
    else:
        # fallback: ultimo estable / primer inestable (sin exigir consecutivos)
        stable_rows   = [r for r in rows if int(r["stable_trend_log"]) == 0]
        unstable_rows = [r for r in rows if int(r["stable_trend_log"]) == 1]
        lambda_minus = float(stable_rows[-1]["epsilon"])   if stable_rows   else float("nan")
        lambda_plus  = float(unstable_rows[0]["epsilon"])  if unstable_rows else float("nan")
        ap_minus     = float(stable_rows[-1]["ap"])        if stable_rows   else float("nan")
        ap_plus      = float(unstable_rows[0]["ap"])       if unstable_rows else float("nan")

    if np.isfinite(lambda_minus) and np.isfinite(lambda_plus):
        lambda_crit_sim = 0.5 * (lambda_minus + lambda_plus)
    else:
        lambda_crit_sim = float("nan")

    ap_crit_sim   = lambda_crit_sim * ap_crit_ref if np.isfinite(lambda_crit_sim) else float("nan")
    percent_error = abs(lambda_crit_sim - 1.0) * 100.0 if np.isfinite(lambda_crit_sim) else float("nan")

    return {
        "lambda_minus":    lambda_minus,
        "lambda_plus":     lambda_plus,
        "ap_minus":        ap_minus,
        "ap_plus":         ap_plus,
        "lambda_crit_sim": lambda_crit_sim,
        "ap_crit_sim":     ap_crit_sim,
        "percent_error":   percent_error,
    }


def _read_case_ap(grp: h5py.Group) -> float:
    """Lee el a_p del caso desde Ap_start o Ap_end."""
    if "$Ap_start$" in grp.attrs:
        return float(grp.attrs["$Ap_start$"])
    if "$Ap_end$" in grp.attrs:
        return float(grp.attrs["$Ap_end$"])
    raise KeyError(f"[{grp.name}] No se encontro Ap_start ni Ap_end en atributos")


def compute_case_force_from_ap(case_ap: float, f_tooth_mm: float, k_cut: float) -> float:
    """Calcula la fuerza teorica del caso a partir de su a_p y constantes de corte."""
    return k_cut * (case_ap * 1e3) * f_tooth_mm


def _available_case_names(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as h5f:
        return sorted([name for name in h5f.keys() if name.startswith("case_")])


def _read_rms_group(case_grp: h5py.Group, signal_name: str):
    rms_path = f"RMS_movil/{signal_name}_rms"
    if rms_path not in case_grp:
        return None
    rms_grp = case_grp[rms_path]
    time_ds = rms_grp.get("time")
    values_ds = rms_grp.get("values")
    if time_ds is None or values_ds is None:
        return None
    return time_ds[()], values_ds[()]


def _plot_case_rms_from_h5(h5_path: str) -> None:
    """Muestra dos figuras con el RMS movil de Axial_disp y Axial_vel para un caso elegido."""
    case_names = _available_case_names(h5_path)
    if not case_names:
        raise ValueError(f"No se encontraron casos en {h5_path}")

    print("Casos disponibles:")
    for idx, case_name_item in enumerate(case_names, start=0):
        print(f"  {idx:02d}. {case_name_item}")

    selection = input("Elige el numero de caso a mostrar: ").strip()
    case_idx = int(selection)
    if case_idx < 0 or case_idx >= len(case_names):
        raise IndexError("Caso seleccionado fuera de rango")

    case_name = case_names[case_idx]

    with h5py.File(h5_path, "r") as h5f:
        case_grp = h5f[case_name]
        disp_data = _read_rms_group(case_grp, "Axial_disp")
        vel_data = _read_rms_group(case_grp, "Axial_vel")

        if disp_data is None or vel_data is None:
            raise KeyError(f"[{case_name}] No se encontro RMS_movil para Axial_disp y/o Axial_vel")

        disp_time, disp_values = disp_data
        vel_time, vel_values = vel_data

    disp_values = np.asarray(disp_values)
    vel_values = np.asarray(vel_values)
    if disp_values.ndim > 1:
        disp_values = disp_values[:, 0]
    if vel_values.ndim > 1:
        vel_values = vel_values[:, 0]

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(disp_time, disp_values, color="tab:blue", lw=1.5)
    ax1.set_title(f"{case_name} - RMS movil de Axial_disp")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("RMS [m]")
    ax1.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(vel_time, vel_values, color="tab:orange", lw=1.5)
    ax2.set_title(f"{case_name} - RMS movil de Axial_vel")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("RMS [m/s]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _read_signal_group(case_grp: h5py.Group, signal_name: str):
    signal_grp = case_grp.get(signal_name)
    if signal_grp is None or not isinstance(signal_grp, h5py.Group):
        return None
    time_ds = signal_grp.get("time")
    values_ds = signal_grp.get("values")
    if time_ds is None or values_ds is None:
        return None
    return time_ds[()], values_ds[()]


def _overwrite_signal_values(case_grp: h5py.Group, signal_name: str, new_values: np.ndarray) -> None:
    signal_grp = case_grp.get(signal_name)
    if signal_grp is None or not isinstance(signal_grp, h5py.Group):
        raise KeyError(f"[{case_grp.name}] No se encontro el grupo {signal_name}")

    values_ds = signal_grp.get("values")
    if values_ds is None:
        raise KeyError(f"[{case_grp.name}/{signal_name}] No se encontro el dataset values")

    values_arr = np.asarray(new_values)
    if values_ds.shape != values_arr.shape:
        raise ValueError(
            f"[{case_grp.name}/{signal_name}] La forma de values no coincide: {values_ds.shape} != {values_arr.shape}"
        )

    values_ds[...] = values_arr


def _plot_case_signals_from_h5(h5_path: str) -> None:
    """Muestra dos figuras con Axial_disp y Axial_vel originales para un caso elegido."""
    case_names = _available_case_names(h5_path)
    if not case_names:
        raise ValueError(f"No se encontraron casos en {h5_path}")

    print("Casos disponibles:")
    for idx, case_name_item in enumerate(case_names, start=0):
        print(f"  {idx:02d}. {case_name_item}")

    selection = input("Elige el numero de caso a mostrar: ").strip()
    case_idx = int(selection)
    if case_idx < 0 or case_idx >= len(case_names):
        raise IndexError("Caso seleccionado fuera de rango")

    case_name = case_names[case_idx]

    with h5py.File(h5_path, "r") as h5f:
        case_grp = h5f[case_name]
        disp_data = _read_signal_group(case_grp, "Axial_disp")
        vel_data = _read_signal_group(case_grp, "Axial_vel")

        if disp_data is None or vel_data is None:
            raise KeyError(f"[{case_name}] No se encontro Axial_disp y/o Axial_vel")

        disp_time, disp_values = disp_data
        vel_time, vel_values = vel_data

    disp_values = np.asarray(disp_values)
    vel_values = np.asarray(vel_values)
    if disp_values.ndim > 1:
        disp_values = disp_values[:, 0]
    if vel_values.ndim > 1:
        vel_values = vel_values[:, 0]

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(disp_time, disp_values, color="tab:blue", lw=1.5)
    ax1.set_title(f"{case_name} - Axial_disp")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Desplazamiento [m]")
    ax1.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(vel_time, vel_values, color="tab:orange", lw=1.5)
    ax2.set_title(f"{case_name} - Axial_vel")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Velocidad [m/s]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def build_sld_1dof(h5_path: Optional[str] = None, spin_rate: Optional[float] = None) -> tuple:
    """Encapsula el bloque de parametros y calculo del caso 1-DOF - 150Hz.

    No modifica `Lobes.py`; solo centraliza aquí la configuracion del ejemplo.
    Retorna (fig, ax, lobes, f_peaks) para reutilizar o testear el resultado.
    """
    from sld_tools import (
        FRFModel,
        AltintasPhaseStrategy,
        LobeCalculator,
        Plotter,
    )

    # —— Parámetros
    w1 = 250.0
    w2 = 150.0
    w_delta = 0.5
    k1 = 2.26e8
    k2 = 2.13e8
    Kf = 1000e6
    zeta1 = 0.012
    zeta2 = 0.01
    theta1_A = 30.0 * np.pi / 180.0
    theta2_A = -45.0 * np.pi / 180.0
    target = 0.0
    num_lobes = 4
    k_list = np.arange(0, num_lobes)
    num_points = 50000

    r1 = lambda w: w / w1
    r2 = lambda w: w / w2

    frf_1DOF_150Hz = FRFModel.one_dof(k2, zeta2, theta2_A, r2)

    # —— Mallado y FRF
    w_min, w_max = np.min([w1, w2]), np.max([w1, w2])
    w = np.linspace(0, w_max * (1 + w_delta), num_points)
    G = frf_1DOF_150Hz.fG(w)
    H = frf_1DOF_150Hz.fH(w)

    # —— Estrategia de fase
    phase = AltintasPhaseStrategy()

    # —— Cálculo de lóbulos
    calculator = LobeCalculator(frf=frf_1DOF_150Hz, phase_strategy=phase)
    lobes, f_peaks = calculator.compute_lobes(w=w, target=target, k_list=k_list, Kf=Kf)

    # —— Gráficas
    plotter = Plotter(style_params=None)
    A_intersection = (8000.0, 25.0)
    B_intersection = (17000.0, 25.0)
    fig_1DOF_150Hz, ax_1DOF_150Hz = plotter.plot_lobes(
        lobes=lobes,
        frequency_peaks=f_peaks,
        mode_index=1,
        title="1-DOF - 150Hz",
        intersections=False,
        A_intersection=A_intersection,
        B_intersection=B_intersection,
        show_min=False,
        save_flag=False,
        save_path="SLD_cases_AD.png",
    )

    ap_theo = float("nan")
    ap_sim = float("nan")
    ap_minus = float("nan")
    ap_plus = float("nan")
    dxl_size = float("nan")


    if h5_path is not None and os.path.isfile(h5_path):
        with h5py.File(h5_path, "r") as h5f:
            ap_theo = float(h5f.attrs.get("stage_1_ap_crit", float("nan")))
            ap_sim = float(h5f.attrs.get("stage_1_ap_crit_sim", float("nan")))
            ap_minus = float(h5f.attrs.get("stage_1_ap_minus", float("nan")))
            ap_plus = float(h5f.attrs.get("stage_1_ap_plus", float("nan")))
            _first = next((n for n in h5f.keys() if n.startswith("case_")), None)
            if _first:
                dxl_size = float(h5f[_first].attrs.get("$dxl_size$", h5f[_first].attrs.get("dxl_size", float("nan"))))
                ax_1DOF_150Hz.set_title(f"1-DOF - 150Hz  |  dxl={dxl_size:.2e} m" if np.isfinite(dxl_size) else "1-DOF - 150Hz",
                                        fontsize=6, )

            case_names = sorted([name for name in h5f.keys() if name.startswith("case_")])
            if spin_rate is not None and np.isfinite(float(spin_rate)):
                x_vals = [float(spin_rate)] * len(case_names)
            else:
                x_vals = []
            y_vals = []
            colors = []
            for case_name in case_names:
                grp = h5f[case_name]
                if spin_rate is None or not np.isfinite(float(spin_rate)):
                    x_vals.append(float(grp.attrs.get("$spin_rate$", grp.attrs.get("spin_rate", np.nan))))
                y_vals.append(float(grp.attrs.get("$Ap_start$", grp.attrs.get("$Ap_end$", np.nan))))
                stable_val = grp.attrs.get("stable_trend_log", np.nan)
                colors.append("red" if int(stable_val) == 1 else "limegreen")

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)*1e3
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)


        if np.any(mask):
            ax_1DOF_150Hz.scatter(
                x_arr[mask],
                y_arr[mask],
                c=np.asarray(colors, dtype=object)[mask],
                s=18,
                marker="o",
                edgecolors="k",
                linewidths=0.35,
                # label="cases (blue=stable, red=unstable)",
                zorder=5,
            )
            # primer inestable (stable_trend_log=1) y ultimo estable (stable_trend_log=0)
            stab_arr = np.asarray([int(c == "red") for c in np.asarray(colors, dtype=object)[mask]])
            trans = next((i for i in range(len(stab_arr) - 1) if stab_arr[i] == 0 and stab_arr[i + 1] == 1), None)

            ax_1DOF_150Hz.plot([x_arr[0],x_arr[0] ], [y_arr[trans], y_arr[trans+1]], c="red", ls="--", lw=0.7, zorder=4)


            ax_1DOF_150Hz.scatter([x_arr[mask][0]], [ap_theo * 1e3], c="red", s=25, marker="x", edgecolors="k", linewidths=0.6, zorder=6, label=rf"$a_{{p,crit,theo}}={ap_theo:.4e}$ m")
            ax_1DOF_150Hz.scatter([x_arr[mask][0]], [ap_sim * 1e3], c="blue", s=18, marker="s", linewidths=0.9, zorder=7, label=rf"$a_{{p,crit,sim}}={ap_sim:.4e}$ m")
            ax_1DOF_150Hz.legend(fontsize=4, loc="lower left", framealpha=0.9)
            ax_1DOF_150Hz.set_xlabel(ax_1DOF_150Hz.get_xlabel(), fontsize=6)
            ax_1DOF_150Hz.set_ylabel(ax_1DOF_150Hz.get_ylabel(), fontsize=6)

    return fig_1DOF_150Hz, ax_1DOF_150Hz, lobes, f_peaks


def _write_stage1_metadata(h5_path: str, ap_crit: float, T_w: float, f_tooth_mm: float, k_cut: float, k_sys: float) -> None:
    """Guarda en doe_results.h5 la metainformacion de la etapa de a_p."""
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"doe_results.h5 not found: {h5_path}")

    with h5py.File(h5_path, "a") as h5f:
        h5f.attrs["stage_1_done"] = True
        h5f.attrs["stage_1_ap_crit"] = float(ap_crit)

        case_names = sorted([name for name in h5f.keys() if name.startswith("case_")])

        for case_name in case_names:
            grp = h5f[case_name]
            case_ap = _read_case_ap(grp)
            case_epsilon_ap_critic = (case_ap / ap_crit) if np.isfinite(case_ap) and ap_crit != 0 else float("nan")
            case_stable_theoric = 0 if np.isfinite(case_ap) and case_ap < ap_crit else 1
            case_force_theoric = compute_case_force_from_ap(case_ap, f_tooth_mm, k_cut)
            case_delta_theoric = compute_static_deflection(case_force_theoric, k_sys)
            rms_group = grp.require_group("RMS_movil")

            out_deflex_group = grp.require_group("Out_Deflex")

            disp_data = read_time_values(grp, "Axial_disp")
            vel_data = read_time_values(grp, "Axial_vel")
            if disp_data is not None and vel_data is not None:
                disp_time, disp_values = disp_data
                vel_time, vel_values = vel_data
                if disp_time.shape != vel_time.shape or not np.allclose(disp_time, vel_time):
                    raise ValueError(f"[{grp.name}] Axial_disp y Axial_vel no comparten el mismo eje temporal")

                disp_values_corrected = np.asarray(disp_values, dtype=float) - float(case_delta_theoric)

                disp_out_deflex_group = out_deflex_group.require_group("Axial_disp_out_deflex")
                vel_out_deflex_group = out_deflex_group.require_group("Axial_vel_out_deflex")

                if "time" in disp_out_deflex_group:
                    del disp_out_deflex_group["time"]
                disp_out_deflex_group.create_dataset("time", data=disp_time)
                if "values" in disp_out_deflex_group:
                    del disp_out_deflex_group["values"]
                disp_out_deflex_group.create_dataset("values", data=disp_values_corrected)

                if "time" in vel_out_deflex_group:
                    del vel_out_deflex_group["time"]
                vel_out_deflex_group.create_dataset("time", data=vel_time)
                if "values" in vel_out_deflex_group:
                    del vel_out_deflex_group["values"]
                vel_out_deflex_group.create_dataset("values", data=vel_values)


                # if "Axial_disp_out_deflex" not in disp_out_deflex_group:
                #     _overwrite_signal_values(grp, "Axial_disp_out_deflex", disp_values_corrected)
                # if "Axial_vel_out_deflex" not in vel_out_deflex_group:
                #     _overwrite_signal_values(grp, "Axial_vel_out_deflex", vel_values)


                window_samples = compute_window_samples(disp_time, T_w)
                rms_time_disp, rms_values_disp = compute_moving_rms(disp_time, disp_values_corrected, window_samples, ignore_initial_time_s=0.15)
                rms_time_vel, rms_values_vel = compute_moving_rms(vel_time, vel_values, window_samples, ignore_initial_time_s=0.15)

                disp_rms_group = rms_group.require_group("Axial_disp_rms")
                vel_rms_group = rms_group.require_group("Axial_vel_rms")





                if "time" in disp_rms_group:
                    del disp_rms_group["time"]
                disp_rms_group.create_dataset("time", data=rms_time_disp)
                if "values" in disp_rms_group:
                    del disp_rms_group["values"]
                disp_rms_group.create_dataset("values", data=rms_values_disp)

                if "time" in vel_rms_group:
                    del vel_rms_group["time"]
                vel_rms_group.create_dataset("time", data=rms_time_vel)
                if "values" in vel_rms_group:
                    del vel_rms_group["values"]
                vel_rms_group.create_dataset("values", data=rms_values_vel)

                disp_trend_log = compute_log_rms_trend(rms_time_disp, rms_values_disp)
                vel_trend_log = compute_log_rms_trend(rms_time_vel, rms_values_vel)
                trend_log_mean = vel_trend_log
                stable_trend_log = classify_trend_from_slope(vel_trend_log)

                rms_group.attrs["window_samples"] = int(window_samples)
                rms_group.attrs["T_w_s"] = float(T_w)
                rms_group.attrs["signal_names"] = np.array(["Axial_disp", "Axial_vel"], dtype="S")

            grp.attrs["ap_crit"] = float(ap_crit)
            grp.attrs["epsilon_ap_critic"] = case_epsilon_ap_critic
            grp.attrs["stable_theoric"] = case_stable_theoric
            grp.attrs["deflex_theoric_m"] = case_delta_theoric
            grp.attrs["force_theoric_N"] = case_force_theoric

            grp.attrs["trend_log_rms_disp"] = disp_trend_log
            grp.attrs["trend_log_rms_vel"] = vel_trend_log
            grp.attrs["trend_log_rms_mean"] = trend_log_mean
            grp.attrs["stable_trend_log"] = stable_trend_log

        case_rows = []
        for case_name in case_names:
            grp = h5f[case_name]
            case_rows.append({
                "case_name": case_name,
                "epsilon": float(grp.attrs.get("epsilon_ap_critic", float("nan"))),
                "ap": float(grp.attrs.get("$Ap_start$", grp.attrs.get("$Ap_end$", float("nan")))),
                "dxl_size": float(grp.attrs.get("$dxl_size$", grp.attrs.get("dxl_size", float("nan")))),
                "stable_theoric": grp.attrs.get("stable_theoric", float("nan")),
                "stable_trend_log": grp.attrs.get("stable_trend_log", float("nan")),
            })

        if case_names:
            first_case_grp = h5f[case_names[0]]
            root_dxl_size = float(first_case_grp.attrs.get("$dxl_size$", first_case_grp.attrs.get("dxl_size", float("nan"))))
            h5f.attrs["stage_1_dxl_size"] = root_dxl_size
        else:
            h5f.attrs["stage_1_dxl_size"] = float("nan")

        limit_result = compute_numeric_limit_from_h5(case_rows=case_rows, ap_crit_ref=ap_crit)

        h5f.attrs["stage_1_lambda_minus"]    = float(limit_result["lambda_minus"])
        h5f.attrs["stage_1_lambda_plus"]     = float(limit_result["lambda_plus"])
        h5f.attrs["stage_1_lambda_crit_sim"] = float(limit_result["lambda_crit_sim"])
        h5f.attrs["stage_1_ap_crit_sim"]     = float(limit_result["ap_crit_sim"])
        h5f.attrs["stage_1_percent_error"]   = float(limit_result["percent_error"])
        h5f.attrs["stage_1_ap_minus"]        = float(limit_result["ap_minus"])
        h5f.attrs["stage_1_ap_plus"]         = float(limit_result["ap_plus"])

        print(f"[OK] Metadata written to {h5_path}")


def main() -> None:
    """Ejecuta la etapa de a_p sin mezclarla con analisis de fuerzas."""
    parser = argparse.ArgumentParser(description="Etapa 1 - Analisis de a_p DOE")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo imprime, no escribe en el HDF5")
    parser.add_argument("--h5", default=None,
                        help="Ruta al doe_results.h5 (por defecto: misma carpeta que este script)")
    parser.add_argument("--plot-rms", action="store_true",
                        help="Lista casos y muestra dos figuras con el RMS movil de Axial_disp y Axial_vel")
    parser.add_argument("--plot-signals", action="store_true",
                        help="Lista casos y muestra dos figuras con Axial_disp y Axial_vel originales")
    parser.add_argument("--plots-lobes", action="store_true",
                        help="Ejecuta el ejemplo 1-DOF - 150Hz de LobeCalculator y muestra su figura")
    parser.add_argument("--plots", action="store_true",
                        help="Muestra fig1 (estabilidad), fig2 (rms_vel vs dxl) y fig5 (ap_crit comparacion)")
    args = parser.parse_args()

    # doe_name = "1_Detection_Limite_Lobes\\DOE_Detection_Limite_Lobes_dxl_10e-5"
    # doe_name = r"3_Sensitivity_dt\DOE_Detection_Limite_Lobes_dt_200"
    doe_name = "1_Detection_Limite_Lobes\\DOE_Detection_Limite_Lobes_dxl_1.25e-5_RUN_10"

    # ===========================================================================
    # CONSTANTES DE CORTE  (editar aqui antes de ejecutar)
    # ===========================================================================
    spin_rate  = 12_094.0     # rpm
    nb_dt_rev  = 200          # pasos de tiempo por revolucion

    ap         = 15e-3        # profundidad axial  [m]
    f_tooth_mm = 0.05         # avance por diente  [mm/diente]
    k_cut      = 1_000.0      # coeficiente de fuerza especifica de corte [N/mm^2]
    k_sys      = 2.13e8       # rigidez equivalente del sistema [N/m]

    # Radio interior / exterior del cono  [m]
    r_int      = 50e-3
    r_ext_1    = 65e-3
    r_ext_2    = 65e-3
    l_cylindre = 150e-3       # longitud del cilindro [m]
    t_start = 0.05             # inicio de ventana para max/min [s]
    t_end   = -1.0            # fin de ventana para max/min [s] (-1 = ultimo tiempo)
    T_w     = 0.1            # ventana temporal para RMS movil / envolvente [s]
    # ===========================================================================

    local_epsilons_10 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    last_epsilon_stable_10 = None# Ultimo epsilon stable

    print( np.linspace(12000, 12000, 7).tolist())

    epsilon_5 = last_epsilon_stable_10 + 0.05 if last_epsilon_stable_10 is not None else -1.0
    if epsilon_5 > 0.0:
        local_epsilons_10.append(epsilon_5)
        local_epsilons_10.sort()

    last_epsilon_stable_5 = None

    # local_epsilons_1 =   np.linspace(last_epsilon_stable_5+0.005, last_epsilon_stable_5 + 0.05 -0.005 , 9).tolist() if last_epsilon_stable_5 is not None else []
    local_epsilons_1 =   np.linspace(last_epsilon_stable_5+0.01, last_epsilon_stable_5 + 0.05 -0.01 , 4).tolist() if last_epsilon_stable_5 is not None else []

    local_epsilons = local_epsilons_10
    ap_crit = 8.6052e-3
    n0 = 12_099.28

    h5_path = args.h5 if args.h5 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), doe_name, "doe_results.h5"
    )
    print(f"Using HDF5 file: {h5_path}")

    if args.plots_lobes:
        fig_lobe, ax_lobe, lobe, f_peak = build_sld_1dof(h5_path, spin_rate=spin_rate)
        plt.show()
        return

    if args.plot_rms:
        _plot_case_rms_from_h5(h5_path)
        return

    if args.plot_signals:
        _plot_case_signals_from_h5(h5_path)
        return

    print("\nstage-1 a_p metadata:")
    print(f"  ap_crit = {ap_crit:.6e} m")
    print(f"  n0      = {n0:.2f} rpm")

    print("\nReferencia epsilon -> a_p:")
    candidate_ap_values = compute_ap_values(ap_crit, eps_list=local_epsilons)
    print("  idx | epsilon   | a_p [m]")
    print("  ----+-----------+-------------")
    for idx, (eps_value, ap_value) in enumerate(zip(local_epsilons, candidate_ap_values), start=0):
        print(f"  {idx:>3d} | {eps_value:>9.6g} | {ap_value:>11.6e}")

    force_ref = k_cut * (ap_crit * 1e3) * f_tooth_mm
    delta_stat_ref = compute_static_deflection(force_ref, k_sys)

    print("\ndeflexion estatica teorica:")
    print(f"  F_ref      = {force_ref:.6f} N")
    print(f"  K_cut      = {k_cut:.6f} N/mm^2")
    print(f"  K_sys      = {k_sys:.6e} N/m")
    print(f"  delta_stat = {delta_stat_ref:.6e} m")
    print(f"  T_w        = {T_w:.6e} s")

    if not args.dry_run:
        _write_stage1_metadata(h5_path, ap_crit, T_w, f_tooth_mm, k_cut, k_sys)

    print("\nCasos guardados en el HDF5:")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"No se encontro el HDF5: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        case_names = sorted([name for name in h5f.keys() if name.startswith("case_")])
        if not case_names:
            print("  [sin casos encontrados]")
        else:
            case_rows = []
            for case_name in case_names:
                grp = h5f[case_name]
                case_ap = float(grp.attrs.get("$Ap_start$", grp.attrs.get("$Ap_end$", np.nan)))
                epsilon_ap = float(grp.attrs.get("epsilon_ap_critic", np.nan))
                dxl_size = float(grp.attrs.get("$dxl_size$", grp.attrs.get("dxl_size", np.nan)))
                stable_theoric_value = grp.attrs.get("stable_theoric", np.nan)
                stable_trend_log_value = grp.attrs.get("stable_trend_log", np.nan)

                case_rows.append({
                    "case_name": case_name,
                    "epsilon": epsilon_ap,
                    "ap": case_ap,
                    "dxl_size": dxl_size,
                    "stable_theoric": stable_theoric_value,
                    "stable_trend_log": stable_trend_log_value,
                })

            case_rows.sort(key=lambda row: row["epsilon"])

            limit_result = compute_numeric_limit_from_h5(case_rows, ap_crit)

            if args.dry_run:
                root_dxl_size        = float(case_rows[0]["dxl_size"]) if case_rows else float("nan")
                root_lambda_minus    = float(limit_result["lambda_minus"])
                root_lambda_plus     = float(limit_result["lambda_plus"])
                root_lambda_crit_sim = float(limit_result["lambda_crit_sim"])
                root_ap_crit_sim     = float(limit_result["ap_crit_sim"])
                root_percent_error   = float(limit_result["percent_error"])
                root_ap_minus        = float(limit_result["ap_minus"])
                root_ap_plus         = float(limit_result["ap_plus"])
            else:
                # Use the already-open h5f handle — avoids double-open on Windows
                root_dxl_size        = float(h5f.attrs.get("stage_1_dxl_size",      case_rows[0]["dxl_size"] if case_rows else float("nan")))
                root_lambda_minus    = float(h5f.attrs.get("stage_1_lambda_minus",   limit_result["lambda_minus"]))
                root_lambda_plus     = float(h5f.attrs.get("stage_1_lambda_plus",    limit_result["lambda_plus"]))
                root_lambda_crit_sim = float(h5f.attrs.get("stage_1_lambda_crit_sim",limit_result["lambda_crit_sim"]))
                root_ap_crit_sim     = float(h5f.attrs.get("stage_1_ap_crit_sim",    limit_result["ap_crit_sim"]))
                root_percent_error   = float(h5f.attrs.get("stage_1_percent_error",  limit_result["percent_error"]))
                root_ap_minus        = float(h5f.attrs.get("stage_1_ap_minus",        limit_result["ap_minus"]))
                root_ap_plus         = float(h5f.attrs.get("stage_1_ap_plus",         limit_result["ap_plus"]))

            print("\nRoot metadata stage-1 (from HDF5 root attrs):")
            print(f"  dxl_size         = {root_dxl_size:.6e}")
            print(f"  epsilon_minus    = {root_lambda_minus:.6f}  |  ap_minus = {root_ap_minus:.6e} m")
            print(f"  epsilon_plus     = {root_lambda_plus:.6f}  |  ap_plus  = {root_ap_plus:.6e} m")
            print(f"  epsilon_crit_sim = {root_lambda_crit_sim:.6f}")
            print(f"  ap_crit_sim [m]  = {root_ap_crit_sim:.6e}")
            print(f"  error_percentual = {root_percent_error:.6f} %")



            print("  idx | case_name | epsilon  | ap [m]       | stable_trend_log")
            print("  ----+-----------+----------+-------------+-----------------")
            for idx, row in enumerate(case_rows, start=0):
                stable_trend_log_label = _stable_trend_label(row["stable_trend_log"])
                print(
                    f"  {idx:>3d} | {row['case_name']:<9} | {float(row['epsilon']):>8.5f} | {float(row['ap']):>11.6e} | "
                    f"{stable_trend_log_label:>15}"
                )

            if args.plots:
                fig1_stability_map(case_rows, ap_crit, h5_path)
                fig2_rms_vel_vs_epsilon(case_rows, h5_path)
                fig5_crit_comparison(h5_path)
                plt.show()


# ==============================================================================
# FIGURAS
# ==============================================================================

def _save_fig(fig, h5_path: str, filename: str) -> str:
    """Guarda PNG en carpeta plots/ junto al HDF5."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(h5_path)), "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def fig1_stability_map(case_rows: list, ap_crit: float, h5_path: str) -> None:
    """Fig1: epsilon_ap_critic (x) vs stable_trend_log (y)."""
    rows = sorted(case_rows, key=lambda r: float(r["epsilon"]))
    eps   = [float(r["epsilon"]) for r in rows]
    stab  = [float(r["stable_trend_log"]) for r in rows]
    with h5py.File(h5_path, "r") as hf:
        dxl_size = float(hf.attrs.get("stage_1_dxl_size", float("nan")))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(eps, stab, color="steelblue", zorder=3)
    ax.axvline(ap_crit / ap_crit, color="gray", linestyle="--", linewidth=0.9, label=r"$\epsilon=1$")
    ap_crit_sim_eps = None
    for i in range(len(stab) - 1):
        if stab[i] == 0 and stab[i + 1] == 1:
            ap_crit_sim_eps = (eps[i] + eps[i + 1]) / 2.0
            break
    if ap_crit_sim_eps is not None:
        ax.axvline(ap_crit_sim_eps, color="red", linestyle=":", linewidth=1.1, label=rf"$\epsilon_{{crit,sim}}={ap_crit_sim_eps:.4f}$")
    ax.set_xlabel(r"$\epsilon = a_p / a_{p,crit}$")
    ax.set_ylabel("stable_trend_log  (0=stable, 1=unstable)")
    dxl_label = f"{dxl_size:.2e}" if np.isfinite(dxl_size) else "N/A"
    ax.set_title(f"Fig 1 - Stability map by epsilon  |  dxl_size = {dxl_label} m")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["stable (0)", "unstable (1)"])
    ax.legend(title="blue=stable, red=unstable")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = _save_fig(fig, h5_path, "fig1_stability_map.png")
    print(f"[fig1] saved -> {out}")
    


def fig2_rms_vel_vs_epsilon(case_rows: list, h5_path: str) -> None:
    """Fig2: epsilon (x) vs trend_log_rms_vel (y), coloured by stability."""
    with h5py.File(h5_path, "r") as hf:
        dxl_size = float(hf.attrs.get("stage_1_dxl_size", float("nan")))
        data = []
        for r in case_rows:
            grp = hf[r["case_name"]]
            rms_vel = float(grp.attrs.get("trend_log_rms_vel", float("nan")))
            data.append({
                "eps": float(r["epsilon"]),
                "rms_vel": rms_vel,
                "stab": int(r["stable_trend_log"]),
            })

    data.sort(key=lambda d: d["eps"])
    stable_pts   = [(d["eps"], d["rms_vel"]) for d in data if d["stab"] == 0]
    unstable_pts = [(d["eps"], d["rms_vel"]) for d in data if d["stab"] == 1]

    # epsilon_crit_sim from root attrs
    eps_crit_sim = None
    with h5py.File(h5_path, "r") as hf2:
        v = hf2.attrs.get("stage_1_lambda_crit_sim", None)
        if v is not None:
            eps_crit_sim = float(v)

    fig, ax = plt.subplots(figsize=(8, 4))
    if stable_pts:
        ax.scatter(*zip(*stable_pts),   color="steelblue", label="stable",   zorder=3)
    if unstable_pts:
        ax.scatter(*zip(*unstable_pts), color="tomato",    label="unstable", zorder=3)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.9, label=r"$\epsilon=1$")
    if eps_crit_sim is not None and np.isfinite(eps_crit_sim):
        ax.axvline(eps_crit_sim, color="red", linestyle=":", linewidth=1.1,
                   label=rf"$\epsilon_{{crit,sim}}={eps_crit_sim:.4f}$")
    ax.set_xlabel(r"$\epsilon = a_p / a_{p,crit}$")
    ax.set_ylabel("trend_log_rms_vel  [log(m/s)/s]")
    dxl_label = f"{dxl_size:.2e}" if np.isfinite(dxl_size) else "N/A"
    ax.set_title(f"Fig 2 - RMS velocity trend vs epsilon  |  dxl_size = {dxl_label} m")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = _save_fig(fig, h5_path, "fig2_rms_vel_vs_epsilon.png")
    print(f"[fig2] saved -> {out}")
    


def fig5_crit_comparison(h5_path: str) -> None:
    """Fig5: comparacion ap_crit_theo vs ap_crit_sim desde atributos raiz."""
    with h5py.File(h5_path, "r") as hf:
        ap_theo  = float(hf.attrs.get("stage_1_ap_crit",      float("nan")))
        ap_sim   = float(hf.attrs.get("stage_1_ap_crit_sim",  float("nan")))
        ap_minus = float(hf.attrs.get("stage_1_ap_minus",     float("nan")))
        ap_plus  = float(hf.attrs.get("stage_1_ap_plus",      float("nan")))
        pct_err  = float(hf.attrs.get("stage_1_percent_error",float("nan")))
        dxl_size = float(hf.attrs.get("stage_1_dxl_size",     float("nan")))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvspan(ap_minus, ap_plus, alpha=0.15, color="orange", label="sim interval")

    theo_label = rf"$a_{{p,crit,theo}}={ap_theo:.4e}$ m"
    sim_label  = rf"$a_{{p,crit,sim}}={ap_sim:.4e}$ m"

    ax.scatter([ap_theo], [0], marker="s", s=110, color="red", edgecolors="k", linewidths=0.6, zorder=6,
               label=theo_label)

    if np.isfinite(ap_minus) and np.isfinite(ap_plus):
        ap_mid = 0.5 * (ap_minus + ap_plus)
        ax.axvline(ap_mid, color="red", linestyle=":", linewidth=1.2, label="limit line")

    ax.scatter([ap_sim], [0], marker="x", s=115, color="black", linewidths=1.6, zorder=7,
               label=sim_label)
    ax.annotate(
        rf"$a_{{p,crit,sim}}={ap_sim:.4e}$ m",
        xy=(ap_sim, 0),
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.9),
    )

    ax.set_xlabel(r"$a_p$ [m]")
    dxl_label = f"{dxl_size:.2e}" if np.isfinite(dxl_size) else "N/A"
    ax.set_title(rf"Fig 5 - Theoretical vs simulated  |  dxl_size = {dxl_label} m")
    ax.set_yticks([])
    ax.legend(title=rf"error = {pct_err:.3f}%")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out = _save_fig(fig, h5_path, "fig5_crit_comparison.png")
    print(f"[fig5] saved -> {out}")
    


if __name__ == '__main__':
    main()
