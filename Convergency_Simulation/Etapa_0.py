#!/usr/bin/env python
# coding: utf-8
"""Etapa 0 - Analisis de fuerzas del DOE.

Lee `doe_results.h5` en la misma carpeta, calcula la fuerza de referencia
analitica F_ref a partir de las constantes de corte, extrae la fuerza
simulada `res_R_p` de cada caso y guarda:
  - `force_mean`                  : media temporal por componente (array 3-elem)
  - `force_error_percent_mean`    : |F_mean - F_ref| / F_ref * 100 por componente
  - `force_error_percent_maxmin`  : max(|F_max-F_ref|, |F_min-F_ref|) / F_ref * 100
  - `force_error_percent_spread`  : |F_max - F_min| / F_ref * 100
  - `force_error_percent_std`     : std(F) / F_ref * 100

Uso:
    python Etapa_0.py              # escribe resultados en doe_results.h5
    python Etapa_0.py --dry-run    # solo imprime, no escribe
    python Etapa_0.py --plots-only # solo figuras sobre HDF5 ya generado
    python Etapa_0.py --no-plots   # desactiva figuras
"""

import os
import argparse
import logging

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Nombre del dataset de fuerza en doe_results.h5
FORCE_SIGNAL = "res_R_p"

# ==============================================================================
# FIGURAS
# ==============================================================================

_STYLE = {
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "lines.linewidth": 1.4,
    "axes.linewidth": 0.8, "grid.linewidth": 0.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
}


def _sci_yaxis(ax) -> None:
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)


def _plain_yaxis(ax) -> None:
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _save_fig(fig, h5_path: str, filename: str) -> str:
    """Guarda una figura PNG en la carpeta plots junto al HDF5."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(h5_path)), "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path)
    return out_path


def _load_cases(h5_path: str) -> list:
    """Lee todos los casos y devuelve la fuerza y los errores ya guardados en HDF5."""
    cases = []
    with h5py.File(h5_path, "r") as hf:
        for name in sorted(hf.keys()):
            if not name.startswith("case_"):
                continue
            grp = hf[name]
            data = read_force_values(grp)
            if data is None:
                continue
            time_arr, values_arr = data
            values_arr = np.asarray(values_arr, dtype=float)
            if values_arr.ndim == 1:
                values_arr = values_arr[:, np.newaxis]
            dxl = float(grp.attrs.get("$dxl_size$", grp.attrs.get("dxl_size", np.nan)))
            error_mean = np.asarray(grp["force_error_percent_mean"]) if "force_error_percent_mean" in grp else None
            error_maxmin = np.asarray(grp["force_error_percent_maxmin"]) if "force_error_percent_maxmin" in grp else None
            error_spread = np.asarray(grp["force_error_percent_spread"]) if "force_error_percent_spread" in grp else None
            error_std = np.asarray(grp["force_error_percent_std"]) if "force_error_percent_std" in grp else None
            cases.append({
                "case_name": name,
                "time": np.asarray(time_arr, dtype=float),
                "force": values_arr[:, 0],
                "dxl_size": dxl,
                "error_mean": error_mean,
                "error_maxmin": error_maxmin,
                "error_spread": error_spread,
                "error_std": error_std,
            })
    return cases


def _cases_from_h5(h5_path: str):
    """Carga los casos una sola vez para reutilizarlos en varias figuras."""
    return _load_cases(h5_path)


def _zorders(cases: list) -> dict:
    """dxl_size mas alto -> zorder mas bajo (fondo). dxl_size mas bajo -> zorder mas alto (frente)."""
    valid = sorted({c["dxl_size"] for c in cases if np.isfinite(c["dxl_size"])}, reverse=True)
    rank = {v: i + 2 for i, v in enumerate(valid)}
    return {c["case_name"]: rank.get(c["dxl_size"], 10) for c in cases}


def fig3_error_summary(cases: list, F_ref: float, h5_path: str, highlight_dxl_size: float | None = None) -> None:
    """Figure 3: error summary by case using precomputed HDF5 datasets."""
    plt.rcParams.update(_STYLE)
    if not cases:
        log.warning("Figure 3: no cases available.")
        return
    cases_s = sorted(cases, key=lambda c: c["dxl_size"] if np.isfinite(c["dxl_size"]) else 1e99)
    labels, e_mean, e_maxmin, e_spread, e_std = [], [], [], [], []
    for c in cases_s:
        labels.append(f"{c['dxl_size']:.2e}" if np.isfinite(c["dxl_size"]) else "nan")
        if c["error_mean"] is None or c["error_maxmin"] is None or c["error_spread"] is None or c["error_std"] is None:
            raise KeyError(f"[{c['case_name']}] missing error datasets in HDF5")
        e_mean.append(float(np.asarray(c["error_mean"]).ravel()[0]))
        e_maxmin.append(float(np.asarray(c["error_maxmin"]).ravel()[0]))
        e_spread.append(float(np.asarray(c["error_spread"]).ravel()[0]))
        e_std.append(float(np.asarray(c["error_std"]).ravel()[0]))
    log.info(
        "Figure 3 control: first dxl_size=%s, error_mean=%.6f%%",
        labels[0],
        e_mean[0],
    )
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Error summary by case")
    highlight_index = None
    if highlight_dxl_size is not None and np.isfinite(highlight_dxl_size):
        for idx, c in enumerate(cases_s):
            if np.isfinite(c["dxl_size"]) and np.isclose(c["dxl_size"], highlight_dxl_size, rtol=0, atol=1e-12):
                highlight_index = idx
                break

    def _bar_colors(values):
        return ["#4c78a8"] * len(values)

    def _bar_edgecolors(values):
        edgecolors = ["k"] * len(values)
        if highlight_index is not None:
            edgecolors[highlight_index] = "red"
        return edgecolors

    def _bar_linewidth(values):
        linewidths = [0.4] * len(values)
        if highlight_index is not None:
            linewidths[highlight_index] = 1.6
        return linewidths

    ax.bar(x - 1.5*w, e_mean,   w, label="Mean",    edgecolor=_bar_edgecolors(e_mean), linewidth=_bar_linewidth(e_mean), color=_bar_colors(e_mean))
    ax.bar(x - 0.5*w, e_maxmin, w, label="Max/Min", edgecolor=_bar_edgecolors(e_maxmin), linewidth=_bar_linewidth(e_maxmin), color=["#f58518"] * len(e_maxmin))
    ax.bar(x + 0.5*w, e_spread, w, label="Spread",  edgecolor=_bar_edgecolors(e_spread), linewidth=_bar_linewidth(e_spread), color=["#54a24b"] * len(e_spread))
    ax.bar(x + 1.5*w, e_std,    w, label="Std",     edgecolor=_bar_edgecolors(e_std), linewidth=_bar_linewidth(e_std), color=["#b279a2"] * len(e_std))
    ax.set_xticks(x)
    tick_labels = ax.set_xticklabels(labels, rotation=45, ha="right")
    
    if highlight_index is not None:
        tick_labels[highlight_index].set_color("red")
    ax.set_xlabel(r"$dxl\_size$")
    ax.set_ylabel("Error [%]")
    _plain_yaxis(ax)
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1.0, label="10%")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = _save_fig(fig, h5_path, "fig3_error_summary.png")
    log.info("Figure 3 saved to %s", out_path)
    plt.show()
  



# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def compute_machining_constants(spin_rate, nb_dt_rev, ap, f_tooth_mm, k_f):
    """Calcula dt, V_f, A_c y F_ref a partir de las constantes de corte.

    Unidades:
        spin_rate   [rpm]
        ap          [m]
        f_tooth_mm  [mm/diente]
        k_f         [N/mm^2]

    Retorna dict con todas las constantes derivadas.
    """
    dt     = 60.0 / spin_rate / nb_dt_rev        # paso de tiempo [s]
    V_f    = spin_rate * f_tooth_mm / 1e3         # velocidad de avance [m/min]
    ap_mm  = ap * 1e3                             # profundidad en mm
    A_c    = ap_mm * f_tooth_mm                   # seccion de corte [mm^2]
    F_ref  = k_f * A_c                            # fuerza de referencia [N]

    return {
        "dt":    dt,
        "V_f":   V_f,
        "A_c":   A_c,
        "F_ref": F_ref,
    }


def _hdf5_find_dataset(group, name):
    """Busqueda recursiva de un Dataset por nombre dentro de un h5py.Group."""
    for key in group:
        item = group[key]
        if key == name:
            if isinstance(item, h5py.Dataset):
                return item
            if isinstance(item, h5py.Group) and "data" in item:
                return item["data"]
        if isinstance(item, h5py.Group):
            result = _hdf5_find_dataset(item, name)
            if result is not None:
                return result
    return None


def read_force_values(case_grp):
    """Lee tiempo y valores del dataset de fuerza desde un grupo HDF5 de caso.

    Acepta tanto:
      - case_grp[FORCE_SIGNAL]["values"]   (formato doe_runner)
      - case_grp[FORCE_SIGNAL]             (dataset directo, por si acaso)

    Retorna (time, values) o None si no encontrado.
    """
    if FORCE_SIGNAL not in case_grp:
        return None
    obj = case_grp[FORCE_SIGNAL]
    if isinstance(obj, h5py.Dataset):
        arr = obj[()]
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Dataset incompatible para fuerza: shape={getattr(arr, 'shape', None)}")
        return arr[:, 0], arr[:, 1:]
    if isinstance(obj, h5py.Group):
        time_ds = obj.get("time")
        values_ds = obj.get("values")
        if time_ds is not None and values_ds is not None:
            return time_ds[()], values_ds[()]
        ds = _hdf5_find_dataset(obj, "values")
        if ds is not None:
            values = ds[()]
            time_ds = _hdf5_find_dataset(obj, "time")
            if time_ds is not None:
                return time_ds[()], values
    return None


def compute_force_stats(values, F_ref):
    """Calcula media por componente y error porcentual respecto a F_ref.

    Si `values` es 1-D, lo trata como una sola componente.
    Si `values` es 2-D (N, k), calcula media por columna.

    Retorna (force_mean, force_error_percent) ambos como ndarray 1-D.
    F_ref puede ser escalar (mismo limite para todas las componentes).
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    force_mean = arr.mean(axis=0)
    force_error_percent = np.abs(force_mean - F_ref) / F_ref * 100.0
    return force_mean, force_error_percent


def compute_force_error_window(time, values, F_ref, t_start, t_end):
    """Calcula el error porcentual max/min dentro de una ventana temporal.

    Usa max(abs(max - F_ref), abs(min - F_ref)) / F_ref * 100 por componente.
    """
    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if values_arr.ndim == 1:
        values_arr = values_arr[:, np.newaxis]

    if t_end == -1:
        t_end = time_arr[-1]

    mask = (time_arr >= t_start) & (time_arr <= t_end)
    if not np.any(mask):
        raise ValueError(f"Ventana temporal vacia: [{t_start}, {t_end}]")

    window_values = values_arr[mask]
    window_max = window_values.max(axis=0)
    window_min = window_values.min(axis=0)
    error_max = np.abs(window_max - F_ref)
    error_min = np.abs(window_min - F_ref)
    return np.maximum(error_max, error_min) / F_ref * 100.0


def compute_force_spread_error_window(time, values, F_ref, t_start, t_end):
    """Calcula el error porcentual del rango max-min dentro de una ventana temporal.

    Usa abs(max - min) / F_ref * 100 por componente.
    """
    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if values_arr.ndim == 1:
        values_arr = values_arr[:, np.newaxis]

    if t_end == -1:
        t_end = time_arr[-1]

    mask = (time_arr >= t_start) & (time_arr <= t_end)
    if not np.any(mask):
        raise ValueError(f"Ventana temporal vacia: [{t_start}, {t_end}]")

    window_values = values_arr[mask]
    window_max = window_values.max(axis=0)
    window_min = window_values.min(axis=0)
    return np.abs(window_max - window_min) / F_ref * 100.0


def compute_force_std_error_window(time, values, F_ref, t_start, t_end):
    """Calcula el error porcentual basado en la desviacion estandar dentro de la ventana.

    Usa std(F) / F_ref * 100 por componente.
    """
    time_arr = np.asarray(time, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if values_arr.ndim == 1:
        values_arr = values_arr[:, np.newaxis]

    if t_end == -1:
        t_end = time_arr[-1]

    mask = (time_arr >= t_start) & (time_arr <= t_end)
    if not np.any(mask):
        raise ValueError(f"Ventana temporal vacia: [{t_start}, {t_end}]")

    window_values = values_arr[mask]
    return window_values.std(axis=0) / F_ref * 100.0


# ==============================================================================
# FUNCION PRINCIPAL
# ==============================================================================

def run(h5_path, constants, dry_run=False):
    """Procesa todos los grupos case_XXX en doe_results.h5."""

    F_ref  = constants["F_ref"]
    dt     = constants["dt"]
    V_f    = constants["V_f"]
    A_c    = constants["A_c"]
    t_start = constants["t_start"]
    t_end = constants["t_end"]

    # log.info("Constantes de corte:")
    # log.info("  dt    = %.4e s", dt)
    # log.info("  V_f   = %.4f m/min", V_f)
    # log.info("  A_c   = %.4f mm^2", A_c)
    # log.info("  F_ref = %.2f N", F_ref)
    # log.info("  t_start = %.6e s", t_start)
    # log.info("  t_end   = %.6e s", t_end)

    if not os.path.isfile(h5_path):
        log.error("No se encontro doe_results.h5 en: %s", h5_path)
        return

    mode = "r" if dry_run else "a"
    with h5py.File(h5_path, mode) as hf:
        case_groups = sorted(
            [k for k in hf.keys() if k.startswith("case_")],
        )
        log.info("Casos encontrados: %d", len(case_groups))

        for grp_name in case_groups:
            grp = hf[grp_name]

            force_data = read_force_values(grp)
            if force_data is None:
                log.warning("[%s] Dataset '%s' no encontrado — omitido.", grp_name, FORCE_SIGNAL)
                continue

            time, values = force_data
            force_mean, force_error_percent_mean = compute_force_stats(values, F_ref)
            force_error_percent_window = compute_force_error_window(time, values, F_ref, t_start, t_end)
            force_error_percent_spread = compute_force_spread_error_window(time, values, F_ref, t_start, t_end)
            force_error_percent_std = compute_force_std_error_window(time, values, F_ref, t_start, t_end)

            attrs_str = dict(grp.attrs)
            # log.info(
            #     "[%s] attrs=%s | F_mean=%s N | error%%=%s",
            #     grp_name, attrs_str, np.round(force_mean, 3), np.round(force_error_percent_window, 3)
            # )

            if dry_run:
                continue

            # Escribir/sobreescribir datasets en el grupo del caso
            for ds_name, data in [("force_mean", force_mean),
                                   ("force_error_percent_mean", force_error_percent_mean),
                                   ("force_error_percent_maxmin", force_error_percent_window),
                                   ("force_error_percent_spread", force_error_percent_spread),
                                   ("force_error_percent_std", force_error_percent_std)]:
                if ds_name in grp:
                    del grp[ds_name]
                grp.create_dataset(ds_name, data=data)

            # Guardar constantes usadas como atributos del grupo
            grp.attrs["F_ref_N"]      = F_ref
            grp.attrs["A_c_mm2"]      = A_c
            grp.attrs["V_f_m_min"]    = V_f
            grp.attrs["dt_s"]         = dt

    log.info("Listo.")


def main():
    parser = argparse.ArgumentParser(description="Etapa 1 — Analisis de fuerzas DOE")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo imprime, no escribe en el HDF5")
    parser.add_argument("--h5", default=None,
                        help="Ruta al doe_results.h5 (por defecto: misma carpeta que este script)")
    parser.add_argument("--plots", action="store_true", default=False,
                        help="Show Figure 3 after the analysis")
    args = parser.parse_args()

    DOE_NAME = "0_Cinematique\\DOE_Dexels_Cinematique"   # nombre de la carpeta de salida  (dir_ref2exe)
    # ===========================================================================
    # CONSTANTES DE CORTE  (editar aqui antes de ejecutar)
    # ===========================================================================
    spin_rate  = 12_094.0     # rpm
    nb_dt_rev  = 200          # pasos de tiempo por revolucion

    ap         = 15e-3        # profundidad axial  [m]
    f_tooth_mm = 0.05         # avance por diente  [mm/diente]
    k_f        = 1_000.0      # coeficiente de fuerza especifica [N/mm^2]

    # Radio interior / exterior del cono  [m]
    r_int      = 50e-3
    r_ext_1    = 65e-3
    r_ext_2    = 65e-3
    l_cylindre = 150e-3       # longitud del cilindro [m]
    t_start = 0.05             # inicio de ventana para max/min [s]
    t_end   = -1.0            # fin de ventana para max/min [s] (-1 = ultimo tiempo)
    # ===========================================================================

    constants = compute_machining_constants(spin_rate, nb_dt_rev, ap, f_tooth_mm, k_f)
    constants["t_start"] = t_start
    constants["t_end"] = t_end

    print("Constantes de corte:")
    print(f"  spin_rate  = {spin_rate:.2f} rpm")
    print(f"  nb_dt_rev  = {nb_dt_rev}")
    print(f"  dt         = {constants['dt']:.6e} s")
    print(f"  ap         = {ap:.6e} m")
    print(f"  f_tooth    = {f_tooth_mm:.6f} mm/diente")
    print(f"  K_f        = {k_f:.2f} N/mm^2")
    print(f"  R_int      = {r_int:.6e} m")
    print(f"  R_ext_1    = {r_ext_1:.6e} m")
    print(f"  R_ext_2    = {r_ext_2:.6e} m")
    print(f"  L_cylindre = {l_cylindre:.6e} m")
    print(f"  t_start    = {t_start:.6e} s")
    print(f"  t_end      = {t_end:.6e} s")
    print(f"  V_f        = {constants['V_f']:.6f} m/min")
    print(f"  A_c        = {constants['A_c']:.6f} mm^2")
    print(f"  F_ref      = {constants['F_ref']:.6f} N")

    h5_path = args.h5 if args.h5 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),DOE_NAME, "doe_results.h5"
    )

    run(h5_path, constants, dry_run=args.dry_run)

    if args.plots:
        F_ref = constants["F_ref"]
        cases = _cases_from_h5(h5_path)
        selected_dxl_size = 20.e-5  # resaltar este tamaño de dexel en la figura
        fig3_error_summary(cases, F_ref, h5_path, highlight_dxl_size=selected_dxl_size)


if __name__ == "__main__":
    main()
