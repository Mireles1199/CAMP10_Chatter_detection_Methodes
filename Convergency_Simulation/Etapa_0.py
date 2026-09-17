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
import ast
import argparse
import logging

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Nombre del dataset de fuerza en doe_results.h5
FORCE_SIGNAL = "res_R_p"

# ==============================================================================
# FIGURAS
# ==============================================================================

# rcParams y tamano de figura: ver skill article-plot-style
# (.claude/skills/article-plot-style/SKILL.md, secciones 0 y 1)
_STYLE = {
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 10, "lines.linewidth": 1.2,
    "lines.markersize": 10,
    "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "legend.frameon": False, "legend.loc": "best",
    "legend.handlelength": 2.0, "legend.borderaxespad": 0.5,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "savefig.transparent": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
}

# FIGSIZE_WIDE = (7.16, 2.6)
FIGSIZE_WIDE = (3.58, 2.6)  # ancho de pagina completa, alto ligeramente mayor para leyenda



def figsize_from_scale(base_figsize: tuple[float, float], scale: float) -> tuple[float, float]:
    """Escala un figsize base (p.ej. FIGSIZE_WIDE) preservando su relacion de aspecto.

    `scale` es un multiplicador simple (1 = tamano original, 1.5, 2, 0.5, ...)
    aplicado por igual a ancho y alto, asi la proporcion del preset original
    (FIGSIZE_SIMPLE/FIGSIZE_WIDE) se mantiene siempre.
    """
    w, h = base_figsize
    return (w * scale, h * scale)


def _sci_yaxis(ax) -> None:
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)


def _plain_yaxis(ax) -> None:
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _lang_text(en: str, fr: str, language: str, sep: str = "\n") -> str:
    """Arma el texto de la figura segun el idioma configurado en main() (FIGURE_LANGUAGE).

    language: "EN" (solo ingles) | "FR" (solo frances) | "both" (bilingue, con prefijo).
    """
    if language == "EN":
        return en
    if language == "FR":
        return fr
    if language == "both":
        return f"[EN] {en}{sep}[FR] {fr}"
    raise ValueError(f"language debe ser 'EN', 'FR' o 'both', recibido: {language!r}")


def _save_fig(fig, h5_path: str, filename: str) -> str:
    """Guarda una figura PNG en la carpeta plots de la ETAPA (un nivel arriba del
    run del DOE), no junto al doe_results.h5.

    Ej.: h5_path=".../0_Cinematique/DOE_Dexels_Cinematique/doe_results.h5"
         -> plots en ".../0_Cinematique/plots/" (no en ".../DOE_Dexels_Cinematique/plots/")
    """
    run_dir = os.path.dirname(os.path.abspath(h5_path))
    stage_dir = os.path.dirname(run_dir)
    out_dir = os.path.join(stage_dir, "plots")
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


def _load_wall_times(doe_run_dir: str) -> list:
    """Lee tiempo de calculo (wall_time_s.txt) y dxl_size (var_val.py) de cada
    subcarpeta numerica del run del DOE.

    IMPORTANTE: el emparejamiento dxl_size <-> wall_time se hace leyendo el
    `var_val.py` de la PROPIA subcarpeta numerica (`<N>/<modelo>/var_val.py`,
    el nombre de `<modelo>` no se asume fijo), NO por indice/orden de carpeta.
    El orden de las subcarpetas numericas (0, 1, 2, ...) NO coincide con el
    orden de los `case_XXX` del HDF5 ni con el orden ascendente de dxl_size
    (verificado: la carpeta 3 tiene dxl_size=1.6e-4 pero la carpeta 6 tiene
    dxl_size=1.28e-4, mas chico) -- por eso no se puede asumir esa correspondencia.

    Retorna una lista de dicts {"dxl_size": float, "wall_time_s": float, "folder": str}.
    """
    entries = []
    if not os.path.isdir(doe_run_dir):
        log.warning("Carpeta del run del DOE no encontrada: %s", doe_run_dir)
        return entries

    for name in sorted(os.listdir(doe_run_dir)):
        sub_dir = os.path.join(doe_run_dir, name)
        if not os.path.isdir(sub_dir) or not name.isdigit():
            continue

        wt_path = os.path.join(sub_dir, "wall_time_s.txt")
        if not os.path.isfile(wt_path):
            continue

        var_val_path = None
        for root, _dirs, files in os.walk(sub_dir):
            if "var_val.py" in files:
                var_val_path = os.path.join(root, "var_val.py")
                break
        if var_val_path is None:
            log.warning("[%s] var_val.py no encontrado -- omitido del costo computacional.", name)
            continue

        with open(wt_path, "r") as f:
            wall_time_s = float(f.read().strip())

        with open(var_val_path, "r") as f:
            text = f.read()
        var_val = ast.literal_eval(text.split("=", 1)[1].strip())
        dxl = float(var_val["$dxl_size$"])

        entries.append({"dxl_size": dxl, "wall_time_s": wall_time_s, "folder": name})

    return entries


def _zorders(cases: list) -> dict:
    """dxl_size mas alto -> zorder mas bajo (fondo). dxl_size mas bajo -> zorder mas alto (frente)."""
    valid = sorted({c["dxl_size"] for c in cases if np.isfinite(c["dxl_size"])}, reverse=True)
    rank = {v: i + 2 for i, v in enumerate(valid)}
    return {c["case_name"]: rank.get(c["dxl_size"], 10) for c in cases}


def fig3_error_summary(cases: list, F_ref: float, h5_path: str, highlight_dxl_size: float | None = None,
                        language: str = "both", figsize: tuple[float, float] = FIGSIZE_WIDE) -> None:
    """Figure 3: error summary by case using precomputed HDF5 datasets.

    language: "EN" | "FR" | "both" -- ver FIGURE_LANGUAGE en main().
    figsize: tamano de la figura -- ver FIGURE_SCALE/figsize_from_scale en main().
    """
    plt.rcParams.update(_STYLE)
    if not cases:
        log.warning("Figure 3: no cases available.")
        return
    cases_s = sorted(cases, key=lambda c: c["dxl_size"] if np.isfinite(c["dxl_size"]) else 1e99)
    dxl_labels, err_mean_bias_pct, err_peak_pct, err_range_pct, err_std_pct = [], [], [], [], []
    for c in cases_s:
        dxl_labels.append(f"{c['dxl_size']:.2e}" if np.isfinite(c["dxl_size"]) else "nan")
        if c["error_mean"] is None or c["error_maxmin"] is None or c["error_spread"] is None or c["error_std"] is None:
            raise KeyError(f"[{c['case_name']}] missing error datasets in HDF5")
        err_mean_bias_pct.append(float(np.asarray(c["error_mean"]).ravel()[0]))
        err_peak_pct.append(float(np.asarray(c["error_maxmin"]).ravel()[0]))
        err_range_pct.append(float(np.asarray(c["error_spread"]).ravel()[0]))
        err_std_pct.append(float(np.asarray(c["error_std"]).ravel()[0]))
    log.info(
        "Figure 3 control: first dxl_size=%s, err_mean_bias_pct=%.6f%%",
        dxl_labels[0],
        err_mean_bias_pct[0],
    )
    x = np.arange(len(dxl_labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.suptitle(_lang_text(
        "Force error vs. dexel discretization size",
        "Erreur de force en fonction de la taille de discrétisation (dexel)",
        language,
    ))
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

    ax.bar(x - 1.5*w, err_mean_bias_pct, w,
           label=_lang_text("Mean bias", "Biais moyen", language, sep=" / "),
           edgecolor=_bar_edgecolors(err_mean_bias_pct), linewidth=_bar_linewidth(err_mean_bias_pct),
           color=_bar_colors(err_mean_bias_pct))
    ax.bar(x - 0.5*w, err_peak_pct, w,
           label=_lang_text("Peak (max/min)", "Pic (max/min)", language, sep=" / "),
           edgecolor=_bar_edgecolors(err_peak_pct), linewidth=_bar_linewidth(err_peak_pct),
           color=["#f58518"] * len(err_peak_pct))
    ax.bar(x + 0.5*w, err_range_pct, w,
           label=_lang_text("Range (max−min)", "Étendue (max−min)", language, sep=" / "),
           edgecolor=_bar_edgecolors(err_range_pct), linewidth=_bar_linewidth(err_range_pct),
           color=["#54a24b"] * len(err_range_pct))
    ax.bar(x + 1.5*w, err_std_pct, w,
           label=_lang_text("Std. deviation", "Écart-type", language, sep=" / "),
           edgecolor=_bar_edgecolors(err_std_pct), linewidth=_bar_linewidth(err_std_pct),
           color=["#b279a2"] * len(err_std_pct))
    ax.set_xticks(x)
    tick_labels = ax.set_xticklabels(dxl_labels, rotation=45, ha="right")

    if highlight_index is not None:
        tick_labels[highlight_index].set_color("red")
    ax.set_xlabel(_lang_text(
        r"Dexel size $\Delta_{\mathrm{dxl}}$ [m]",
        r"Taille du dexel $\Delta_{\mathrm{dxl}}$ [m]",
        language,
    ))
    ax.set_ylabel(_lang_text("Error [%]", "Erreur [%]", language))
    _plain_yaxis(ax)
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1.0,
               label=_lang_text("10% threshold", "Seuil de 10 %", language, sep=" / "))
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
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
    parser.add_argument("--plots", action="store_true", default=True,
                        help="Show Figure 3 after the analysis")
    args = parser.parse_args()

    DOE_NAME = "0_Cinematique\\DOE_Dexels_Cinematique"   # nombre de la carpeta de salida  (dir_ref2exe)

    # Idioma del texto de las figuras (titulo, ejes, leyenda): "EN" | "FR" | "both"
    FIGURE_LANGUAGE = "FR"

    # Multiplicador simple del tamano de figura (1 = FIGSIZE_WIDE tal cual,
    # 1.5, 2, 0.5, etc.) -- mantiene siempre la proporcion ancho/alto.
    FIGURE_SCALE = 2.0

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
        fig3_error_summary(cases, F_ref, h5_path, highlight_dxl_size=selected_dxl_size,
                           language=FIGURE_LANGUAGE,
                           figsize=figsize_from_scale(FIGSIZE_WIDE, FIGURE_SCALE))


if __name__ == "__main__":
    main()
