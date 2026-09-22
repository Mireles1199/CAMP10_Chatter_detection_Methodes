#!/usr/bin/env python
# coding: utf-8
"""Etapa 2 - Analisis posterior del DOE.

Base inicial con las mismas constantes de la etapa 1.
La logica especifica de la etapa 2 se completara despues.
"""

import os
import argparse
from typing import List, Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt


def read_convergence_data(folders: List[str]) -> List[dict]:
    """Lee dxl_size, ap_crit_sim/theo, ap_minus/plus, percent_error y lambda_* de cada HDF5."""
    rows = []
    for folder in folders:
        h5_path = os.path.join(folder, "doe_results.h5")
        if not os.path.isfile(h5_path):
            print(f"[WARN] No encontrado: {h5_path}")
            continue
        with h5py.File(h5_path, "r") as h5f:
            dxl_size        = float(h5f.attrs.get("stage_1_dxl_size",       float("nan")))
            ap_crit_sim     = float(h5f.attrs.get("stage_1_ap_crit_sim",    float("nan")))
            ap_crit_theo    = float(h5f.attrs.get("stage_1_ap_crit",        float("nan")))
            ap_minus        = float(h5f.attrs.get("stage_1_ap_minus",       float("nan")))
            ap_plus         = float(h5f.attrs.get("stage_1_ap_plus",        float("nan")))
            percent_error   = float(h5f.attrs.get("stage_1_percent_error",  float("nan")))
            lambda_crit_sim = float(h5f.attrs.get("stage_1_lambda_crit_sim", float("nan")))
            lambda_minus    = float(h5f.attrs.get("stage_1_lambda_minus",   float("nan")))
            lambda_plus     = float(h5f.attrs.get("stage_1_lambda_plus",    float("nan")))
        rows.append({
            "folder":          folder,
            "dxl_size":        dxl_size,
            "ap_crit_sim":     ap_crit_sim,
            "ap_crit_theo":    ap_crit_theo,
            "ap_minus":        ap_minus,
            "ap_plus":         ap_plus,
            "percent_error":   percent_error,
            "lambda_crit_sim": lambda_crit_sim,
            "lambda_minus":    lambda_minus,
            "lambda_plus":     lambda_plus,
        })
    rows.sort(key=lambda r: r["dxl_size"])
    return rows




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


def plot_epsilon_convergence(data: List[dict], language: str = "both",
                              outlier_eps_tol: float = 0.15) -> plt.Figure:
    """Influencia del tamano de dexel en el limite detectado (epsilon) vs. el teorico.

    Eje X: dxl_size [mm] (log). Eje Y: epsilon = a_p / a_p,crit,theo. Cada dexel se
    grafica como epsilon_crit_sim (centro de la biseccion) con barra de error
    [lambda_minus, lambda_plus]. Esa banda es el criterio de parada de la biseccion
    (~0.01 de epsilon, fijo en todas las corridas) -- no mide precision que varie
    con el dexel; sirve como piso de resolucion del metodo de deteccion, para poder
    distinguir un desvio real del centro (mayor que la banda, atribuible al dexel)
    de ruido propio del criterio de deteccion (dentro de la banda).

    Eje Y partido en dos paneles (broken axis): los dexels con |epsilon_crit_sim - 1|
    > outlier_eps_tol (dexels muy gruesos donde la deteccion se degrada del todo) van
    en el panel superior; el resto -- donde esta la convergencia fina que interesa --
    en el panel inferior, con su propio zoom. Sin esto un solo outlier aplana la
    escala y no deja distinguir los puntos finos entre si.
    """
    plt.rcParams.update(ARTICLE_RCPARAMS)

    dxl       = np.asarray([r["dxl_size"]        for r in data], dtype=float) * 1e3  # -> mm
    lam_crit  = np.asarray([r["lambda_crit_sim"] for r in data], dtype=float)
    lam_minus = np.asarray([r["lambda_minus"]    for r in data], dtype=float)
    lam_plus  = np.asarray([r["lambda_plus"]     for r in data], dtype=float)

    yerr_lo = np.where(np.isfinite(lam_crit - lam_minus), lam_crit - lam_minus, 0.0)
    yerr_hi = np.where(np.isfinite(lam_plus - lam_crit), lam_plus - lam_crit, 0.0)

    is_outlier = np.isfinite(lam_crit) & (np.abs(lam_crit - 1.0) > outlier_eps_tol)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=figsize_from_scale(FIGSIZE_SIMPLE, FIGSCALE_SIMPLE),
        gridspec_kw={"height_ratios": [1.4, 2.2], "hspace": 0.08},
        constrained_layout=True,
    )

    theo_label = _lang_text(r"Theoretical limit $\eta=1$",
                             r"Limite théorique $\eta=1$", language, sep=" / ")
    band_label = _lang_text(r"Detection resolution [$\lambda_-,\lambda_+$]",
                             r"Résolution de détection [$\lambda_-,\lambda_+$]",
                             language, sep=" / ")
    sim_label = _lang_text(r"Simulated $\eta_{crit,sim}$",
                            r"$\eta_{crit,sim}$ simulé",
                            language, sep=" / ")

    def _band(ax, mask):
        """Une los limites superior/inferior de los puntos de `mask` (>=2 puntos) y
        colorea la zona entre ambas curvas -- envolvente continua de [lambda_minus,
        lambda_plus], complementaria a la barra de error de cada punto individual."""
        order = np.argsort(dxl[mask])
        xs, los, his = dxl[mask][order], lam_minus[mask][order], lam_plus[mask][order]
        valid = np.isfinite(los) & np.isfinite(his)
        if np.count_nonzero(valid) < 2:
            return
        ax.fill_between(xs[valid], los[valid], his[valid],
                         color="steelblue", alpha=0.35, zorder=0, label=band_label)

    _band(ax_top, is_outlier)
    _band(ax_bot, ~is_outlier & np.isfinite(lam_crit))

    for ax in (ax_top, ax_bot):
        ax.axhline(1.0, color="crimson", linewidth=1.2, linestyle="--",
                   zorder=1, label=theo_label)
        # Barra de error (linea vertical + punto) en azul, sin los capsize por
        # defecto -- los topes se dibujan aparte abajo, naranja arriba / verde abajo.
        ax.errorbar(dxl, lam_crit, yerr=[yerr_lo, yerr_hi],
                    fmt="o", color="steelblue", ecolor="steelblue",
                    capsize=0, elinewidth=1.2, ms=7,
                    zorder=2, label=sim_label)
        # Tope superior (lambda_plus) en naranja, tope inferior (lambda_minus) en verde.
        ax.scatter(dxl, lam_plus, marker="_", s=90, linewidths=1.6,
                   color="darkorange", zorder=3)
        ax.scatter(dxl, lam_minus, marker="_", s=90, linewidths=1.6,
                   color="green", zorder=3)
        ax.set_xscale("log")

    # Rango del panel superior: solo los outliers (con margen).
    if np.any(is_outlier):
        out_vals = np.concatenate([lam_minus[is_outlier], lam_plus[is_outlier], lam_crit[is_outlier]])
        out_vals = out_vals[np.isfinite(out_vals)]
        pad = 0.08 * (np.max(out_vals) - np.min(out_vals) + 1e-9)
        ax_top.set_ylim(np.min(out_vals) - pad, np.max(out_vals) + pad)

    # Rango del panel inferior: el resto (zoom), siempre incluye epsilon=1.
    main_mask = ~is_outlier & np.isfinite(lam_crit)
    main_vals = np.concatenate([lam_minus[main_mask], lam_plus[main_mask], lam_crit[main_mask], [1.0]])
    main_vals = main_vals[np.isfinite(main_vals)]
    pad = 0.25 * (np.max(main_vals) - np.min(main_vals) + 1e-9)
    ax_bot.set_ylim(np.min(main_vals) - pad, np.max(main_vals) + pad)

    # Nota corta sobre la linea epsilon=1: a que valor absoluto de a_p corresponde.
    # "theo" como superindice (a_{p,crit}^{theo}) en vez de metido en el subindice.
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

    # Definicion de lambda_-/lambda_+: lambda_- es el epsilon del ultimo caso
    # simulado estable, lambda_+ el del primer caso simulado inestable. Va arriba
    # del todo en ax_top (debajo del titulo, encima de la leyenda) para no chocar
    # con ninguno de los dos.
    lambda_def_text = _lang_text(
        r"$\lambda_-$: last stable $\eta$  /  $\lambda_+$: first unstable $\eta$",
        r"$\lambda_-$ : dernier $\eta$ stable  /  $\lambda_+$ : premier $\eta$ instable",
        language, sep="\n")
    ax_top.text(
        0.02, 0.98, lambda_def_text,
        transform=ax_top.transAxes, color="steelblue",
        fontsize=plt.rcParams["legend.fontsize"] * 0.85, va="top", ha="left",
    )

    # Corte visual entre paneles (broken axis).
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

    # Eje X: escala log estandar (potencias de 10), sin tocar sus ticks/labels.
    # Los 9 tamanos de dexel reales se marcan DENTRO de cada panel (no como ticks
    # del eje, para no chocar con las etiquetas 10^-2/10^-1/10^0 de abajo):
    # ax_bot tiene margen vacio debajo de los datos (por el padding del ylim) --
    # ahi va una marquita + el valor, horizontal, chico y en negro. ax_top solo
    # tiene el outlier, asi que se anota directo al lado del punto.
    # Formato "Xe-5": mismo convenio que el nombre de carpeta (dxl en metros x1e5).
    def _e5_label(dxl_mm: float) -> str:
        return f"{dxl_mm * 100:.4g}e-5"

    main_mask_labels = ~is_outlier & np.isfinite(lam_crit)
    y_tick_frac = 0.05
    for x in dxl[main_mask_labels]:
        ax_bot.plot([x, x], [0.0, y_tick_frac], transform=ax_bot.get_xaxis_transform(),
                    color="black", linewidth=0.8, zorder=4)
        ax_bot.text(x, y_tick_frac + 0.015, _e5_label(x),
                    transform=ax_bot.get_xaxis_transform(),
                    rotation=45, va="bottom", ha="left", color="black",
                    fontsize=plt.rcParams["xtick.labelsize"] * 0.6, zorder=4)


    ax_bot.set_xlabel(_lang_text("Dexel size [mm]", "Taille du dexel [mm]", language))
    fig.supylabel(r"$\eta$")
    ax_top.set_title(_lang_text(
        "Detected limit vs. dexel size",
        "Limite détectée vs. taille du dexel",
        language))

    # Leyenda combinada (la banda solo tiene handle en ax_bot cuando ax_top tiene
    # un unico outlier y _band no dibuja nada ahi) -- se muestra en ax_top, que
    # tiene espacio vacio a la izquierda sin tapar datos.
    bot_handles, bot_labels = ax_bot.get_legend_handles_labels()
    top_handles, top_labels = ax_top.get_legend_handles_labels()
    label_to_handle = dict(zip(bot_labels, bot_handles))
    label_to_handle.update(dict(zip(top_labels, top_handles)))
    ordered_labels = [l for l in (theo_label, band_label, sim_label) if l in label_to_handle]
    ax_top.legend([label_to_handle[l] for l in ordered_labels], ordered_labels, loc="lower left")

    return fig


def format_dexel_table(base_dexel_m: float, factors: List[int]) -> List[dict]:
    """Genera una tabla de tamaños de dexel alrededor de un valor base.

    Para cada factor entero f se calcula:
    - dexel_base / f
    - dexel_base * f
    """
    table_rows = []
    for factor in factors:
        if factor <= 0:
            raise ValueError("Los factores de dexel deben ser enteros positivos")
        table_rows.append({
            "factor": factor,
            "dexel_down_m": base_dexel_m / factor,
            "dexel_up_m": base_dexel_m * factor,
        })
    return table_rows


def dexel_folder_name(dxl_m: float, suffix: str = "_RUN_10") -> str:
    """Nombre de carpeta DOE para un tamaño de dexel dado, ej. 20e-5 -> 'DOE_Detection_Limite_Lobes_dxl_20e-5_RUN_10'."""
    label = format(dxl_m * 1e5, "g")
    return f"DOE_Detection_Limite_Lobes_dxl_{label}e-5{suffix}"


def build_convergence_folders(
    cases_dir: str,
    base_dexel_m: float,
    factors: List[float],
    suffix: str,
) -> List[str]:
    """Construye las rutas de las carpetas DOE dentro de cases_dir a partir de dexel_base_m y los factores del barrido."""
    return [
        os.path.join(cases_dir, dexel_folder_name(base_dexel_m * factor, suffix))
        for factor in factors
    ]

# ==============================================================================
# ESTILO DE FIGURA (ver skill article-plot-style) Y RUTA COMPARTIDA DE SALIDA
# ==============================================================================

# Carpeta compartida donde se guardan las figuras de todas las etapas (mismo
# convenio que Etapa_1.py); el .tex la referencia via \graphicspath{{../plots/}}.
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Latex", "plots")
FIG_PREFIX = "etapa2_"

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


def main() -> None:
    """Punto de entrada de la etapa 2."""
    parser = argparse.ArgumentParser(description="Etapa 2 - Analisis posterior del DOE")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime, no escribe en el HDF5")
    parser.add_argument("--h5", default=None, help="Ruta al doe_results.h5")
    parser.add_argument("--plots", action="store_true", help="Figura de convergencia ap_crit_sim vs dxl_size")
    args = parser.parse_args()

    doe_name = "1_Detection_Limite_Lobes\\DOE_Detection_Limite_Lobes"

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
    t_start = 0.05            # inicio de ventana para max/min [s]
    t_end   = -1.0            # fin de ventana para max/min [s] (-1 = ultimo tiempo)

    ap_crit = 8.6052e-3
    n0 = 12_099.28
    dexel_base_m = 20e-5
    dexel_factors = [1/16., 1/8., 1/4., 1/2., 1, 2, 4, 8, 16]

    # Idioma del texto de la figura de convergencia: "EN" | "FR" | "both" (ver skill
    # article-plot-style). El .tex de la tesis esta en frances -- cambiar a "FR" para
    # la version final.
    FIGURE_LANGUAGE = "FR"

    # ===========================================================================
    # CARPETAS DE CASOS DE CONVERGENCIA  (editar aqui antes de ejecutar --plots)
    # cases_dir  : carpeta que contiene TODAS las carpetas de caso (una por tamaño de dexel)
    # run_suffix : sufijo comun a esas carpetas, ej. "_RUN_10"
    # Los datos pesados viven en el checkout madre (no en este worktree), por eso
    # cases_dir apunta ahi en vez de ser relativo a este script.
    # ===========================================================================
    cases_dir = (
        r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
        r"\CAMP10_Chatter_detection_Methodes\Convergency_Simulation\1_Detection_Limite_Lobes"
    )
    run_suffix = "_RUN_10"

    h5_path = args.h5 if args.h5 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), doe_name, "doe_results.h5"
    )

    print("\nEtapa 2 - Dexels Senbility")
    print(f"  h5_path    = {h5_path}\n")
    print(f"  ap_crit    = {ap_crit:.6e} m")
    print(f"  n0         = {n0:.2f} rpm")

    print("\nTabla de dexel base:")
    print(f"  dexel_base = {dexel_base_m:.6e} m")
    print("  idx | factor | dexel_base / factor [m] | dexel_base * factor [m]")
    print("  ----+--------+-------------------------+------------------------")
    for idx, row in enumerate(format_dexel_table(dexel_base_m, dexel_factors), start=0):
        print(
            f"  {idx:>3d} | {row['factor']:>6.2f} | {row['dexel_down_m']:>23.6e} | {row['dexel_up_m']:>22.6e}"
        )

    if args.dry_run:
        print("[dry-run] No se escribe nada aun.")
        return

    # Carpetas de caso (una por tamaño de dexel, cada una con doe_results.h5), generadas
    # a partir de cases_dir / run_suffix y de dexel_base_m / dexel_factors (arriba).
    convergence_folders = build_convergence_folders(cases_dir, dexel_base_m, dexel_factors, run_suffix)

    if args.plots:
        conv_data = read_convergence_data(convergence_folders)
        if not conv_data:
            print("[WARN] No se encontraron datos validos en las carpetas indicadas.")
        else:
            print(f"\nDatos de convergencia ({len(conv_data)} puntos):")
            print(f"  {'dxl [mm]':>12}  {'eps_crit_sim':>12}  {'eps_minus':>10}  {'eps_plus':>10}")
            for r in conv_data:
                print(f"  {r['dxl_size']*1e3:>12.4e}  {r['lambda_crit_sim']:>12.5f}  {r['lambda_minus']:>10.5f}  {r['lambda_plus']:>10.5f}")
            fig = plot_epsilon_convergence(conv_data, language=FIGURE_LANGUAGE)
            if PLOT_SHOW:
                plt.show()
            out_path = _save_fig(fig, "convergence_epsilon_vs_dxl.png")
            print(f"[OK] Figura guardada en: {out_path}")
        return


if __name__ == '__main__':
    main()
