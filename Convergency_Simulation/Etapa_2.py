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
    """Lee dxl_size, ap_crit_sim, ap_theo, ap_minus, ap_plus y percent_error de cada HDF5."""
    rows = []
    for folder in folders:
        h5_path = os.path.join(folder, "doe_results.h5")
        if not os.path.isfile(h5_path):
            print(f"[WARN] No encontrado: {h5_path}")
            continue
        with h5py.File(h5_path, "r") as h5f:
            dxl_size      = float(h5f.attrs.get("stage_1_dxl_size",      float("nan")))
            ap_crit_sim   = float(h5f.attrs.get("stage_1_ap_crit_sim",   float("nan")))
            ap_crit_theo  = float(h5f.attrs.get("stage_1_ap_crit",       float("nan")))
            ap_minus      = float(h5f.attrs.get("stage_1_ap_minus",      float("nan")))
            ap_plus       = float(h5f.attrs.get("stage_1_ap_plus",       float("nan")))
            percent_error = float(h5f.attrs.get("stage_1_percent_error", float("nan")))
            lambda_crit   = float(h5f.attrs.get("stage_1_lambda_crit_sim", float("nan")))
        rows.append({
            "folder":        folder,
            "dxl_size":      dxl_size,
            "ap_crit_sim":   ap_crit_sim,
            "ap_crit_theo":  ap_crit_theo,
            "ap_minus":      ap_minus,
            "ap_plus":       ap_plus,
            "percent_error": percent_error,
            "lambda_crit":   lambda_crit,
        })
    rows.sort(key=lambda r: r["dxl_size"])
    return rows


def plot_dxl_convergence(data: List[dict]) -> plt.Figure:
    """Figura de convergencia: ap_crit_sim vs dxl_size con banda de transicion y error."""
    font_size = 15
    plt.rcParams.update({"font.size": font_size})
    dxl   = np.asarray([r["dxl_size"]     for r in data], dtype=float)
    ap_s  = np.asarray([r["ap_crit_sim"]  for r in data], dtype=float) * 1e3   # -> mm
    ap_t  = np.asarray([r["ap_crit_theo"] for r in data], dtype=float) * 1e3
    ap_m  = np.asarray([r["ap_minus"]     for r in data], dtype=float) * 1e3
    ap_p  = np.asarray([r["ap_plus"]      for r in data], dtype=float) * 1e3
    err   = np.asarray([r["percent_error"] for r in data], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Banda de transicion
    ax1.fill_between(dxl * 1e3, ap_m, ap_p, alpha=0.18, color="steelblue", label="Transition band [ap_minus, ap_plus]")

    # ap_crit_sim con barras de error
    yerr_lo = np.where(np.isfinite(ap_s - ap_m), ap_s - ap_m, 0.0)
    yerr_hi = np.where(np.isfinite(ap_p - ap_s), ap_p - ap_s, 0.0)
    ax1.errorbar(dxl * 1e3, ap_s, yerr=[yerr_lo, yerr_hi],
                 fmt="o-", color="steelblue", capsize=4, lw=1.4, ms=5,
                 label=r"$a_{p,crit,sim}$")

    # Referencia teorica
    if np.any(np.isfinite(ap_t)):
        ax1.axhline(ap_t[np.isfinite(ap_t)][0], color="red", ls="--", lw=1.2,
                    label=rf"$a_{{p,crit,theo}} = {ap_t[np.isfinite(ap_t)][0]:.3f}$ mm")

    ax1.set_xscale("log")
    ax1.set_xlabel("dxl size [mm]", fontsize=font_size * 1.125)
    ax1.set_ylabel(r"$a_{p,crit}$ [mm]", fontsize=font_size * 1.125)
    ax1.set_title("Convergence of $a_{p,crit,sim}$ vs dexel size", fontsize=font_size * 1.25)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.tick_params(labelsize=font_size)

    # Eje secundario: error porcentual
    ax2 = ax1.twinx()
    mask_e = np.isfinite(err)
    ax2.plot(dxl[mask_e] * 1e3, err[mask_e], "s--", color="darkorange",
             ms=4, lw=1.0, label=r"% error vs $\epsilon=1$")
    ax2.set_ylabel(r"% error  $|\epsilon_{crit} - 1| \times 100$", fontsize=font_size, color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=font_size * 0.875)

    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=font_size * 0.875, loc="upper left", framealpha=0.9)

    fig.tight_layout()
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

    # ===========================================================================
    # CARPETAS DE CASOS  (una carpeta por tamaño de dexel, cada una con doe_results.h5)
    # ===========================================================================
    _base = os.path.dirname(os.path.abspath(__file__))
    convergence_folders = [
        os.path.join(_base, "1_Detection_Limite_Lobes", "DOE_Detection_Limite_Lobes_dxl_5e-5"),
        os.path.join(_base, "1_Detection_Limite_Lobes", "DOE_Detection_Limite_Lobes_dxl_10e-5"),
        os.path.join(_base, "1_Detection_Limite_Lobes", "DOE_Detection_Limite_Lobes_dxl_20e-5"),
        os.path.join(_base, "1_Detection_Limite_Lobes", "DOE_Detection_Limite_Lobes_dxl_15e-5"),
        os.path.join(_base, "1_Detection_Limite_Lobes", "DOE_Detection_Limite_Lobes_dxl_25e-5"),
        # Agrega aqui las rutas absolutas de tus carpetas:
    ]

    if args.plots:
        conv_data = read_convergence_data(convergence_folders)
        if not conv_data:
            print("[WARN] No se encontraron datos validos en las carpetas indicadas.")
        else:
            print(f"\nDatos de convergencia ({len(conv_data)} puntos):")
            print(f"  {'dxl [mm]':>12}  {'ap_sim [mm]':>12}  {'ap_theo [mm]':>12}  {'% error':>8}")
            for r in conv_data:
                print(f"  {r['dxl_size']*1e3:>12.4e}  {r['ap_crit_sim']*1e3:>12.5f}  {r['ap_crit_theo']*1e3:>12.5f}  {r['percent_error']:>8.3f}")
            plot_dxl_convergence(conv_data)
            plt.show()
        return


if __name__ == '__main__':
    main()
