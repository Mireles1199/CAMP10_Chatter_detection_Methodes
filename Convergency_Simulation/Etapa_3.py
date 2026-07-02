#!/usr/bin/env python3
# coding: utf-8
"""
Etapa_3.py — Estudio de sensibilidad a la discretizacion temporal (N)
Genera listas de N (discretizaciones por revolución) y de a_p para una malla
de epsilones alrededor de 1.0.
No realiza lecturas ni escrituras de HDF5 en esta versión inicial.
"""

import os
from typing import List
import argparse

import numpy as np
import h5py
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Generadores
# -------------------------------------------------------

def generate_N_list(N_base: int, factors: List[int]) -> List[int]:
    """Genera lista de N = N_base / f para cada factor f.
    Se asegura que cada N sea entero >= 1 y preserva orden de factors.
    """
    ns = []
    for f in factors:
        if f <= 0:
            raise ValueError("factors deben ser enteros positivos")
        n_val = max(1, N_base / f)
        ns.append(n_val)
    # Garantizar que N_base aparece (si factors no incluyen 1)
    if 1 not in factors:
        if N_base not in ns:
            ns.insert(0, N_base)
    return ns


def generate_ap_from_eps(eps_list: List[float], ap_crit: float) -> List[float]:
    return [float(eps) * float(ap_crit) for eps in eps_list]


# -------------------------------------------------------
# Lectura y plot de convergencia dt
# -------------------------------------------------------

def read_dt_convergence_data(folders: List[str]) -> List[dict]:
    """Lee nb_dt_rev, ap_crit_sim, ap_theo, ap_minus, ap_plus y percent_error de cada HDF5."""
    rows = []
    for folder in folders:
        h5_path = os.path.join(folder, "doe_results.h5")
        if not os.path.isfile(h5_path):
            print(f"[WARN] No encontrado: {h5_path}")
            continue
        with h5py.File(h5_path, "r") as h5f:
            ap_crit_sim   = float(h5f.attrs.get("stage_1_ap_crit_sim",   float("nan")))
            ap_crit_theo  = float(h5f.attrs.get("stage_1_ap_crit",       float("nan")))
            ap_minus      = float(h5f.attrs.get("stage_1_ap_minus",      float("nan")))
            ap_plus       = float(h5f.attrs.get("stage_1_ap_plus",       float("nan")))
            percent_error = float(h5f.attrs.get("stage_1_percent_error", float("nan")))
            # nb_dt_rev: leer del primer caso
            nb_dt_rev = float("nan")
            first = next((n for n in sorted(h5f.keys()) if n.startswith("case_")), None)
            if first:
                grp = h5f[first]
                nb_dt_rev = float(grp.attrs.get("$nb_dt_rev$", grp.attrs.get("nb_dt_rev", float("nan"))))
        rows.append({
            "folder":        folder,
            "nb_dt_rev":     nb_dt_rev,
            "ap_crit_sim":   ap_crit_sim,
            "ap_crit_theo":  ap_crit_theo,
            "ap_minus":      ap_minus,
            "ap_plus":       ap_plus,
            "percent_error": percent_error,
        })
    rows.sort(key=lambda r: r["nb_dt_rev"])
    return rows


def plot_dt_convergence(data: List[dict]) -> plt.Figure:
    """Figura de convergencia: ap_crit_sim vs nb_dt_rev con banda de transicion y error."""
    font_size = 15
    plt.rcParams.update({"font.size": font_size})

    nb   = np.asarray([r["nb_dt_rev"]    for r in data], dtype=float)
    ap_s = np.asarray([r["ap_crit_sim"]  for r in data], dtype=float) * 1e3
    ap_t = np.asarray([r["ap_crit_theo"] for r in data], dtype=float) * 1e3
    ap_m = np.asarray([r["ap_minus"]     for r in data], dtype=float) * 1e3
    ap_p = np.asarray([r["ap_plus"]      for r in data], dtype=float) * 1e3
    err  = np.asarray([r["percent_error"] for r in data], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Banda de transicion
    ax1.fill_between(nb, ap_m, ap_p, alpha=0.18, color="steelblue",
                     label=r"Transition band [$a_{p,-}$, $a_{p,+}$]")

    # ap_crit_sim con barras de error
    yerr_lo = np.where(np.isfinite(ap_s - ap_m), ap_s - ap_m, 0.0)
    yerr_hi = np.where(np.isfinite(ap_p - ap_s), ap_p - ap_s, 0.0)
    ax1.errorbar(nb, ap_s, yerr=[yerr_lo, yerr_hi],
                 fmt="o-", color="steelblue", capsize=4, lw=1.4, ms=5,
                 label=r"$a_{p,crit,sim}$")

    # Referencia teorica
    if np.any(np.isfinite(ap_t)):
        val_t = ap_t[np.isfinite(ap_t)][0]
        ax1.axhline(val_t, color="red", ls="--", lw=1.2,
                    label=rf"$a_{{p,crit,theo}} = {val_t:.3f}$ mm")

    ax1.set_xscale("log")
    ax1.set_xlabel(r"$N_{dt}$ (steps / rev)", fontsize=font_size * 1.125)
    ax1.set_ylabel(r"$a_{p,crit}$ [mm]", fontsize=font_size * 1.125)
    ax1.set_title(r"Convergence of $a_{p,crit,sim}$ vs $N_{dt}$", fontsize=font_size * 1.25)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.tick_params(labelsize=font_size)

    # Eje secundario: error porcentual
    ax2 = ax1.twinx()
    mask_e = np.isfinite(err)
    ax2.plot(nb[mask_e], err[mask_e], "s--", color="darkorange",
             ms=4, lw=1.0, label=r"% error $|\epsilon_{crit}-1|\times100$")
    ax2.set_ylabel(r"% error  $|\epsilon_{crit} - 1| \times 100$",
                   fontsize=font_size, color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=font_size * 0.875)

    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=font_size * 0.875, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig


# -------------------------------------------------------
# CLI y ejecución simple (sin I/O)
# -------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Etapa 3 — sensibilidad a la discretizacion N")
    # defaults will be provided by caller (main) — keep args optional
    p.add_argument("--n-base", type=int, help="N_base (discretizaciones por rev)")
    p.add_argument("--factors", type=int, nargs="*", help="Factores para dividir N_base")
    p.add_argument("--eps", type=float, nargs="*", help="Lista de epsilones")
    p.add_argument("--ap-crit", type=float, help="a_p critico teorico [m]")
    p.add_argument("--plots", action="store_true", help="Figura de convergencia ap_crit_sim vs N_dt")
    return p.parse_args()


def main():
    # Local defaults (no globals)
    DEFAULT_N_BASE = 200
    DEFAULT_FACTORS = [1, 2, 4, 8, 16]
    DEFAULT_EPSILONS = [
        0.90, 0.95, 0.98, 0.982, 0.984, 0.986, 0.988, 0.99,
        0.995, 1.00, 1.005, 1.01, 1.02, 1.05, 1.10,
    ]
    DEFAULT_AP_CRIT_THEO = 8.6052e-3

    args = parse_args()

    n_base = args.n_base if args.n_base is not None else DEFAULT_N_BASE
    factors = args.factors if args.factors is not None else DEFAULT_FACTORS
    eps_list = args.eps if args.eps is not None else DEFAULT_EPSILONS
    ap_crit = args.ap_crit if args.ap_crit is not None else DEFAULT_AP_CRIT_THEO

    N_values = generate_N_list(N_base=n_base, factors=factors)
    ap_values = generate_ap_from_eps(eps_list, ap_crit)

    print("Etapa 3 — sensibilidad a la discretizacion temporal")
    print(f"  N_base = {n_base}")
    print(f"  factors = {factors}")
    print(f"  N_values = {N_values}")
    print("\nTabla N (factores -> N values):")
    print("  idx | factor | N_value")
    print("  ----+--------+--------")
    for idx, (f, nval) in enumerate(zip(factors, N_values)):
        print(f"  {idx:>3d} | {f:>6.2g} | {nval:>6.2f}")

    print("\nEpsilons (malla):")
    print("  idx | eps    | ap [m]")
    print("  ----+--------+--------------------")
    for idx, (eps, ap) in enumerate(zip(eps_list, ap_values)):
        print(f"  {idx:>3d} | {eps:>6.6g} | {ap:>18.6e}")

    # ===========================================================================
    # CARPETAS DE CASOS  (una carpeta por N_dt, cada una con doe_results.h5)
    # ===========================================================================
    _base = os.path.dirname(os.path.abspath(__file__))
    convergence_folders = [
        os.path.join(_base, "3_Sensitivity_dt", "DOE_Detection_Limite_Lobes_dt_50"),
        os.path.join(_base, "3_Sensitivity_dt", "DOE_Detection_Limite_Lobes_dt_100"),
        os.path.join(_base, "3_Sensitivity_dt", "DOE_Detection_Limite_Lobes_dt_200"),
        os.path.join(_base, "3_Sensitivity_dt", "DOE_Detection_Limite_Lobes_dt_25"),
        os.path.join(_base, "3_Sensitivity_dt", "DOE_Detection_Limite_Lobes_dt_12.5"),
        # Agrega aqui las rutas absolutas de tus carpetas:
    ]

    if args.plots:
        conv_data = read_dt_convergence_data(convergence_folders)
        if not conv_data:
            print("[WARN] No se encontraron datos validos en las carpetas indicadas.")
        else:
            print(f"\nDatos de convergencia ({len(conv_data)} puntos):")
            print(f"  {'N_dt':>8}  {'ap_sim [mm]':>12}  {'ap_theo [mm]':>12}  {'% error':>8}")
            for r in conv_data:
                print(f"  {r['nb_dt_rev']:>8.0f}  {r['ap_crit_sim']*1e3:>12.5f}  {r['ap_crit_theo']*1e3:>12.5f}  {r['percent_error']:>8.3f}")
            plot_dt_convergence(conv_data)
            plt.show()


if __name__ == "__main__":
    main()
