#!/usr/bin/env python
# coding: utf-8
"""
doe_unified_selector.py — Selector interactivo unificado para todos los HDF5 DOE.
===================================================================================
Soporta 5 formatos de archivos .h5:

  1. doe_results.h5               — señales (Axial_disp, Axial_vel) por caso DOE
  2. doe_noise_results.h5         — señales degradadas por nivel de ruido SNR
  3. doe_indicator_results.h5     — indicadores + señales por caso DOE
  4. doe_noise_indicator_results.h5 — indicadores por nivel de ruido
  5. doe_model_snr_results.h5     — SNR_mod_dB por señal y caso DOE

Layout de la ventana:
  [Tabla (izq)] | [Señales / I_t (centro)] | [Summary plots (dcha)]

Uso:
    python doe_unified_selector.py
    python doe_unified_selector.py --h5 PATH/doe_indicator_results.h5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# ── Backend ANTES de cualquier import de pyplot ──────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")

import h5py
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ── Import de plotters existentes ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# doe_plotter convergence functions (return Figure)
from doe_plotter import (
    configurar_estilo_global as _cfg_estilo,
    SIGNAL_YLABELS,
    color_azul,
    color_orange,
    color_verde,
    color_red,
    plot_convergence,
    plot_convergence_error_ref,
    plot_convergence_error_consec,
    plot_convergence_time,
)

# doe_indicator_plotter functions (some create plt figures, don't return them)
from doe_indicator_plotter import (
    load_indicator_results,
    _plot_td_single,
    _plot_td_per_run,
    plot_It_overlay,
    _resolve_label_key,
    _x_values as _ind_x_values,
    _RUN_COLORS,
    DECIMATE as _IND_DECIMATE,
    _sanitize,
)

# doe_model_snr_plotter functions
from doe_model_snr_plotter import (
    load_snr_results,
    plot_snr_vs_param,
    _detect_param_key as _snr_detect_param_key,
)

# funciones de doe_noise_plotter (para el panel de resumen de TYPE_NOISE_IND)
from doe_noise_plotter import (
    gather_detection_rows        as _noise_gather_df,
    plot_td_for_indicator        as _noise_plot_td_ind,
    plot_td_lollipop             as _noise_plot_lollipop,
    plot_delay_vs_snr            as _noise_plot_delay,
    gather_indicator_curves      as _noise_gather_curves,
    plot_it_overlay              as _noise_plot_it_overlay,
)

# estilo article-plot-style (skill) -- fuente unica de verdad para figuras de exportacion
import plot_style
try:
    import sld_model  # pyright: ignore[reportMissingImports]  # lóbulos de estabilidad (necesita sld_tools); sin él el visor funciona igual
except Exception:
    sld_model = None

_cfg_estilo()

# ── Constantes de tipo de formato ──────────────────────────────────────────────────────
TYPE_DOE_RESULTS   = "doe_results"
TYPE_DOE_NOISE     = "doe_noise"
TYPE_DOE_INDICATOR = "doe_indicator"
TYPE_NOISE_IND     = "doe_noise_ind"
TYPE_MODEL_SNR     = "doe_model_snr"
# reference_dataset.py (Fase 1/2A): stable/unstable, no "case_*" -- formatos aparte,
# con un visualizador propio (ReferenceViewerApp), no encajan en el resto de esta app.
TYPE_REFERENCE_DATASET  = "reference_dataset"
TYPE_REFERENCE_COMBINED = "reference_combined"

_TYPE_LABELS = {
    TYPE_DOE_RESULTS  : "DOE Results  (signals)",
    TYPE_DOE_NOISE    : "DOE Noise  (signals + SNR)",
    TYPE_DOE_INDICATOR: "DOE Indicator Results",
    TYPE_NOISE_IND    : "DOE Noise Indicators",
    TYPE_MODEL_SNR    : "DOE Model SNR",
    TYPE_REFERENCE_DATASET : "Reference Dataset  (segments per case)",
    TYPE_REFERENCE_COMBINED: "Reference Combined  (signal per label+channel)",
}

DECIMATE = 1   # decimación para plots de señales en panel central

# ── Nombre bonito + unidades por canal, para ReferenceViewerApp ─────────────────
# Base: SIGNAL_YLABELS (ya definido en doe_plotter.py) para Axial_disp/Axial_vel,
# extendido acá con los demás canales que reference_dataset.py puede autodetectar
# (Axial_acc, res_R_p, los *_out_deflex de Out_Deflex) sin tocar doe_plotter.py.
_REFERENCE_CHANNEL_INFO = {
    "Axial_disp": ("Axial Displacement", "m"),
    "Axial_vel":  ("Axial Velocity", "m/s"),
    "Axial_acc":  ("Axial Acceleration", "m/s²"),
    "res_R_p":    ("Resultant Force", "N"),
    "Axial_disp_out_deflex": ("Axial Displacement (deflection-corrected)", "m"),
    "Axial_vel_out_deflex":  ("Axial Velocity (deflection-corrected)", "m/s"),
}


def _channel_title(channel: str) -> str:
    """Nombre legible del canal (sin unidades), para usar como título de plot."""
    return _REFERENCE_CHANNEL_INFO.get(channel, (channel, ""))[0]


def _channel_ylabel(channel: str) -> str:
    """'Nombre [unidad]' del canal, para ejes -- ej. 'Axial Velocity [m/s]'."""
    name, unit = _REFERENCE_CHANNEL_INFO.get(channel, (channel, ""))
    return f"{name} [{unit}]" if unit else name


# ==============================================================================
# DETECCION DE FORMATO
# ==============================================================================

def detect_h5_type(h5_path: str) -> str:
    """Auto-detecta el tipo de formato de un HDF5 del DOE."""
    with h5py.File(h5_path, "r") as f:
        groups = list(f.keys())
        if not groups:
            return TYPE_DOE_RESULTS

        # reference_dataset.py: to_hdf5() -> stable/unstable/gray anidado (case/pieza);
        # save_combined() -> stable__<canal>/unstable__<canal>/gray__<canal> plano, con t/y directo.
        # (gray se reconoce para que el archivo abra; los visores solo dibujan stable/unstable)
        if groups and all(g in ("stable", "unstable", "gray") for g in groups):
            return TYPE_REFERENCE_DATASET
        if groups and all(g.startswith(("stable__", "unstable__", "gray__")) for g in groups):
            return TYPE_REFERENCE_COMBINED

        has_case_groups = any(g.startswith("case_") for g in groups)
        has_snr_groups  = any(g.startswith("snr_") or g == "control" for g in groups)

        first_grp   = f[groups[0]]
        first_attrs = dict(first_grp.attrs)

        # doe_model_snr: case_* con atributos SNR_mod_dB_*
        has_snr_attrs = any(k.startswith("SNR_mod_dB_") for k in first_attrs)
        if has_case_groups and has_snr_attrs:
            return TYPE_MODEL_SNR

        # Buscar subgrupos con t + I_t (indicadores) en el primer grupo
        has_run_subgroups = False
        if isinstance(first_grp, h5py.Group):
            for sub_name in list(first_grp.keys())[:8]:
                if sub_name in {"Axial_disp", "Axial_vel"}:
                    continue
                sub = first_grp[sub_name]
                if isinstance(sub, h5py.Group) and "t" in sub and "I_t" in sub:
                    has_run_subgroups = True
                    break

        # doe_indicator: case_* + subgrupos run
        if has_case_groups and has_run_subgroups:
            return TYPE_DOE_INDICATOR

        # doe_noise_ind: snr_*/control + subgrupos run
        if has_snr_groups and has_run_subgroups:
            return TYPE_NOISE_IND

        # doe_noise: snr_*/control sólo con señales
        if has_snr_groups:
            return TYPE_DOE_NOISE

        # Por defecto: doe_results (case_* con señales directas)
        return TYPE_DOE_RESULTS


# ==============================================================================
# LOADERS NORMALIZADOS
# ==============================================================================
# Estructura normalizada de cada dict de caso:
#   group     : str        — nombre del grupo HDF5
#   label_key : str        — clave DOE principal (eje X tabla)
#   label_val : float      — valor numérico del eje X
#   var_val   : dict       — todas las variables DOE del caso
#   signals   : dict       — {"Axial_disp": (t, y), "Axial_vel": (t, y)} o {}
#   forces    : dict       — {"res_R_p": (t, y)} o {}
#   runs      : dict       — {run_name: {t, I_t, t_d, attrs}} o {}
#   snr       : dict       — {"Axial_disp": float, ...} o {}
#   dt_us     : float|None — delta t en µs (solo para $nb_dt_rev$)
#   wall_time_s: float|None
# ==============================================================================

_SIGNAL_NAMES = {"Axial_disp", "Axial_vel", "Axial_acc"}   # acc solo si el archivo la trae
_FORCE_NAMES = {"res_R_p"}
_OUT_DEFLEX_GROUP = "Out_Deflex"
_OUT_DEFLEX_NAMES = {"Axial_disp_out_deflex", "Axial_vel_out_deflex"}

_LINE_TARGET_ALL = "Todas (tab activo)"
_LINE_TARGETS = [_LINE_TARGET_ALL, "Señales: disp", "Señales: vel", "Señales: acc",
                 "Fuerzas: F1", "Fuerzas: F2", "Fuerzas: F3",
                 "Deflex: disp", "Deflex: vel", "I_t"]
_LINE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                "#46f0f0", "#f032e6", "#9a6324", "#000075", "#808000"]


def _read_signals(grp: h5py.Group) -> Dict[str, Any]:
    """Lee señales de tiempo de un grupo HDF5 → {name: (t, y)}."""
    signals = {}
    for sig in _SIGNAL_NAMES:
        if sig in grp:
            sg = grp[sig]
            if isinstance(sg, h5py.Group) and "time" in sg and "values" in sg:
                signals[sig] = (sg["time"][()], sg["values"][()])
    return signals


def _read_forces(grp: h5py.Group) -> Dict[str, Any]:
    """Lee fuerzas de tiempo de un grupo HDF5 → {name: (t, y)}."""
    forces = {}
    for sig in _FORCE_NAMES:
        if sig in grp:
            sg = grp[sig]
            if isinstance(sg, h5py.Group) and "time" in sg and "values" in sg:
                forces[sig] = (sg["time"][()], sg["values"][()])
    return forces


def _read_out_deflex(grp: h5py.Group) -> Dict[str, Any]:
    """Lee señales de Out_Deflex de un grupo HDF5 → {name: (t, y)} o {}."""
    out = {}
    if _OUT_DEFLEX_GROUP not in grp:
        return out
    od_grp = grp[_OUT_DEFLEX_GROUP]
    if not isinstance(od_grp, h5py.Group):
        return out
    for sig in _OUT_DEFLEX_NAMES:
        if sig in od_grp:
            sg = od_grp[sig]
            if isinstance(sg, h5py.Group) and "time" in sg and "values" in sg:
                out[sig] = (sg["time"][()], sg["values"][()])
    return out


def _read_runs(grp: h5py.Group) -> Dict[str, Any]:
    """Lee subgrupos de run (indicadores) de un grupo HDF5."""
    runs = {}
    for rname in grp.keys():
        if rname in _SIGNAL_NAMES:
            continue
        rgrp = grp[rname]
        if not isinstance(rgrp, h5py.Group):
            continue
        if "t" not in rgrp or "I_t" not in rgrp:
            continue
        runs[rname] = {
            "t":          rgrp["t"][()]          if "t"          in rgrp else np.array([]),
            "I_t":        rgrp["I_t"][()]        if "I_t"        in rgrp else np.array([]),
            "t_d":        rgrp["t_d"][()]        if "t_d"        in rgrp else np.array([]),
            "attrs":      dict(rgrp.attrs),
        }
    return runs


def _auto_label_key(var_val: dict) -> str:
    """Auto-detecta la clave DOE más informativa (la primera disponible)."""
    if var_val:
        return next(iter(var_val))
    return "case_idx"


def _best_label_key(cases: List[Dict]) -> str:
    """Elige la clave DOE con mayor variación *relativa* entre todos los casos
    (coeficiente de variación = spread / |media|).  Esto evita que variables
    con valores absolutos grandes (spin_rate ~12000) ganen sobre variables que
    cambian mucho en términos relativos (nb_dt_rev 1→8).  Si la media es 0 se
    usa el spread absoluto como fallback para esa clave."""
    all_keys: set = set()
    for c in cases:
        all_keys.update(c.get("var_val", {}).keys())

    best_key, best_score = "case_idx", -1.0
    kap = [c["var_val"].get("kappa") for c in cases]
    try:   # kappa first: it is what the labels and the SLD speak about
        if len({round(float(k), 9) for k in kap if k is not None}) > 1:
            return "kappa"
    except (TypeError, ValueError):
        pass
    for k in sorted(all_keys):
        vals = []
        for c in cases:
            try:
                vals.append(float(c["var_val"].get(k, float("nan"))))
            except (TypeError, ValueError):
                pass
        valid = [v for v in vals if np.isfinite(v)]
        if len(valid) < 2:
            continue
        spread = max(valid) - min(valid)
        if spread == 0.0:
            continue
        mean_abs = abs(np.mean(valid))
        # Dispersión relativa (CV); usa absoluta si la media ≈ 0
        score = spread / mean_abs if mean_abs > 1e-12 else spread
        if score > best_score:
            best_score, best_key = score, k
    return best_key


def load_doe_results(h5_path: str) -> List[Dict]:
    """Loader para doe_results.h5 (case_* + señales + $..$ attrs)."""
    cases = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(k for k in f.keys() if k.startswith("case_")):
            grp      = f[grp_name]
            attrs    = dict(grp.attrs)
            var_val  = {k.strip("$"): v for k, v in attrs.items() if k != "wall_time_s"}
            wall_t   = float(attrs["wall_time_s"]) if "wall_time_s" in attrs else None
            lk       = _auto_label_key(var_val)
            try:
                lv = float(var_val.get(lk, float("nan")))
            except (TypeError, ValueError):
                lv = float("nan")
            cases.append({
                "group":       grp_name,
                "label_key":   lk,
                "label_val":   lv,
                "var_val":     var_val,
                "signals":     _read_signals(grp),
                "forces":      _read_forces(grp),
                "out_deflex":  _read_out_deflex(grp),
                "runs":        {},
                "snr":         {},
                "dt_us":       None,
                "wall_time_s": wall_t,
                # Tambien copia las señales como claves de primer nivel (compat con plot_convergence)
                "Axial_disp":  None,
                "Axial_vel":   None,
            })
            # Make signals accessible at top level (doe_plotter compat)
            for sig in _SIGNAL_NAMES:
                cases[-1][sig] = cases[-1]["signals"].get(sig)

    # Re-normaliza label_key/label_val con la clave de mayor variación
    if cases:
        best_lk = _best_label_key(cases)
        for c in cases:
            c["label_key"] = best_lk
            try:
                c["label_val"] = float(c["var_val"].get(best_lk, float("nan")))
            except (TypeError, ValueError):
                c["label_val"] = float("nan")

    cases.sort(key=lambda c: c["label_val"] if not np.isnan(c["label_val"]) else float("inf"))
    return cases


def load_doe_noise(h5_path: str) -> List[Dict]:
    """Loader para doe_noise_results.h5 (control + snr_* + señales)."""
    cases = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp   = f[grp_name]
            attrs = dict(grp.attrs)
            snr   = float(attrs.get("snr_db", 0.0)) if grp_name != "control" else float("inf")
            cases.append({
                "group":       grp_name,
                "label_key":   "snr_db",
                "label_val":   snr,
                "var_val":     {"snr_db": snr},
                "signals":     _read_signals(grp),
                "forces":      _read_forces(grp),
                "runs":        {},
                "snr":         {},
                "dt_us":       None,
                "wall_time_s": None,
                "Axial_disp":  None,
                "Axial_vel":   None,
            })
            for sig in _SIGNAL_NAMES:
                cases[-1][sig] = cases[-1]["signals"].get(sig)
    cases.sort(key=lambda c: c["label_val"] if np.isfinite(c["label_val"]) else float("inf"))
    return cases


def load_doe_indicator_unified(h5_path: str) -> List[Dict]:
    """Loader para doe_indicator_results.h5 — reutiliza load_indicator_results
    y añade la clave 'signals' y compatibilidad top-level."""
    raw = load_indicator_results(h5_path)
    for c in raw:
        # load_indicator_results ya lee señales dentro del grupo como subgrupos
        # pero NO las pone en c["signals"]; las añadimos aquí
        c.setdefault("signals", {})
        c.setdefault("snr", {})
        c.setdefault("dt_us", None)
        c.setdefault("wall_time_s", None)
        # top-level compat
        c["Axial_disp"] = c["signals"].get("Axial_disp")
        c["Axial_vel"]  = c["signals"].get("Axial_vel")
    # Vuelve a leer las señales del HDF5 (load_indicator_results no las lee)
    with h5py.File(h5_path, "r") as f:
        for c in raw:
            grp = f.get(c["group"])
            if grp is not None:
                for k in ("kappa", "Ap_mm", "true_label", "label_strategy", "kappa_start", "kappa_end", "t_onset",
                          "Ap_end_mm"):
                    if k in grp.attrs:
                        v = grp.attrs[k]
                        c["var_val"][k] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)
                sigs = _read_signals(grp)
                c["signals"] = sigs
                c["Axial_disp"] = sigs.get("Axial_disp")
                c["Axial_vel"]  = sigs.get("Axial_vel")

    # Re-normaliza label_key/label_val con la clave de mayor variación relativa
    # (el atributo guardado en el HDF5 puede apuntar a la variable equivocada
    # si el DOE fue generado con otra variable fija como spin_rate)
    if raw:
        best_lk = _best_label_key(raw)
        for c in raw:
            c["label_key"] = best_lk
            try:
                c["label_val"] = float(c["var_val"].get(best_lk, float("nan")))
            except (TypeError, ValueError):
                c["label_val"] = float("nan")
        raw.sort(key=lambda c: c["label_val"] if np.isfinite(c["label_val"]) else float("inf"))

    return raw


def load_noise_indicator_unified(h5_path: str) -> List[Dict]:
    """Loader para doe_noise_indicator_results.h5 (control + snr_* + runs)."""
    cases = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp   = f[grp_name]
            attrs = dict(grp.attrs)
            snr   = float(attrs.get("snr_db", 0.0)) if grp_name != "control" else float("inf")
            cases.append({
                "group":       grp_name,
                "label_key":   "snr_db",
                "label_val":   snr,
                "var_val":     {"snr_db": snr},
                "signals":     _read_signals(grp),
                "forces":      _read_forces(grp),
                "runs":        _read_runs(grp),
                "snr":         {},
                "dt_us":       None,
                "wall_time_s": None,
                "Axial_disp":  None,
                "Axial_vel":   None,
            })
            for sig in _SIGNAL_NAMES:
                cases[-1][sig] = cases[-1]["signals"].get(sig)
    cases.sort(key=lambda c: c["label_val"] if np.isfinite(c["label_val"]) else float("inf"))
    return cases


def load_model_snr_unified(h5_path: str) -> List[Dict]:
    """Loader para doe_model_snr_results.h5 — reutiliza load_snr_results."""
    raw = load_snr_results(h5_path)
    for c in raw:
        # Normalizar: añadir claves estándar
        c.setdefault("signals", {})
        c.setdefault("runs", {})
        # label_key / label_val
        pk = _snr_detect_param_key(raw) or "case_idx"
        c["label_key"] = pk
        try:
            c["label_val"] = float(c["var_val"].get(pk, c.get("case_idx", float("nan"))))
        except (TypeError, ValueError):
            c["label_val"] = float("nan")
        c.setdefault("dt_us", None)
        c.setdefault("wall_time_s", None)
        c["snr"]       = c.get("snr_by_signal", {})
        c["Axial_disp"] = None
        c["Axial_vel"]  = None
    return raw


def load_h5_unified(h5_path: str, h5_type: str) -> List[Dict]:
    """Dispatcher: carga cualquier formato HDF5 → lista normalizada de casos."""
    loaders = {
        TYPE_DOE_RESULTS  : load_doe_results,
        TYPE_DOE_NOISE    : load_doe_noise,
        TYPE_DOE_INDICATOR: load_doe_indicator_unified,
        TYPE_NOISE_IND    : load_noise_indicator_unified,
        TYPE_MODEL_SNR    : load_model_snr_unified,
    }
    cases = loaders[h5_type](h5_path)
    _apply_ramps(cases, h5_path)
    _assign_case_colors(cases, qualitative=False)
    return cases


# ── rampas de Ap (PLAN_ramps.md) ──────────────────────────────────────────────────────────────────────────────
TRUTH_SHADE = {"stable": "#0072B2", "gray": "#999999", "unstable": "#D55E00"}


def _is_ramp_vv(vv: dict) -> bool:
    try:
        return abs(float(vv["Ap_end"]) - float(vv["Ap_start"])) > 1e-9
    except (KeyError, TypeError, ValueError):
        return False


def _truth_from_dataset(path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """{case: [(t0, t1, label)]} of a reference_dataset*.h5 (one channel: the first one met per case)."""
    out: Dict[str, List[Tuple[float, float, str]]] = {}
    chan: Dict[str, str] = {}
    try:
        with h5py.File(path, "r") as f:
            for lab in f:
                for case, g in f[lab].items():
                    for piece in g.values():
                        a = piece.attrs
                        ch = str(a.get("channel", ""))
                        if chan.setdefault(case, ch) == ch and "t0" in a:
                            out.setdefault(case, []).append((float(a["t0"]), float(a["t1"]), lab))
    except OSError:
        return {}
    return {c: sorted(v) for c, v in out.items()}


def _apply_ramps(cases: List[Dict], h5_path: str) -> None:
    """Ramp cases (Ap_end != Ap_start): their single 'kappa' is ignored (NaN), the table shows kappa_start /
    kappa_end; with kappa as label key, a ramp takes its start kappa (ramps sort by the kappa where they start).
    Each ramp gets 'truth_iv' (the intervals of its ground truth: datasets of a validation file, else the
    labelled dataset named in the file) and 't_onset' (start of the first unstable interval), to draw them."""
    ramps = [c for c in cases if _is_ramp_vv(c.get("var_val", {}))]
    if not ramps:
        return
    truth_ds = None
    try:
        with h5py.File(h5_path, "r") as f:
            lab_path = f.attrs.get("label_dataset")
            for c in ramps:
                g = f.get(c["group"])
                if g is not None and "truth_t0" in g and "truth_label" in g:
                    c["truth_iv"] = list(zip(g["truth_t0"][()].tolist(), g["truth_t1"][()].tolist(),
                                             g["truth_label"].asstr()[()].tolist()))
    except OSError:
        lab_path = None
    for c in ramps:
        vv = c["var_val"]
        c["ramp"] = True
        if "kappa" in vv:
            vv["kappa"] = float("nan")
        if "truth_iv" not in c and lab_path:
            if truth_ds is None:
                truth_ds = _truth_from_dataset(str(lab_path.decode() if isinstance(lab_path, bytes) else lab_path))
            if c["group"] in truth_ds:
                c["truth_iv"] = truth_ds[c["group"]]
        t_on = vv.get("t_onset")
        if t_on is None or not np.isfinite(float(t_on)):
            t_on = min((a for a, _, lab in c.get("truth_iv", []) if lab == "unstable"), default=None)
        c["t_onset"] = None if t_on is None else float(t_on)

    def kap(c):   # kappa of a constant case, start kappa of a ramp
        try:
            return float(c["var_val"].get("kappa_start" if c.get("ramp") else "kappa", float("nan")))
        except (TypeError, ValueError):
            return float("nan")
    ks = [kap(c) for c in cases]
    # the key is kappa whenever every case has one (a ramp its start): never t_onset or another ramp-only attribute
    if all(np.isfinite(k) for k in ks) and len({round(k, 9) for k in ks}) > 1 or cases[0].get("label_key") == "kappa":
        for c, k in zip(cases, ks):
            c["label_key"], c["label_val"] = "kappa", k
    cases.sort(key=lambda c: c["label_val"] if np.isfinite(c.get("label_val", float("nan"))) else float("inf"))


def _case_legend(c: dict, lk: str, lv: float) -> str:
    """Legend text of a case: 'kappa=1.03', a ramp 'kappa 0.58->1.74', else the group."""
    vv = c.get("var_val", {})
    if c.get("ramp"):
        try:
            return f"kappa {float(vv['kappa_start']):.3g}->{float(vv['kappa_end']):.3g}"
        except (KeyError, TypeError, ValueError):
            return f"{c.get('group', '?')} (ramp)"
    base = f"{_col_header(lk)}={lv:.3g}" if np.isfinite(lv) else c.get("group", "?")
    # a validation made with --gray stable|unstable scores the gray cases as such: their legend keeps saying they were gray
    return base + (" (gray)" if str(vv.get("gray")) in ("1", "1.0", "True") else "")


def _case_onset(c: dict):
    """t_onset [s] of a case: where its ground truth turns unstable (a ramp: the first unstable window; a constant case of a
    validation file: the first sample over the amplitude limit of the labelling), or None."""
    v = c.get("t_onset") if c.get("ramp") else c.get("var_val", {}).get("t_onset")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _draw_truth_marks(axes, c: dict, color, shade: bool) -> None:
    """Vertical line where the ground truth of a case turns unstable (t_onset) and, for a ramp with one case shown, the
    stable / gray / unstable intervals of the truth shaded. Nothing for a case with no t_onset (a stable one)."""
    t_on = _case_onset(c)
    for ax in axes:
        if shade and c.get("ramp"):
            for t0, t1, lab in c.get("truth_iv", []):
                ax.axvspan(t0, t1, color=TRUTH_SHADE.get(lab, "#999999"), alpha=0.08, lw=0, zorder=0)
        if t_on is not None:
            ax.axvline(t_on, color=color, ls=":", lw=1.8, zorder=8,
                       label=f"truth turns unstable ({t_on:.2f} s)" if shade else None)


def _assign_case_colors(cases: List[Dict], qualitative: bool = False) -> None:
    """Asigna un color persistente a cada caso en c['_color'] y c['_color_hex'].
    Control → siempre rojo.
    qualitative=True  → tab20 por índice (casos con valores cercanos se distinguen bien).
    qualitative=False → viridis normalizado por valor numérico."""
    non_ctrl = [c for c in cases if c.get("group", "") != "control"]

    if qualitative:
        cmap = matplotlib.colormaps["tab20"]
        for i, c in enumerate(non_ctrl):
            rgba = cmap(i % cmap.N)
            c["_color"]     = rgba
            c["_color_hex"] = mcolors.to_hex(rgba)
        for c in cases:
            if c.get("group", "") == "control":
                c["_color"]     = (0.85, 0.05, 0.05, 1.0)
                c["_color_hex"] = "#d90d0d"
        return
    cmap = matplotlib.colormaps["viridis"]
    vals = [c.get("label_val", float("nan")) for c in non_ctrl]
    finite_vals = [v for v in vals if np.isfinite(v)]
    if len(finite_vals) >= 2:
        vmin, vmax = min(finite_vals), max(finite_vals)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=max(len(non_ctrl) - 1, 1))
        # asigna por índice cuando todos los valores son iguales
        for i, c in enumerate(non_ctrl):
            rgba = cmap(norm(i))
            c["_color"]     = rgba
            c["_color_hex"] = mcolors.to_hex(rgba)
        for c in cases:
            if c.get("group", "") == "control":
                c["_color"]     = (0.85, 0.05, 0.05, 1.0)
                c["_color_hex"] = "#d90d0d"
        return
    for c in cases:
        if c.get("group", "") == "control":
            c["_color"]     = (0.85, 0.05, 0.05, 1.0)
            c["_color_hex"] = "#d90d0d"
        else:
            lv = c.get("label_val", float("nan"))
            rgba = cmap(norm(lv)) if np.isfinite(lv) else (0.5, 0.5, 0.5, 1.0)
            c["_color"]     = rgba
            c["_color_hex"] = mcolors.to_hex(rgba)


# ==============================================================================
# AUXILIARES
# ==============================================================================

def _fmt_val(v) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
        if np.isnan(fv):
            return "NaN"
        if np.isinf(fv):
            return "∞"
        return f"{fv:.2e}"
    except (TypeError, ValueError):
        return str(v)


def _exact(v) -> str:
    """Valor tal cual está guardado en el .h5, sin redondear: de un número, la cadena más corta que lo
    reproduce exactamente (repr); de un vector, todos sus elementos (hasta 50)."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, np.ndarray):
        if v.size > 50:
            return f"array{v.shape} {v.dtype}"
        return "[" + ", ".join(_exact(x) for x in v.ravel().tolist()) + "]"
    return str(v)


def _col_header(key: str) -> str:
    return key.replace("$", "")


def _all_var_keys(cases: List[Dict]) -> List[str]:
    """Todas las claves de var_val presentes en los casos (label_key primero)."""
    keys: set = set()
    for c in cases:
        keys.update(c.get("var_val", {}).keys())
    lk = cases[0]["label_key"] if cases else ""
    ordered = []
    if lk and lk in keys:
        ordered.append(lk)
        keys.discard(lk)
    ordered.extend(sorted(keys))
    return ordered


def _capture_new_figure(func, *args, **kwargs) -> Optional[Figure]:
    """Llama func, captura la última figura matplotlib creada."""
    before = set(plt.get_fignums())
    result = func(*args, **kwargs)
    after  = set(plt.get_fignums())
    new_nums = sorted(after - before)
    if isinstance(result, Figure):
        return result
    if new_nums:
        return plt.figure(new_nums[-1])
    return None


def _embed_figure(fig: Figure, canvas_frame: tk.Frame,
                  toolbar_frame: tk.Frame, fig_holder: dict, slot: str) -> None:
    """Destruye widgets previos y embebe una figura matplotlib en Tkinter."""
    plt.close(fig)   # detach de pyplot window manager (no destroy el objeto)

    # Limpiar previo
    old = fig_holder.get(slot)
    if old is not None:
        try:
            plt.close(old)
        except Exception:
            pass
    fig_holder[slot] = fig

    for w in list(canvas_frame.winfo_children()):
        try:
            w.destroy()
        except Exception:
            pass
    for w in list(toolbar_frame.winfo_children()):
        try:
            w.destroy()
        except Exception:
            pass

    canvas  = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(fill=tk.X)


# ==============================================================================
# DIALOGO DE SELECCION DE COLUMNAS (reusado de doe_selector.py)
# ==============================================================================

class ColumnsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, all_keys: list, visible_keys: list) -> None:
        super().__init__(parent)
        self.title("Columnas visibles")
        self.resizable(False, False)
        self.grab_set()
        self._result = None
        self._vars: dict = {}

        ttk.Label(self, text="Visible columns in the table:",
                  font=("Arial", 10, "bold")).pack(padx=14, pady=(10, 4), anchor=tk.W)

        # Botones rápidos
        quick = ttk.Frame(self)
        quick.pack(fill=tk.X, padx=14, pady=(0, 4))
        ttk.Button(quick, text="☑ All",
                   command=lambda: [v.set(True)  for v in self._vars.values()]).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick, text="☐ None",
                   command=lambda: [v.set(False) for v in self._vars.values()]).pack(side=tk.LEFT, padx=2)

        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, padx=14, pady=4)

        ttk.Label(frm, text="case  (fixed)", foreground="#888888").pack(anchor=tk.W, pady=1)

        for key in all_keys:
            var = tk.BooleanVar(value=(key in visible_keys))
            self._vars[key] = var
            ttk.Checkbutton(frm, text=_col_header(key), variable=var).pack(
                anchor=tk.W, pady=1)

        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=14, pady=(6, 10))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=4)

        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{px}+{py}")

    def _apply(self) -> None:
        self._result = [k for k, v in self._vars.items() if v.get()]
        self.destroy()

    @property
    def result(self):
        return self._result


# ==============================================================================
# ENTRADAS DE FIGURAS DE RESUMEN POR TIPO
# ==============================================================================

# Each entry: (label, callable_or_str, kwargs_template)
# callable_or_str: una referencia a función o una clave string especial

def _make_summary_entries(h5_type: str, cases: list, h5_path: str):
    """Devuelve la lista de (label, func, extra_args_dict) para el combobox de resumen."""
    entries = []

    if h5_type == TYPE_DOE_RESULTS:
        for lbl, fn, args in [
            ("Convergencia RMS — Axial_disp",      plot_convergence,              ("Axial_disp", "rms")),
            ("Convergencia RMS — Axial_vel",        plot_convergence,              ("Axial_vel",  "rms")),
            ("Convergencia Max — Axial_disp",       plot_convergence,              ("Axial_disp", "max")),
            ("Convergencia Max — Axial_vel",        plot_convergence,              ("Axial_vel",  "max")),
            ("Error más fino RMS — Axial_disp",     plot_convergence_error_ref,    ("Axial_disp", "rms")),
            ("Error más fino RMS — Axial_vel",      plot_convergence_error_ref,    ("Axial_vel",  "rms")),
            ("Ganancia RMS — Axial_disp",           plot_convergence_error_consec, ("Axial_disp", "rms")),
            ("Ganancia RMS — Axial_vel",            plot_convergence_error_consec, ("Axial_vel",  "rms")),
            ("Tiempo de ejecución",                 plot_convergence_time,         ()),
        ]:
            entries.append((lbl, fn, args))

    elif h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND):
        lk       = _resolve_label_key(cases)
        xs       = _ind_x_values(cases, lk)
        all_runs = sorted({rn for c in cases for rn in c.get("runs", {})})

        if h5_type == TYPE_NOISE_IND:
            # Figuras correctas para DOE de ruido: usan doe_noise_plotter
            # Usar los run names completos como clave (son los indicadores reales del HDF5)
            for ind in all_runs:
                entries.append((
                    f"{ind} — t_d vs SNR",
                    "_noise_td_ind",
                    {"h5_path": h5_path, "indicator": ind, "td_col": "t_d"},
                ))
            entries.append(("Lollipop t_d vs indicador",  "_noise_lollipop", {"h5_path": h5_path}))
            entries.append(("Retraso (t_d - t_gt) vs SNR", "_noise_delay",    {"h5_path": h5_path}))
            for ind in all_runs:
                entries.append((f"I_t overlay — {ind}", "_noise_it_overlay",
                                {"h5_path": h5_path, "indicator": ind}))
        else:
            # Figura global (todos los indicadores juntos)
            entries.append((
                f"t_d vs {_col_header(lk)}  [todos]",
                _plot_td_single,
                {"cases": cases, "xs": xs, "all_runs": all_runs,
                 "label_key": lk, "out_dir": None},
            ))
            # Una figura por indicador
            for rn in all_runs:
                entries.append((
                    f"{rn} — t_d vs {_col_header(lk)}",
                    _plot_td_per_run,
                    {"cases": cases, "xs": xs, "run_name": rn,
                     "label_key": lk, "out_dir": None},
                ))
            entries.append((
                "I_t overlay — t_d",
                plot_It_overlay,
                {"cases": cases, "label_key": lk,
                 "run_name_filter": None, "out_dir": None},
            ))

    elif h5_type == TYPE_DOE_NOISE:
        entries.append(("Overlay Axial_disp por SNR", "_noise_overlay", {"signal": "Axial_disp"}))
        entries.append(("Overlay Axial_vel por SNR",  "_noise_overlay", {"signal": "Axial_vel"}))

    elif h5_type == TYPE_MODEL_SNR:
        pk = _snr_detect_param_key(cases) or "case_idx"
        entries.append((f"SNR_mod_dB vs {_col_header(pk)}", plot_snr_vs_param,
                        {"cases": cases, "param_key": pk, "out_dir": None}))

    # SLD: casos del DOE (spin_rate, Ap) sobre los lóbulos de cada preset de sld_model.MODELS
    if sld_model and h5_type in (TYPE_DOE_RESULTS, TYPE_DOE_INDICATOR):
        used = _sim_models(h5_path)   # only the SLD of the model the cases were simulated with (all if the file does not say)
        presets = [p for p in sld_model.MODELS if p in used] or list(sld_model.MODELS)
        for p in presets:
            m = sld_model.MODELS[p]
            entries.append((f"SLD — {p} [todos los modos]", sld_model.plot_sld, {"cases": cases, "preset": p}))
            if len(m["modes"]) > 1:
                for j, f in enumerate(sorted(x[0] for x in m["modes"])):
                    entries.append((f"SLD — {p} [modo {f:.0f} Hz]", sld_model.plot_sld,
                                    {"cases": cases, "preset": p, "seg": j}))
        # doe_validation_results.h5: casos coloreados por TP/TN/FN/FP de cada indicador ($outcome_<run>$)
        for rn in sorted({k[len("outcome_"):] for c in cases for k in c.get("var_val", {}) if k.startswith("outcome_")}):
            for p in presets:
                entries.append((f"SLD — {p} [outcome {rn}]", sld_model.plot_sld,
                                {"cases": cases, "preset": p, "outcome_run": rn}))

    # doe_validation_results.h5 (has /ranking): the validation figures of validation_figures.py, FIRST in the list (same API
    # as the SLD: fn(h5_path=..., out_dir=None) -> Figure with _keep_size; a figure without data raises, shown as a viewer error)
    if _is_validation_h5(h5_path):
        import validation_figures as vf
        entries = [(f"Validation — {n}", fn, {"h5_path": h5_path}) for n, fn in vf.FIGURES.items()] + entries

    return entries


def _fig_style() -> tuple:
    """(language, scale) the figure modules are using now (the defaults of the Figures window)."""
    m = sys.modules.get("validation_figures") or sys.modules.get("sld_model")
    return getattr(m, "LANGUAGE", "EN"), float(getattr(m, "FIGSCALE", 1.5))


def _apply_fig_style(language: str, scale: float) -> None:
    """Set the language (EN | FR | both) and the scale (multiplier of the plot_style presets) of the figure modules."""
    for name in ("validation_figures", "sld_model"):
        m = sys.modules.get(name)
        if m is not None:
            m.LANGUAGE, m.FIGSCALE = language, scale


def _sim_models(path: str) -> set:
    """SLD models (attr sim_model) of the cases of an .h5; empty if the file does not say."""
    try:
        with h5py.File(path, "r") as f:
            vals = [f[g].attrs.get("sim_model") for g in list(f.keys())[:50] if isinstance(f[g], h5py.Group)]
    except OSError:
        return set()
    return {v.decode() if isinstance(v, bytes) else str(v) for v in vals if v is not None}


def _is_validation_h5(path: str) -> bool:
    """True for a doe_validation_results.h5 (it has the /ranking group)."""
    try:
        with h5py.File(path, "r") as f:
            return "ranking" in f
    except OSError:
        return False


def _build_noise_overlay_fig(cases: list, signal: str) -> Optional[Figure]:
    """Crea una figura de overlay de señales coloreadas por snr_db (doe_noise)."""
    vals = [c["label_val"] for c in cases if np.isfinite(c["label_val"])]
    cmap = matplotlib.colormaps["viridis"]
    if len(vals) > 1:
        norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals))
    else:
        norm = mcolors.Normalize(vmin=0, vmax=100)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(7, 4))
    for c in cases:
        data = c["signals"].get(signal)
        if data is None:
            continue
        t, y = data
        lv = c.get("label_val")
        is_control = (c.get("group", "") == "control")
        if is_control or not np.isfinite(lv):
            # control: dibuja con estilo distintivo y lo incluye en la leyenda
            color = "red"
            ax.plot(t[::DECIMATE], y[::DECIMATE], color=color, lw=2.2, alpha=1.0,
                    label="control", zorder=6, rasterized=True)
            # Tambien marca el control en el plot (anotacion chica)
            try:
                mid = int(len(t) // 2)
                ax.scatter([t[mid]], [y[mid]], marker="D", color=color, s=30, zorder=7)
            except Exception:
                pass
        else:
            color = cmap(norm(lv))
            label = f"SNR={lv:.0f} dB"
            ax.plot(t[::DECIMATE], y[::DECIMATE], color=color, lw=1.4, alpha=0.85,
                    label=label, rasterized=True)

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("SNR (dB)", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(SIGNAL_YLABELS.get(signal, signal))
    ax.set_title(f"{signal} — overlay por SNR")
    # ax.grid(True, linestyle=":", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.grid(False)
    fig.tight_layout()
    return fig


def _run_indicator_prefix(run_name: str) -> str:
    """Devuelve el prefijo de indicador de un run_name (maxent, rms_cv, ssq, green, etc.)."""
    m = re.match(r"^(maxent|rms_cv|ssq[^_]*|green[^_]*|sst_svd[^_]*)", run_name)
    if m:
        return m.group(1)
    parts = run_name.split("_")
    return parts[0] if parts else run_name


def _variable_keys_with_variation(cases: List[Dict]) -> List[str]:
    """Devuelve las claves de var_val que tienen al menos 2 valores distintos entre los casos."""
    all_keys: set = set()
    for c in cases:
        all_keys.update(c.get("var_val", {}).keys())
    varying = []
    for k in sorted(all_keys):
        vals = {c["var_val"].get(k) for c in cases if k in c.get("var_val", {})}
        if len(vals) >= 2:
            varying.append(k)
    return varying


def _indicator_limits(rn: str, attrs: dict) -> list:
    """Decision limits of an indicator on its I(t), from the attributes of its run (checked against the first detections:
    I(t) crosses them at t_d). SST: lim_sup (and lim_inf); RMS-CV: the CV threshold; MaxEnt-SPRT: the two bounds
    ln((1-beta)/alpha) and ln(beta/(1-alpha)). [] when the indicator stores none (Green)."""
    def num(*keys):
        for k in keys:
            try:
                v = float(attrs[k])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(v):
                return v
        return None
    p = _run_indicator_prefix(rn)
    if p == "ssq":
        return [v for v in (num("meta_lim_sup"), num("meta_lim_inf")) if v is not None]
    if p == "rms_cv":
        v = num("meta_cv_threshold_used")
        return [] if v is None else [v]
    if p == "maxent":
        a, b = num("pp_alpha", "meta_alpha"), num("pp_beta", "meta_beta")
        if a and b and 0 < a < 1 and 0 < b < 1:
            return [float(np.log((1 - b) / a)), float(np.log(b / (1 - a)))]
    return []


def _run_delay(run_data: dict):
    """delay_onset_s = first detection - t_onset [s] (negative: the alarm came before the amplitude of the truth), if the
    file (validation) has it for this run, else None."""
    try:
        v = float(run_data.get("attrs", {}).get("delay_onset_s"))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _it_plot_yscale(runs_to_show: List[str]) -> str:
    """Escala Y para I_t: log solo para green* y sst_svd*; resto lineal."""
    if not runs_to_show:
        return "linear"
    prefixes = {_run_indicator_prefix(rn) for rn in runs_to_show}
    log_prefixes = {"green", "sst_svd", "ssq"}
    return "log" if prefixes and prefixes.issubset(log_prefixes) else "linear"


# ==============================================================================
# APLICACION PRINCIPAL
# ==============================================================================

class DoeSelectorUnifiedApp:
    """Visor de un .h5 del DOE: Treeview + panel central (señales/I_t) + summary.

    `container`: frame donde vive (una pestaña de TabbedViewer); por defecto `root` (toda la ventana).
    `on_open`: si se da, el botón de abrir llama a esto (TabbedViewer abre el archivo en OTRA pestaña)
    en vez de reemplazar este visor.
    """

    _LEFT_WIDTH   = 420
    _CENTER_WIDTH = 860
    _RIGHT_WIDTH  = 480

    def __init__(self, root: tk.Tk, h5_path: str, container: Optional[tk.Widget] = None,
                 on_open=None) -> None:
        self.root     = root
        self.container = container if container is not None else root
        self._on_open  = on_open
        self._fig_holder: dict = {}    # slot → Figure para Guardar PNG
        self._cbar       = None
        self._force_cbar = None
        self._deflex_cbar = None
        self._It_cbar     = None
        self._sort_col: Optional[str] = None
        self._sort_rev: bool           = False
        self._iid_to_case: dict = {}
        self._manual_control_group: Optional[str] = None  # grupo elegido como control manual
        self._ref_lines: List[dict] = []  # {"kind": "v"|"h", "value": float, "target": str, "color": str}
        self._plotted_iids: set = set()  # ultima seleccion realmente graficada (para "+ Agregar al plot")

        self._load_file(h5_path)
        self._build_ui()

    # ── Carga de archivo ──────────────────────────────────────────────────────────
    def _load_file(self, h5_path: str) -> None:
        self.h5_path  = h5_path
        self.h5_type  = detect_h5_type(h5_path)
        self.cases    = load_h5_unified(h5_path, self.h5_type)  # colors already assigned
        self.doe_name = os.path.basename(os.path.dirname(h5_path))
        self._all_keys     = _all_var_keys(self.cases)
        self._visible_keys = list(self._all_keys)
        # Nombres de run disponibles (para tipos indicador)
        self._all_runs = sorted({rn for c in self.cases for rn in c.get("runs", {})})

        # Sincronizar doe_plotter.LABEL_KEY con la variable real del DOE cargado.
        # Las funciones plot_convergence / plot_overlay usan ese global directamente.
        import doe_plotter as _dp
        if self.cases:
            _dp.LABEL_KEY = self.cases[0].get("label_key", _dp.LABEL_KEY)

        # Entradas de resumen (figuras precalculadas por tipo)
        self._summary_entries = _make_summary_entries(self.h5_type, self.cases, h5_path)
        self._summary_labels  = [e[0] for e in self._summary_entries]
        # Filter variables
        self._filter_var_str: Optional[tk.StringVar] = None

        # Refresca el combobox de label-key si la UI ya esta construida
        if hasattr(self, "_label_key_combo"):
            self._refresh_label_key_combo()

    # ── Construcción de UI ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        if self.container is self.root:   # dueño de toda la ventana (en una pestaña lo hace TabbedViewer)
            self.root.title(
                f"{_TYPE_LABELS.get(self.h5_type, self.h5_type)}  —  "
                f"{os.path.basename(self.h5_path)}"
            )
            self.root.minsize(1100, 580)
            self.root.state("zoomed")

        self._has_deflex = (
            self.h5_type == TYPE_DOE_RESULTS
            and any(bool(c.get("out_deflex")) for c in self.cases)
        )
        self._has_acc = any("Axial_acc" in c.get("signals", {}) for c in self.cases)
        self._build_topbar()
        self._build_layout()
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    def _build_topbar(self) -> None:
        bar = ttk.Frame(self.container, padding=(4, 2))
        bar.pack(side=tk.TOP, fill=tk.X)

        if self._on_open is None:   # en una pestaña, el botón de abrir es el de la barra de TabbedViewer
            ttk.Button(bar, text="📂  Open .h5", command=self._open_file).pack(side=tk.LEFT, padx=4)
        self._type_label = ttk.Label(
            bar,
            text=f"Type: {_TYPE_LABELS.get(self.h5_type, self.h5_type)}  |  "
                 f"{len(self.cases)} cases  |  {os.path.basename(self.h5_path)}",
            foreground="#444444", font=("Arial", 9),
        )
        self._type_label.pack(side=tk.LEFT, padx=8)
        ttk.Button(bar, text="🔍  Inspect", command=self._open_inspector).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="💾  Export…", command=lambda: self._open_export(self._active_panel_name())).pack(
            side=tk.LEFT, padx=4)
        role = file_role(self.h5_path)
        if role:
            tk.Label(self.container, text=role, bg="#fff8e1", fg="#5d4037", anchor="w", padx=8,
                     font=("Arial", 9, "bold")).pack(side=tk.TOP, fill=tk.X)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._persistent_color_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bar, text="🎨 Fixed color per case",
            variable=self._persistent_color_var,
            command=self._replot_active_tab,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(bar, text="α:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._sig_alpha_var = tk.DoubleVar(value=0.5)
        _sc = ttk.Scale(bar, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                         variable=self._sig_alpha_var, length=80)
        _sc.pack(side=tk.LEFT)
        _sc.bind("<ButtonRelease-1>", lambda _e: self._replot_active_tab())
        ttk.Label(bar, text="lw:", font=("Arial", 8)).pack(side=tk.LEFT, padx=(6, 0))
        self._sig_lw_var = tk.DoubleVar(value=0.9)
        _sc = ttk.Scale(bar, from_=0.3, to=3.0, orient=tk.HORIZONTAL,
                         variable=self._sig_lw_var, length=70)
        _sc.pack(side=tk.LEFT)
        _sc.bind("<ButtonRelease-1>", lambda _e: self._replot_active_tab())

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(bar, text="ctrl α:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._ctrl_alpha_var = tk.DoubleVar(value=1.0)
        _sc = ttk.Scale(bar, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                         variable=self._ctrl_alpha_var, length=70)
        _sc.pack(side=tk.LEFT)
        _sc.bind("<ButtonRelease-1>", lambda _e: self._replot_active_tab())
        ttk.Label(bar, text="ctrl z:", font=("Arial", 8)).pack(side=tk.LEFT, padx=(6, 0))
        self._ctrl_zo_var = tk.IntVar(value=100)
        _sc = ttk.Scale(bar, from_=1, to=200, orient=tk.HORIZONTAL,
                         variable=self._ctrl_zo_var, length=70)
        _sc.pack(side=tk.LEFT)
        _sc.bind("<ButtonRelease-1>", lambda _e: self._replot_active_tab())

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._invert_order_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bar, text="⇅ Invert zorder",
            variable=self._invert_order_var,
            command=self._replot_active_tab,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(bar, text="X axis:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._label_key_var = tk.StringVar(value="auto")
        self._label_key_combo = ttk.Combobox(
            bar, textvariable=self._label_key_var,
            state="readonly", width=18,
        )
        self._label_key_combo.pack(side=tk.LEFT, padx=4)
        self._label_key_combo.bind("<<ComboboxSelected>>", self._on_label_key_change)
        self._refresh_label_key_combo()

        bar2 = ttk.Frame(self.container, padding=(4, 2))
        bar2.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar2, text="line x=", font=("Arial", 8)).pack(side=tk.LEFT)
        self._vline_entry = ttk.Entry(bar2, width=8)
        self._vline_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(bar2, text="y=", font=("Arial", 8)).pack(side=tk.LEFT)
        self._hline_entry = ttk.Entry(bar2, width=8)
        self._hline_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(bar2, text="in:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._line_target_var = tk.StringVar(value=_LINE_TARGET_ALL)
        ttk.Combobox(bar2, textvariable=self._line_target_var, values=_LINE_TARGETS,
                     state="readonly", width=14).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(bar2, text="note:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._line_note_entry = ttk.Entry(bar2, width=12)
        self._line_note_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bar2, text="+ Line", command=self._add_reference_lines).pack(side=tk.LEFT, padx=2)
        self._line_remove_var = tk.StringVar()
        self._line_remove_combo = ttk.Combobox(bar2, textvariable=self._line_remove_var,
                                               state="readonly", width=20)
        self._line_remove_combo.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(bar2, text="Delete sel.", command=self._remove_selected_line).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar2, text="Delete all", command=self._clear_reference_lines).pack(side=tk.LEFT, padx=2)

    def _refresh_label_key_combo(self) -> None:
        """Actualiza las opciones del combobox con las variables que tienen variación."""
        varying = _variable_keys_with_variation(self.cases)
        options = ["auto"] + [_col_header(k) for k in varying]
        self._label_key_combo["values"] = options
        # Preseleccionar la key actualmente activa
        current = _col_header(self.cases[0].get("label_key", "")) if self.cases else "auto"
        if current in options:
            self._label_key_var.set(current)
        else:
            self._label_key_var.set("auto")

    def _on_label_key_change(self, _event=None) -> None:
        """Cambia la variable de etiqueta/color en todos los casos y refresca la UI."""
        chosen = self._label_key_var.get()
        if chosen == "auto":
            new_key = _best_label_key(self.cases)
        else:
            # Buscar la clave original con $ que coincida (strip $)
            varying = _variable_keys_with_variation(self.cases)
            new_key = next((k for k in varying if _col_header(k) == chosen), chosen)

        for c in self.cases:
            c["label_key"] = new_key
            c["label_val"] = float(c.get("var_val", {}).get(new_key, float("nan")))

        _assign_case_colors(self.cases, qualitative=False)

        # Actualizar LABEL_KEY del doe_plotter
        import doe_plotter as _dp
        _dp.LABEL_KEY = new_key

        self._populate_tree(self._filtered_cases())
        self._replot_all_tabs()

    def _replot_all_tabs(self) -> None:
        """Redibuja TODOS los paneles disponibles con la seleccion actual (ej. tras cambiar Eje X)."""
        if not self.tree.selection():
            return
        for fn in self._available_plot_fns():
            fn()

    def _active_tab_info(self):
        """(plot_fn, axes_dict, canvas) del tab actualmente activo, o (None, {}, None)."""
        if hasattr(self, "_nb"):
            current = self._nb.select()
            if hasattr(self, "_sig_tab") and current == str(self._sig_tab):
                return (self._plot_signals,
                        {n: ax for _, ax, n in self._sig_axes()},
                        self.sig_canvas)
            if hasattr(self, "_force_tab") and current == str(self._force_tab):
                return (self._plot_forces,
                        {"Fuerzas: F1": self.ax_force_1, "Fuerzas: F2": self.ax_force_2,
                         "Fuerzas: F3": self.ax_force_3},
                        self.force_canvas)
            if hasattr(self, "_It_tab") and current == str(self._It_tab):
                return self._plot_It, {"I_t": self.ax_It}, self.It_canvas
            if hasattr(self, "_deflex_tab") and current == str(self._deflex_tab):
                return (self._plot_deflex,
                        {"Deflex: disp": self.ax_deflex_d, "Deflex: vel": self.ax_deflex_v},
                        self.deflex_canvas)
        elif hasattr(self, "sig_canvas"):
            return (self._plot_signals,
                    {n: ax for _, ax, n in self._sig_axes()},
                    self.sig_canvas)
        return None, {}, None

    def _replot_active_tab(self) -> None:
        """Redibuja el tab con la selección actual (tras cambiar la propiedad de color/leyenda)."""
        if not self.tree.selection():
            return
        fn, _, _ = self._active_tab_info()
        if fn is not None:
            fn()

    def _replot_preserving_zoom(self) -> None:
        """Redibuja el tab activo pero mantiene el zoom/pan que ya tenia (usado al agregar/borrar lineas)."""
        _, axes, _ = self._active_tab_info()
        saved = {key: (ax.get_xlim(), ax.get_ylim()) for key, ax in axes.items()}
        self._replot_active_tab()
        _, axes2, canvas2 = self._active_tab_info()
        for key, ax in axes2.items():
            if key in saved:
                xlim, ylim = saved[key]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        if canvas2 is not None:
            canvas2.draw()

    def _add_reference_lines(self) -> None:
        """Agrega los valores de los campos x=/y= como lineas de referencia y redibuja."""
        target = self._line_target_var.get() or _LINE_TARGET_ALL
        note = self._line_note_entry.get().strip() or None
        new_entries = []
        v_txt = self._vline_entry.get().strip()
        if v_txt:
            try:
                color = _LINE_COLORS[len(self._ref_lines) % len(_LINE_COLORS)]
                entry = {"kind": "v", "value": float(v_txt), "target": target, "color": color, "note": note}
                self._ref_lines.append(entry)
                new_entries.append(entry)
            except ValueError:
                messagebox.showwarning("Invalid value", f"'{v_txt}' is not a number.", parent=self.root)
        h_txt = self._hline_entry.get().strip()
        if h_txt:
            try:
                color = _LINE_COLORS[len(self._ref_lines) % len(_LINE_COLORS)]
                entry = {"kind": "h", "value": float(h_txt), "target": target, "color": color, "note": note}
                self._ref_lines.append(entry)
                new_entries.append(entry)
            except ValueError:
                messagebox.showwarning("Invalid value", f"'{h_txt}' is not a number.", parent=self.root)
        if new_entries:
            self._vline_entry.delete(0, tk.END)
            self._hline_entry.delete(0, tk.END)
            self._line_note_entry.delete(0, tk.END)
            self._refresh_line_remove_combo()
            _, axes, _ = self._active_tab_info()
            if all(self._line_in_view(e, axes) for e in new_entries):
                self._replot_preserving_zoom()
            else:
                self._replot_active_tab()  # la linea nueva queda fuera del zoom actual -> autoescala

    def _line_in_view(self, entry: dict, axes: dict) -> bool:
        target_axes = list(axes.values()) if entry["target"] == _LINE_TARGET_ALL else (
            [axes[entry["target"]]] if entry["target"] in axes else [])
        for ax in target_axes:
            lo, hi = ax.get_xlim() if entry["kind"] == "v" else ax.get_ylim()
            if not (min(lo, hi) <= entry["value"] <= max(lo, hi)):
                return False
        return True

    def _clear_reference_lines(self) -> None:
        self._ref_lines.clear()
        self._refresh_line_remove_combo()
        self._replot_preserving_zoom()

    def _refresh_line_remove_combo(self) -> None:
        labels = []
        for i, e in enumerate(self._ref_lines):
            base = f"{i}: {'x' if e['kind'] == 'v' else 'y'}={e['value']:g}"
            note_txt = f"  nota='{e['note']}'" if e.get("note") else ""
            labels.append(f"{base}{note_txt}  [{e['target']}]")
        self._line_remove_combo["values"] = labels
        self._line_remove_var.set(labels[-1] if labels else "")

    def _remove_selected_line(self) -> None:
        sel = self._line_remove_var.get()
        if not sel:
            return
        idx = int(sel.split(":", 1)[0])
        del self._ref_lines[idx]
        self._refresh_line_remove_combo()
        self._replot_preserving_zoom()

    def _draw_reference_lines(self, axes: dict) -> None:
        """axes: {nombre_target: Axes} de los ejes del plot que se esta dibujando ahora."""
        for entry in self._ref_lines:
            target = entry["target"]
            targets = list(axes.values()) if target == _LINE_TARGET_ALL else (
                [axes[target]] if target in axes else [])
            label = entry.get("note") or f"{entry['value']:g}"
            for ax in targets:
                if entry["kind"] == "v":
                    ax.axvline(entry["value"], color=entry["color"], lw=1.2, linestyle="--", zorder=10)
                    ax.text(entry["value"], 0.98, label, transform=ax.get_xaxis_transform(),
                            va="top", ha="right", color=entry["color"], rotation=90,
                            fontsize=14, zorder=11, clip_on=True)
                else:
                    ax.axhline(entry["value"], color=entry["color"], lw=1.2, linestyle="--", zorder=10)
                    ax.text(0.02, entry["value"], label, transform=ax.get_yaxis_transform(),
                            va="bottom", ha="left", color=entry["color"],
                            fontsize=14, zorder=11, clip_on=True)

    def _build_layout(self) -> None:
        self.paned = tk.PanedWindow(
            self.container, orient=tk.HORIZONTAL,
            sashwidth=5, sashrelief=tk.RAISED,
        )
        self.paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.left_frame   = ttk.Frame(self.paned)
        self.center_frame = ttk.Frame(self.paned)
        self.paned.add(self.left_frame,   minsize=240, width=self._LEFT_WIDTH)
        self.paned.add(self.center_frame, minsize=400, width=self._CENTER_WIDTH)
        self.right_frame  = ttk.Frame(self.paned)
        self.paned.add(self.right_frame,  minsize=300, width=self._RIGHT_WIDTH)

    # ── PANEL DERECHO: figuras de resumen (convergencia, t_d, SLD, ...) ─────────────
    def _build_right_panel(self) -> None:
        rf  = self.right_frame
        top = ttk.Frame(rf, padding=(4, 4))
        top.pack(fill=tk.X)
        # height: el desplegable de Tk muestra 10 filas por defecto y escondía las últimas (las SLD)
        # solo las SLD cuando las hay; en los demás tipos de archivo, todas las figuras de resumen
        labels = [l for l in self._summary_labels if l.startswith("SLD")] or self._summary_labels
        self._sum_combo = ttk.Combobox(top, values=labels, state="readonly",
                                       height=max(10, min(len(labels), 30)))
        if labels:
            self._sum_combo.current(0)
        self._sum_combo.pack(fill=tk.X)
        btns = ttk.Frame(top)
        btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btns, text="▶ Preview", command=self._refresh_summary).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save…", command=lambda: self._open_export(self._sum_combo.get())).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Figures…", command=lambda: self._open_export(
            next((e[0] for e in self._summary_entries if not e[0].startswith("SLD")), None))).pack(side=tk.LEFT)
        if sld_model:   # vertical axis of the SLD: Ap [mm] or kappa = Ap / SLD limit at that speed (also for its export)
            self._kappa_axis = tk.BooleanVar(value=sld_model.Y_AXIS == "kappa")

            def flip_axis():
                sld_model.Y_AXIS = "kappa" if self._kappa_axis.get() else "Ap"
                if self._sum_combo.get().startswith("SLD"):
                    self._refresh_summary()
            ttk.Checkbutton(btns, text="κ axis", variable=self._kappa_axis, command=flip_axis).pack(side=tk.LEFT, padx=6)
        self._sum_toolbar_frame = ttk.Frame(rf)
        self._sum_toolbar_frame.pack(fill=tk.X)
        self._sum_canvas_frame = ttk.Frame(rf)
        self._sum_canvas_frame.pack(fill=tk.BOTH, expand=True)

    # ── PANEL IZQUIERDO ────────────────────────────────────────────────────────────
    def _build_left_panel(self) -> None:
        lf = self.left_frame

        # Header
        ttk.Label(lf, text=self.doe_name,
                  font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 0))
        ttk.Label(
            lf,
            text=f"{len(self.cases)} cases  ·  {_TYPE_LABELS.get(self.h5_type, '')}",
            font=("Arial", 9), foreground="#555555",
        ).pack(anchor=tk.W, padx=8, pady=(0, 4))

        # Search bar + Columns button
        bar = ttk.Frame(lf)
        bar.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._filter_var_str = tk.StringVar()
        self._filter_var_str.trace_add("write", lambda *_: self._apply_filter())
        ttk.Entry(bar, textvariable=self._filter_var_str,
                  font=("Arial", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(bar, text=" 🔍", font=("Arial", 9)).pack(side=tk.LEFT)
        ttk.Button(bar, text="Columns…",
                   command=self._show_columns_dialog).pack(side=tk.LEFT, padx=(6, 0))

        # Filtro de run (solo para tipos indicador)
        if self.h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND):
            self._build_run_filter(lf)

        # Treeview container
        self._tree_frame = ttk.Frame(lf)
        self._tree_frame.pack(fill=tk.BOTH, expand=True, padx=8)
        self._build_tree()

        # Action buttons
        btn = ttk.Frame(lf)
        btn.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn, text="Select all",
                   command=self._select_all).pack(fill=tk.X, pady=1)
        ttk.Button(btn, text="Clear selection",
                   command=self._clear_sel).pack(fill=tk.X, pady=1)
        ttk.Separator(btn).pack(fill=tk.X, pady=4)
        ttk.Button(btn, text="Plot (all tabs) ▶",
                   command=self._plot_active_tab).pack(fill=tk.X, pady=1, ipady=3)
        ttk.Button(btn, text="+ Add selection to plot",
                   command=self._add_selection_to_plot).pack(fill=tk.X, pady=1, ipady=2)
        ttk.Button(btn, text="Clear plot",
                   command=self._clear_signal_plot).pack(fill=tk.X, pady=1)
        ttk.Separator(btn).pack(fill=tk.X, pady=4)
        self._ctrl_label_var = tk.StringVar(value="Manual control: none")
        ttk.Label(btn, textvariable=self._ctrl_label_var,
                  foreground="#cc0000", font=("Arial", 8)).pack(anchor=tk.W)
        ttk.Button(btn, text="⭐ Mark as control",
                   command=self._set_manual_control).pack(fill=tk.X, pady=1)
        ttk.Button(btn, text="✖ Remove manual control",
                   command=self._clear_manual_control).pack(fill=tk.X, pady=1)

    def _set_manual_control(self) -> None:
        """Marca el caso seleccionado en el árbol como control manual."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Select a case first.", parent=self.root)
            return
        if len(sel) > 1:
            messagebox.showwarning("Multiple selection", "Select only one case.", parent=self.root)
            return
        c = self._iid_to_case.get(sel[0])
        if c is None:
            return
        self._manual_control_group = c["group"]
        self._ctrl_label_var.set(f"Manual control: {c['group']}")
        self._populate_tree(self._filtered_cases())

    def _clear_manual_control(self) -> None:
        """Elimina el control manual."""
        self._manual_control_group = None
        self._ctrl_label_var.set("Manual control: none")
        self._populate_tree(self._filtered_cases())

    def _is_control(self, c: dict) -> bool:
        """True si el caso es control real O control manual asignado en GUI."""
        if c.get("group", "") == "control":
            return True
        if self._manual_control_group and c.get("group", "") == self._manual_control_group:
            return True
        return False

    def _mark_values_on_colorbar(self, cbar, marks: list) -> None:
        """marks: lista de (label_val, color). Dibuja una linea por caso graficado, con
        borde negro para que resalte contra el degradado del colorbar."""
        if cbar is None:
            return
        for v, color in marks:
            if not np.isfinite(v):
                continue
            cbar.ax.axvline(v, color="black", lw=3.5, zorder=9)
            cbar.ax.axvline(v, color=color, lw=1.8, zorder=10)

    def _build_run_filter(self, parent: ttk.Frame) -> None:
        """Panel de selección de indicadores (checkboxes) para tipos indicador."""
        frm = ttk.LabelFrame(parent, text="  Indicators  ", padding=4)
        frm.pack(fill=tk.X, padx=8, pady=(0, 4))

        indicators = self._extract_indicators()
        palette = matplotlib.colormaps.get_cmap("tab10")
        self._ind_color_map = {
            ind: palette(i % palette.N)
            for i, ind in enumerate(indicators)
        }

        # Checkbox "Todos"
        self._ind_all_var = tk.BooleanVar(value=True)
        self._ind_all_chk = ttk.Checkbutton(
            frm, text="(all)", variable=self._ind_all_var,
            command=self._on_ind_all_toggle,
        )
        self._ind_all_chk.pack(anchor=tk.W)

        # Un checkbox por indicador
        self._ind_check_vars: Dict[str, tk.BooleanVar] = {}
        for ind in indicators:
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(
                frm, text=ind, variable=var,
                command=self._on_ind_check_toggle,
            )
            chk.pack(anchor=tk.W, padx=(12, 0))
            self._ind_check_vars[ind] = var

        ttk.Separator(frm).pack(fill=tk.X, pady=3)


    def _extract_indicators(self) -> List[str]:
        """Extrae los prefijos de indicador de los run_names."""
        prefixes = set()
        for rn in self._all_runs:
            # Intenta extraer el prefijo (maxent, rms_cv, ssq, green_fixed, etc.)
            # Patrón: indicador termina antes de _revo o _fcycle o _dec
            m = re.match(r"^(maxent|rms_cv|ssq[^_]*|green[^_]*)", rn)
            if m:
                prefixes.add(m.group(1))
            else:
                parts = rn.split("_")
                if parts:
                    prefixes.add(parts[0])
        return sorted(prefixes)

    def _on_ind_all_toggle(self) -> None:
        """Activa/desactiva todos los checkboxes de indicadores."""
        val = self._ind_all_var.get()
        for v in self._ind_check_vars.values():
            v.set(val)

    def _on_ind_check_toggle(self) -> None:
        """Sincroniza el checkbox 'todos' según el estado individual."""
        all_on = all(v.get() for v in self._ind_check_vars.values())
        self._ind_all_var.set(all_on)

    def _indicator_attrs(self, rn: str) -> dict:
        """Attributes of the run `rn` with the law and thresholds of the indicator (meta_* / pp_*): the same in every case.
        A validation file keeps only a few of them: then they are read, attributes only, from the indicator results file
        next to it (root attr indicator_results_file)."""
        cache = self.__dict__.setdefault("_ind_attrs", {})
        if rn in cache:
            return cache[rn]
        attrs: dict = {}
        for c in self.cases:
            a = c.get("runs", {}).get(rn, {}).get("attrs", {})
            if any(str(k).startswith("meta_") for k in a):
                attrs = dict(a)
                break
        if not attrs:
            try:
                with h5py.File(self.h5_path, "r") as f:
                    name = str(f.attrs.get("indicator_results_file", ""))
                path = os.path.join(os.path.dirname(self.h5_path), name)
                if name and os.path.isfile(path):
                    with h5py.File(path, "r") as f:
                        for g in f:
                            if isinstance(f[g], h5py.Group) and rn in f[g]:
                                attrs = dict(f[g][rn].attrs)
                                break
            except OSError:
                pass
        cache[rn] = attrs
        return attrs

    def _selected_run_filter(self) -> Optional[str]:
        """Retorna None (compatibilidad; la lógica real está en _get_runs_to_show)."""
        return None

    def _get_runs_to_show(self) -> List[str]:
        """Retorna los runs activos según los checkboxes de indicadores."""
        if not hasattr(self, "_ind_check_vars") or not self._ind_check_vars:
            return self._all_runs
        selected_inds = [ind for ind, v in self._ind_check_vars.items() if v.get()]
        if not selected_inds:
            return self._all_runs  # ninguno marcado → todos
        return [r for r in self._all_runs if any(r.startswith(ind) for ind in selected_inds)]

    # ── TREEVIEW ──────────────────────────────────────────────────────────────
    def _build_tree(self) -> None:
        for attr in ("tree", "_sb_y", "_sb_x"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.destroy()
                except Exception:
                    pass

        # Arma columnas: "case" + claves visibles de var_val + t_d por run (indicadores)
        cols = ["case"] + self._visible_keys
        if self.h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND) and self._all_runs:
            with_td = [rn for rn in self._all_runs
                       if any(c.get("runs", {}).get(rn, {}).get("t_d", np.array([])).size for c in self.cases)]
            for rn in (with_td or self._all_runs)[:4]:   # runs that detected something first
                cols += [f"td_{rn}"]
        if self.h5_type == TYPE_MODEL_SNR and self.cases:
            snr_keys = sorted({k for c in self.cases for k in c.get("snr", {})})
            cols += [f"snr_{s}" for s in snr_keys]
        self._cols = cols

        tf = self._tree_frame
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(tf, columns=cols, show="headings",
                                 selectmode="extended")
        for col in cols:
            if col == "case":
                hdr, w = "case", 68
            elif col.startswith("td_"):
                hdr = "t_d:" + col[3:][:10]
                w   = 90
            elif col.startswith("snr_"):
                hdr = "SNR:" + col[4:]
                w   = 90
            else:
                hdr = _col_header(col)
                w   = 90
            self.tree.heading(col, text=hdr,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=w, minwidth=40, anchor=tk.CENTER, stretch=True)

        self._sb_y = ttk.Scrollbar(tf, orient=tk.VERTICAL,   command=self.tree.yview)
        self._sb_x = ttk.Scrollbar(tf, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=self._sb_y.set, xscrollcommand=self._sb_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self._sb_y.grid(row=0, column=1, sticky="ns")
        self._sb_x.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<Double-1>", lambda _: self._plot_active_tab())
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._refresh_inspector(), add="+")   # inspector en vivo

        self._populate_tree(self._filtered_cases())

    def _populate_tree(self, cases_to_show: list) -> None:
        prev_selected = {id(self._iid_to_case[i]) for i in self.tree.selection()
                          if i in self._iid_to_case}
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._iid_to_case.clear()

        to_reselect = []
        for c in cases_to_show:
            row = []
            is_control = (c.get("group", "") == "control")
            for col in self._cols:
                if col == "case":
                    row.append(c["group"])
                elif col.startswith("td_"):
                    rn   = col[3:]
                    td   = c.get("runs", {}).get(rn, {}).get("t_d", np.array([]))
                    row.append(f"{td[0]:.2e} s" if td.size > 0 else "—")
                elif col.startswith("snr_"):
                    sig  = col[4:]
                    snr  = c.get("snr", {}).get(sig, float("nan"))
                    row.append(f"{snr:.2e} dB" if not np.isnan(snr) else "—")
                elif col == "snr_db":
                    raw_v = c.get("var_val", {}).get("snr_db", float("nan"))
                    if is_control or not np.isfinite(float(raw_v) if raw_v is not None else float("nan")):
                        row.append("control")
                    else:
                        row.append(f"{float(raw_v):.2e} dB")
                else:
                    raw_v = c.get("var_val", {}).get(col)
                    if col == "snr_db" and is_control:
                        row.append("control")
                    else:
                        row.append(_fmt_val(raw_v))
            iid = self.tree.insert("", tk.END, values=row)
            is_manual_ctrl = (self._manual_control_group and
                              c.get("group", "") == self._manual_control_group)
            if is_control or is_manual_ctrl:
                self.tree.tag_configure("control_row", foreground="#cc0000", font=("Arial", 9, "bold"))
                self.tree.item(iid, tags=("control_row",))
            self._iid_to_case[iid] = c
            if id(c) in prev_selected:
                to_reselect.append(iid)

        if to_reselect:
            self.tree.selection_set(to_reselect)

    def _sort_by(self, col: str) -> None:
        reverse          = (self._sort_col == col) and not self._sort_rev
        self._sort_col   = col
        self._sort_rev   = reverse
        rows = [(self.tree.set(iid, col), iid) for iid in self.tree.get_children()]

        def _key(item):
            try:
                return (0, float(item[0].rstrip("sBd")))
            except (ValueError, TypeError):
                return (1, str(item[0]))

        rows.sort(key=_key, reverse=reverse)
        for idx, (_, iid) in enumerate(rows):
            self.tree.move(iid, "", idx)
        for c in self._cols:
            hdr = self.tree.heading(c)["text"].rstrip(" ▲▼")
            arrow = (" ▼" if reverse else " ▲") if c == col else ""
            self.tree.heading(c, text=hdr + arrow, command=lambda cc=c: self._sort_by(cc))

    def _filtered_cases(self) -> list:
        query = (self._filter_var_str.get().strip().lower()
                 if self._filter_var_str else "")
        if not query:
            return self.cases
        return [c for c in self.cases if self._case_matches(c, query)]

    def _case_matches(self, c: dict, query: str) -> bool:
        tokens = [c["group"].lower()]
        for v in c.get("var_val", {}).values():
            tokens.append(_fmt_val(v).lower())
        return any(query in t for t in tokens)

    def _apply_filter(self) -> None:
        self._populate_tree(self._filtered_cases())

    def _show_columns_dialog(self) -> None:
        dlg = ColumnsDialog(self.root, self._all_keys, list(self._visible_keys))
        self.root.wait_window(dlg)
        if dlg.result is not None:
            self._visible_keys = list(dlg.result)
            self._build_tree()

    def _select_all(self) -> None:
        self.tree.selection_set(self.tree.get_children())

    def _clear_sel(self) -> None:
        self.tree.selection_remove(self.tree.get_children())

    def _available_plot_fns(self) -> list:
        """Todas las funciones de plot disponibles para este archivo (no solo la del tab activo)."""
        fns = []
        if hasattr(self, "sig_canvas"):
            fns.append(self._plot_signals)
        if hasattr(self, "force_canvas"):
            fns.append(self._plot_forces)
        if hasattr(self, "deflex_canvas"):
            fns.append(self._plot_deflex)
        if hasattr(self, "It_canvas"):
            fns.append(self._plot_It)
        return fns

    def _plot_active_tab(self) -> None:
        """Grafica (reemplazando) la seleccion actual en TODAS las pestañas disponibles."""
        fns = self._available_plot_fns()
        if not fns:
            messagebox.showinfo("No panel", "This format has no plottable panel here "
                                             "(use the ▶ buttons on the slots).", parent=self.root)
            return
        for fn in fns:
            fn()

    def _add_selection_to_plot(self) -> None:
        """Une la seleccion actual con lo ultimo graficado y redibuja en TODAS las pestañas disponibles."""
        new_sel = set(self.tree.selection())
        if not new_sel:
            messagebox.showwarning("No selection", "Select at least one case to add.",
                                   parent=self.root)
            return
        self.tree.selection_set(list(self._plotted_iids | new_sel))
        for fn in self._available_plot_fns():
            fn()

    # ── PANEL CENTRAL ──────────────────────────────────────────────────────────
    def _build_center_panel(self) -> None:
        cf = self.center_frame

        has_signals = self.h5_type in (TYPE_DOE_RESULTS, TYPE_DOE_NOISE,
                                        TYPE_DOE_INDICATOR, TYPE_NOISE_IND)
        has_forces  = self.h5_type == TYPE_DOE_RESULTS
        has_runs    = self.h5_type in (TYPE_DOE_INDICATOR, TYPE_NOISE_IND)
        has_deflex  = self._has_deflex
        has_snr_only = self.h5_type == TYPE_MODEL_SNR

        if has_signals and has_forces and has_runs:
            self._nb = ttk.Notebook(cf)
            self._nb.pack(fill=tk.BOTH, expand=True)
            self._sig_tab   = ttk.Frame(self._nb)
            self._force_tab = ttk.Frame(self._nb)
            self._It_tab    = ttk.Frame(self._nb)
            self._nb.add(self._sig_tab,   text=" Signals ")
            self._nb.add(self._force_tab, text=" Forces ")
            self._nb.add(self._It_tab,    text=" I_t(t) ")
            self._build_signal_canvas(self._sig_tab)
            self._build_force_canvas(self._force_tab)
            self._build_It_canvas(self._It_tab)
            if has_deflex:
                self._deflex_tab = ttk.Frame(self._nb)
                self._nb.add(self._deflex_tab, text=" Out Deflex ")
                self._build_deflex_canvas(self._deflex_tab)

        elif has_signals and has_forces:
            self._nb = ttk.Notebook(cf)
            self._nb.pack(fill=tk.BOTH, expand=True)
            self._sig_tab   = ttk.Frame(self._nb)
            self._force_tab = ttk.Frame(self._nb)
            self._nb.add(self._sig_tab,   text=" Signals ")
            self._nb.add(self._force_tab, text=" Forces ")
            self._build_signal_canvas(self._sig_tab)
            self._build_force_canvas(self._force_tab)
            if has_deflex:
                self._deflex_tab = ttk.Frame(self._nb)
                self._nb.add(self._deflex_tab, text=" Out Deflex ")
                self._build_deflex_canvas(self._deflex_tab)

        elif has_signals and has_runs:
            # Notebook con dos tabs
            self._nb = ttk.Notebook(cf)
            self._nb.pack(fill=tk.BOTH, expand=True)
            self._sig_tab = ttk.Frame(self._nb)
            self._It_tab  = ttk.Frame(self._nb)
            self._nb.add(self._sig_tab, text=" Signals ")
            self._nb.add(self._It_tab,  text=" I_t(t) ")
            self._build_signal_canvas(self._sig_tab)
            self._build_It_canvas(self._It_tab)
        elif has_signals and self.h5_type == TYPE_NOISE_IND:
            # Para el tipo noise_indicator, muestra dos slots verticales de I_t con checkboxes de run
            self._build_noise_indicator_slots(cf)
        elif has_signals:
            self._sig_tab = cf
            self._build_signal_canvas(cf)
        elif has_snr_only:
            ttk.Label(cf, text="Signals not available in doe_model_snr_results.h5.\n"
                                "Use the summary panel for the SNR figures.",
                      foreground="#777777", font=("Arial", 11),
                      anchor=tk.CENTER, justify=tk.CENTER).pack(expand=True)

    def _build_deflex_canvas(self, parent: tk.Frame) -> None:
        """Crea dos subplots embebidos para Out_Deflex (disp y vel sin deflexion estatica)."""
        self.deflex_fig  = Figure(constrained_layout=True)
        self.ax_deflex_d = self.deflex_fig.add_subplot(2, 1, 1)
        self.ax_deflex_v = self.deflex_fig.add_subplot(2, 1, 2, sharex=self.ax_deflex_d)
        self._init_deflex_axes()

        self.deflex_canvas = FigureCanvasTkAgg(self.deflex_fig, master=parent)
        self.deflex_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.deflex_toolbar = NavigationToolbar2Tk(self.deflex_canvas, parent, pack_toolbar=False)
        self.deflex_toolbar.update()
        self.deflex_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.deflex_canvas.draw()

    def _init_deflex_axes(self) -> None:
        self.ax_deflex_d.set_ylabel(SIGNAL_YLABELS.get("Axial_disp", "Axial disp (out deflex)"), fontsize=14)
        self.ax_deflex_d.grid(False)
        self.ax_deflex_d.tick_params(labelbottom=False, labelsize=12)
        self.ax_deflex_d.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_deflex_v.set_ylabel(SIGNAL_YLABELS.get("Axial_vel", "Axial vel (out deflex)"), fontsize=14)
        self.ax_deflex_v.set_xlabel("Time (s)", fontsize=14)
        self.ax_deflex_v.grid(False)
        self.ax_deflex_v.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.deflex_fig.suptitle("Select cases and press  Plot ▶")

    def _build_force_canvas(self, parent: tk.Frame) -> None:
        """Crea tres subplots embebidos para res_R_p (Fx, Fy, Fz)."""
        self.force_fig = Figure(constrained_layout=True)
        self.ax_force_1 = self.force_fig.add_subplot(3, 1, 1)
        self.ax_force_2 = self.force_fig.add_subplot(3, 1, 2, sharex=self.ax_force_1)
        self.ax_force_3 = self.force_fig.add_subplot(3, 1, 3, sharex=self.ax_force_1)
        self._init_force_axes()

        self.force_canvas = FigureCanvasTkAgg(self.force_fig, master=parent)
        self.force_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.force_toolbar = NavigationToolbar2Tk(self.force_canvas, parent, pack_toolbar=False)
        self.force_toolbar.update()
        self.force_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.force_canvas.draw()

    def _init_force_axes(self) -> None:
        labels = ["Fx", "Fy", "Fz"]
        axes = [self.ax_force_1, self.ax_force_2, self.ax_force_3]
        for i, ax in enumerate(axes):
            ax.set_ylabel(labels[i], fontsize=14)
            ax.grid(False)
            ax.tick_params(labelsize=12)
            if i < 2:
                ax.tick_params(labelbottom=False)
        self.ax_force_3.set_xlabel("Time (s)", fontsize=14)
        self.force_fig.suptitle("Select cases and press  Plot ▶")

    def _sig_axes(self) -> list:
        """[(señal, Axes, nombre de target)] de la pestaña Signals; la aceleración solo si el archivo la trae."""
        out = [("Axial_disp", self.ax_disp, "Señales: disp"), ("Axial_vel", self.ax_vel, "Señales: vel")]
        if getattr(self, "ax_acc", None) is not None:
            out.append(("Axial_acc", self.ax_acc, "Señales: acc"))
        return out

    def _build_signal_canvas(self, parent: tk.Frame) -> None:
        """Crea los subplots embebidos: Axial_disp + Axial_vel (+ Axial_acc si algún caso la trae)."""
        n = 3 if getattr(self, "_has_acc", False) else 2
        self.sig_fig     = Figure(constrained_layout=True)
        self.ax_disp     = self.sig_fig.add_subplot(n, 1, 1)
        self.ax_vel      = self.sig_fig.add_subplot(n, 1, 2, sharex=self.ax_disp)
        self.ax_acc      = self.sig_fig.add_subplot(n, 1, 3, sharex=self.ax_disp) if n == 3 else None
        self._init_signal_axes()

        self.sig_canvas = FigureCanvasTkAgg(self.sig_fig, master=parent)
        self.sig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.sig_toolbar = NavigationToolbar2Tk(self.sig_canvas, parent, pack_toolbar=False)
        self.sig_toolbar.update()
        self.sig_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.sig_canvas.draw()

    def _build_It_canvas(self, parent: tk.Frame) -> None:
        """Crea el subplot de I_t(t) embebido."""
        self.It_fig = Figure(constrained_layout=True)
        self.ax_It  = self.It_fig.add_subplot(1, 1, 1)
        self._init_It_axis()

        self.It_canvas = FigureCanvasTkAgg(self.It_fig, master=parent)
        self.It_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.It_toolbar = NavigationToolbar2Tk(self.It_canvas, parent, pack_toolbar=False)
        self.It_toolbar.update()
        self.It_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.It_canvas.draw()

    def _build_noise_indicator_slots(self, parent: tk.Frame) -> None:
        """Crea dos subfiguras verticales, cada una con selector de indicador
        y checkboxes multi-run para hacer overlay de I_t(t).
        """
        top_zone = ttk.LabelFrame(parent, text=" I_t - Top ", padding=4)
        bot_zone = ttk.LabelFrame(parent, text=" I_t - Bottom ", padding=4)
        top_zone.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        bot_zone.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Controles y canvas para cada zona
        def make_slot(zone, slot):
            ctrl = ttk.Frame(zone)
            ctrl.pack(fill=tk.X)
            ttk.Label(ctrl, text="Indicator:").pack(side=tk.LEFT)
            inds = self._extract_indicators()
            ind_var = tk.StringVar(value=inds[0] if inds else "")
            ind_combo = ttk.Combobox(ctrl, values=inds, textvariable=ind_var, state="readonly", width=20)
            ind_combo.pack(side=tk.LEFT, padx=4)
            ttk.Button(ctrl, text="Refresh runs", command=lambda: self._populate_run_checks(slot)).pack(side=tk.LEFT, padx=4)
            ttk.Button(ctrl, text="Plot ▶", command=lambda: self._plot_It_slot(slot)).pack(side=tk.LEFT, padx=4)

            # Frame con scroll para los checkboxes de run
            box_frame = ttk.Frame(zone)
            box_frame.pack(fill=tk.BOTH, expand=False, pady=(4,2))
            canvas = tk.Canvas(box_frame, height=120)
            sb = ttk.Scrollbar(box_frame, orient=tk.VERTICAL, command=canvas.yview)
            inner = ttk.Frame(canvas)
            inner_id = canvas.create_window((0,0), window=inner, anchor='nw')
            canvas.configure(yscrollcommand=sb.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.LEFT, fill=tk.Y)

            inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

            # Canvas para la figura
            fig_frame = ttk.Frame(zone)
            fig_frame.pack(fill=tk.BOTH, expand=True)
            fig = Figure(constrained_layout=True)
            ax = fig.add_subplot(1,1,1)
            canvas_fig = FigureCanvasTkAgg(fig, master=fig_frame)
            canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas_fig, fig_frame, pack_toolbar=False)
            toolbar.update(); toolbar.pack(fill=tk.X)

            # store slot state
            self.__dict__[f"_{slot}_ind_combo"] = ind_combo
            self.__dict__[f"_{slot}_run_frame"] = inner
            self.__dict__[f"_{slot}_fig"] = fig
            self.__dict__[f"_{slot}_ax"] = ax
            self.__dict__[f"_{slot}_canvas"] = canvas_fig

        make_slot(top_zone, "topslot")
        make_slot(bot_zone, "botslot")

        # populate runs initially
        self._populate_run_checks("topslot")
        self._populate_run_checks("botslot")

    def _populate_run_checks(self, slot: str) -> None:
        frame = self.__dict__.get(f"_{slot}_run_frame")
        combo = self.__dict__.get(f"_{slot}_ind_combo")
        if frame is None or combo is None:
            return
        # clear
        for w in list(frame.winfo_children()):
            w.destroy()
        ind = combo.get()
        runs = [r for r in self._all_runs if r.startswith(ind)] if ind else list(self._all_runs)
        self.__dict__[f"_{slot}_run_vars"] = {}
        for r in runs:
            var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(frame, text=r, variable=var)
            chk.pack(anchor=tk.W)
            self.__dict__[f"_{slot}_run_vars"] [r] = var

    def _plot_It_slot(self, slot: str) -> None:
        run_vars = self.__dict__.get(f"_{slot}_run_vars", {})
        selected_runs = [r for r, v in run_vars.items() if v.get()]
        if not selected_runs:
            messagebox.showwarning("No selection", "Select at least one run.", parent=self.root)
            return
        ax = self.__dict__.get(f"_{slot}_ax")
        fig = self.__dict__.get(f"_{slot}_fig")
        canvas_fig = self.__dict__.get(f"_{slot}_canvas")
        if ax is None or fig is None or canvas_fig is None:
            return
        ax.cla()
        # color map
        vals = [c.get("label_val", float("nan")) for c in self.cases]
        vals_f = [v for v in vals if np.isfinite(v)]
        cmap_It = matplotlib.colormaps["viridis"]
        norm_It = mcolors.Normalize(vmin=min(vals_f) if vals_f else 0, vmax=max(vals_f) if vals_f else 1)

        for c in self.cases:
            pv = c.get("label_val", float("nan"))
            is_ctrl = (c.get("group", "") == "control")
            if is_ctrl:
                color = (0.85, 0.05, 0.05)   # rojo siempre para control
            elif np.isfinite(pv):
                color = cmap_It(norm_It(pv))
            else:
                color = (0.5, 0.5, 0.5)
            for rn in selected_runs:
                run_data = c.get("runs", {}).get(rn)
                if run_data is None:
                    continue
                t = run_data.get("t", np.array([]))
                I_t = run_data.get("I_t", np.array([]))
                if t.size == 0 or I_t.size == 0:
                    continue
                lbl = f"control|{rn}" if is_ctrl else f"{c.get('label_val','?'):.1f}dB|{rn}"
                lw  = 2.2 if is_ctrl else 1.6
                ax.plot(t[::_IND_DECIMATE], I_t[::_IND_DECIMATE], color=color, lw=lw,
                        alpha=1.0 if is_ctrl else 0.9, label=lbl, zorder=5 if is_ctrl else 3)
                td = run_data.get("t_d", np.array([]))
                if td.size>0:
                    ax.axvline(td[0], color=color, lw=2.0, linestyle="--", zorder=6 if is_ctrl else 2)

        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$I(t)$")
        ax.set_yscale(_it_plot_yscale(selected_runs))
        # ax.grid(True, linestyle=":", color="#d0d0d0", linewidth=0.6, alpha=0.7)
        ax.grid(False)
        fig.tight_layout()
        canvas_fig.draw()

    def _init_signal_axes(self) -> None:
        axes = self._sig_axes()
        for k, (sig, ax, _) in enumerate(axes):
            last = k == len(axes) - 1
            ax.set_ylabel(SIGNAL_YLABELS.get(sig) or _channel_ylabel(sig), fontsize=14)
            ax.grid(False)
            ax.tick_params(labelbottom=last, labelsize=12)
            if last:
                ax.set_xlabel("Time (s)", fontsize=14)
        self.sig_fig.suptitle("Select cases and press  Plot ▶")

    def _init_It_axis(self) -> None:
        self.ax_It.set_xlabel(r"$t$ (s)", fontsize=14)
        self.ax_It.set_ylabel(r"$I(t)$", fontsize=14)
        # self.ax_It.grid(True, linestyle=":", color="#bfbfbf", linewidth=0.6, alpha=0.6)
        self.ax_It.grid(False)
        self.ax_It.tick_params(labelsize=12)
        self.It_fig.suptitle("Select cases and press  Plot ▶")

    # ── PLOT DE SEÑALES ───────────────────────────────────────────────────────────
    def _plot_signals(self) -> None:
        if not hasattr(self, "sig_canvas"):
            messagebox.showinfo("No panel", "This format has no signals panel.",
                                parent=self.root)
            return
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning("No selection",
                                   "Select at least one case in the table.",
                                   parent=self.root)
            return
        selected = [self._iid_to_case[i] for i in sel_iids if i in self._iid_to_case]
        if not selected:
            return

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        for _, ax, _n in self._sig_axes():
            ax.cla()

        use_fixed = getattr(self, "_persistent_color_var", None)
        use_fixed = use_fixed.get() if use_fixed is not None else True

        # Colormap dinámico — se recalcula solo sobre los casos seleccionados
        if not use_fixed:
            dyn_vals = [c["label_val"] for c in selected
                        if c.get("group", "") != "control" and np.isfinite(c.get("label_val", float("nan")))]
            if len(dyn_vals) >= 2:
                dyn_norm = mcolors.Normalize(vmin=min(dyn_vals), vmax=max(dyn_vals))
            else:
                dyn_norm = mcolors.Normalize(vmin=0, vmax=1)
            dyn_cmap = matplotlib.colormaps["viridis"]
            dyn_idx  = {id(c): i for i, c in enumerate(
                [c for c in selected if c.get("group", "") != "control"])}

        _invert = getattr(self, "_invert_order_var", None)
        _invert = _invert.get() if _invert is not None else False
        _lw_nc  = self._sig_lw_var.get()  if hasattr(self, "_sig_lw_var")  else 0.9
        _alp_nc = self._sig_alpha_var.get() if hasattr(self, "_sig_alpha_var") else 0.5
        # No-control primero (zorder incremental), control siempre al final (zorder 100)
        _non_ctrl = [c for c in selected if not self._is_control(c)]
        _ctrl_lst = [c for c in selected if self._is_control(c)]
        _non_ctrl = list(reversed(_non_ctrl)) if _invert else _non_ctrl
        _plot_order = _non_ctrl + _ctrl_lst
        _cbar_marks = []
        for ci, c in enumerate(_plot_order):
            is_ctrl  = self._is_control(c)
            if is_ctrl:
                clr = "red"
            elif use_fixed:
                clr = c.get("_color", color_azul)
            else:
                lv_dyn = c.get("label_val", float("nan"))
                clr = dyn_cmap(dyn_norm(lv_dyn)) if np.isfinite(lv_dyn) else color_azul
            lk   = c.get("label_key", "")
            lv   = c.get("label_val", float("nan"))
            _cbar_marks.append((lv, clr))
            lbl = "control" if is_ctrl else _case_legend(c, lk, lv)
            lw   = 2.2   if is_ctrl else _lw_nc
            _ctrl_zo  = int(self._ctrl_zo_var.get())  if hasattr(self, "_ctrl_zo_var")  else 100
            _ctrl_alp = self._ctrl_alpha_var.get() if hasattr(self, "_ctrl_alpha_var") else 1.0
            zo    = _ctrl_zo  if is_ctrl else (3 + ci)
            alpha = _ctrl_alp if is_ctrl else _alp_nc
            for sig, ax, _n in self._sig_axes():
                data = c.get("signals", {}).get(sig) or c.get(sig)
                if data is None:
                    continue
                t, y = data
                ax.plot(t[::DECIMATE], y[::DECIMATE], color=clr, lw=lw,
                        alpha=alpha, label=lbl,
                        zorder=zo, rasterized=True)
            _draw_truth_marks([ax for _, ax, _n in self._sig_axes()], c, clr, shade=len(selected) == 1)

        axes_sig = self._sig_axes()
        for k, (sig, ax, _n) in enumerate(axes_sig):
            last = k == len(axes_sig) - 1
            ax.set_ylabel(SIGNAL_YLABELS.get(sig) or _channel_ylabel(sig), fontsize=14)
            ax.grid(False)
            ax.tick_params(labelbottom=last, labelsize=12)
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            if last:
                ax.set_xlabel("Time (s)", fontsize=14)
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

        n = len(selected)
        lk_disp = _col_header(selected[0]["label_key"]) if selected else ""
        if n <= 12:
            for _, ax, _n in axes_sig:
                ax.legend(fontsize=9, framealpha=0.7, loc="upper left")

        # Colorbar horizontal — escala global (fijo) o sobre seleccionados (dinámico)
        if use_fixed:
            cbar_cases = self.cases
        else:
            cbar_cases = selected
        cb_vals = [c.get("label_val", float("nan"))
                   for c in cbar_cases if c.get("group", "") != "control"]
        finite_vals = [v for v in cb_vals if np.isfinite(v)]
        if len(finite_vals) >= 2:
            cmap_s = matplotlib.colormaps["viridis"]
            norm_s = mcolors.Normalize(vmin=min(finite_vals), vmax=max(finite_vals))
            sm = cm.ScalarMappable(cmap=cmap_s, norm=norm_s)
            sm.set_array([])
            if self._cbar is not None:
                try:
                    self._cbar.remove()
                except Exception:
                    pass
            self._cbar = self.sig_fig.colorbar(
                sm, ax=[ax for _, ax, _n in self._sig_axes()],
                label=lk_disp, shrink=0.85,
                orientation="horizontal", pad=0.08,
            )
            self._cbar.formatter = mticker.FormatStrFormatter("%.3g")
            self._cbar.update_ticks()
            self._mark_values_on_colorbar(self._cbar, _cbar_marks)

        self.sig_fig.suptitle(f"{lk_disp}  —  {n} case(s)")
        self._draw_reference_lines({n: ax for _, ax, n in self._sig_axes()})
        self.sig_canvas.draw()
        self.sig_toolbar.update()  # refresca "Home" a la vista actual (con las lineas nuevas incluidas)
        self._plotted_iids = set(self.tree.selection())

    def _plot_forces(self) -> None:
        if not hasattr(self, "force_canvas"):
            messagebox.showinfo("No panel", "This format has no forces panel.", parent=self.root)
            return
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning("No selection", "Select at least one case in the table.", parent=self.root)
            return
        selected = [self._iid_to_case[i] for i in sel_iids if i in self._iid_to_case]
        if not selected:
            return

        for ax in (self.ax_force_1, self.ax_force_2, self.ax_force_3):
            ax.cla()

        use_fixed = self._persistent_color_var.get() if hasattr(self, "_persistent_color_var") else True
        _invert = getattr(self, "_invert_order_var", None)
        _invert = _invert.get() if _invert is not None else False
        _alp_nc = self._sig_alpha_var.get() if hasattr(self, "_sig_alpha_var") else 0.5
        _lw_nc = self._sig_lw_var.get() if hasattr(self, "_sig_lw_var") else 0.9
        _ctrl_alp = self._ctrl_alpha_var.get() if hasattr(self, "_ctrl_alpha_var") else 1.0
        _ctrl_zo = int(self._ctrl_zo_var.get()) if hasattr(self, "_ctrl_zo_var") else 100

        non_ctrl = [c for c in selected if not self._is_control(c)]
        ctrl_lst = [c for c in selected if self._is_control(c)]
        non_ctrl = list(reversed(non_ctrl)) if _invert else non_ctrl
        plot_order = non_ctrl + ctrl_lst

        if not use_fixed:
            dyn_vals = [c.get("label_val", float("nan")) for c in non_ctrl if np.isfinite(c.get("label_val", float("nan")))]
            if len(dyn_vals) >= 2:
                dyn_norm = mcolors.Normalize(vmin=min(dyn_vals), vmax=max(dyn_vals))
            else:
                dyn_norm = mcolors.Normalize(vmin=0, vmax=1)
            dyn_cmap = matplotlib.colormaps["viridis"]

        _cbar_marks_f = []
        for ci, c in enumerate(plot_order):
            data = c.get("forces", {}).get("res_R_p")
            if data is None:
                continue
            t, y = data
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr[:, np.newaxis]
            if y_arr.shape[1] < 3:
                continue
            is_ctrl = self._is_control(c)
            if is_ctrl:
                color = "red"
            elif use_fixed:
                color = c.get("_color", color_azul)
            else:
                lv = c.get("label_val", float("nan"))
                color = dyn_cmap(dyn_norm(lv)) if np.isfinite(lv) else color_azul
            lk = c.get("label_key", "")
            lv = c.get("label_val", float("nan"))
            _cbar_marks_f.append((lv, color))
            label = "control" if is_ctrl else (
                f"{_col_header(lk)}={lv:.3g}" if np.isfinite(lv) else c.get("group", "?")
            )
            lw = 2.2 if is_ctrl else _lw_nc
            alpha = _ctrl_alp if is_ctrl else _alp_nc
            zorder = _ctrl_zo if is_ctrl else (3 + ci)
            self.ax_force_1.plot(t[::DECIMATE], y_arr[::DECIMATE, 0], color=color, lw=lw, alpha=alpha, label=label, zorder=zorder)
            self.ax_force_2.plot(t[::DECIMATE], y_arr[::DECIMATE, 1], color=color, lw=lw, alpha=alpha, label=label, zorder=zorder)
            self.ax_force_3.plot(t[::DECIMATE], y_arr[::DECIMATE, 2], color=color, lw=lw, alpha=alpha, label=label, zorder=zorder)

        self.ax_force_1.tick_params(labelbottom=False)
        self.ax_force_2.tick_params(labelbottom=False)
        self.ax_force_1.legend(fontsize=8, framealpha=0.7, loc="upper left")
        self.ax_force_2.legend(fontsize=8, framealpha=0.7, loc="upper left")
        self.ax_force_3.legend(fontsize=8, framealpha=0.7, loc="upper left")

        # Colorbar
        lk_f = _col_header(selected[0]["label_key"]) if selected else ""
        if use_fixed:
            cbar_cases_f = self.cases
        else:
            cbar_cases_f = selected
        cb_vals_f = [c.get("label_val", float("nan")) for c in cbar_cases_f if c.get("group", "") != "control"]
        finite_f  = [v for v in cb_vals_f if np.isfinite(v)]
        if self._force_cbar is not None:
            try:
                self._force_cbar.remove()
            except Exception:
                pass
            self._force_cbar = None
        if len(finite_f) >= 2:
            sm_f = cm.ScalarMappable(cmap=matplotlib.colormaps["viridis"],
                                     norm=mcolors.Normalize(vmin=min(finite_f), vmax=max(finite_f)))
            sm_f.set_array([])
            self._force_cbar = self.force_fig.colorbar(
                sm_f, ax=[self.ax_force_1, self.ax_force_2, self.ax_force_3],
                label=lk_f, shrink=0.85, orientation="horizontal", pad=0.08)
            self._force_cbar.formatter = mticker.FormatStrFormatter("%.3g")
            self._force_cbar.update_ticks()
            self._mark_values_on_colorbar(self._force_cbar, _cbar_marks_f)

        self.force_fig.suptitle(f"res_R_p — {len(selected)} case(s)")
        self._draw_reference_lines({"Fuerzas: F1": self.ax_force_1, "Fuerzas: F2": self.ax_force_2,
                                    "Fuerzas: F3": self.ax_force_3})
        self.force_canvas.draw()
        self.force_toolbar.update()
        self._plotted_iids = set(self.tree.selection())

    def _plot_deflex(self) -> None:
        if not hasattr(self, "deflex_canvas"):
            messagebox.showinfo("No panel", "This file has no Out_Deflex data.", parent=self.root)
            return
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning("No selection", "Select at least one case in the table.", parent=self.root)
            return
        selected = [self._iid_to_case[i] for i in sel_iids if i in self._iid_to_case]
        if not selected:
            return

        self.ax_deflex_d.cla()
        self.ax_deflex_v.cla()

        use_fixed = self._persistent_color_var.get() if hasattr(self, "_persistent_color_var") else True
        _invert   = getattr(self, "_invert_order_var", None)
        _invert   = _invert.get() if _invert is not None else False
        _alp_nc   = self._sig_alpha_var.get()  if hasattr(self, "_sig_alpha_var")  else 0.5
        _lw_nc    = self._sig_lw_var.get()     if hasattr(self, "_sig_lw_var")    else 0.9
        _ctrl_alp = self._ctrl_alpha_var.get() if hasattr(self, "_ctrl_alpha_var") else 1.0
        _ctrl_zo  = int(self._ctrl_zo_var.get()) if hasattr(self, "_ctrl_zo_var") else 100

        non_ctrl   = [c for c in selected if not self._is_control(c)]
        ctrl_lst   = [c for c in selected if self._is_control(c)]
        non_ctrl   = list(reversed(non_ctrl)) if _invert else non_ctrl
        plot_order = non_ctrl + ctrl_lst

        if not use_fixed:
            dyn_vals = [c.get("label_val", float("nan")) for c in non_ctrl
                        if np.isfinite(c.get("label_val", float("nan")))]
            if len(dyn_vals) >= 2:
                dyn_norm = mcolors.Normalize(vmin=min(dyn_vals), vmax=max(dyn_vals))
            else:
                dyn_norm = mcolors.Normalize(vmin=0, vmax=1)
            dyn_cmap = matplotlib.colormaps["viridis"]

        _disp_key = "Axial_disp_out_deflex"
        _vel_key  = "Axial_vel_out_deflex"

        _cbar_marks_d = []
        for ci, c in enumerate(plot_order):
            od = c.get("out_deflex", {})
            is_ctrl = self._is_control(c)
            color   = "red" if is_ctrl else (
                c.get("_color", color_azul) if use_fixed else
                (dyn_cmap(dyn_norm(c["label_val"])) if np.isfinite(c.get("label_val", float("nan"))) else color_azul)
            )
            lk  = c.get("label_key", "")
            lv  = c.get("label_val", float("nan"))
            _cbar_marks_d.append((lv, color))
            lbl = "control" if is_ctrl else (
                f"{_col_header(lk)}={lv:.3g}" if np.isfinite(lv) else c.get("group", "?")
            )
            lw    = 2.2 if is_ctrl else _lw_nc
            alpha = _ctrl_alp if is_ctrl else _alp_nc
            zo    = _ctrl_zo  if is_ctrl else (3 + ci)

            for sig, ax in ((_disp_key, self.ax_deflex_d), (_vel_key, self.ax_deflex_v)):
                data = od.get(sig)
                if data is None:
                    continue
                t, y = data
                ax.plot(t[::DECIMATE], y[::DECIMATE], color=color, lw=lw,
                        alpha=alpha, label=lbl, zorder=zo, rasterized=True)

        self.ax_deflex_d.set_ylabel(SIGNAL_YLABELS.get("Axial_disp", "Axial disp (out deflex)"), fontsize=14)
        self.ax_deflex_d.tick_params(labelbottom=False, labelsize=12)
        self.ax_deflex_d.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_deflex_d.grid(False)
        self.ax_deflex_v.set_ylabel(SIGNAL_YLABELS.get("Axial_vel", "Axial vel (out deflex)"), fontsize=14)
        self.ax_deflex_v.set_xlabel("Time (s)", fontsize=14)
        self.ax_deflex_v.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.ax_deflex_v.grid(False)

        n = len(selected)
        lk_disp = _col_header(selected[0]["label_key"]) if selected else ""
        if n <= 12:
            self.ax_deflex_d.legend(fontsize=9, framealpha=0.7, loc="upper left")
            self.ax_deflex_v.legend(fontsize=9, framealpha=0.7, loc="upper left")

        # Colorbar
        if use_fixed:
            cbar_cases_d = self.cases
        else:
            cbar_cases_d = selected
        cb_vals_d = [c.get("label_val", float("nan")) for c in cbar_cases_d if c.get("group", "") != "control"]
        finite_d  = [v for v in cb_vals_d if np.isfinite(v)]
        if self._deflex_cbar is not None:
            try:
                self._deflex_cbar.remove()
            except Exception:
                pass
            self._deflex_cbar = None
        if len(finite_d) >= 2:
            sm_d = cm.ScalarMappable(cmap=matplotlib.colormaps["viridis"],
                                     norm=mcolors.Normalize(vmin=min(finite_d), vmax=max(finite_d)))
            sm_d.set_array([])
            self._deflex_cbar = self.deflex_fig.colorbar(
                sm_d, ax=[self.ax_deflex_d, self.ax_deflex_v],
                label=lk_disp, shrink=0.85, orientation="horizontal", pad=0.08)
            self._deflex_cbar.formatter = mticker.FormatStrFormatter("%.3g")
            self._deflex_cbar.update_ticks()
            self._mark_values_on_colorbar(self._deflex_cbar, _cbar_marks_d)

        self.deflex_fig.suptitle(f"Out Deflex  —  {lk_disp}  —  {n} case(s)")
        self._draw_reference_lines({"Deflex: disp": self.ax_deflex_d, "Deflex: vel": self.ax_deflex_v})
        self.deflex_canvas.draw()
        self.deflex_toolbar.update()
        self._plotted_iids = set(self.tree.selection())

    def _clear_signal_plot(self) -> None:
        if not hasattr(self, "sig_canvas"):
            return
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        for _, ax, _n in self._sig_axes():
            ax.cla()
        self._init_signal_axes()
        self.sig_canvas.draw()
        self._plotted_iids.clear()

    # ── PLOT DE I_t ──────────────────────────────────────────────────────────────
    def _plot_It(self) -> None:
        if not hasattr(self, "It_canvas"):
            return
        sel_iids = self.tree.selection()
        if not sel_iids:
            messagebox.showwarning("No selection",
                                   "Select at least one case in the table.",
                                   parent=self.root)
            return
        selected = [self._iid_to_case[i] for i in sel_iids if i in self._iid_to_case]
        if not selected:
            return

        run_filter = self._selected_run_filter()
        td_key     = "t_d"

        # Filtrar runs según indicadores seleccionados
        runs_to_show = self._get_runs_to_show() if hasattr(self, "_get_runs_to_show") else self._all_runs
        if not runs_to_show:
            runs_to_show = self._all_runs

        vals = [c.get("label_val", float("nan")) for c in selected]

        if self._It_cbar is not None:
            try:
                self._It_cbar.remove()
            except Exception:
                pass
            self._It_cbar = None
        self.ax_It.cla()

        lk_disp = _col_header(selected[0]["label_key"]) if selected else ""
        plotted = td_drawn = False
        cbar_marks, deltas = [], []

        # Decide coloring strategy:
        #   · varios indicadores  → color por indicador (tab10)
        #   · un solo indicador   → color por caso (viridis, fijo o dinámico según toggle)
        n_indicators = len({_run_indicator_prefix(rn) for rn in runs_to_show})
        color_by_case = (n_indicators == 1)
        use_fixed = self._persistent_color_var.get() if hasattr(self, "_persistent_color_var") else True

        # Paleta dinámica (sólo casos no-control) para color_by_case + no fijo
        non_ctrl_cases = [c for c in selected if not self._is_control(c)]
        if not use_fixed and non_ctrl_cases:
            _dyn_cmap = matplotlib.colormaps.get_cmap("viridis")
            _dyn_colors = {id(c): _dyn_cmap(i / max(len(non_ctrl_cases) - 1, 1))
                           for i, c in enumerate(non_ctrl_cases)}
        else:
            _dyn_colors = {}

        _it_alp_nc  = self._sig_alpha_var.get()  if hasattr(self, "_sig_alpha_var")  else 0.5
        _it_lw_nc   = self._sig_lw_var.get()      if hasattr(self, "_sig_lw_var")      else 0.9
        _it_ctrl_alp = self._ctrl_alpha_var.get() if hasattr(self, "_ctrl_alpha_var") else 1.0
        _it_ctrl_zo  = int(self._ctrl_zo_var.get()) if hasattr(self, "_ctrl_zo_var")  else 100
        _invert_it = getattr(self, "_invert_order_var", None)
        _invert_it = _invert_it.get() if _invert_it is not None else False
        # No-control primero, control al final (siempre encima)
        _nc_pairs   = [(c, v) for c, v in zip(selected, vals) if not self._is_control(c)]
        _ctrl_pairs = [(c, v) for c, v in zip(selected, vals) if self._is_control(c)]
        _nc_pairs   = list(reversed(_nc_pairs)) if _invert_it else _nc_pairs
        _it_order   = _nc_pairs + _ctrl_pairs
        n_sel = max(len(selected), 1)
        for ci, (c, pv) in enumerate(_it_order):
            is_ctrl = self._is_control(c)
            case_alpha = _it_alp_nc
            for rn in runs_to_show:
                run_data = c.get("runs", {}).get(rn)
                if run_data is None:
                    continue
                t   = run_data["t"]
                I_t = run_data["I_t"]
                if t.size == 0 or I_t.size == 0:
                    continue
                if is_ctrl:
                    if color_by_case:
                        color = "red"   # un indicador → control siempre rojo
                    else:
                        ind_prefix = _run_indicator_prefix(rn)
                        ind_color  = self._ind_color_map.get(ind_prefix, None) if hasattr(self, "_ind_color_map") else None
                        color = ind_color if ind_color is not None else "red"
                elif color_by_case:
                    # Respeta toggle fijo/dinámico igual que señales
                    if use_fixed:
                        color = c.get("_color", (0.5, 0.5, 0.5))
                    else:
                        color = _dyn_colors.get(id(c), c.get("_color", (0.5, 0.5, 0.5)))
                else:
                    ind_prefix = _run_indicator_prefix(rn)
                    color = self._ind_color_map.get(ind_prefix, (0.5, 0.5, 0.5)) if hasattr(self, "_ind_color_map") else c.get("_color", (0.5, 0.5, 0.5))
                lv_str = "control" if is_ctrl else (f"{pv:.3g}" if np.isfinite(pv) else "?")
                lw = 2.2 if is_ctrl else _it_lw_nc
                delay = _run_delay(run_data)   # validation files: first detection - t_onset
                if delay is not None:
                    deltas.append(delay)
                if color_by_case and not is_ctrl:
                    cbar_marks.append((pv, color))
                self.ax_It.plot(t[::_IND_DECIMATE], I_t[::_IND_DECIMATE],
                                color=color, lw=lw,
                                alpha=_it_ctrl_alp if is_ctrl else case_alpha,
                                label=(f"{_case_legend(c, c.get('label_key', ''), pv)} | {rn}" if c.get("ramp")
                                       else f"{lk_disp}={lv_str} | {rn}")
                                      + ("" if delay is None else f"   Δ = {delay:+.2f} s"),
                                zorder=_it_ctrl_zo if is_ctrl else (3 + ci),
                                rasterized=True)
                # t_d vline
                for key, style in (("t_d", "--"),):
                    td = run_data.get(key, np.array([]))
                    if td.size > 0:
                        td_drawn = True
                        self.ax_It.axvline(td[0], color=color, lw=2.2,
                                           linestyle=style,
                                           alpha=_it_ctrl_alp if is_ctrl else case_alpha,
                                           zorder=6)
                        if t.size > 1:
                            y_td = float(np.interp(td[0], t, I_t))
                            self.ax_It.scatter([td[0]], [y_td], s=24, color=color,
                                               edgecolor="black", linewidths=0.3, zorder=7)
                plotted = True
            _draw_truth_marks([self.ax_It], c, c.get("_color", "k"), shade=len(selected) == 1)

        self.ax_It.set_xlabel(r"$t$ (s)", fontsize=14)
        self.ax_It.set_ylabel(r"$I(t)$", fontsize=14)
        # self.ax_It.grid(False, linestyle="--", alpha=0.3)
        self.ax_It.grid(False)
        self.ax_It.set_yscale(_it_plot_yscale(runs_to_show))

        # decision limits of each indicator (the same for every case: it learns once), dash-dot, in its colour
        lim_drawn = False
        for rn in runs_to_show:
            if not any(rn in c.get("runs", {}) for c in selected):
                continue
            lcol = "k" if color_by_case else self._ind_color_map.get(_run_indicator_prefix(rn), "k")
            for v in _indicator_limits(rn, self._indicator_attrs(rn)):
                if self.ax_It.get_yscale() == "log" and v <= 0:
                    continue
                self.ax_It.axhline(v, color=lcol, ls="-.", lw=1.4, alpha=0.9, zorder=2)
                lim_drawn = True

        run_txt = run_filter or "(all)"
        self.ax_It.set_title(f"I_t(t)  —  run: {run_txt}", fontsize=13)
        self.It_fig.suptitle(f"{lk_disp}  —  {len(selected)} case(s)")   # as the signals panels (replaces the start-up text)

        # what the vertical lines are (same colour as the curve they belong to): dashed + dot = first detection t_d of
        # that indicator; dotted = t_onset, where the ground truth of a ramp turns unstable (only when several cases are
        # shown: with one case that line already carries its own label)
        from matplotlib.lines import Line2D
        proxies = []
        if td_drawn:
            proxies.append(Line2D([0], [0], color="0.35", ls="--", lw=2.2, marker="o", ms=4,
                                  label=r"$t_d$: first detection of the indicator"))
        if len(selected) > 1 and any(_case_onset(c) is not None for c in selected):
            proxies.append(Line2D([0], [0], color="0.35", ls=":", lw=2.2,
                                  label=r"$t_{onset}$: truth turns unstable (labelling amplitude reached)"))
        if lim_drawn:
            proxies.append(Line2D([0], [0], color="0.35", ls="-.", lw=1.4, label="detection limit of the indicator"))
        if deltas:
            proxies.append(Line2D([0], [0], color="none", label=r"$\Delta = t_d - t_{onset}$ (< 0: detected before the amplitude)"))
        n = len(selected) * len(runs_to_show)
        if n <= 10 and plotted:
            handles, labels = self.ax_It.get_legend_handles_labels()
            self.ax_It.legend(handles=handles + proxies, fontsize=14, loc="upper left")
        elif proxies:   # too many curves for a legend: still say what the vertical lines mean
            self.ax_It.legend(handles=proxies, fontsize=11, loc="upper left")

        # colour bar of the cases (as the signals panels): only when the curves are coloured by case (one indicator)
        if color_by_case and plotted:
            cbar_cases = self.cases if use_fixed else selected
            finite = [v for v in (c.get("label_val", float("nan")) for c in cbar_cases if not self._is_control(c))
                      if np.isfinite(v)]
            if len(finite) >= 2:
                sm = cm.ScalarMappable(cmap=matplotlib.colormaps["viridis"],
                                       norm=mcolors.Normalize(vmin=min(finite), vmax=max(finite)))
                sm.set_array([])
                self._It_cbar = self.It_fig.colorbar(sm, ax=self.ax_It, label=lk_disp, shrink=0.85,
                                                     orientation="horizontal", pad=0.08)
                self._It_cbar.formatter = mticker.FormatStrFormatter("%.3g")
                self._It_cbar.update_ticks()
                self._mark_values_on_colorbar(self._It_cbar, cbar_marks)

        self._draw_reference_lines({"I_t": self.ax_It})
        self.It_canvas.draw()
        self.It_toolbar.update()
        self._plotted_iids = set(self.tree.selection())

    # ── PLOT DE RESUMEN ──────────────────────────────────────────────────────────
    # ── INSPECTOR: valores exactos de los atributos de los casos seleccionados ─────────
    def _inspector_data(self):
        """([grupos], {atributo: {grupo: texto exacto}}, {atributo: tipo}) de la selección actual,
        leído DIRECTO del .h5 (no de la copia en memoria ni de la tabla, que redondea a 3 cifras)."""
        groups = []
        for iid in self.tree.selection():
            c = self._iid_to_case.get(iid)
            if c is not None and c.get("group") not in groups:
                groups.append(c["group"])
        rows: Dict[str, Dict[str, str]] = {}
        types: Dict[str, str] = {}
        with h5py.File(self.h5_path, "r") as f:
            for g in groups:
                if g not in f:
                    continue
                grp = f[g]
                items = [("", dict(grp.attrs))]
                if self._insp_runs.get():   # atributos de cada corrida de indicador (pp_*, meta_*, ...)
                    for rn in grp.keys():
                        sub = grp[rn]
                        if isinstance(sub, h5py.Group) and "I_t" in sub and "t" in sub:
                            items.append((rn + " ▸ ", dict(sub.attrs)))
                for prefix, attrs in items:
                    for k, v in attrs.items():
                        key = prefix + str(k)
                        rows.setdefault(key, {})[g] = _exact(v)
                        types.setdefault(key, type(v).__name__ if not hasattr(v, "dtype") else str(v.dtype))
        return groups, rows, types

    def _open_inspector(self) -> None:
        win = getattr(self, "_inspector", None)
        if win is not None and win.winfo_exists():
            win.lift()
            self._refresh_inspector()
            return
        win = tk.Toplevel(self.root)
        win.title(f"Inspect  —  {os.path.basename(self.h5_path)}")
        win.geometry("980x560")
        self._inspector = win
        self._insp_diff = tk.BooleanVar(value=False)
        self._insp_runs = tk.BooleanVar(value=False)
        self._insp_filter = tk.StringVar()

        bar = ttk.Frame(win, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="Filter:").pack(side=tk.LEFT)
        ent = ttk.Entry(bar, textvariable=self._insp_filter, width=22)
        ent.pack(side=tk.LEFT, padx=(2, 10))
        self._insp_filter.trace_add("write", lambda *_: self._refresh_inspector())
        self._insp_diff_cb = ttk.Checkbutton(bar, text="Only attributes that differ", variable=self._insp_diff,
                                             command=self._refresh_inspector)
        self._insp_diff_cb.pack(side=tk.LEFT, padx=4)
        if any(c.get("runs") for c in self.cases):
            ttk.Checkbutton(bar, text="Include indicator runs", variable=self._insp_runs,
                            command=self._refresh_inspector).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Copy table", command=self._inspector_copy).pack(side=tk.RIGHT, padx=4)
        self._insp_status = ttk.Label(win, foreground="#555555", padding=(8, 2))
        self._insp_status.pack(side=tk.BOTTOM, fill=tk.X)

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self._insp_tree = ttk.Treeview(frame, show="headings", selectmode="extended")
        sy = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._insp_tree.yview)
        sx = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self._insp_tree.xview)
        self._insp_tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self._insp_tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        self._insp_tree.tag_configure("diff", background="#fff3b0")
        self._insp_tree.tag_configure("run", foreground="#555555")
        self._insp_tree.bind("<Double-1>", self._inspector_copy_cell)
        win.bind("<Control-c>", lambda _e: self._inspector_copy())
        win.protocol("WM_DELETE_WINDOW", self._close_inspector)
        self._refresh_inspector()

    def _close_inspector(self) -> None:
        win = getattr(self, "_inspector", None)
        self._inspector = None
        if win is not None:
            win.destroy()

    def _refresh_inspector(self) -> None:
        win = getattr(self, "_inspector", None)
        if win is None or not win.winfo_exists():
            return
        groups, rows, types = self._inspector_data()
        multi = len(groups) > 1
        self._insp_diff_cb.state(["!disabled"] if multi else ["disabled"])
        cols = ["attribute", "type"] + groups
        tv = self._insp_tree
        tv.delete(*tv.get_children())
        tv.configure(columns=cols)
        tv.heading("attribute", text="attribute")
        tv.column("attribute", width=230, minwidth=120, stretch=False, anchor=tk.W)
        tv.heading("type", text="type")
        tv.column("type", width=70, minwidth=50, stretch=False, anchor=tk.W)
        for g in groups:
            tv.heading(g, text=g)
            tv.column(g, width=190, minwidth=90, stretch=True, anchor=tk.W)

        flt = self._insp_filter.get().strip().lower()
        shown = ndiff = 0
        # parámetros del DOE ($...$) primero, luego el resto; las corridas de indicador al final
        for key in sorted(rows, key=lambda k: (" ▸ " in k, not k.startswith("$"), k)):
            vals = [rows[key].get(g, "—") for g in groups]
            differs = multi and len(set(vals)) > 1
            ndiff += differs
            if (flt and flt not in key.lower()) or (self._insp_diff.get() and multi and not differs):
                continue
            tags = (["diff"] if differs else []) + (["run"] if " ▸ " in key else [])
            tv.insert("", tk.END, values=[key, types.get(key, "")] + vals, tags=tags)
            shown += 1
        if not groups:
            msg = "Select one or more cases in the table to see their exact attributes."
        else:
            msg = (f"{len(groups)} case(s) · {shown}/{len(rows)} attributes shown"
                   + (f" · {ndiff} differ (highlighted)" if multi else "")
                   + "   |   values exactly as stored in the .h5 · double-click copies a value · Ctrl+C copies the table")
        self._insp_status.config(text=msg)

    def _inspector_text(self) -> str:
        tv = self._insp_tree
        head = [tv.heading(c, "text") for c in tv["columns"]]
        lines = ["\t".join(head)] + ["\t".join(str(x) for x in tv.item(i, "values")) for i in tv.get_children()]
        return "\n".join(lines)

    def _inspector_copy(self) -> None:
        self._inspector.clipboard_clear()
        self._inspector.clipboard_append(self._inspector_text())
        self._insp_status.config(text="Table copied to the clipboard (tab-separated: paste into Excel).")

    def _inspector_copy_cell(self, event) -> None:
        tv = self._insp_tree
        row, col = tv.identify_row(event.y), tv.identify_column(event.x)
        if not row or not col:
            return
        val = str(tv.item(row, "values")[int(col[1:]) - 1])
        self._inspector.clipboard_clear()
        self._inspector.clipboard_append(val)
        self._insp_status.config(text=f"Copied: {val}")

    def _sync_globals(self) -> None:
        """doe_plotter.LABEL_KEY es global del modulo (lo usan las figuras de convergencia): con varias
        pestañas abiertas se fija al de ESTE archivo al activar la pestaña y antes de cada figura de resumen."""
        import doe_plotter as _dp
        if self.cases:
            _dp.LABEL_KEY = self.cases[0].get("label_key", _dp.LABEL_KEY)

    def _make_summary_figure(self, entry) -> Optional[Figure]:
        """Figure of a summary entry (label, func, extra); None if there is none, raises if it cannot be made."""
        _label, func, extra = entry
        if isinstance(extra, tuple):
            # plot_convergence*(cases, *args) — returns a Figure
            return _capture_new_figure(func, self.cases, *extra)
        if isinstance(extra, dict):
            if func == "_noise_overlay":
                return _build_noise_overlay_fig(self.cases, extra["signal"])
            kw = dict(extra)
            kw["out_dir"] = None   # solo previsualizar
            return _capture_new_figure(func, **kw)
        return None

    # (tab attribute, figure attribute, name in the export window) of the panels of the centre
    _PANELS = (("_sig_tab", "sig_fig", "Panel — Signals"), ("_force_tab", "force_fig", "Panel — Forces"),
               ("_It_tab", "It_fig", "Panel — I(t)"), ("_deflex_tab", "deflex_fig", "Panel — Out Deflex"))

    def _export_items(self) -> list:
        """Every figure of this viewer for the export window: the panels as they are on screen (exported as a copy)
        and every summary figure (SLD and validation draw their own language; the others are translated)."""
        from figures_window import Item
        items = [Item(name, (lambda a=attr: getattr(self, a)), live=True)
                 for _tab, attr, name in self._PANELS if hasattr(self, attr)]
        items.append(Item("Panel — right (current)", lambda: self._fig_holder.get("summary"), live=True))

        def gen(entry):
            self._sync_globals()
            return self._make_summary_figure(entry)
        items += [Item(e[0], (lambda e=e: gen(e)), native=e[0].startswith(("SLD", "Validation")))
                  for e in self._summary_entries]
        return items

    def _active_panel_name(self) -> Optional[str]:
        current = self._nb.select() if hasattr(self, "_nb") else None
        for tab, _attr, name in self._PANELS:
            if hasattr(self, tab) and current == str(getattr(self, tab)):
                return name
        return "Panel — Signals" if hasattr(self, "sig_fig") else None

    def _open_export(self, select: Optional[str] = None) -> None:
        """The one export window of this viewer (figures_window.py): every panel and every summary figure, with size,
        scale, language, dpi, format, folder and name. Opened on `select`."""
        win = getattr(self, "_export_win", None)
        if win is not None and win.win.winfo_exists():
            win.select(select)
            return
        from figures_window import FiguresWindow
        lang, scale = _fig_style()
        if _is_validation_h5(self.h5_path):   # one folder per gray mode (validation_figures.figs_dir)
            import validation_figures as vf
            folder = os.path.basename(vf.figs_dir(self.h5_path))
        else:
            folder = "figs_indicators"
        self._export_win = FiguresWindow(
            self.root, f"Export — {os.path.basename(os.path.dirname(self.h5_path))} / {os.path.basename(self.h5_path)}",
            self._export_items(), style=_apply_fig_style, out_dir=os.path.join(os.path.dirname(self.h5_path), folder),
            language=lang, scale=scale, select=select)

    def _refresh_summary(self) -> None:
        self._sync_globals()
        if not self._summary_entries:
            return
        choice = self._sum_combo.get()
        entry  = next((e for e in self._summary_entries if e[0] == choice), None)
        if entry is None:
            return

        label = entry[0]
        fig = None

        try:
            fig = self._make_summary_figure(entry)

            if fig is None:
                messagebox.showwarning("No figure",
                                       f"Could not generate the figure:\n{label}",
                                       parent=self.root)
                return

            if not getattr(fig, "_keep_size", False):   # las SLD traen su FIGSIZE x FIGSCALE
                fig.set_size_inches(4.5, 4.5)
            _embed_figure(fig, self._sum_canvas_frame,
                          self._sum_toolbar_frame, self._fig_holder, "summary")

        except Exception as exc:
            messagebox.showerror("Error generating figure",
                                 f"{type(exc).__name__}: {exc}", parent=self.root)

    # ── ABRIR ARCHIVO ─────────────────────────────────────────────────────────────
    def _open_file(self) -> None:
        if self._on_open is not None:   # en una pestaña: el archivo nuevo va a OTRA pestaña
            self._on_open()
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open DOE HDF5 file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.h5_path),
        )
        if not path:
            return

        if detect_h5_type(path) in (TYPE_REFERENCE_DATASET, TYPE_REFERENCE_COMBINED):
            _launch_app_for(self.root, path)
            return

        try:
            self._load_file(path)
        except Exception as exc:
            messagebox.showerror("Error loading", str(exc), parent=self.root)
            return

        # Rebuild UI
        for w in self.container.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self._fig_holder.clear()
        self._cbar    = None
        self._sort_col = None
        self._sort_rev = False
        self._iid_to_case.clear()
        self._build_ui()


# ==============================================================================
# REFERENCE DATASET / COMBINED — loaders livianos (solo attrs, t/y bajo demanda)
# ==============================================================================

def _amp_limits(attrs: Dict[str, Any], channel: str) -> Optional[Tuple[float, float, float, str]]:
    """(base, lim_inf_pct, lim_sup_pct, nombre_base) si el tramo se etiquetó con
    `reference_dataset.py --strategy amplitude` sobre ESTE canal (attrs labeling_*
    que copia build); None si no -- otra estrategia, otro canal (los límites están
    en unidades de la señal etiquetada) o un .h5 generado antes del bloque labeling."""
    if attrs.get("labeling_strategy") != "amplitude" or attrs.get("labeling_signal") != channel:
        return None
    try:
        base_attr = str(attrs["labeling_base_attr"])
        base = float(attrs[base_attr]) * float(attrs["labeling_base_scale"])
        return base, float(attrs["labeling_lim_inf_pct"]), float(attrs["labeling_lim_sup_pct"]), base_attr.strip("$")
    except (KeyError, TypeError, ValueError):
        return None


def _draw_amp_limits(ax, rows: List[Dict[str, Any]], vertical: bool = False) -> None:
    """Líneas ±lim_inf / ±lim_sup del criterio max|y| de --strategy amplitude
    (horizontales en la señal, verticales en la distribución) -- una vez por valor
    distinto entre los tramos de `rows` (la base puede variar entre casos)."""
    draw = ax.axvline if vertical else ax.axhline
    seen = set()
    for r in rows:
        lim = r.get("amp_limits")
        if lim is None:
            continue
        base, inf_pct, sup_pct, base_name = lim
        for pct, ls in ((inf_pct, ":"), (sup_pct, "-.")):
            v = base * pct / 100.0
            if v in seen:
                continue
            seen.add(v)
            draw(v, color="black", ls=ls, lw=1.2, label=f"±{pct:g}% {base_name}")
            draw(-v, color="black", ls=ls, lw=1.2)


SHOW_NORMAL = [True]   # interruptor "Fitted normal" del visor de datasets (todas las vistas)


def _plot_normal_fit(ax, y_flat: np.ndarray, color) -> None:
    """Curva normal N(μ, σ) ajustada a `y_flat`, discontinua sobre su histograma (si SHOW_NORMAL)."""
    if not SHOW_NORMAL[0]:
        return
    mu, sigma = float(np.mean(y_flat)), float(np.std(y_flat))
    if sigma > 0:
        x = np.linspace(y_flat.min(), y_flat.max(), 300)
        pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color=color, lw=1.4, ls="--")


def file_role(h5_path: str) -> str:
    """Qué es este archivo, para la banda superior del visor: experimento que lo hizo, su papel (dataset
    etiquetado = verdad / con el que aprenden los indicadores; resultados de indicadores y con qué se entrenaron)."""
    try:
        with h5py.File(h5_path, "r") as f:
            a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
            lab, ramp_cases = None, set()
            for g in ("stable", "unstable", "gray"):   # a labelled dataset (a label group can be empty)
                for cname, case in (f[g].items() if g in f else []):
                    for piece in case.values():
                        if lab is None:
                            lab = {k: piece.attrs[k] for k in piece.attrs if str(k).startswith("labeling_")}
                        if _is_ramp_vv({"Ap_start": piece.attrs.get("$Ap_start$"), "Ap_end": piece.attrs.get("$Ap_end$")}):
                            ramp_cases.add(cname)
                        break
            for k in f:
                if k.startswith("case_") and _is_ramp_vv({"Ap_start": f[k].attrs.get("$Ap_start$"),
                                                         "Ap_end": f[k].attrs.get("$Ap_end$")}):
                    ramp_cases.add(k)
    except OSError:
        return ""
    bits = []
    if ramp_cases:
        bits.append(f"{len(ramp_cases)} RAMP case(s) of Ap (kappa_start -> kappa_end; dotted line = where the truth "
                    f"turns unstable)")
    if a.get("experiment"):
        bits.append(f"experiment {a['experiment']} · stage {a.get('experiment_stage', '?')}")
    if a.get("experiment_role"):
        bits.append(str(a["experiment_role"]))
    if lab is not None:
        bits.append("LABELLED DATASET (ground truth)" + (f": labels by {lab.get('labeling_strategy')} on "
                                                         f"{lab.get('labeling_signal', '?')}" if lab else ""))
        ref = a.get("experiment_reference")
        if ref and os.path.normcase(os.path.abspath(str(ref))) == os.path.normcase(os.path.abspath(h5_path)):
            bits.append("TRAINING data: the indicators of this experiment learn their thresholds from it")
    elif a.get("reference_dataset"):
        bits.append(f"indicators trained on {os.path.basename(str(a['reference_dataset']))}")
        if a.get("label_strategy"):
            bits.append(f"true label of each case: {a.get('label_strategy')} labelling (column true_label)")
    return "   |   ".join(bits)


def _index_reference_dataset(h5_path: str) -> List[Dict[str, Any]]:
    """Lee attrs de cada tramo de un reference_dataset.h5 (to_hdf5 anidado) -- sin t/y."""
    rows: List[Dict[str, Any]] = []
    with h5py.File(h5_path, "r") as f:
        for label in ("stable", "unstable", "gray"):
            if label not in f:
                continue
            for case_name in f[label].keys():
                case_grp = f[label][case_name]
                for piece_name in case_grp.keys():
                    attrs = dict(case_grp[piece_name].attrs)
                    channel = attrs.get("channel") or piece_name.rsplit("__", 1)[0]
                    idx_str = piece_name.rsplit("__", 1)[-1]
                    kappa = attrs.get("kappa")
                    ap = attrs.get("$Ap_start$")
                    row = {
                        "label": label, "case": case_name, "channel": str(channel),
                        "idx": int(idx_str) if idx_str.isdigit() else 0,
                        "piece_name": piece_name,
                        "t0": float(attrs.get("t0", 0.0)), "t1": float(attrs.get("t1", 0.0)),
                        "kappa": float(kappa) if kappa is not None else None,
                        "Ap_mm": float(ap) * 1e3 if ap is not None else None,
                        "amp_limits": _amp_limits(attrs, str(channel)),
                    }
                    if _is_ramp_vv({"Ap_start": ap, "Ap_end": attrs.get("$Ap_end$")}):
                        # a piece of a ramp: kappa and Ap at its two ends (its single 'kappa' is ignored)
                        k0, k1 = attrs.get("kappa_t0"), attrs.get("kappa_t1")
                        a0, a1 = attrs.get("Ap_start_mm"), attrs.get("Ap_end_mm")
                        row["kappa"] = float(k0) if k0 is not None else None
                        row["kappa_txt"] = f"{float(k0):.3f}->{float(k1):.3f}" if k0 is not None and k1 is not None else "ramp"
                        row["Ap_mm"] = float(a0) if a0 is not None else row["Ap_mm"]
                        row["ap_txt"] = f"{float(a0):.2f}->{float(a1):.2f}" if a0 is not None and a1 is not None else ""
                    rows.append(row)
    return rows


def _load_piece_ty(h5_path: str, label: str, case: str, piece_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        g = f[label][case][piece_name]
        return g["t"][()], g["y"][()]


def _index_reference_combined(h5_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """(label, canal) -> {group, n_pieces} de un reference_combined.h5 (save_combined) -- sin t/y."""
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with h5py.File(h5_path, "r") as f:
        for grp_name in f.keys():
            attrs = dict(f[grp_name].attrs)
            label, channel = attrs.get("label"), attrs.get("channel")
            if label is None or channel is None:
                continue
            idx[(label, str(channel))] = {"group": grp_name, "n_pieces": int(attrs.get("n_pieces", 0))}
    return idx


def _load_combined_group(h5_path: str, grp_name: str) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        g = f[grp_name]
        return {
            "t": g["t"][()], "y": g["y"][()], "fs": float(g.attrs.get("fs", 1.0)),
            "piece_lengths": [int(n) for n in g["piece_lengths"][()]] if "piece_lengths" in g else [],
            "source_ids": (
                [s.decode() if isinstance(s, bytes) else str(s) for s in g["source_ids"][()]]
                if "source_ids" in g else []
            ),
        }


def _decimate_for_plot(t: np.ndarray, y: np.ndarray, max_points: int = 20_000) -> Tuple[np.ndarray, np.ndarray]:
    n = len(t)
    if n <= max_points:
        return t, y
    stride = max(1, n // max_points)
    return t[::stride], y[::stride]


def _case_from_source_id(source_id: str) -> str:
    """'case_007/Axial_vel#case_007/Axial_vel__000' -> 'case_007'."""
    return source_id.split("#")[0].split("/")[0]


def _case_color_map(cases: List[str]):
    """Un color distinto por caso único (ciclo tab20), consistente en toda la sesión."""
    cmap = cm.get_cmap("tab20")
    unique = sorted(set(cases))
    return {c: cmap(i % 20) for i, c in enumerate(unique)}


class ReferenceViewerApp:
    """Visualizador de reference_dataset.py -- Fase 1 (tramos) y Fase 2A (combinado),
    y cualquier variante futura de reference_dataset.py que se agregue como un
    h5_type mas (el dispatch por tipo abajo es el unico lugar a extender).

    Ventana separada de DoeSelectorUnifiedApp a propósito: el esquema de datos
    (label/case/tramo o label/canal) no tiene nada que ver con "casos DOE" y
    reusar el Treeview/topbar de esa clase confundiría más de lo que ayudaría.
    Mismo lanzador/auto-detección/FileDialog que el resto de la app.

    `container`: frame donde se empaquetan los widgets -- por defecto `root`
    (dueño de toda la ventana, comportamiento de siempre). Si se pasa un
    frame distinto (una pestaña de Notebook, ver `_launch_app_for`), la
    instancia vive ahí en vez de en la ventana completa, para poder alternar
    entre varios reference_dataset.py (tramos, combinado, o lo que venga
    después) sin reabrir el diálogo de archivo.
    """

    def __init__(self, root: tk.Tk, h5_path: str, h5_type: str, container: Optional[tk.Widget] = None,
                 on_open=None) -> None:
        self.root = root
        self.container = container if container is not None else root
        self._on_open = on_open   # en TabbedViewer: abrir un archivo agrega una pestaña
        self.h5_path = h5_path
        self.h5_type = h5_type

        if container is None:
            self.root.title(f"{_TYPE_LABELS.get(h5_type, h5_type)}  —  {os.path.basename(h5_path)}")
            self.root.minsize(1000, 600)
            self.root.state("zoomed")

        role = file_role(h5_path)
        if role:
            tk.Label(self.container, text=role, bg="#fff8e1", fg="#5d4037", anchor="w", padx=8,
                     font=("Arial", 9, "bold")).pack(side=tk.TOP, fill=tk.X)
        if h5_type == TYPE_REFERENCE_DATASET:
            self._build_tramos_ui()
        else:
            self._build_combinado_ui()

    def _add_normal_toggle(self, bar, replot) -> None:
        """Interruptor de la ley normal ajustada sobre los histogramas."""
        self._normal_var = tk.BooleanVar(value=SHOW_NORMAL[0])

        def flip():
            SHOW_NORMAL[0] = self._normal_var.get()
            replot()
        ttk.Checkbutton(bar, text="Fitted normal", variable=self._normal_var, command=flip).pack(side=tk.LEFT, padx=4)

    # ── ABRIR ARCHIVO (comun a las dos vistas) ────────────────────────────────
    def _open_file(self) -> None:
        if self._on_open is not None:
            self._on_open()
            return
        paths = filedialog.askopenfilenames(
            parent=self.root, title="Open DOE HDF5 file(s) -- select several for a tab per file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.h5_path),
        )
        if not paths:
            return
        try:
            _launch_app_for(self.root, paths)
        except Exception as exc:
            messagebox.showerror("Error loading", str(exc), parent=self.root)

    # ── EXPORTAR FIGURA (comun a las dos vistas) ──────────────────────────────
    def _open_export(self) -> None:
        """The one export window (figures_window.py) with the figures of this view: the article-style version of the
        selection (built like before: article-plot-style) and the view as it is on screen; saved in figs_reference/."""
        win = getattr(self, "_export_win", None)
        if win is not None and win.win.winfo_exists():
            win.select(None)
            return
        from figures_window import Item, FiguresWindow
        if self.h5_type == TYPE_REFERENCE_DATASET:
            items = [Item("Segments — article (selection)", self._tramos_article_figure),
                     Item("Segments — screen", lambda: self._fig_tramos, live=True)]
        else:
            n = self._combinado_pages()
            items = [Item("Combined — article" + (f" (page {p + 1}/{n})" if n > 1 else ""),
                          (lambda p=p: self._combinado_article_figure(p))) for p in range(n)]
            items.append(Item("Combined — screen", lambda: self._fig_comb, live=True))
        lang, scale = _fig_style()
        self._export_win = FiguresWindow(
            self.root, f"Export — {os.path.basename(self.h5_path)}", items, style=_apply_fig_style,
            out_dir=os.path.join(os.path.dirname(self.h5_path), "figs_reference"), language=lang, scale=scale)

    @staticmethod
    def _apply_sci_y(ax) -> None:
        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(fmt)

    def _tramos_article_figure(self) -> Figure:
        """Reconstruye la selección actual (overlay, mismo layout que la vista interactiva)
        como figura nueva en estilo article-plot-style (skill: plot_style.py) -- la figura
        interactiva tal cual se exporta aparte ("Segments — screen")."""
        sel = self._tree.selection()
        if not sel:
            raise ValueError("select at least one segment first")
        pieces = [self._index[int(iid)] for iid in sel]

        show_signal = self._tramos_show_signal_var.get()
        show_dist = self._tramos_show_distribution_var.get()
        row_kinds = [k for k, on in (("signal", show_signal), ("distribution", show_dist)) if on] or ["signal"]
        detailed = len(pieces) <= self._TRAMOS_DETAIL_LIMIT
        channels = sorted({p["channel"] for p in pieces})
        channel_txt = channels[0] if len(channels) == 1 else "mixed_channels"
        ylabel_text = _channel_ylabel(channels[0]) if len(channels) == 1 else "value"

        with matplotlib.rc_context(plot_style.ARTICLE_RCPARAMS):
            figsize = plot_style.FIGSIZE_WIDE if len(row_kinds) == 2 else plot_style.FIGSIZE_SIMPLE
            fig = Figure(figsize=figsize, constrained_layout=True)
            axs = fig.subplots(1, len(row_kinds))
            axes = dict(zip(row_kinds, axs if len(row_kinds) > 1 else [axs]))

            for r in pieces:
                t, y = _load_piece_ty(self.h5_path, r["label"], r["case"], r["piece_name"])
                y_flat = np.asarray(y).ravel()
                mu, sigma = float(np.mean(y_flat)), float(np.std(y_flat))
                color, hatch = {
                    "stable": (plot_style.COLOR_STABLE, None),
                    "gray":   (plot_style.COLOR_GRAY, plot_style.HATCH_GRAY),
                }.get(r["label"], (plot_style.COLOR_UNSTABLE, plot_style.HATCH_UNSTABLE))
                piece_label = f"{r['case']} ({r['label']})"

                if "signal" in axes:
                    t_dec, y_dec = _decimate_for_plot(t, y)
                    axes["signal"].plot(t_dec, y_dec, color=color, lw=1.0, alpha=0.8, label=piece_label)

                if "distribution" in axes:
                    axes["distribution"].hist(
                        y_flat, bins=40, density=True, color=color, alpha=0.35, hatch=hatch,
                        edgecolor=color, label=piece_label if detailed else None,
                    )
                    _plot_normal_fit(axes["distribution"], y_flat, color)
                    if detailed and sigma > 0:
                        axes["distribution"].axvline(mu, color=color, lw=1.2)

            # límites en unidades de UN canal -> no con canales mezclados
            if len(channels) == 1 and self._tramos_show_limits_var.get():
                for kind, ax in axes.items():
                    _draw_amp_limits(ax, pieces, vertical=(kind == "distribution"))

            if "signal" in axes:
                ax = axes["signal"]
                ax.set_xlabel("t [s]")
                ax.set_ylabel(ylabel_text)
                self._apply_sci_y(ax)
                if len(pieces) <= 8:
                    ax.legend()

            if "distribution" in axes:
                ax = axes["distribution"]
                ax.set_xlabel(ylabel_text, labelpad=14)
                ax.set_ylabel("Density")
                if detailed:
                    ax.legend()

            return fig

    def _combinado_pages(self) -> int:
        """Pages of the article version of the combined view: one per grid page with 'Grid by case', else 1."""
        if not self._combinado_grid_var.get():
            return 1
        _pieces, cases = self._combinado_pieces_by_case()
        return max(1, -(-len(cases) // self._combinado_grid_page_size()))

    def _combinado_grid_figure(self, channel: str, show_dist: bool, page: int) -> Figure:
        """Version article-plot-style del grid por case (ver _plot_combinado_grid) --
        una figura por pagina (mismo page_size que la vista interactiva), en vez de una
        sola imagen gigante con todos los cases."""
        pieces_by_label, cases = self._combinado_pieces_by_case()
        if not cases:
            raise ValueError("no data for this channel")
        ylabel = _channel_ylabel(channel)
        page_size = self._combinado_grid_page_size()
        page_cases = cases[page * page_size:(page + 1) * page_size]
        with matplotlib.rc_context(plot_style.ARTICLE_RCPARAMS):
            fig = Figure(figsize=plot_style.figsize_grid(len(page_cases), 2), constrained_layout=True)
            axes_grid = fig.subplots(2, len(page_cases), squeeze=False)
            for col, case in enumerate(page_cases):
                for row, (label, color, hatch) in enumerate((
                    ("stable", plot_style.COLOR_STABLE, None),
                    ("unstable", plot_style.COLOR_UNSTABLE, plot_style.HATCH_UNSTABLE),
                )):
                    ax = axes_grid[row][col]
                    piece = pieces_by_label[label].get(case)
                    if piece is None:
                        ax.axis("off")
                        continue
                    t_piece, y_piece = piece
                    if show_dist:
                        ax.hist(np.asarray(y_piece).ravel(), bins=30, density=True,
                                color=color, alpha=0.5, hatch=hatch, edgecolor=color)
                        ax.set_xlabel(ylabel, labelpad=14)
                        if col == 0:
                            ax.set_ylabel("Density")
                    else:
                        t_dec, y_dec = _decimate_for_plot(t_piece, y_piece)
                        ax.plot(t_dec, y_dec, color=color, lw=1.0)
                        ax.set_xlabel("t [s]")
                        if col == 0:
                            ax.set_ylabel(ylabel)
                        self._apply_sci_y(ax)
                axes_grid[0][col].set_title(case)
        return fig

    def _combinado_article_figure(self, page: int = 0) -> Figure:
        """Reconstruye stable/unstable como figura nueva en estilo article-plot-style
        (skill: plot_style.py), lado a lado (FIGSIZE_WIDE = 2 paneles), no la figura
        interactiva tal cual. Si el toggle "Grid by case" esta activo, la pagina `page`
        de esa vista en su lugar (ver _combinado_grid_figure)."""
        channel = self._channel_var.get()
        if not channel or not any(self._combined_data.get(l) for l in ("stable", "unstable")):
            raise ValueError("pick a channel first")
        show_dist = self._show_distribution_var.get()
        if self._combinado_grid_var.get():
            return self._combinado_grid_figure(channel, show_dist, page)

        with matplotlib.rc_context(plot_style.ARTICLE_RCPARAMS):
            fig = Figure(figsize=plot_style.FIGSIZE_WIDE, constrained_layout=True)
            ax_stable, ax_unstable = fig.subplots(1, 2)
            for ax, label, color, hatch in (
                (ax_stable, "stable", plot_style.COLOR_STABLE, None),
                (ax_unstable, "unstable", plot_style.COLOR_UNSTABLE, plot_style.HATCH_UNSTABLE),
            ):
                ax.set_title(label)
                data = self._combined_data.get(label)
                if not data or len(data["y"]) == 0:
                    continue
                y = np.asarray(data["y"]).ravel()
                if show_dist:
                    mu, sigma = float(np.mean(y)), float(np.std(y))
                    ax.hist(y, bins=60, density=True, color=color, alpha=0.5, hatch=hatch, edgecolor=color)
                    if sigma > 0:
                        xg = np.linspace(y.min(), y.max(), 300)
                        pdf = np.exp(-0.5 * ((xg - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                        ax.plot(xg, pdf, color="black", lw=1.0, ls="--")
                    ax.set_xlabel(_channel_ylabel(channel), labelpad=14)
                    ax.set_ylabel("Density")
                else:
                    t_dec, y_dec = _decimate_for_plot(data["t"], data["y"])
                    ax.plot(t_dec, y_dec, color=color, lw=1.0)
                    ax.set_xlabel("t [s]")
                    ax.set_ylabel(_channel_ylabel(channel))
                    self._apply_sci_y(ax)
        return fig

    # ══════════════════════════════ PESTAÑA "TRAMOS" ═══════════════════════════════
    def _build_tramos_ui(self) -> None:
        self._index = _index_reference_dataset(self.h5_path)

        bar = ttk.Frame(self.container, padding=(4, 2))
        bar.pack(side=tk.TOP, fill=tk.X)
        if self._on_open is None:
            ttk.Button(bar, text="📂  Open .h5", command=self._open_file).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            bar, text=f"{len(self._index)} segments  |  {os.path.basename(self.h5_path)}",
            foreground="#444444", font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=8)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        labels_present = sorted({r["label"] for r in self._index})
        channels_present = sorted({r["channel"] for r in self._index})
        ttk.Label(bar, text="Label:", font=("Arial", 9)).pack(side=tk.LEFT)
        self._label_filter_var = tk.StringVar(value="(all)")
        _lc = ttk.Combobox(bar, textvariable=self._label_filter_var, state="readonly", width=10,
                            values=["(all)"] + labels_present)
        _lc.pack(side=tk.LEFT, padx=(2, 8))
        _lc.bind("<<ComboboxSelected>>", self._refresh_tree)
        ttk.Label(bar, text="Channel:", font=("Arial", 9)).pack(side=tk.LEFT)
        self._channel_filter_var = tk.StringVar(value="(all)")
        _cc = ttk.Combobox(bar, textvariable=self._channel_filter_var, state="readonly", width=14,
                            values=["(all)"] + channels_present)
        _cc.pack(side=tk.LEFT, padx=2)
        _cc.bind("<<ComboboxSelected>>", self._refresh_tree)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._tramos_show_signal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bar, text="〰️ Show signal", variable=self._tramos_show_signal_var,
            command=self._plot_selected_tramos,
        ).pack(side=tk.LEFT, padx=4)
        self._tramos_show_distribution_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bar, text="📊 Show distribution", variable=self._tramos_show_distribution_var,
            command=self._plot_selected_tramos,
        ).pack(side=tk.LEFT, padx=4)
        # límites operacionales (±lim_inf / ±lim_sup de --strategy amplitude): el toggle
        # solo aparece si el .h5 se etiquetó por amplitud; activado por defecto
        self._tramos_show_limits_var = tk.BooleanVar(value=True)
        if any(r["amp_limits"] for r in self._index):
            ttk.Checkbutton(
                bar, text="📏 Operational limits", variable=self._tramos_show_limits_var,
                command=self._plot_selected_tramos,
            ).pack(side=tk.LEFT, padx=4)

        self._add_normal_toggle(bar, self._plot_selected_tramos)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(bar, text="💾 Export figure", command=self._open_export).pack(side=tk.LEFT, padx=4)

        body = ttk.Panedwindow(self.container, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body)
        body.add(left, weight=1)
        right = ttk.Frame(body)
        body.add(right, weight=3)
        stats_frame = ttk.Frame(body)
        body.add(stats_frame, weight=1)
        # weight solo afecta el resize, no el ancho inicial, y el ancho de la ventana
        # "zoomed" (o de la pestaña, si esta embebida en un Notebook) tarda un poco en
        # asentarse -- reaplicar en cada resize del contenedor (no se dispara al
        # arrastrar el sash a mano, solo al cambiar el tamaño real del contenedor).
        def _sync_left_pane_width(_event=None):
            w = self.container.winfo_width()
            if w > 100:
                try:
                    body.sashpos(0, int(w * 0.32))
                    body.sashpos(1, int(w * 0.82))
                except tk.TclError:
                    pass
        self.container.bind("<Configure>", _sync_left_pane_width)
        self.container.after(50, _sync_left_pane_width)

        self._tree_cols = ("label", "case", "channel", "idx", "t0", "t1", "duration", "kappa", "Ap [mm]")
        widths = (60, 90, 100, 40, 65, 65, 75, 60, 65)
        self._tree_sort_col: Optional[str] = None
        self._tree_sort_rev = False
        self._tree = ttk.Treeview(left, columns=self._tree_cols, show="headings", selectmode="extended")
        for c, w in zip(self._tree_cols, widths):
            self._tree.heading(c, text=c, command=lambda cc=c: self._sort_tree_by(cc))
            self._tree.column(c, width=w, anchor=tk.CENTER)
        self._tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._plot_selected_tramos())

        self._fig_tramos = Figure(figsize=(7, 5), dpi=100)
        self._fig_tramos.add_subplot(111)
        self._canvas_tramos = FigureCanvasTkAgg(self._fig_tramos, master=right)
        self._canvas_tramos.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas_tramos, right).update()

        ttk.Label(stats_frame, text="Mean / Std", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=4, pady=(4, 0))
        self._tramos_stats_text = tk.Text(stats_frame, wrap=tk.WORD, width=28, font=("Consolas", 9))
        self._tramos_stats_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._refresh_tree()

    def _refresh_tree(self, *_args) -> None:
        self._tree.delete(*self._tree.get_children())
        lab = self._label_filter_var.get()
        ch = self._channel_filter_var.get()
        for i, r in enumerate(self._index):
            if lab != "(all)" and r["label"] != lab:
                continue
            if ch != "(all)" and r["channel"] != ch:
                continue
            kappa_txt = r.get("kappa_txt") or (f"{r['kappa']:.3f}" if r["kappa"] is not None else "")
            ap_txt = r.get("ap_txt") or (f"{r['Ap_mm']:.3f}" if r.get("Ap_mm") is not None else "")
            self._tree.insert("", tk.END, iid=str(i), values=(
                r["label"], r["case"], r["channel"], r["idx"],
                f"{r['t0']:.3f}", f"{r['t1']:.3f}", f"{r['t1'] - r['t0']:.3f}", kappa_txt, ap_txt,
            ))
        if self._tree_sort_col:
            self._sort_tree_by(self._tree_sort_col, toggle=False)

    def _sort_tree_by(self, col: str, toggle: bool = True) -> None:
        """Ordena la tabla de tramos al clickear un encabezado (clic de nuevo -> invierte)."""
        if toggle:
            self._tree_sort_rev = (self._tree_sort_col == col) and not self._tree_sort_rev
            self._tree_sort_col = col
        rows = [(self._tree.set(iid, col), iid) for iid in self._tree.get_children()]

        def _key(item):
            try:
                return (0, float(item[0]))
            except (ValueError, TypeError):
                return (1, str(item[0]))

        rows.sort(key=_key, reverse=self._tree_sort_rev)
        for idx, (_, iid) in enumerate(rows):
            self._tree.move(iid, "", idx)
        for c in self._tree_cols:
            hdr = self._tree.heading(c)["text"].rstrip(" ▲▼")
            arrow = (" ▼" if self._tree_sort_rev else " ▲") if c == self._tree_sort_col else ""
            self._tree.heading(c, text=hdr + arrow, command=lambda cc=c: self._sort_tree_by(cc))

    _TRAMOS_DETAIL_LIMIT = 3  # por encima de esto, sin lineas mu+-sigma/stats detalladas en la leyenda

    def _plot_selected_tramos(self) -> None:
        sel = self._tree.selection()
        self._fig_tramos.clf()

        show_signal = self._tramos_show_signal_var.get()
        show_dist = self._tramos_show_distribution_var.get()
        panels = [p for p, on in (("signal", show_signal), ("distribution", show_dist)) if on] or ["signal"]
        axes = {kind: self._fig_tramos.add_subplot(len(panels), 1, i + 1) for i, kind in enumerate(panels)}

        if not sel:
            self._canvas_tramos.draw_idle()
            self._update_tramos_stats_panel([])
            return

        cmap = cm.get_cmap("tab10")
        detailed = len(sel) <= self._TRAMOS_DETAIL_LIMIT
        selected_channels = {self._index[int(iid)]["channel"] for iid in sel}
        stats = []
        for i, iid in enumerate(sel):
            r = self._index[int(iid)]
            t, y = _load_piece_ty(self.h5_path, r["label"], r["case"], r["piece_name"])
            color = cmap(i % 10)  # color propio por tramo seleccionado, para distinguirlos entre si
            piece_label = f"{r['case']}/{r['channel']}__{r['idx']:03d} ({r['label']})"
            y_flat = np.asarray(y).ravel()
            mu, sigma = float(np.mean(y_flat)), float(np.std(y_flat))
            lim = r["amp_limits"]
            amp = float(np.abs(y_flat).max()) if lim else 0.0
            amp_txt = f"max|y| = {amp:.3g} ({100.0 * amp / lim[0]:.1f}% {lim[3]})" if lim else ""
            stats.append((piece_label, mu, sigma, amp_txt))

            if "signal" in axes:
                t_dec, y_dec = _decimate_for_plot(t, y)
                style = {"stable": "-", "gray": ":"}.get(r["label"], "--")  # el label se sigue viendo por el trazo
                axes["signal"].plot(t_dec, y_dec, color=color, ls=style, lw=1.1, alpha=0.9, label=piece_label)

            if "distribution" in axes:
                dist_label = f"{piece_label}  (μ={mu:.3g}, σ²={sigma ** 2:.3g})" if detailed else piece_label
                axes["distribution"].hist(y_flat, bins=60, density=True, color=color, alpha=0.4, label=dist_label)
                _plot_normal_fit(axes["distribution"], y_flat, color)
                if detailed and sigma > 0:
                    axes["distribution"].axvline(mu, color=color, lw=1.4, ls="-")
                    axes["distribution"].axvline(mu - sigma, color=color, lw=1.0, ls=":")
                    axes["distribution"].axvline(mu + sigma, color=color, lw=1.0, ls=":")

        if len(selected_channels) == 1:
            channel = next(iter(selected_channels))
            plot_title = _channel_title(channel)
            plot_ylabel = _channel_ylabel(channel)
            if self._tramos_show_limits_var.get():
                sel_rows = [self._index[int(iid)] for iid in sel]
                for kind, ax in axes.items():
                    _draw_amp_limits(ax, sel_rows, vertical=(kind == "distribution"))
        else:
            plot_title = "Selected segments (mixed channels)"
            plot_ylabel = "value"

        _FS_TITLE, _FS_AXIS, _FS_LEGEND, _FS_TICK = 18, 16, 12, 13

        if "signal" in axes:
            ax = axes["signal"]
            ax.set_title(plot_title, fontsize=_FS_TITLE)
            ax.set_xlabel("t [s]", fontsize=_FS_AXIS)
            ax.set_ylabel(plot_ylabel, fontsize=_FS_AXIS)
            if len(sel) <= 15:
                ax.legend(fontsize=_FS_LEGEND, loc="best")
            ax.tick_params(axis="both", labelsize=_FS_TICK)

        if "distribution" in axes:
            ax = axes["distribution"]
            ax.set_title(plot_title, fontsize=_FS_TITLE)
            ax.set_xlabel(plot_ylabel, fontsize=_FS_AXIS)
            ax.set_ylabel("Density", fontsize=_FS_AXIS)
            if len(sel) <= 15:
                ax.legend(fontsize=_FS_LEGEND, loc="best")
            ax.tick_params(axis="both", labelsize=_FS_TICK)

        self._fig_tramos.tight_layout()
        self._canvas_tramos.draw_idle()
        self._update_tramos_stats_panel(stats)

    def _update_tramos_stats_panel(self, stats) -> None:
        self._tramos_stats_text.delete("1.0", tk.END)
        if not stats:
            self._tramos_stats_text.insert(tk.END, "(no segments selected)\n")
            return
        self._tramos_stats_text.insert(tk.END, "== Mean (μ) ==\n")
        for piece_label, mu, _sigma, _amp_txt in stats:
            self._tramos_stats_text.insert(tk.END, f"{piece_label}\n  μ = {mu:.4g}\n\n")
        self._tramos_stats_text.insert(tk.END, "== Std (σ) ==\n")
        for piece_label, _mu, sigma, _amp_txt in stats:
            self._tramos_stats_text.insert(tk.END, f"{piece_label}\n  σ = {sigma:.4g}\n\n")
        if any(amp_txt for *_, amp_txt in stats):
            self._tramos_stats_text.insert(tk.END, "== Amplitude (--strategy amplitude) ==\n")
            for piece_label, _mu, _sigma, amp_txt in stats:
                if amp_txt:
                    self._tramos_stats_text.insert(tk.END, f"{piece_label}\n  {amp_txt}\n\n")

    # ══════════════════════════════ PESTAÑA "COMBINADO" ════════════════════════════
    def _build_combinado_ui(self) -> None:
        self._combined_index = _index_reference_combined(self.h5_path)
        self._combined_data: Dict[str, Dict[str, Any]] = {}
        channels = sorted({ch for (_, ch) in self._combined_index})

        bar = ttk.Frame(self.container, padding=(4, 2))
        bar.pack(side=tk.TOP, fill=tk.X)
        if self._on_open is None:
            ttk.Button(bar, text="📂  Open .h5", command=self._open_file).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            bar, text=f"{len(channels)} channels  |  {os.path.basename(self.h5_path)}",
            foreground="#444444", font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=8)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(bar, text="Channel:", font=("Arial", 9)).pack(side=tk.LEFT)
        self._channel_var = tk.StringVar(value=channels[0] if channels else "")
        _chc = ttk.Combobox(bar, textvariable=self._channel_var, state="readonly", width=16, values=channels)
        _chc.pack(side=tk.LEFT, padx=(2, 8))
        _chc.bind("<<ComboboxSelected>>", self._load_and_plot_channel)

        self._color_by_piece_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bar, text="🎨 Color segments", variable=self._color_by_piece_var,
            command=self._replot_combinado,
        ).pack(side=tk.LEFT, padx=4)
        self._add_normal_toggle(bar, self._replot_combinado)
        self._combinado_grid_page = 0
        self._combinado_grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bar, text="🔲 Grid by case", variable=self._combinado_grid_var,
            command=self._on_combinado_grid_toggle,
        ).pack(side=tk.LEFT, padx=4)
        self._combinado_grid_label_var = tk.StringVar(value="Both")
        _glc = ttk.Combobox(
            bar, textvariable=self._combinado_grid_label_var, state="readonly", width=9,
            values=["Both", "Stable", "Unstable"],
        )
        _glc.pack(side=tk.LEFT, padx=2)
        _glc.bind("<<ComboboxSelected>>", lambda _e: self._on_combinado_grid_toggle())
        ttk.Button(bar, text="◀", width=2, command=self._combinado_grid_prev_page).pack(side=tk.LEFT, padx=(6, 0))
        self._combinado_page_label_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self._combinado_page_label_var, font=("Arial", 9)).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="▶", width=2, command=self._combinado_grid_next_page).pack(side=tk.LEFT)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._show_distribution_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bar, text="📊 Show distribution", variable=self._show_distribution_var,
            command=self._replot_combinado,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(bar, text="💾 Export figure", command=self._open_export).pack(side=tk.LEFT, padx=4)

        body = ttk.Panedwindow(self.container, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        plot_frame = ttk.Frame(body)
        body.add(plot_frame, weight=3)
        meta_frame = ttk.Frame(body)
        body.add(meta_frame, weight=1)
        stats_frame = ttk.Frame(body)
        body.add(stats_frame, weight=1)

        self._fig_comb = Figure(figsize=(9, 6), dpi=100)
        self._canvas_comb = FigureCanvasTkAgg(self._fig_comb, master=plot_frame)
        self._canvas_comb.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas_comb, plot_frame).update()

        ttk.Label(meta_frame, text="Metadata", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=4, pady=(4, 0))
        self._meta_text = tk.Text(meta_frame, wrap=tk.WORD, width=38, font=("Consolas", 9))
        self._meta_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        ttk.Label(stats_frame, text="Mean / Variance", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=4, pady=(4, 0))
        self._combinado_stats_text = tk.Text(stats_frame, wrap=tk.WORD, width=28, font=("Consolas", 9))
        self._combinado_stats_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        if channels:
            self._load_and_plot_channel()

    def _load_and_plot_channel(self, *_args) -> None:
        channel = self._channel_var.get()
        self._combined_data = {}
        for label in ("stable", "unstable"):
            entry = self._combined_index.get((label, channel))
            if entry is not None:
                self._combined_data[label] = _load_combined_group(self.h5_path, entry["group"])
        self._combinado_grid_page = 0
        self._replot_combinado()

    def _on_combinado_grid_toggle(self) -> None:
        self._combinado_grid_page = 0
        self._replot_combinado()

    def _combinado_grid_prev_page(self) -> None:
        if self._combinado_grid_page > 0:
            self._combinado_grid_page -= 1
            self._replot_combinado()

    def _combinado_grid_next_page(self) -> None:
        self._combinado_grid_page += 1  # se clampa dentro de _plot_combinado_grid
        self._replot_combinado()

    def _combinado_grid_page_size(self) -> int:
        """Cuantos cases entran legibles en el ancho actual del panel de plot --
        usado solo por el export (que sigue paginando por ancho, columnas anchas
        con labels completos); la vista interactiva usa _combinado_grid_shape."""
        w = self._canvas_comb.get_tk_widget().winfo_width()
        if w < 50:  # todavia no tiene tamaño real (recien construido)
            w = 900
        return max(1, w // self._COMBINADO_GRID_MIN_COL_PX)

    _COMBINADO_UNIT_MIN_PX = 90            # tamaño minimo (ancho y alto) de cada unidad del grid
    _COMBINADO_UNIT_ASPECT_RANGE = (0.75, 1.6)  # ancho/alto aceptable de una unidad ("casi cuadrado, un poco rectangulo")

    def _combinado_grid_shape(self, n_labels_per_unit: int) -> Tuple[int, int]:
        """Elige (nrows, ncols) EN UNIDADES para la vista interactiva -- una unidad es
        1 subplot si el filtro es Stable/Unstable solo, o el PAR stable+unstable
        apilado si es "Both" (`n_labels_per_unit` = 1 o 2). Busca, entre las formas
        que caben en el panel real, la que maximiza cuantas unidades entran
        manteniendo cada unidad casi cuadrada (_COMBINADO_UNIT_ASPECT_RANGE)."""
        w = self._canvas_comb.get_tk_widget().winfo_width()
        h = self._canvas_comb.get_tk_widget().winfo_height()
        if w < 50:
            w = 900
        if h < 50:
            h = 500

        max_cols = max(1, w // self._COMBINADO_UNIT_MIN_PX)
        max_rows = max(1, h // (self._COMBINADO_UNIT_MIN_PX * n_labels_per_unit))
        lo, hi = self._COMBINADO_UNIT_ASPECT_RANGE

        best_shape = (1, 1)
        best_units = 0
        for ncols in range(1, max_cols + 1):
            for nrows in range(1, max_rows + 1):
                aspect = (w / ncols) / (h / nrows)
                if lo <= aspect <= hi:
                    units = ncols * nrows
                    if units > best_units:
                        best_units = units
                        best_shape = (nrows, ncols)
        return best_shape

    def _combinado_grid_tight_shape(
        self, n_items: int, max_rows: int, max_cols: int
    ) -> Tuple[int, int]:
        """(nrows, ncols) que cubre exactamente `n_items` unidades con el minimo de
        celdas vacias, sin superar la capacidad del panel (`max_rows`/`max_cols`,
        salida de `_combinado_grid_shape`) y manteniendo aspecto casi-cuadrado.
        Evita que una pagina con pocos cases (ej. 14 Stable) quede con la mayoria
        de las celdas de la grilla de capacidad maxima vacias."""
        w = self._canvas_comb.get_tk_widget().winfo_width()
        h = self._canvas_comb.get_tk_widget().winfo_height()
        if w < 50:
            w = 900
        if h < 50:
            h = 500
        lo, hi = self._COMBINADO_UNIT_ASPECT_RANGE

        best_shape = (max_rows, max_cols)
        best_waste = max_rows * max_cols - n_items
        for ncols in range(1, max_cols + 1):
            nrows = min(max_rows, -(-n_items // ncols))
            if nrows < 1 or nrows * ncols < n_items:
                continue
            aspect = (w / ncols) / (h / nrows)
            if not (lo <= aspect <= hi):
                continue
            waste = nrows * ncols - n_items
            if waste < best_waste:
                best_waste = waste
                best_shape = (nrows, ncols)
        return best_shape

    def _replot_combinado(self) -> None:
        self._fig_comb.clf()
        if self._combinado_grid_var.get():
            self._plot_combinado_grid()
        else:
            self._plot_combinado_normal()
        self._fig_comb.tight_layout()
        self._canvas_comb.draw_idle()
        self._update_meta_text()

    def _plot_combinado_normal(self) -> None:
        show_dist = self._show_distribution_var.get()
        channel = self._channel_var.get()
        title_base = _channel_title(channel)
        ylabel = _channel_ylabel(channel)
        _FS_TITLE, _FS_AXIS, _FS_LEGEND, _FS_TICK = 18, 16, 12, 13

        stats = []
        for row, (label, color) in enumerate((("stable", color_verde), ("unstable", color_red))):
            ax = self._fig_comb.add_subplot(2, 1, row + 1)
            data = self._combined_data.get(label)
            ax.set_title(f"{title_base} — {label}", fontsize=_FS_TITLE)
            ax.tick_params(axis="both", labelsize=_FS_TICK)
            if not data or len(data["y"]) == 0:
                continue

            y_flat = np.asarray(data["y"]).ravel()
            mu, sigma = float(np.mean(y_flat)), float(np.std(y_flat))
            stats.append((label, mu, sigma ** 2))

            if show_dist:
                self._plot_distribution(ax, data["y"], color, xlabel=ylabel)
                continue

            t, y, fs = data["t"], data["y"], data["fs"]
            t_dec, y_dec = _decimate_for_plot(t, y)
            ax.plot(t_dec, y_dec, color=color, lw=0.7)
            if self._color_by_piece_var.get() and data["piece_lengths"] and data["source_ids"]:
                bounds = np.cumsum([0] + data["piece_lengths"]) / fs
                cases = [_case_from_source_id(sid) for sid in data["source_ids"]]
                case_color = _case_color_map(cases)
                for i, case in enumerate(cases):
                    ax.axvspan(bounds[i], bounds[i + 1], color=case_color[case], alpha=0.25, zorder=0)
                # Leyenda compacta (color -> case) en vez de texto sobre cada banda -- con
                # muchas piezas angostas, el texto rotado quedaba ilegible por más grande que fuera.
                unique_cases = sorted(case_color)
                handles = [mpatches.Patch(color=case_color[c], label=c) for c in unique_cases]
                ax.legend(
                    handles=handles, loc="upper right", fontsize=_FS_LEGEND, ncol=min(len(handles), 4) or 1,
                    framealpha=0.85, borderaxespad=0.3, handlelength=1.2, columnspacing=0.8,
                )
            ax.set_xlabel("t [s]", fontsize=_FS_AXIS)  # concatenación de tramos -- ya no es tiempo real de ensayo
            ax.set_ylabel(ylabel, fontsize=_FS_AXIS)

        self._update_combinado_stats_panel(stats)

    def _combinado_pieces_by_case(self) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Corta cada señal concatenada (stable/unstable) en sus piezas de origen,
        usando piece_lengths/source_ids -- (pieces_by_label[label][case] = (t, y), cases).

        `t` de cada pieza se rebasea para arrancar en 0 -- data["t"] es el eje
        sintético de TODA la concatenación (t = índice/fs), así que una pieza que
        no sea la primera hereda el offset acumulado de las piezas anteriores si no
        se resta (ej. la 2da pieza mostraría t=[14, 28] en vez de [0, 14])."""
        pieces_by_label: Dict[str, Dict[str, Any]] = {"stable": {}, "unstable": {}}
        cases: set = set()
        for label in ("stable", "unstable"):
            data = self._combined_data.get(label)
            if not data or not data.get("piece_lengths"):
                continue
            idx = 0
            for length, sid in zip(data["piece_lengths"], data["source_ids"]):
                case = _case_from_source_id(sid)
                t_piece = data["t"][idx:idx + length]
                t_piece = t_piece - t_piece[0] if len(t_piece) else t_piece
                pieces_by_label[label][case] = (t_piece, data["y"][idx:idx + length])
                cases.add(case)
                idx += length
        return pieces_by_label, sorted(cases)

    _COMBINADO_GRID_MIN_COL_PX = 140  # ancho minimo por columna en pantalla (sin texto de eje, mas compacto)

    def _plot_combinado_grid(self) -> None:
        """Grid por case -- un mini-panel por case, cortando la señal concatenada con
        piece_lengths/source_ids. Alternativa a las bandas de color sobre la curva
        única, para comparar cases lado a lado. Grid nxm casi-cuadrado en UNIDADES
        (ver _combinado_grid_shape) -- una unidad es el case solo si el filtro es
        Stable/Unstable, o el PAR stable+unstable apilado si es "Both". Paginado
        (_combinado_grid_{prev,next}_page) en vez de envolver en bloques -- con 30+
        cases, mostrarlos todos de una da una imagen gigante e ilegible tanto en
        pantalla como al exportar.

        Vista rápida/sucia a propósito: sin texto de ejes (t [s], nombre de canal)
        repetido en cada panel, solo ticks chicos como referencia de escala y el
        nombre del case como título -- prioriza que se vea la FORMA de muchas curvas
        a la vez. El detalle completo (labels, unidades) queda para el export en
        estilo article-plot-style, no para esta vista."""
        show_dist = self._show_distribution_var.get()
        label_filter = self._combinado_grid_label_var.get()
        labels_shown = (
            ["stable", "unstable"] if label_filter == "Both"
            else ["stable"] if label_filter == "Stable" else ["unstable"]
        )
        label_color = {"stable": color_verde, "unstable": color_red}
        n_labels = len(labels_shown)

        pieces_by_label, all_cases = self._combinado_pieces_by_case()
        cases = [c for c in all_cases if any(c in pieces_by_label[l] for l in labels_shown)]
        if not cases:
            ax = self._fig_comb.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, "No data for this channel/label.", ha="center", va="center", fontsize=14)
            self._combinado_page_label_var.set("")
            self._update_combinado_stats_panel([])
            return

        grid_rows, grid_cols = self._combinado_grid_shape(n_labels)
        units_per_page = grid_rows * grid_cols
        n_pages = -(-len(cases) // units_per_page)
        self._combinado_grid_page = max(0, min(self._combinado_grid_page, n_pages - 1))
        start = self._combinado_grid_page * units_per_page
        page_cases = cases[start:start + units_per_page]
        self._combinado_page_label_var.set(f"Page {self._combinado_grid_page + 1}/{n_pages}  ({len(cases)} cases)")

        grid_rows, grid_cols = self._combinado_grid_tight_shape(len(page_cases), grid_rows, grid_cols)
        axes_grid = self._fig_comb.subplots(grid_rows * n_labels, grid_cols, squeeze=False)
        stats = []
        for i in range(grid_rows * grid_cols):
            unit_row, unit_col = divmod(i, grid_cols)
            if i >= len(page_cases):
                for row in range(n_labels):
                    axes_grid[unit_row * n_labels + row][unit_col].axis("off")
                continue
            case = page_cases[i]
            for row, label in enumerate(labels_shown):
                ax = axes_grid[unit_row * n_labels + row][unit_col]
                color = label_color[label]
                piece = pieces_by_label[label].get(case)
                if piece is None:
                    ax.axis("off")
                    continue
                t_piece, y_piece = piece
                y_flat = np.asarray(y_piece).ravel()
                mu, sigma = float(np.mean(y_flat)), float(np.std(y_flat))
                stats.append((f"{case} ({label})", mu, sigma ** 2))
                if show_dist:
                    ax.hist(y_flat, bins=30, density=True, color=color, alpha=0.5, edgecolor="none")
                    if sigma > 0:
                        x = np.linspace(y_flat.min(), y_flat.max(), 200)
                        pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                        ax.plot(x, pdf, color="black", lw=1.0, ls="--")
                else:
                    t_dec, y_dec = _decimate_for_plot(t_piece, y_piece)
                    ax.plot(t_dec, y_dec, color=color, lw=1.0)
                ax.tick_params(axis="both", labelsize=7)
            axes_grid[unit_row * n_labels][unit_col].set_title(case, fontsize=10)
        self._update_combinado_stats_panel(stats)

    @staticmethod
    def _plot_distribution(ax, y: np.ndarray, color, xlabel: str = "signal value") -> None:
        """Histograma de `y` + curva gaussiana ajustada, para ver a ojo qué tan normal es la forma."""
        _FS_AXIS, _FS_LEGEND, _FS_TEXT = 16, 12, 12
        y = np.asarray(y).ravel()
        mu, sigma = float(np.mean(y)), float(np.std(y))
        ax.hist(y, bins=80, density=True, color=color, alpha=0.5, edgecolor="none")
        if sigma > 0 and SHOW_NORMAL[0]:
            x = np.linspace(y.min(), y.max(), 300)
            pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            ax.plot(x, pdf, color="black", lw=1.2, ls="--", label="Fitted normal")
            skew = float(np.mean(((y - mu) / sigma) ** 3))
            ax.text(
                0.02, 0.95, f"μ={mu:.3g}\nσ={sigma:.3g}\nskew={skew:.3g}",
                transform=ax.transAxes, fontsize=_FS_TEXT, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.7"),
            )
            ax.legend(fontsize=_FS_LEGEND, loc="upper right")
        ax.set_xlabel(xlabel, fontsize=_FS_AXIS)
        ax.set_ylabel("Density", fontsize=_FS_AXIS)

    def _update_meta_text(self) -> None:
        self._meta_text.delete("1.0", tk.END)
        for label in ("stable", "unstable"):
            data = self._combined_data.get(label)
            self._meta_text.insert(tk.END, f"== {label} ==\n")
            if not data:
                self._meta_text.insert(tk.END, "  (no data for this channel)\n\n")
                continue
            n_pieces = len(data["source_ids"]) or len(data["piece_lengths"])
            dur = len(data["y"]) / data["fs"] if data["fs"] else 0.0
            self._meta_text.insert(tk.END, f"pieces: {n_pieces}\n")
            self._meta_text.insert(tk.END, f"total duration: {dur:.2f} s\n")
            if data["source_ids"]:
                self._meta_text.insert(tk.END, "source segments:\n")
                for sid in data["source_ids"]:
                    self._meta_text.insert(tk.END, f"  - {sid}\n")
            self._meta_text.insert(tk.END, "\n")

    def _update_combinado_stats_panel(self, stats) -> None:
        """`stats`: lista de (item_label, mu, variance). Mismo formato que
        _update_tramos_stats_panel (dos secciones agrupadas) pero con varianza
        en vez de desvío estándar."""
        self._combinado_stats_text.delete("1.0", tk.END)
        if not stats:
            self._combinado_stats_text.insert(tk.END, "(no data)\n")
            return
        self._combinado_stats_text.insert(tk.END, "== Mean (μ) ==\n")
        for item_label, mu, _var in stats:
            self._combinado_stats_text.insert(tk.END, f"{item_label}\n  μ = {mu:.4g}\n\n")
        self._combinado_stats_text.insert(tk.END, "== Variance (σ²) ==\n")
        for item_label, _mu, var in stats:
            self._combinado_stats_text.insert(tk.END, f"{item_label}\n  σ² = {var:.4g}\n\n")


class TabbedViewer:
    """Ventana con una pestaña por .h5: cada pestaña es un visor completo e independiente
    (DoeSelectorUnifiedApp o ReferenceViewerApp, con su tabla, selección, figuras y resumen propios).

    Abrir:  botón "Open .h5" de la barra de arriba o Ctrl+O -> el archivo va a una pestaña NUEVA.
    Cerrar: botón "Close tab", Ctrl+W o clic con la rueda sobre la pestaña.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.apps: Dict[str, Any] = {}   # nombre del frame de la pestaña -> visor
        self._tab_base: Dict[str, str] = {}   # nombre del frame -> texto de la pestaña sin la marca de activa
        self._last_dir: Optional[str] = None
        for w in root.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        root.title("DOE viewer")
        root.minsize(1100, 650)
        root.state("zoomed")

        bar = ttk.Frame(root, padding=(4, 2))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="📂  Open .h5 (new tab)", command=self.open_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="✖  Close tab", command=self.close_current).pack(side=tk.LEFT, padx=4)
        self._hint = ttk.Label(bar, foreground="#666666",
                               text="Ctrl+O open · Ctrl+W close · middle-click a tab to close it")
        self._hint.pack(side=tk.LEFT, padx=12)

        # rotulo con el archivo de la pestaña activa: en el tema de Windows las pestañas casi no se distinguen
        self._banner = tk.Label(root, anchor="w", padx=10, pady=3, font=("Segoe UI", 10, "bold"))
        self._banner.pack(side=tk.TOP, fill=tk.X)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.nb.bind("<Button-2>", self._on_middle_click)
        root.bind("<Control-o>", lambda _e: self.open_dialog())
        root.bind("<Control-w>", lambda _e: self.close_current())

    @staticmethod
    def _tab_text(path: str, h5_type: str) -> str:
        # todos los resultados se llaman doe_results.h5 & cia: la carpeta del DOE es lo que los distingue
        kind = {TYPE_REFERENCE_DATASET: "Segments", TYPE_REFERENCE_COMBINED: "Combined"}.get(
            h5_type, _TYPE_LABELS.get(h5_type, h5_type).split("(")[0].strip())
        where = os.path.basename(os.path.dirname(os.path.abspath(path)))
        text = f"{kind} — {where}/{os.path.basename(path)}"
        return text if len(text) <= 70 else text[:67] + "..."

    def add(self, path: str):
        """Carga `path` en una pestaña nueva y la activa. Si falla, avisa y no deja la pestaña a medias."""
        path = os.path.abspath(path)
        tab = ttk.Frame(self.nb)
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            h5_type = detect_h5_type(path)
            self._tab_base[str(tab)] = self._tab_text(path, h5_type)
            self.nb.add(tab, text=self._tab_base[str(tab)])
            if h5_type in (TYPE_REFERENCE_DATASET, TYPE_REFERENCE_COMBINED):
                app = ReferenceViewerApp(self.root, path, h5_type, container=tab, on_open=self.open_dialog)
            else:
                app = DoeSelectorUnifiedApp(self.root, path, container=tab, on_open=self.open_dialog)
        except Exception as exc:
            try:
                self.nb.forget(tab)
            except tk.TclError:
                pass
            tab.destroy()
            messagebox.showerror("Error loading", f"{os.path.basename(path)}:\n{type(exc).__name__}: {exc}",
                                 parent=self.root)
            return None
        finally:
            self.root.config(cursor="")
        self.apps[str(tab)] = app
        self._last_dir = os.path.dirname(path)
        self.nb.select(tab)
        self._on_tab_changed()
        return app

    def open_dialog(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root, title="Open .h5 (one tab per file)",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
            initialdir=self._last_dir,
        )
        for p in paths:
            self.add(p)

    def close_tab(self, tab_id: str) -> None:
        app = self.apps.pop(str(tab_id), None)
        getattr(app, "_close_inspector", lambda: None)()   # el inspector de esa pestaña se cierra con ella
        self._tab_base.pop(str(tab_id), None)
        try:
            self.nb.forget(tab_id)
            self.root.nametowidget(tab_id).destroy()   # libera la tabla, las figuras y los datos de ese .h5
        except (tk.TclError, KeyError):
            pass
        self._on_tab_changed()

    def close_current(self) -> None:
        cur = self.nb.select()
        if cur:
            self.close_tab(cur)

    def _on_middle_click(self, event) -> None:
        try:
            idx = self.nb.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return
        self.close_tab(self.nb.tabs()[idx])

    def _on_tab_changed(self, _event=None) -> None:
        cur = self.nb.select()
        for tid in self.nb.tabs():   # "▶" delante del título de la pestaña activa
            self.nb.tab(tid, text=("▶ " if tid == cur else "") + self._tab_base.get(tid, ""))
        app = self.apps.get(cur) if cur else None
        if app is None:
            self.root.title("DOE viewer  —  Open a .h5 (Ctrl+O)")
            self._banner.config(text="No file open  —  Ctrl+O to open a .h5", bg="#9e9e9e", fg="white")
            return
        self.root.title(f"{_TYPE_LABELS.get(app.h5_type, app.h5_type)}  —  {app.h5_path}")
        self._banner.config(
            text=f"▶  {_TYPE_LABELS.get(app.h5_type, app.h5_type).split('(')[0].strip()}   |   {app.h5_path}",
            bg="#1f6feb", fg="white")
        sync = getattr(app, "_sync_globals", None)
        if sync is not None:
            sync()


def _launch_app_for(root: tk.Tk, h5_paths) -> "TabbedViewer":
    """Construye la ventana con una pestaña por archivo (también con uno solo, para poder sumar más
    después con "Open .h5"). Devuelve el TabbedViewer."""
    if isinstance(h5_paths, str):
        h5_paths = [h5_paths]
    viewer = TabbedViewer(root)
    for path in h5_paths:
        viewer.add(path)
    if len(h5_paths) > 1:
        viewer.nb.select(0)   # empieza en el primero
    return viewer


# ==============================================================================
# CLI + MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="doe_unified_selector — Selector interactivo unificado para HDF5 DOE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  python doe_unified_selector.py
  python doe_unified_selector.py --h5 DOE_xxx/doe_results.h5
  python doe_unified_selector.py --h5 DOE_xxx/doe_indicator_results.h5
  python doe_unified_selector.py --h5 DOE_xxx/doe_noise_indicator_results.h5
  python doe_unified_selector.py --h5 DOE_xxx/doe_model_snr_results.h5
  python doe_unified_selector.py --h5 reference_dataset.h5 reference_combined.h5
""",
    )
    p.add_argument("--h5", default=None, metavar="PATH", nargs="+",
                   help="Ruta(s) al/los archivo(s) .h5 a abrir (omitir → FileDialog). "
                        "Una pestaña por archivo; luego se pueden abrir más con 'Open .h5' o Ctrl+O.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    h5_paths = args.h5
    if not h5_paths:
        # Sin parent explicito: tkinter crea y maneja su propio root implicito,
        # mas confiable en Windows que un root manual withdraw()-eado.
        h5_paths = filedialog.askopenfilenames(
            title="Select one or more DOE HDF5 files",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not h5_paths:
            print("[INFO] No se seleccionó ningún archivo. Saliendo.")
            return

    missing = [p for p in h5_paths if not os.path.isfile(p)]
    if missing:
        print(f"[ERROR] Archivo(s) no encontrado(s): {missing}")
        return

    for p in h5_paths:
        print(f"[INFO] Cargando: {p} ({_TYPE_LABELS.get(detect_h5_type(p), '?')})")

    root = tk.Tk()
    root.update()  # pinta la ventana ya, antes de la carga pesada del .h5
    _launch_app_for(root, h5_paths)
    root.mainloop()


def _selftest() -> None:
    """Loading functions with ramps of Ap (no window): kappa of a ramp ignored, sorted by its start kappa, the
    truth intervals / t_onset of indicator and validation files, the file band, the pieces of a dataset."""
    import tempfile
    d = tempfile.mkdtemp(prefix="unified_sel_")
    t = np.linspace(0.0, 10.0, 101)
    doe, ind, val, lab = (os.path.join(d, n) for n in ("doe_results.h5", "ind.h5", "val.h5", "lab.h5"))
    cases = {"case_000": (0.009, 0.009, {"kappa": 1.05}),
             "case_001": (0.005, 0.015, {"kappa": 0.58, "kappa_start": 0.58, "kappa_end": 1.74}),
             "case_002": (0.015, 0.005, {"kappa": 1.74, "kappa_start": 1.74, "kappa_end": 0.58})}
    with h5py.File(lab, "w") as f:
        for c, iv in (("case_001", [(0.0, 6.0, "stable"), (6.0, 10.0, "unstable")]),
                      ("case_002", [(0.0, 4.0, "unstable"), (4.0, 10.0, "stable")]), ("case_000", [(0.0, 10.0, "unstable")])):
            for i, (a, b, lb) in enumerate(iv):
                p = f.require_group(f"{lb}/{c}").create_dataset(f"Axial_disp__{i:03d}", data=[0.0])
                a0, a1, extra = cases[c]
                p.attrs.update({"t0": a, "t1": b, "channel": "Axial_disp", "$Ap_start$": a0, "$Ap_end$": a1,
                                "kappa_t0": 0.58 + 0.116 * a, "kappa_t1": 0.58 + 0.116 * b,
                                "Ap_start_mm": 5 + a, "Ap_end_mm": 5 + b, "labeling_strategy": "amplitude", **extra})
    for path in (doe, ind, val):
        with h5py.File(path, "w") as f:
            if path == ind:
                f.attrs["label_dataset"] = lab
            for c, (a0, a1, extra) in cases.items():
                g = f.create_group(c)
                g.attrs.update({"$Ap_start$": a0, "$Ap_end$": a1, "$spin_rate$": 12000.0, **extra})
                g.create_dataset("Axial_disp/time", data=t)
                g.create_dataset("Axial_disp/values", data=np.sin(t))
                if path != doe:
                    r = g.create_group("maxent_x")
                    r["t"], r["I_t"], r["t_d"] = t, np.cos(t), [7.0]
                if path == val:
                    g.attrs.update({"$kappa$": np.nan if a0 != a1 else extra["kappa"], "$truth$": "mixed"})
                    iv = [(0.0, 6.0, "stable"), (6.0, 10.0, "unstable")]
                    g["truth_t0"], g["truth_t1"] = [x[0] for x in iv], [x[1] for x in iv]
                    g.create_dataset("truth_label", data=np.array([x[2] for x in iv], dtype=object),
                                     dtype=h5py.string_dtype())
                    if a0 != a1:
                        g.attrs["$t_onset$"] = 6.0
    for path in (doe, ind, val):
        cs = load_h5_unified(path, detect_h5_type(path))
        by = {c["group"]: c for c in cs}
        assert [c["group"] for c in cs] == ["case_001", "case_000", "case_002"], (path, [c["group"] for c in cs])
        r = by["case_001"]
        assert r["ramp"] and np.isnan(r["var_val"]["kappa"]) and r["label_val"] == 0.58, (path, r["var_val"])
        assert _case_legend(r, "kappa", r["label_val"]) == "kappa 0.58->1.74" and not by["case_000"].get("ramp")
        if path == doe:
            assert "truth_iv" not in r and r["t_onset"] is None
        else:
            assert r["truth_iv"][0] == (0.0, 6.0, "stable") and r["t_onset"] == 6.0, (path, r.get("truth_iv"))
        if path == ind:
            assert by["case_002"]["t_onset"] == 0.0 and by["case_002"]["truth_iv"][0][2] == "unstable"
    fig = Figure()
    ax = fig.add_subplot(111)
    _draw_truth_marks([ax], by["case_001"], "C0", shade=True)
    assert len(ax.lines) == 1 and ax.lines[0].get_xdata()[0] == 6.0 and len(ax.patches) == 2
    _draw_truth_marks([ax], by["case_000"], "C0", shade=True)          # a constant case without t_onset: nothing drawn
    assert len(ax.lines) == 1
    _draw_truth_marks([ax], {"var_val": {"t_onset": 0.9}}, "C0", shade=False)   # a constant case of a validation: t_onset
    assert len(ax.lines) == 2 and ax.lines[1].get_xdata()[0] == 0.9
    assert _case_onset({"var_val": {"t_onset": float("nan")}}) is None and _case_onset({"var_val": {}}) is None
    assert _case_onset({"ramp": True, "t_onset": 2.0, "var_val": {}}) == 2.0
    # decision limits of the indicators (checked on real files: I(t) crosses them at t_d) and the delay of a run
    assert _indicator_limits("ssq_revo_x", {"meta_lim_sup": 11.2, "meta_lim_inf": -4.3}) == [11.2, -4.3]
    assert _indicator_limits("rms_cv_x", {"meta_cv_threshold_used": 0.0011}) == [0.0011]
    lm = _indicator_limits("maxent_revo_x", {"pp_alpha": 0.00135, "pp_beta": 0.00135})
    assert abs(lm[0] - 6.6063) < 1e-3 and abs(lm[1] + 6.6063) < 1e-3, lm
    assert _indicator_limits("green_fixed_x", {"pp_z_sigma": 3.0}) == [] and _indicator_limits("ssq_x", {}) == []
    assert _run_delay({"attrs": {"delay_onset_s": -0.45}}) == -0.45 and _run_delay({"attrs": {}}) is None
    assert "3 RAMP" not in file_role(doe) and "2 RAMP case(s)" in file_role(doe) and "2 RAMP case(s)" in file_role(lab)
    rows = {(x["case"], x["t0"]): x for x in _index_reference_dataset(lab)}
    assert rows[("case_001", 6.0)]["kappa_txt"] == "1.276->1.740" and rows[("case_001", 6.0)]["ap_txt"] == "11.00->15.00"
    assert "kappa_txt" not in rows[("case_000", 0.0)]
    # ramps without a single 'kappa' (what doe_runner extract writes) and a t_onset that varies: the key is still
    # kappa (the start kappa of a ramp), never t_onset
    with h5py.File(ind, "a") as f:
        for c in ("case_001", "case_002"):
            del f[c].attrs["kappa"]
            f[c].attrs["t_onset"] = 6.0 if c == "case_001" else 0.5
    cs = load_h5_unified(ind, detect_h5_type(ind))
    assert all(c["label_key"] == "kappa" for c in cs), [c["label_key"] for c in cs]
    assert [c["group"] for c in cs] == ["case_001", "case_000", "case_002"] and cs[0]["label_val"] == 0.58
    print("doe_unified_selector selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
