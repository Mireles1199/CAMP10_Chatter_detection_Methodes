"""Green Integral Fixed-Window — Pedagogical tutorial.

Generates ONE phase diagram for the selected window, showing:
  - the shaded area computed by the Shoelace formula,
  - the orbit trace,
  - the start marker (●, green) and end marker (★, red),
  - the artificial closing line end→start,
  - the rotation direction arrow.

The window is selected via _WIN_IDX (index) or _WIN_T (approximate time).

Usage
-----
    cd indicators/green_integral
    python examples/Green_Integral_FixedWindow_Tutorial.py
"""

from __future__ import annotations

import os
import sys
import pathlib
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as _scipy_norm

# -- Path setup -------------------------------------------------------------
_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral.logging_setup import configure_logging, LOGGING_LEVELS
configure_logging(level=LOGGING_LEVELS["warning"])  # silent during tutorial

from green_integral import (
    HDF5Reader,
    StdSignalData,       # ← interfaz estándar (misma que MaxEnt / RMS-CV)
    IndicatorResult,     # ← resultado estándar
    run_green_std,       # ← runner con interfaz f_cycle / N_cycles_per_seg / step_cycles
)
from green_integral.utils.signal_filter import savgol_filter_window

# -- Canonical color palette (SKILL.md Â§3) ----------------------------------
r, g, b = colorsys.hls_to_rgb(346 / 360, 0.45, 0.99);  color_red    = (r, g, b)
r, g, b = colorsys.hls_to_rgb(36  / 360, 0.45, 0.99);  color_orange = (r, g, b)
r, g, b = colorsys.hls_to_rgb(279 / 360, 0.36, 0.99);  color_purple = (r, g, b)
r, g, b = colorsys.hls_to_rgb(98  / 360, 0.36, 0.99);  color_verde  = (r, g, b)
r, g, b = colorsys.hls_to_rgb(206.957 / 360, 0.40941, 0.55603); color_azul = (r, g, b)

# -- Figure helpers (SKILL.md Â§1â€“2) -----------------------------------------
def fig_size(scale: float = 1.0, ncols: int = 1, base_width: float = 3.4):
    """Return (width, height) following the canonical aspect ratio (Ã—0.70)."""
    width = base_width * ncols * scale
    return (width, width * 0.70)


def configure_global_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",          "font.size": 9,
        "axes.titlesize": 25,            "axes.labelsize": 25,
        "xtick.labelsize": 23,           "ytick.labelsize": 23,
        "legend.fontsize": 23,
        "lines.linewidth": 1.25,         "lines.markersize": 6,
        "axes.linewidth": 0.8,           "grid.linewidth": 0.5,
        "xtick.major.width": 0.8,        "ytick.major.width": 0.8,
        "xtick.direction": "in",         "ytick.direction": "in",
        "xtick.major.size": 4,           "ytick.major.size": 4,
        "xtick.minor.size": 2.5,         "ytick.minor.size": 2.5,
        "xtick.minor.width": 0.6,        "ytick.minor.width": 0.6,
        "mathtext.fontset": "stix",      "axes.formatter.use_mathtext": True,
        "legend.frameon": False,         "legend.loc": "best",
        "figure.dpi": 100,               "savefig.dpi": 300,
        "savefig.bbox": "tight",         "savefig.pad_inches": 0.02,
        "savefig.transparent": True,
        "figure.facecolor": "white",     "axes.facecolor": "white",
    })


configure_global_style()

# -- _draw_vlines helper (SKILL.md Â§5) --------------------------------------
def _draw_vlines(ax, vlines, default_color: str = "black",
                 default_ls: str = "--") -> None:
    """Draw vertical event lines with optional rotated labels.

    Each entry in *vlines* may be:
      float                  â†’ dashed line, no label
      (float, label)         â†’ dashed line + rotated label (default_color)
      (float, label, color)  â†’ dashed line + rotated label (custom color)
    """
    if vlines is None:
        return
    for entry in vlines:
        if isinstance(entry, (list, tuple)):
            vx    = float(entry[0])
            label = str(entry[1]) if len(entry) > 1 else None
            color = entry[2]      if len(entry) > 2 else default_color
        else:
            vx, label, color = float(entry), None, default_color
        ax.axvline(x=vx, color=color, linestyle=default_ls, lw=1.2)
        if label:
            ax.text(
                vx, 0.97, f"  {label}",
                rotation=90, va="top", ha="right", fontsize=16,
                color=color, transform=ax.get_xaxis_transform(),
            )


# -- Local shoelace (mirrors runner_fixed._shoelace) ------------------------
def _shoelace(x: np.ndarray, v: np.ndarray) -> float:
    """Signed shoelace area of the closed orbit (q, dq/dt)."""
    if len(x) < 3:
        return float("nan")
    return 0.5 * abs(float(
        np.dot(x, np.roll(v, -1)) - np.dot(v, np.roll(x, -1))
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# CASE SELECTOR — change only this line to switch between signals
# ═══════════════════════════════════════════════════════════════════════════════
_ACTIVE_CASE = "chatter_15mm"   # "cono" | "stable_5mm" | "chatter_15mm"

_BASE = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
_CASES = {
    "cono": {
        "hdf5":               rf"{_BASE}\2DOF_Cono\1DOF_150Hz\out.hdf5",
        "name":               "cono",
        "t_range":            (0.05, 16.0),
        "t_gt":               5.36577,
        "f_modal":            200.0,
        "num_T":              4,
        "dt_step":            0.005,
        "use_area_threshold": True,
        "training_intervals": [
            (0.05,    5.36577, "stable"),
            (5.36577, 16.0,    "chatter"),
        ],
        "t_stable":           4.0,
        "t_chatter":          8.0,
        "win_t":              6,
    },
    "stable_5mm": {
        "hdf5":               (rf"{_BASE}\Chatter-Criteria\CAMP8-Ventanna_Glisante"
                               r"\Nessy2m_Case_Test_Explicit\1DOF_150Hz_5mm\1DOF_150Hz\out.hdf5"),
        "name":               "5mm_stable",
        "t_range":            (0.05, 16.0),
        "t_gt":               None,
        "f_modal":            150.0,
        "num_T":              3,
        "dt_step":            0.005,
        "use_area_threshold": False,
        "training_intervals": [(0.05, 16.0, "stable")],
        "t_stable":           5.0,
        "t_chatter":          10.0,
        "win_t":              5.0,
    },
    "chatter_15mm": {
        "hdf5":               (rf"{_BASE}\Chatter-Criteria\CAMP8-Ventanna_Glisante"
                               r"\Nessy2m_Case_Test_Explicit\1DOF_150Hz_15mm\1DOF_150Hz\out.hdf5"),
        "name":               "15mm_chatter",
        "t_range":            (0.01, 0.88),
        "t_gt":               0.05,
        "f_modal":            150.0,
        "num_T":              16,
        "dt_step":            0.005,
        "use_area_threshold": False,
        "training_intervals": [
            (0.00, 0.00, "stable"),
            (0.05, 0.88, "chatter"),
        ],
        "t_stable":           0.025,
        "t_chatter":          0.50,
        "win_t":              0.40,
    },
}

_cfg        = _CASES[_ACTIVE_CASE]
_HDF5_PATH  = _cfg["hdf5"]
_SIG_NAME   = _cfg["name"]
_T0, _T1    = _cfg["t_range"]
_T_GT       = _cfg["t_gt"]           # None when no chatter expected
_F_MODAL    = _cfg["f_modal"]
_NUM_T      = _cfg["num_T"]
_DT_STEP    = _cfg["dt_step"]
_USE_THR    = _cfg.get("use_area_threshold", True)  # True only for cono
_TRAIN_IV   = _cfg.get("training_intervals", [])
_T_STABLE   = _cfg.get("t_stable", _T0)
_T_CHATTER  = _cfg.get("t_chatter", _T1)

# -------------------------------------------------------------------------------
# 1.  SIGNAL LOADING
# -------------------------------------------------------------------------------
data         = HDF5Reader(_HDF5_PATH)
tool_dyn     = data.get_element("tool_dyn/data")
t_raw        = tool_dyn[:, 0]
x_raw        = tool_dyn[:, 1]
v_raw        = data.get_element("tool_dyn_o/data")[:, 1]

_mask = (t_raw >= _T0) & (t_raw <= _T1)
t_arr = t_raw[_mask]
q_arr = x_raw[_mask]
v_arr = v_raw[_mask]

# StdSignalData: interfaz estándar CAMP10
# signal_analysis = desplazamiento; velocidad en meta["velocity"]
sig_std = StdSignalData(
    t_analysis=t_arr,
    signal_analysis=q_arr,
    path=_HDF5_PATH,
    fs=1.0 / float(t_arr[1] - t_arr[0]),
    meta={"velocity": v_arr, "name": _SIG_NAME},
)

# -------------------------------------------------------------------------------
# 2.  RUN INDICATOR
# -------------------------------------------------------------------------------
# Config estándar CAMP10 para FixedWindow — interfaz unificada f_cycle.
# f_cycle = f_modal → ventana por periodo modal  (T_cycle = T_modal)
# Equivalencia: dt = _DT_STEP × T_modal = step_cycles / f_cycle
config_std_fixed = {
    "func":       "FixedWindow",
    "params_physical": {
        "f_modal":          _F_MODAL,               # Hz — filtro bandpass y ciclo
        "f_cycle":          _F_MODAL,               # Hz — ventana por periodo modal
        "N_cycles_per_seg": _NUM_T,                 # ciclos por ventana
        "step_cycles":      _DT_STEP * _F_MODAL,    # step en ciclos modales
        "data_filtrated":       False,
        "lambda_ewma":          None,
        "accumulate":           False,
        "G_memory":             None,
        "sigma_method":         "ratio",
        "sigma_local_n":        5,
        "area_noise_eps":       1e-30,
        "use_area_threshold":   _USE_THR,
        "training_intervals":   _TRAIN_IV,
        "z_sigma":              3.0,
        "debug_level":          0,
    },
}

result_std = run_green_std(sig_std, config_std_fixed)

# Extraer campos desde IndicatorResult (interfaz estándar)
# result_std.t    = t_wins (tiempos de inicio de ventana)
# result_std.I_t  = sigma_ewma (exponente de Lyapunov suavizado)
# result_std.t_d  = tiempo de detección
# result_std.meta["raw_result"] = FixedWindowResult (acceso a areas, sigma, global_data)
result_fw = result_std.meta["raw_result"]   # FixedWindowResult completo

t_wins = np.asarray(result_std.t)             # == result_fw.t_wins
areas  = np.asarray(result_fw.areas)
sigma  = np.asarray(result_std.I_t)           # sigma_ewma (indicador estándar)
t_d    = result_std.t_d
thr    = (result_fw.global_data or {}).get("area_mu_3sigma", {})

# -- Derived signal geometry ------------------------------------------------
dt_sig   = float(t_arr[1] - t_arr[0])
T_window = _NUM_T / _F_MODAL          # [s]
N_win    = max(3, int(round(T_window / dt_sig)))
step     = max(1, int(round(_DT_STEP / dt_sig)))
valid    = np.isfinite(areas) & (areas > 0)

# -- Auto vlines (canonical) ------------------------------------------------
auto_vlines: list = []
if _T_GT is not None:
    auto_vlines.append((_T_GT, rf"$t_{{gt}}={_T_GT:.3f}$ s", "black"))
if t_d is not None:
    _ref = _T_GT if _T_GT is not None else 0.0
    _td_lbl = (
        rf"$t_d^+={t_d:.3f}$ s" if t_d > _ref
        else rf"$t_d={t_d:.3f}$ s"
    )
    auto_vlines.append((t_d, _td_lbl, color_orange))


# -- Window extraction helper -----------------------------------------------
def _get_window(t_target: float):
    """Return filtered (t_w, q_w, v_w, i_win) for the window closest to t_target."""
    i_win = int(np.argmin(np.abs(t_wins - t_target)))
    i_sig = i_win * step
    i_end = min(i_sig + N_win, len(t_arr))
    t_w = t_arr[i_sig:i_end].copy()
    q_w = q_arr[i_sig:i_end].copy()
    v_w = v_arr[i_sig:i_end].copy()
    if len(q_w) >= 7:
        q_w = savgol_filter_window(q_w)
        v_w = savgol_filter_window(v_w)
    return t_w, q_w, v_w, i_win


# -- Representative windows -------------------------------------------------
t_w_s, q_w_s, v_w_s, i_s = _get_window(_T_STABLE)
t_w_c, q_w_c, v_w_c, i_c = _get_window(_T_CHATTER)

A_stable  = _shoelace(q_w_s, v_w_s)
A_chatter = _shoelace(q_w_c, v_w_c)

# Shared axis limits for phase portrait comparison
_q_max = max(np.max(np.abs(q_w_s)), np.max(np.abs(q_w_c))) * 1.15
_v_max = max(np.max(np.abs(v_w_s)), np.max(np.abs(v_w_c))) * 1.15

print(
    f"  t_d = {t_d} s  |  A_stable = {A_stable:.3e}  "
    f"|  A_chatter = {A_chatter:.3e}  "
    f"|  ratio = {A_chatter/A_stable:.1f}x"
)

# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW TO VISUALISE  ← change here to explore different windows
# ═══════════════════════════════════════════════════════════════════════════════
_WIN_IDX    = None    # int (0-based) to select by index; None → use _WIN_T
_WIN_T      = _cfg["win_t"]  # default from case; override freely
_WIN_N_T    = _NUM_T  # modal periods included in the window (default = _NUM_T)
_WIN_FILTER = False   # True → apply Savitzky-Golay filter before Shoelace

# ── Resolve window ───────────────────────────────────────────────────────
if _WIN_IDX is not None:
    _i_win = int(np.clip(_WIN_IDX, 0, len(t_wins) - 1))
else:
    _i_win = int(np.argmin(np.abs(t_wins - _WIN_T)))

_i_sig    = _i_win * step
_N_nt     = max(3, int(round((_WIN_N_T / _F_MODAL) / dt_sig)))
_i_end    = min(_i_sig + _N_nt, len(t_arr))
_q_w      = q_arr[_i_sig:_i_end].copy()
_v_w      = v_arr[_i_sig:_i_end].copy()
if _WIN_FILTER and len(_q_w) >= 7:
    _q_w = savgol_filter_window(_q_w)
    _v_w = savgol_filter_window(_v_w)

_t_center = float(t_wins[_i_win])
_n_pts    = len(_q_w)
_A_k      = _shoelace(_q_w, _v_w)
_T_win_ms = _WIN_N_T / _F_MODAL * 1e3

# ── Automatic colour: blue = stable, orange = chatter ─────────────────────
_ref_t     = t_d if t_d is not None else (_T_GT if _T_GT is not None else float("inf"))
_orb_color = color_azul if _t_center < _ref_t else color_orange
_region    = "Stable" if (_T_GT is None or _t_center < _T_GT) else "Chatter"

# ── Axis limits: tight on real data + margin ────────────────────────────
_q_all  = np.append(_q_w, [_q_w[-1], _q_w[0]])   # orbit + closing vertices
_v_all  = np.append(_v_w, [_v_w[-1], _v_w[0]])
_q_span = max(float(np.max(_q_all) - np.min(_q_all)), 1e-20)
_v_span = max(float(np.max(_v_all) - np.min(_v_all)), 1e-20)
_mg     = 0.15
_xlim   = (float(np.min(_q_all)) - _q_span * _mg,
            float(np.max(_q_all)) + _q_span * _mg)
_ylim   = (float(np.min(_v_all)) - _v_span * _mg,
            float(np.max(_v_all)) + _v_span * _mg)

# ── Samples per modal period (shared by orbit + markers) ──────────────────
_T_modal_pts = max(1, int(round(1.0 / (_F_MODAL * dt_sig))))

# Colour per cycle: perceptually uniform gradient k=0→N_T-1
from matplotlib.colors import Normalize as _Norm
_cmap_cyc = plt.cm.plasma
_norm_cyc = _Norm(vmin=0, vmax=max(_WIN_N_T - 1, 1))

# Line style rotated each cycle (colour + style = double distinction)
_CYCLE_LS = ["-", "--", "-.", (0, (1, 1))]

# ═══════════════════════════════════════════════════════════════════════════════
# FIG — Phase diagram for the selected window
# ═══════════════════════════════════════════════════════════════════════════════
configure_global_style()
fig, ax = plt.subplots(figsize=fig_size(scale=2.8, ncols=1, base_width=3.4))

# 1. Shaded area — computed by the Shoelace formula
ax.fill(_q_w, _v_w, alpha=0.22, color=_orb_color, zorder=1,
        label=rf"Area  $A_k = {_A_k:.3e}$ m·m/s")

# 2. Orbit trace — plasma(k) colour + rotated line style
for _k in range(_WIN_N_T):
    _ks = _k * _T_modal_pts
    _ke = min((_k + 1) * _T_modal_pts + 1, len(_q_w))   # +1 para continuidad
    if _ks >= len(_q_w):
        break
    _col_k = _cmap_cyc(_norm_cyc(_k))
    ax.plot(
        _q_w[_ks:_ke], _v_w[_ks:_ke],
        color=_col_k, lw=2.0, zorder=2,
        ls=_CYCLE_LS[_k % len(_CYCLE_LS)],
        label=(rf"Cycle $k=0$…${_WIN_N_T-1}$" if _k == 0 else None),
    )

# 3. Artificial closing line  end → start  (distinct style)
ax.plot(
    [_q_w[-1], _q_w[0]], [_v_w[-1], _v_w[0]],
    color=color_red, lw=1.8, ls=(0, (5, 2)),
    zorder=3, label="Artificial closure  (end → start)",
)

# 4. Start marker — green circle
ax.scatter(
    [_q_w[0]], [_v_w[0]],
    color=color_verde, s=120, marker="o", zorder=6,
    edgecolors="white", linewidths=0.8,
    label=r"Start  $(q_0,\,\dot{q}_0)$",
)

# 5. End marker — red star
ax.scatter(
    [_q_w[-1]], [_v_w[-1]],
    color=color_red, s=160, marker="*", zorder=6,
    edgecolors="white", linewidths=0.5,
    label=r"End  $(q_{N-1},\,\dot{q}_{N-1})$",
)

# 6. Cycle-start markers  (k = 1 … N_T−1) — colour = plasma(k)
_period_idx = [k * _T_modal_pts for k in range(1, _WIN_N_T)
               if k * _T_modal_pts < len(_q_w)]
_first_marker_label = True
for _k_m, _i_m in enumerate(_period_idx, start=1):
    ax.scatter(
        [_q_w[_i_m]], [_v_w[_i_m]],
        color=_cmap_cyc(_norm_cyc(_k_m)), s=60, marker="D", zorder=5,
        edgecolors="white", linewidths=0.6,
        label=(rf"Cycle start $k$" if _first_marker_label else None),
    )
    _first_marker_label = False

# 7. Rotation direction arrow (at ~35 % of the path, colour of the middle cycle)
_mid      = max(1, int(len(_q_w) * 0.35))
_k_arrow  = min(_mid // max(_T_modal_pts, 1), _WIN_N_T - 1)
_col_arr  = _cmap_cyc(_norm_cyc(_k_arrow))
ax.annotate(
    "", xy=(_q_w[_mid], _v_w[_mid]),
    xytext=(_q_w[_mid - 1], _v_w[_mid - 1]),
    arrowprops=dict(arrowstyle="-|>", color=_col_arr, lw=2.4),
    zorder=5,
)

# ── Axes ──────────────────────────────────────────────────────────────────
ax.axhline(0, color="lightgray", lw=0.6, ls=":")
ax.axvline(0, color="lightgray", lw=0.6, ls=":")
ax.set_xlim(*_xlim)
ax.set_ylim(*_ylim)
ax.set_xlabel(r"$q$ [m]", fontsize=20)
ax.set_ylabel(r"$\dot{q}$ [m/s]", fontsize=20)
ax.tick_params(labelsize=16)

# ── Title ─────────────────────────────────────────────────────────────────
ax.set_title(
    rf"Phase orbit — window $i = {_i_win}$"
    rf"   ($t \approx {_t_center:.3f}$ s,  {_region})"
    "\n"
    rf"$N_T = {_WIN_N_T}$,  $T_{{win}} = {_T_win_ms:.0f}$ ms,"
    rf"  $N_{{pts}} = {_n_pts}$,  $A_k = {_A_k:.3e}$ m·m/s",
    fontsize=15,
)
# ── Cycle colorbar (k = 0 … N_T−1) ──────────────────────────────────────────
_sm_cyc = plt.cm.ScalarMappable(cmap=_cmap_cyc, norm=_norm_cyc)
_sm_cyc.set_array([])
_cb = fig.colorbar(_sm_cyc, ax=ax, shrink=0.75, pad=0.02, aspect=28)
_cb.set_label(r"Cycle $k$", fontsize=14)
_cb_ticks = sorted(set([0, _WIN_N_T // 4, _WIN_N_T // 2,
                         3 * _WIN_N_T // 4, _WIN_N_T - 1]))
_cb.set_ticks(_cb_ticks)
_cb.ax.tick_params(labelsize=12)
# ── Legend ───────────────────────────────────────────────────────────────
ax.legend(loc="lower right", fontsize=13, frameon=True,
          framealpha=0.88, edgecolor="lightgray")

# ── Time reference (upper-left corner) ───────────────────────────────────
_ref_txt = rf"$t_{{gt}} = {_T_GT:.3f}$ s" if _T_GT is not None else ""
if t_d is not None:
    _sep = "\n" if _ref_txt else ""
    _ref_txt += _sep + rf"$t_d = {t_d:.3f}$ s"
ax.text(
    0.02, 0.98, _ref_txt,
    transform=ax.transAxes, va="top", fontsize=13, color="gray",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.80),
)

plt.tight_layout()
_fname = f"fig_orbit_win{_i_win}.png"
# fig.savefig(_fname, dpi=300, bbox_inches="tight")
# print(f"Saved: {_fname}")
# plt.show(block=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Area Ak vs time  (all windows)
# ═══════════════════════════════════════════════════════════════════════════════
configure_global_style()
fig2, ax2 = plt.subplots(figsize=fig_size(scale=2.8, ncols=1, base_width=3.4))

# ── Colour each point by region (stable=blue, chatter=orange) ────────────
if _T_GT is not None:
    _mask_s = valid & (t_wins <= _T_GT)
    _mask_c = valid & (t_wins >  _T_GT)
else:
    _mask_s = valid
    _mask_c = np.zeros_like(valid, dtype=bool)

ax2.semilogy(t_wins[_mask_s], areas[_mask_s],
             color=color_azul,   lw=1.4, label="Stable region")
ax2.semilogy(t_wins[_mask_c], areas[_mask_c],
             color=color_orange, lw=1.4, label="Chatter region")

# ── Threshold lines (mu, mu±z·sigma) ─────────────────────────────────────
if thr:
    z_lbl   = f"{thr['z']:.0f}"
    y_upper = 10 ** thr["upper"]
    y_lower = 10 ** thr["lower"]
    y_mu    = 10 ** thr["mu"]
    ax2.axhline(y_upper, color=color_red,   ls="--", lw=1.4)
    ax2.text(0.01, y_upper, rf"$\mu+{z_lbl}\sigma={thr['upper']:.3g}$",
             transform=ax2.get_yaxis_transform(),
             color=color_red, ha='left', va='bottom', fontsize=14)
    ax2.axhline(y_lower, color=color_red,   ls=":",  lw=1.2)
    ax2.text(0.01, y_lower, rf"$\mu-{z_lbl}\sigma={thr['lower']:.3g}$",
             transform=ax2.get_yaxis_transform(),
             color=color_red, ha='left', va='top', fontsize=14)
    ax2.axhline(y_mu, color=color_verde, ls="-", lw=1.0)
    ax2.text(0.01, y_mu, rf"$\mu={thr['mu']:.3g}$",
             transform=ax2.get_yaxis_transform(),
             color=color_verde, ha='left', va='bottom', fontsize=14)

# ── Mark the selected window ──────────────────────────────────────────────
ax2.axvline(_t_center, color=color_purple, ls="--", lw=1.2)
ax2.text(_t_center, 0.97, rf"  $t_{{win}}={_t_center:.3f}$ s",
         rotation=90, va="top", ha="right", fontsize=13,
         color=color_purple, transform=ax2.get_xaxis_transform())
ax2.scatter([_t_center], [_A_k], color=color_purple, s=80, zorder=6,
            label=rf"Selected window ($A_k={_A_k:.2e}$)")

# ── Event vlines ──────────────────────────────────────────────────────────
_draw_vlines(ax2, auto_vlines)

# ── Axes / labels ─────────────────────────────────────────────────────────
ax2.set_xlabel(r"Time $t$ [s]", fontsize=22)
ax2.set_ylabel(r"Area $A_k$ [m$\cdot$m/s]", fontsize=22)
ax2.tick_params(labelsize=18)
ax2.set_title(
    rf"Green-integral area per window — {_SIG_NAME}"
    "\n"
    rf"$N_T={_NUM_T}$,  $T_{{win}}={_NUM_T/_F_MODAL*1e3:.0f}$ ms,"
    rf"  step $={_DT_STEP*1e3:.1f}$ ms",
    fontsize=15,
)
ax2.legend(fontsize=14)
ax2.grid(which="both", ls=":", lw=0.4, color="lightgray")

plt.tight_layout()
_fname2 = f"fig_areas_{_SIG_NAME}.png"
# fig2.savefig(_fname2, dpi=300, bbox_inches="tight")
# print(f"Saved: {_fname2}")
plt.show(block=True)

