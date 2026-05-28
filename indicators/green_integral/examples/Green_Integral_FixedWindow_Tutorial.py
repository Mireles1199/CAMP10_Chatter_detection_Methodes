"""Green Integral Fixed-Window â€” Tutorial pedagÃ³gico.

Genera UN diagrama de fase de la ventana seleccionada, mostrando:
  - el área sombreada que calcula el Shoelace,
  - la traza de la órbita,
  - el marcador de inicio (●, verde) y de fin (★, rojo),
  - la línea de cierre artificial fin→inicio,
  - la flecha de sentido de giro.

La ventana se elige mediante _WIN_IDX (índice) o _WIN_T (tiempo aproximado).

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
    SignalData,
    run_fixed_window,
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


def configurar_estilo_global() -> None:
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


configurar_estilo_global()

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


# -------------------------------------------------------------------------------
# 1.  SIGNAL LOADING
# -------------------------------------------------------------------------------
_DIR_CONO = (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    r"\2DOF_Cono\1DOF_150Hz"
)
_HDF5_PATH = os.path.join(_DIR_CONO, "out.hdf5")

data         = HDF5Reader(_HDF5_PATH)
tool_dyn     = data.get_element("tool_dyn/data")
t_raw        = tool_dyn[:, 0]
x_raw        = tool_dyn[:, 1]
v_raw        = data.get_element("tool_dyn_o/data")[:, 1]

_T0, _T1 = 0.05, 16.0
_mask = (t_raw >= _T0) & (t_raw <= _T1)
t_arr = t_raw[_mask]
q_arr = x_raw[_mask]
v_arr = v_raw[_mask]

_T_GT: float = 5.36577          # [s]  â€” onset anotado del chatter

sig = SignalData(t=t_arr, displacement=q_arr, velocity=v_arr, name="cono")

# -------------------------------------------------------------------------------
# 2.  RUN INDICATOR
# -------------------------------------------------------------------------------
_F_MODAL = 150.0      # [Hz]
_NUM_T   = 6         # N periodos por ventana
_DT_STEP = 0.005      # [s] paso de ventana

config_fixed = {
    "func": "FixedWindow",
    "params": {
        "f_modal":            _F_MODAL,
        "num_T":              _NUM_T,
        "dt":                 _DT_STEP,
        "data_filtrated":     False,
        "lambda_ewma":        None,
        "accumulate":         False,
        "G_memory":           None,
        "sigma_method":       "ratio",
        "sigma_local_n":      5,
        "area_noise_eps":     1e-30,
        "use_area_threshold": True,
        "training_intervals": [
            (0.05,  _T_GT, "stable"),
            (_T_GT, 10.0,  "chatter"),
        ],
        "z_sigma":            3.0,
        "debug_level":        0,
    },
}

result = run_fixed_window(sig, config_fixed)

t_wins = np.asarray(result.t_wins)
areas  = np.asarray(result.areas)
sigma  = np.asarray(result.sigma)
t_d    = result.t_d
thr    = (result.global_data or {}).get("area_mu_3sigma", {})

# -- Derived signal geometry ------------------------------------------------
dt_sig   = float(t_arr[1] - t_arr[0])
T_window = _NUM_T / _F_MODAL          # [s]
N_win    = max(3, int(round(T_window / dt_sig)))
step     = max(1, int(round(_DT_STEP / dt_sig)))
valid    = np.isfinite(areas) & (areas > 0)

# -- Auto vlines (canonical) ------------------------------------------------
auto_vlines: list = [(_T_GT, rf"$t_{{gt}}={_T_GT:.3f}$ s", "black")]
if t_d is not None:
    _td_lbl = (
        rf"$t_d^+={t_d:.3f}$ s" if t_d > _T_GT
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
_T_STABLE  = 2.0   # well inside stable region
_T_CHATTER = 8.0   # well inside chatter region

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
# VENTANA A VISUALIZAR  ← cambiar aquí para explorar distintas ventanas
# ═══════════════════════════════════════════════════════════════════════════════
_WIN_IDX    = None   # int (0-based) para elegir por índice; None → usar _WIN_T
_WIN_T      = 4.00   # tiempo aproximado [s]; ignorado si _WIN_IDX no es None
_WIN_N_T    = _NUM_T # periodos modales incluidos en la ventana (default = 16)
_WIN_FILTER = False   # True → aplica filtro Savitzky-Golay antes del Shoelace

# ── Resolver ventana ───────────────────────────────────────────────────────
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

# ── Color automático: azul = estable, naranja = chatter ───────────────────
_ref_t     = t_d if t_d is not None else _T_GT
_orb_color = color_azul if _t_center < _ref_t else color_orange
_region    = "Estable" if _t_center < _T_GT else "Chatter"

# ── Límites de ejes: tight sobre los datos reales + margen ──────────────
_q_all  = np.append(_q_w, [_q_w[-1], _q_w[0]])   # órbita + vértices cierre
_v_all  = np.append(_v_w, [_v_w[-1], _v_w[0]])
_q_span = max(float(np.max(_q_all) - np.min(_q_all)), 1e-20)
_v_span = max(float(np.max(_v_all) - np.min(_v_all)), 1e-20)
_mg     = 0.15
_xlim   = (float(np.min(_q_all)) - _q_span * _mg,
            float(np.max(_q_all)) + _q_span * _mg)
_ylim   = (float(np.min(_v_all)) - _v_span * _mg,
            float(np.max(_v_all)) + _v_span * _mg)

# ── Muestras por periodo modal (compartido por órbita + markers) ─────────────
_T_modal_pts = max(1, int(round(1.0 / (_F_MODAL * dt_sig))))

# Color por ciclo: gradiente perceptualmente uniforme k=0→N_T-1
from matplotlib.colors import Normalize as _Norm
_cmap_cyc = plt.cm.plasma
_norm_cyc = _Norm(vmin=0, vmax=max(_WIN_N_T - 1, 1))

# Estilo de línea que rota cada ciclo (color + estilo = doble distincción)
_CYCLE_LS = ["-", "--", "-.", (0, (1, 1))]

# ═══════════════════════════════════════════════════════════════════════════════
# FIG — Diagrama de fase de la ventana seleccionada
# ═══════════════════════════════════════════════════════════════════════════════
configurar_estilo_global()
fig, ax = plt.subplots(figsize=fig_size(scale=2.8, ncols=1, base_width=3.4))

# 1. Área sombreada — lo que calcula el Shoelace
ax.fill(_q_w, _v_w, alpha=0.22, color=_orb_color, zorder=1,
        label=rf"Área  $A_k = {_A_k:.3e}$ m·m/s")

# 2. Traza de la órbita — color plasma(k) + estilo de línea rotado
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
        label=(rf"Ciclo $k=0$…${_WIN_N_T-1}$" if _k == 0 else None),
    )

# 3. Línea de cierre artificial  fin → inicio  (estilo distinto)
ax.plot(
    [_q_w[-1], _q_w[0]], [_v_w[-1], _v_w[0]],
    color=color_red, lw=1.8, ls=(0, (5, 2)),
    zorder=3, label="Cierre artificial  (fin → inicio)",
)

# 4. Marcador inicio — círculo verde
ax.scatter(
    [_q_w[0]], [_v_w[0]],
    color=color_verde, s=120, marker="o", zorder=6,
    edgecolors="white", linewidths=0.8,
    label=r"Inicio  $(q_0,\,\dot{q}_0)$",
)

# 5. Marcador final — estrella roja
ax.scatter(
    [_q_w[-1]], [_v_w[-1]],
    color=color_red, s=160, marker="*", zorder=6,
    edgecolors="white", linewidths=0.5,
    label=r"Fin  $(q_{N-1},\,\dot{q}_{N-1})$",
)

# 6. Marcadores de inicio de cada ciclo  (k = 1 … N_T−1) — color = plasma(k)
_period_idx = [k * _T_modal_pts for k in range(1, _WIN_N_T)
               if k * _T_modal_pts < len(_q_w)]
_first_marker_label = True
for _k_m, _i_m in enumerate(_period_idx, start=1):
    ax.scatter(
        [_q_w[_i_m]], [_v_w[_i_m]],
        color=_cmap_cyc(_norm_cyc(_k_m)), s=60, marker="D", zorder=5,
        edgecolors="white", linewidths=0.6,
        label=(rf"Inicio ciclo $k$" if _first_marker_label else None),
    )
    _first_marker_label = False

# 7. Flecha de sentido de giro (al ~35 % del recorrido, color del ciclo medio)
_mid      = max(1, int(len(_q_w) * 0.35))
_k_arrow  = min(_mid // max(_T_modal_pts, 1), _WIN_N_T - 1)
_col_arr  = _cmap_cyc(_norm_cyc(_k_arrow))
ax.annotate(
    "", xy=(_q_w[_mid], _v_w[_mid]),
    xytext=(_q_w[_mid - 1], _v_w[_mid - 1]),
    arrowprops=dict(arrowstyle="-|>", color=_col_arr, lw=2.4),
    zorder=5,
)

# ── Ejes ──────────────────────────────────────────────────────────────────
ax.axhline(0, color="lightgray", lw=0.6, ls=":")
ax.axvline(0, color="lightgray", lw=0.6, ls=":")
ax.set_xlim(*_xlim)
ax.set_ylim(*_ylim)
ax.set_xlabel(r"$q$ [m]", fontsize=20)
ax.set_ylabel(r"$\dot{q}$ [m/s]", fontsize=20)
ax.tick_params(labelsize=16)

# ── Título ────────────────────────────────────────────────────────────────
ax.set_title(
    rf"Órbita de fase — ventana $i = {_i_win}$"
    rf"   ($t \approx {_t_center:.3f}$ s,  {_region})"
    "\n"
    rf"$N_T = {_WIN_N_T}$,  $T_{{win}} = {_T_win_ms:.0f}$ ms,"
    rf"  $N_{{pts}} = {_n_pts}$,  $A_k = {_A_k:.3e}$ m·m/s",
    fontsize=15,
)
# ── Colorbar de ciclos (k = 0 … N_T−1) ───────────────────────────────────────
_sm_cyc = plt.cm.ScalarMappable(cmap=_cmap_cyc, norm=_norm_cyc)
_sm_cyc.set_array([])
_cb = fig.colorbar(_sm_cyc, ax=ax, shrink=0.75, pad=0.02, aspect=28)
_cb.set_label(r"Ciclo $k$", fontsize=14)
_cb_ticks = sorted(set([0, _WIN_N_T // 4, _WIN_N_T // 2,
                         3 * _WIN_N_T // 4, _WIN_N_T - 1]))
_cb.set_ticks(_cb_ticks)
_cb.ax.tick_params(labelsize=12)
# ── Leyenda ───────────────────────────────────────────────────────────────
ax.legend(loc="lower right", fontsize=13, frameon=True,
          framealpha=0.88, edgecolor="lightgray")

# ── Referencia temporal (esquina superior izquierda) ─────────────────────
_ref_txt = rf"$t_{{gt}} = {_T_GT:.3f}$ s"
if t_d is not None:
    _ref_txt += "\n" + rf"$t_d = {t_d:.3f}$ s"
ax.text(
    0.02, 0.98, _ref_txt,
    transform=ax.transAxes, va="top", fontsize=13, color="gray",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.80),
)

plt.tight_layout()
_fname = f"fig_orbit_win{_i_win}.png"
fig.savefig(_fname, dpi=300, bbox_inches="tight")
print(f"Guardado: {_fname}")
plt.show(block=True)

