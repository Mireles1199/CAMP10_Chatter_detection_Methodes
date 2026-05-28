"""Phase-Area Indicator — medida continua de distancia al límite de estabilidad.

Contexto
--------
Sistema DDE puramente regenerativo (1DOF, cono, ap variable):

    m ẍ + c ẋ + k x = Kc·ap(t)·[x(t-T) − x(t)]

La órbita en el plano de fase (x, ẋ) tiene un área proporcional a la energía
del modo chatter.  Cuando ap → ap_lim, el amortiguamiento efectivo
ζ_eff → 0 y el área diverge.  El área es por tanto un proxy continuo de
"distancia al límite de estabilidad".

Tres estimadores del área por ventana deslizante
-------------------------------------------------
  A_shoelace  — fórmula de Shoelace (teorema de Green) dividida entre
                el número de ciclos en la ventana.  Exacta para curvas
                cerradas; aquí aproxima el área media por ciclo.

  A_ellipse   — π · √det(Σ)  donde Σ es la matriz de covarianza 2×2 de
                (x, ẋ) en la ventana.  Equivale al área de la elipse de
                inercia (robusta a inclinación y excentricidad).

  A_rms       — π · RMS_x · RMS_v  (aproximación analítica para una
                elipse pura con semiejes √2·RMS_x y √2·RMS_v).

Normalización opcional
----------------------
  Si se proporciona ap(t), se calcula también:
      A_norm = A / (Kc · ap(t_w))
  que cancela la contribución de la fuerza creciente y deja visible solo
  el efecto de ζ_eff → 0.

Parámetros configurables
------------------------
  N_CYCLES   — longitud de ventana en ciclos de f_n  (default: 20)
  STEP_CYCLES — paso entre ventanas en ciclos de f_n  (default: 5)

Uso
---
    python examples/phase_area_indicator.py
"""

from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import colorsys

# ── Path setup ──────────────────────────────────────────────────────────────
_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral import HDF5Reader
from green_integral.lib.runner_fixed import run_fixed_window
from green_integral.utils.types import SignalData, FixedWindowConfig

# ══════════════════════════════════════════════════════════════════════════════
# Estilo CAMP10
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib as mpl

mpl.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        110,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Paleta de colores ────────────────────────────────────────────────────────
def _hls(h_deg, l, s):
    r, g, b = colorsys.hls_to_rgb(h_deg / 360, l, s)
    return (r, g, b)

color_azul   = _hls(207, 0.409, 0.556)   # zona estable
color_orange = _hls(36,  0.45,  0.99)    # zona chatter
color_verde  = _hls(98,  0.36,  0.99)    # A_shoelace
color_purple = _hls(279, 0.36,  0.99)    # A_ellipse
color_red    = _hls(346, 0.45,  0.99)    # A_rms
color_brown  = _hls(30,  0.30,  0.70)    # A_norm

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros del experimento — AJUSTA AQUÍ
# ══════════════════════════════════════════════════════════════════════════════
_spd_var = (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
    r"\CAMP8-Ventanna_Glisante\Nessy2m_Case_Test_Explicit"
    r"\1DOF_150Hz_20mm_7.5k-12kSpdS_100_F-0_05_L-50mm_Statico\1DOF_150Hz"
)

_DIR_CONO = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"
work_space_5mm   = 'D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Chatter-Criteria/CAMP8-Ventanna_Glisante/Nessy2m_Case_Test_Explicit/1DOF_150Hz_5mm/1DOF_150Hz'
work_space_15mm   = 'D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Chatter-Criteria/CAMP8-Ventanna_Glisante/Nessy2m_Case_Test_Explicit/1DOF_150Hz_15mm/1DOF_150Hz'


_DIR_USE   = _DIR_CONO


_T_START   = 0.05       # [s] inicio de análisis
_T_END     = 15      # [s] fin de análisis
_T_GT      = 5.36577    # [s] onset ground-truth
# _T_GT      = 0    # [s] onset ground-truth
_F_N       = 150.0      # [Hz] frecuencia natural (modo chatter)
_KC        = 1.0        # [N/m²] coeficiente de corte — ajusta si lo conoces
                        #        solo afecta A_norm; 1.0 → A_norm en unidades relativas

# Cono: ap lineal de AP0 (inicio) a AP_MAX (fin del corte)
_AP0       = 5e-3       # [m]  5  mm
_AP_MAX    = 15e-3      # [m]  5  mm

# ── Parámetros de la ventana deslizante ─────────────────────────────────────
N_CYCLES    = 5    # ciclos de f_n por ventana   → ~133 ms a 150 Hz
STEP_CYCLES = 1    # paso entre ventanas en ciclos

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cut(t_full, x_full, t0, t1):
    m = (t_full >= t0) & (t_full <= t1)
    return t_full[m], x_full[m]


def _ap_linear(t):
    """ap(t) lineal del cono [m]."""
    alpha = (_AP_MAX - _AP0) / (_T_END - _T_START)
    return _AP0 + alpha * (t - _T_START)


def _area_ellipse(x, v):
    """Área de la elipse de inercia: π · √det(Σ) donde Σ = cov([x, v])."""
    x0 = x - x.mean()
    v0 = v - v.mean()
    cov = np.cov(x0, v0)
    det = max(np.linalg.det(cov), 0.0)
    return np.pi * np.sqrt(det)


def _area_rms(x, v):
    """Aproximación analítica: π · RMS_x · RMS_v."""
    return np.pi * np.sqrt(np.mean(x**2)) * np.sqrt(np.mean(v**2))


def _area_green_fixed(t, x, v, f_n, n_cycles, step_cycles):
    """Área por ventana usando run_fixed_window + estimadores ellipse y RMS.

    Returns
    -------
    t_wins   : tiempos de inicio de cada ventana [s]
    A_gi     : área Shoelace (Green-Fixed) por ventana [m·m/s]
    A_el     : área elipse (π√det Σ) por ventana [m·m/s]
    A_rm     : área RMS (π·RMS_x·RMS_v) por ventana [m·m/s]
    sigma    : exponente de Lyapunov σ̂ por ventana [1/s]
    sigma_ew : σ̂ suavizado con EWMA (lambda=0.3)
    """
    signal = SignalData(t=t, displacement=x, velocity=v, name="cono_fixed")
    T_modal = 1.0 / f_n
    dt_step = step_cycles * T_modal

    cfg = FixedWindowConfig(
        f_modal          = f_n,
        num_T            = n_cycles,
        dt               = dt_step,
        data_filtrated   = True,
        lambda_ewma      = None,
        accumulate       = False,
        sigma_method     = "ratio",
        area_noise_eps   = 1e-30,
    )
    result = run_fixed_window(signal, cfg)

    # Compute ellipse & RMS on the same windows
    dt_sig = float(t[1] - t[0])
    N_win  = max(3, int(round(cfg.T_window / dt_sig)))
    dt_samp = max(1, int(round(dt_step / dt_sig)))
    A_el_list, A_rm_list = [], []
    i = 0
    while i + N_win <= len(t):
        xw = x[i:i + N_win]
        vw = v[i:i + N_win]
        A_el_list.append(_area_ellipse(xw, vw))
        A_rm_list.append(_area_rms(xw, vw))
        i += dt_samp

    A_el = np.array(A_el_list)
    A_rm = np.array(A_rm_list)
    # Trim/pad to match run_fixed_window window count
    n = len(result.t_wins)
    A_el = A_el[:n]
    A_rm = A_rm[:n]

    return result.t_wins, result.areas, A_el, A_rm, result.sigma, result.sigma_ewma


def _mk_fig(title, figsize=(14, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.manager.set_window_title(title)
    return fig, ax


def _draw_tgt(ax, t_gt):
    ax.axvline(t_gt, color='k', lw=1.4, ls='--', zorder=5)
    ax.text(t_gt + 0.05, ax.get_ylim()[1] * 0.97,
            f'$t_{{GT}}={t_gt:.2f}$ s', va='top', fontsize=9)


def _shade(ax, t_start, t_gt, t_end):
    ax.axvspan(t_start, t_gt,  alpha=0.06, color=color_azul)
    ax.axvspan(t_gt,    t_end, alpha=0.06, color=color_orange)


# ══════════════════════════════════════════════════════════════════════════════
# Carga de señal
# ══════════════════════════════════════════════════════════════════════════════
_data_path = os.path.abspath(os.path.join(_DIR_USE, "out.hdf5"))
_data      = HDF5Reader(_data_path)

_raw    = _data.get_element("tool_dyn/data")
_t_full = _raw[:, 0]
_x_full = _raw[:, 1]                                      # desplazamiento [m]
_v_full = _data.get_element("tool_dyn_o/data")[:, 1]     # velocidad [m/s]

_fs = 1.0 / float(_t_full[1] - _t_full[0])
print(f"fs = {_fs:.1f} Hz  |  duración = {_t_full[-1]:.2f} s")

t, x = _cut(_t_full, _x_full, _T_START, _T_END)
_, v = _cut(_t_full, _v_full, _T_START, _T_END)

ap_t = _ap_linear(t)
ap_gt = _ap_linear(_T_GT)
print(f"ap en onset  t_GT = {_T_GT:.3f} s  →  ap_lim ≈ {ap_gt*1e3:.2f} mm")

# ══════════════════════════════════════════════════════════════════════════════
# Cómputo del indicador
# ══════════════════════════════════════════════════════════════════════════════
print(f"Ventana: {N_CYCLES} ciclos = {N_CYCLES / _F_N * 1e3:.1f} ms  |  "
      f"paso: {STEP_CYCLES} ciclos = {STEP_CYCLES / _F_N * 1e3:.1f} ms")

t_w, A_gi, A_el, A_rm, sigma, sigma_ew = _area_green_fixed(
    t, x, v, _F_N, N_CYCLES, STEP_CYCLES
)

# Normalización por Kc·ap(t)  →  aísla 1/ζ_eff
ap_w   = _ap_linear(t_w)
denom  = _KC * ap_w
A_gi_n = A_gi / denom
A_el_n = A_el / denom
A_rm_n = A_rm / denom

# Máscara estable / chatter en la ventana
m_st = t_w <= _T_GT
m_ch = t_w >  _T_GT

_XLIM = (_T_START, _T_END)

# ══════════════════════════════════════════════════════════════════════════════
# Figuras
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig A0 — Señal cruda (contexto) ─────────────────────────────────────────
fig_a0, ax_a0 = _mk_fig('Fig A0 — Señal cruda x(t)  (contexto)')
_shade(ax_a0, _T_START, _T_GT, _T_END)
ax_a0.plot(t[t <= _T_GT],  x[t <= _T_GT],  color=color_azul,   lw=0.5, label='estable')
ax_a0.plot(t[t >  _T_GT],  x[t >  _T_GT],  color=color_orange, lw=0.5, label='chatter')
ax_a0.set_xlabel('Tiempo [s]')
ax_a0.set_ylabel('$x$ [m]')
ax_a0.set_title('Fig A0 — Desplazamiento crudo  |  cono 1DOF 150 Hz', fontsize=12)
ax_a0.legend(ncol=2)
ax_a0.set_xlim(_XLIM)
_draw_tgt(ax_a0, _T_GT)

# ── Fig A1 — Tres estimadores de área (sin normalizar) ──────────────────────
fig_a1, ax_a1 = _mk_fig(
    f'Fig A1 — Área de órbita  (ventana={N_CYCLES} ciclos, paso={STEP_CYCLES} ciclos)')
_shade(ax_a1, _T_START, _T_GT, _T_END)
ax_a1.plot(t_w, A_gi, color=color_verde,  lw=1.5, label='Green-Fixed (Shoelace)')
ax_a1.plot(t_w, A_el, color=color_purple, lw=1.5, label=r'Elipse  $\pi\sqrt{\det\Sigma}$', ls='--')
ax_a1.plot(t_w, A_rm, color=color_red,    lw=1.5, label=r'RMS  $\pi\cdot\mathrm{RMS}_x\cdot\mathrm{RMS}_v$', ls=':')
ax_a1.set_xlabel('Tiempo [s]')
ax_a1.set_ylabel(r'Área  $[\mathrm{m}^2/\mathrm{s}]$')
ax_a1.set_title(
    rf'Fig A1 — Área de la órbita (x, ẋ)  |  $N={N_CYCLES}$ ciclos  |  sin normalizar',
    fontsize=12)
ax_a1.legend(ncol=3)
ax_a1.set_xlim(_XLIM)
_draw_tgt(ax_a1, _T_GT)

# ── Fig A2 — Tres estimadores normalizados + exponente de Lyapunov ────────────
fig_a2, (ax_a2a, ax_a2b) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig_a2.canvas.manager.set_window_title(
    'Fig A2 — Área normalizada  +  exponente de Lyapunov')

_shade(ax_a2a, _T_START, _T_GT, _T_END)
ax_a2a.plot(t_w, A_gi_n, color=color_verde,  lw=1.5, label='Green-Fixed / $K_c a_p$')
ax_a2a.plot(t_w, A_el_n, color=color_purple, lw=1.5, label=r'Elipse / $K_c a_p$', ls='--')
ax_a2a.plot(t_w, A_rm_n, color=color_red,    lw=1.5, label=r'RMS / $K_c a_p$', ls=':')
ax_a2a.set_ylabel(r'$A\,/\,(K_c a_p)$  $[\mathrm{m}/\mathrm{s}\cdot\mathrm{N}^{-1}]$')
ax_a2a.set_title(
    r'Área normalizada  $A/(K_c a_p)$  — cancela efecto de fuerza creciente',
    fontsize=11)
ax_a2a.legend(ncol=3)
ax_a2a.axvline(_T_GT, color='k', lw=1.2, ls='--')

_shade(ax_a2b, _T_START, _T_GT, _T_END)
ax_a2b.plot(t_w[m_st], sigma_ew[m_st], color=color_azul,   lw=1.5, label=r'$\hat{\sigma}_\mathrm{EWMA}$ (estable)')
ax_a2b.plot(t_w[m_ch], sigma_ew[m_ch], color=color_orange, lw=1.5, label=r'$\hat{\sigma}_\mathrm{EWMA}$ (chatter)')
ax_a2b.axhline(0, color='k', lw=0.8, ls=':')
ax_a2b.set_ylabel(r'$\hat{\sigma}$  [1/s]')
ax_a2b.set_xlabel('Tiempo [s]')
ax_a2b.set_title(r'Exponente de Lyapunov $\hat{\sigma}$ (Green-Fixed, EWMA $\lambda=0.3$)', fontsize=11)
ax_a2b.legend()
ax_a2b.axvline(_T_GT, color='k', lw=1.2, ls='--')
ax_a2b.set_xlim(_XLIM)

fig_a2.suptitle(
    rf'Fig A2 — Estimadores normalizados  |  $N={N_CYCLES}$ ciclos  |  cono 1DOF 150 Hz',
    fontsize=13, fontweight='bold')
fig_a2.tight_layout()

# ── Fig A3 — Panel combinado: señal + área + ap(t) ───────────────────────────
fig_a3, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig_a3.canvas.manager.set_window_title(
    'Fig A3 — Panel combinado: señal | área Green-Fixed | ap(t)')

# Subplot 1: señal
_shade(axes[0], _T_START, _T_GT, _T_END)
axes[0].plot(t[t <= _T_GT], x[t <= _T_GT], color=color_azul,   lw=0.5, label='estable')
axes[0].plot(t[t >  _T_GT], x[t >  _T_GT], color=color_orange, lw=0.5, label='chatter')
axes[0].set_ylabel('$x$ [m]')
axes[0].set_title('Desplazamiento crudo', fontsize=11)
axes[0].legend(ncol=2, loc='upper left')
axes[0].axvline(_T_GT, color='k', lw=1.2, ls='--')

# Subplot 2: área elipse (determinante), más robusta
_shade(axes[1], _T_START, _T_GT, _T_END)
axes[1].plot(t_w[m_st], A_el[m_st], color=color_azul,   lw=1.5)
axes[1].plot(t_w[m_ch], A_el[m_ch], color=color_orange, lw=1.5)
axes[1].set_ylabel(r'$A_\mathrm{elipse}$  $[\mathrm{m}^2/\mathrm{s}]$')
axes[1].set_title(
    rf'Área elipse  $\pi\sqrt{{\det\Sigma}}$  |  $N={N_CYCLES}$ ciclos',
    fontsize=11)
axes[1].axvline(_T_GT, color='k', lw=1.2, ls='--')

# Subplot 3: ap(t)
axes[2].plot(t, ap_t * 1e3, color='dimgray', lw=1.5)
axes[2].axhline(ap_gt * 1e3, color='k', lw=1.0, ls=':', label=f'$a_{{p,\\mathrm{{lim}}}}={ap_gt*1e3:.1f}$ mm')
axes[2].set_ylabel('$a_p(t)$  [mm]')
axes[2].set_xlabel('Tiempo [s]')
axes[2].set_title('Profundidad de corte del cono (lineal)', fontsize=11)
axes[2].legend()
axes[2].axvline(_T_GT, color='k', lw=1.2, ls='--')
axes[2].set_xlim(_XLIM)

fig_a3.suptitle(
    'Fig A3 — Indicador de área de órbita  |  cono 1DOF 150 Hz',
    fontsize=13, fontweight='bold')
fig_a3.tight_layout()

# Diagnóstico en consola
i_gt = np.argmin(np.abs(t_w - _T_GT))
i_0  = 0
print("\n── Diagnóstico del indicador ──")
print(f"{'':20s}  {'t_start':>12s}  {'t_GT':>12s}  {'t_END':>12s}")
print(f"{'A_gi (Green-Fixed)':20s}  {A_gi[i_0]:.3e}  {A_gi[i_gt]:.3e}  {A_gi[-1]:.3e}  m²/s")
print(f"{'A_el (elipse)':20s}  {A_el[i_0]:.3e}  {A_el[i_gt]:.3e}  {A_el[-1]:.3e}  m²/s")
print(f"{'A_rm (RMS)':20s}  {A_rm[i_0]:.3e}  {A_rm[i_gt]:.3e}  {A_rm[-1]:.3e}  m²/s")
print(f"  σ̂_EWMA en t_GT : {sigma_ew[i_gt]:.4f}  1/s")
print(f"  σ̂_EWMA al final : {sigma_ew[-1]:.4f}  1/s")

plt.show()
