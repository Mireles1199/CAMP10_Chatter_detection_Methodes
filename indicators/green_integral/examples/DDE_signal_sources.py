"""DDE Signal Sources — 3 fuentes de variación de amplitud en zona estable.

Referencia teórica
------------------
La ecuación del cono es un DDE no autónomo:

    m ẍ(t) + c ẋ(t) + k x(t) = Kc·ap(t)·[x(t-T) − x(t)]

Las variaciones de amplitud en zona estable (antes del onset de chatter)
tienen tres fuentes distintas, todas derivables de la ecuación anterior:

  Fuente 1 — Forzamiento regenerativo ≠ 0
      El lado derecho K_c·a_p·x(t-T) es una excitación real aunque a_p < a_p_lim.
      La herramienta NO está en reposo: vibra como oscilador forzado.
      → Evidencia: envolvente de Hilbert A(t) es no nula en toda la zona estable.

  Fuente 2 — Susceptibilidad dinámica crece al acercarse al límite
      |H(ω_n)| = 1 / (2·m·ω_n²·ζ_eff(a_p))  con  ζ_eff → 0 cuando a_p → a_p_lim.
      La misma fuerza produce respuesta cada vez mayor conforme avanza el corte.
      → Evidencia: RMS local crece con tendencia positiva antes de t_gt.

  Fuente 3 — Batido entre armónicos de la fuerza de corte y ω_n
      F_c(t) contiene armónicos en n·f_r.  La respuesta se modula a f_beat = |n·f_r − f_n|.
      → Evidencia: espectro de la envolvente A(t) tiene pico en f_beat = |f_r − f_n|.

Uso
---
    cd indicators/green_integral
    pip install -e .
    python examples/DDE_signal_sources.py
"""

from __future__ import annotations

import os
import sys
import pathlib
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import colorsys
from scipy.signal import butter, sosfiltfilt, hilbert, welch

# ── Path setup ─────────────────────────────────────────────────────────────
_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral import HDF5Reader

# ══════════════════════════════════════════════════════════════════════════════
# Canonical CAMP10 plot style
# ══════════════════════════════════════════════════════════════════════════════

def configurar_estilo_global() -> None:
    local_style = {
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'lines.linewidth': 1.25,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'mathtext.fontset': 'stix',
        'axes.formatter.use_mathtext': True,
        'legend.frameon': False,
        'legend.loc': 'best',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    }
    plt.rcParams.update(local_style)


configurar_estilo_global()

# ── Color palette ──────────────────────────────────────────────────────────
r, g, b = colorsys.hls_to_rgb(206.957 / 360, 0.40941, 0.55603)
color_azul   = (r, g, b)   # stable signal

r, g, b = colorsys.hls_to_rgb(36 / 360, 0.45, 0.99)
color_orange = (r, g, b)   # chatter signal / t_d

r, g, b = colorsys.hls_to_rgb(98 / 360, 0.36, 0.99)
color_verde  = (r, g, b)   # envelope / Hilbert

r, g, b = colorsys.hls_to_rgb(279 / 360, 0.36, 0.99)
color_purple = (r, g, b)   # RMS / Source 2

r, g, b = colorsys.hls_to_rgb(346 / 360, 0.45, 0.99)
color_red    = (r, g, b)   # beat frequency marks / trend

# ══════════════════════════════════════════════════════════════════════════════
# Experiment parameters
# ══════════════════════════════════════════════════════════════════════════════
work_space_5mm   = 'D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Chatter-Criteria/CAMP8-Ventanna_Glisante/Nessy2m_Case_Test_Explicit/1DOF_150Hz_5mm/1DOF_150Hz'
spd_var =r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP8-Ventanna_Glisante\Nessy2m_Case_Test_Explicit\1DOF_150Hz_20mm_7.5k-12kSpdS_100_F-0_05_L-50mm_Statico\1DOF_150Hz"

_DIR_CONO = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"

_DIR_USE = spd_var
_T_START  = 0.05      # [s] start of analysis
_T_END    = 10      # [s] end of analysis
_T_GT     = 5.36577   # [s] ground-truth chatter onset
_F_N      = 150.0     # [Hz] natural / chatter frequency
_F_R      = 200.0     # [Hz] spindle frequency  (12000 RPM / 60)
_BW_FILT  = 50     # [Hz] half-bandwidth of bandpass filter around f_n

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _cut(
    t: np.ndarray,
    x: np.ndarray,
    t0: float,
    t1: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays masked to [t0, t1]."""
    m = (t >= t0) & (t <= t1)
    return t[m], x[m]


def _bandpass(x: np.ndarray, fs: float, fc: float, bw: float) -> np.ndarray:
    """4th-order Butterworth bandpass centred at fc ± bw [Hz] — SOS form for stability."""
    lo = max(fc - bw, 1.0)
    hi = min(fc + bw, fs / 2.0 - 1.0)
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


def _hilbert_envelope(x_filt: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude envelope via analytic signal."""
    return np.abs(hilbert(x_filt))


def _sliding_rms(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Causal sliding-window RMS (non-overlapping kernel via convolution)."""
    N = max(3, int(window_s * fs))
    kernel = np.ones(N) / N
    return np.sqrt(np.convolve(x ** 2, kernel, mode='same'))


def _draw_tgt(ax: plt.Axes, t_gt: float) -> None:
    """Draw ground-truth vertical line with rotated label."""
    ax.axvline(x=t_gt, color='black', ls='--', lw=1.2)
    ax.text(
        t_gt, 0.97, f"  $t_{{gt}}={t_gt:.3f}$ s",
        rotation=90, va='top', ha='right', fontsize=10,
        color='black', transform=ax.get_xaxis_transform(),
    )


def _shade_regions(
    ax: plt.Axes,
    t_start: float,
    t_gt: float,
    t_end: float,
) -> None:
    """Light background shading for stable / chatter zones."""
    ax.axvspan(t_start, t_gt,  alpha=0.07, color=color_azul,   zorder=0)
    ax.axvspan(t_gt,    t_end, alpha=0.07, color=color_orange, zorder=0)


# ══════════════════════════════════════════════════════════════════════════════
# Load signal
# ══════════════════════════════════════════════════════════════════════════════
_data_path = os.path.abspath(os.path.join(_DIR_USE, "out.hdf5"))
_data      = HDF5Reader(_data_path)

_raw       = _data.get_element("tool_dyn/data")
_t_full    = _raw[:, 0]
_x_full    = _raw[:, 1]                               # displacement [m]
_v_full    = _data.get_element("tool_dyn_o/data")[:, 1]   # velocity  [m/s]

_fs = 1.0 / float(_t_full[1] - _t_full[0])
print(f"fs = {_fs:.1f} Hz  |  duration = {_t_full[-1]:.2f} s")

# Analysis window
t, x = _cut(_t_full, _x_full, _T_START, _T_END)
_, v = _cut(_t_full, _v_full, _T_START, _T_END)

# Stable-only slice (for Fuente 3 spectrum)
t_s, v_s = _cut(_t_full, _v_full, _T_START, _T_GT)

# ══════════════════════════════════════════════════════════════════════════════
# Computations
# ══════════════════════════════════════════════════════════════════════════════

# --- Bandpass around f_n (shared base for all three sources) ----------------
v_filt  = _bandpass(v,   _fs, _F_N, _BW_FILT)
v_s_filt = _bandpass(v_s, _fs, _F_N, _BW_FILT)

# ── Fuente 1: Hilbert envelope ───────────────────────────────────────────────
A_full   = _hilbert_envelope(v_filt)    # entire window
A_stable = _hilbert_envelope(v_s_filt)  # stable zone only

# ── Fuente 2: sliding RMS — proxy for |H(ω_n)| · F_c  ──────────────────────
T_modal    = 1.0 / _F_N            # [s]  ≈ 6.67 ms
rms_window = 10.0 * T_modal        # [s]  ≈ 66.7 ms
rms_local  = _sliding_rms(v_filt, _fs, rms_window)

# Linear trend in stable zone (for overlay)
_t_fit_start = _T_START + 1.0        # skip transitorio inicial (~2.7 s hasta min, fit desde 1.05 s)
_t_fit_end   = _T_GT   - 0.3         # stop 0.3 s before onset
m_fit = (t >= _t_fit_start) & (t <= _t_fit_end)
poly_rms = np.polyfit(t[m_fit], rms_local[m_fit], 1)  # degree 1

# ── Fuente 3: PSD of envelope (Welch) ────────────────────────────────────────
# Beating requires BOTH f_n AND f_r to be present → use a wider bandpass
# that spans [f_n - margin, f_r + margin], i.e. includes 150 Hz AND 200 Hz.
_FC_WIDE = (_F_N + _F_R) / 2          # 175 Hz — midpoint between f_n and f_r
_BW_WIDE = (_F_R - _F_N) / 2 + 30.0  # 55 Hz  — half-width with ±30 Hz margin
v_s_wide = _bandpass(v_s, _fs, _FC_WIDE, _BW_WIDE)   # stable zone, 120–230 Hz
A_beat   = _hilbert_envelope(v_s_wide)

# Remove DC so Welch captures the modulation frequencies only
A_ac = A_beat - np.mean(A_beat)

nperseg = min(len(A_ac) // 4, int(_fs * 0.5))
f_psd, Pxx = welch(A_ac, fs=_fs, nperseg=nperseg, noverlap=nperseg // 2)

# Expected beat frequencies: |n·f_r − f_n| for n = 1, 2, 3, ...
_N_HARM = 5
f_beats = sorted({
    abs(n * _F_R - _F_N)
    for n in range(1, _N_HARM + 1)
    if 0 < abs(n * _F_R - _F_N) < _fs / 2
})

# ── Desplazamiento: computaciones paralelas a velocidad ────────────────────
x_filt       = _bandpass(x,   _fs, _F_N, _BW_FILT)
_, x_s_full  = _cut(_t_full, _x_full, _T_START, _T_GT)
x_s_filt     = _bandpass(x_s_full, _fs, _F_N, _BW_FILT)

# Fuente 1 — envolvente Hilbert del desplazamiento
A_full_x = _hilbert_envelope(x_filt)

# Fuente 2 — RMS local del desplazamiento
rms_local_x = _sliding_rms(x_filt, _fs, rms_window)
poly_rms_x  = np.polyfit(t[m_fit], rms_local_x[m_fit], 1)

# Fuente 3 — batido en desplazamiento
x_s_wide         = _bandpass(x_s_full, _fs, _FC_WIDE, _BW_WIDE)
A_beat_x         = _hilbert_envelope(x_s_wide)
A_ac_x           = A_beat_x - np.mean(A_beat_x)
f_psd_x, Pxx_x   = welch(A_ac_x, fs=_fs, nperseg=nperseg, noverlap=nperseg // 2)

# ══════════════════════════════════════════════════════════════════════════════
# Figures — 1 figure per source
# ══════════════════════════════════════════════════════════════════════════════
_XLIM        = (_T_START, _T_END)
_BEAT_COLORS = [color_red, color_orange, color_purple, color_verde, 'saddlebrown']
f_max_plot   = min(300.0, _fs / 2.0)


def _mk_fig(win_name: str, figsize=(13, 5.4)):
    """Single-axes figure with named window and reserved bottom space for info."""
    fig, ax = plt.subplots(1, 1, figsize=figsize, num=win_name,
                           constrained_layout=False)
    fig.subplots_adjust(bottom=0.10, top=0.89, left=0.08, right=0.97)
    return fig, ax



# ── Fig 0 — Señal cruda ─────────────────────────────────────────────────────
fig0, ax0 = _mk_fig('Fig 0 — Senal cruda  (contexto temporal)')
_shade_regions(ax0, _T_START, _T_GT, _T_END)
m_st = t <= _T_GT
ax0.plot(t[m_st],  v[m_st],  color=color_azul,   lw=0.5, label='estable')
ax0.plot(t[~m_st], v[~m_st], color=color_orange, lw=0.5, label='chatter')
_draw_tgt(ax0, _T_GT)
ax0.set_xlabel('Tiempo [s]')
ax0.set_ylabel(r'$\dot{x}\ [\mathrm{m/s}]$')
ax0.set_title('Fig 0 — Velocidad cruda  |  cono 1DOF 150 Hz  (contexto temporal)', fontsize=12)
ax0.legend(loc='upper left', ncol=2)
ax0.set_xlim(_XLIM)

# ── Fig 0b — Desplazamiento crudo ───────────────────────────────────────────
fig0b, ax0b = _mk_fig('Fig 0b — Desplazamiento crudo  x(t)  (contexto temporal)')
_shade_regions(ax0b, _T_START, _T_GT, _T_END)
ax0b.plot(t[m_st],  x[m_st],  color=color_azul,   lw=0.5, label='estable')
ax0b.plot(t[~m_st], x[~m_st], color=color_orange, lw=0.5, label='chatter')
_draw_tgt(ax0b, _T_GT)
ax0b.set_xlabel('Tiempo [s]')
ax0b.set_ylabel(r'$x\ [\mathrm{m}]$')
ax0b.set_title('Fig 0b — Desplazamiento crudo  |  cono 1DOF 150 Hz  (contexto temporal)', fontsize=12)
ax0b.legend(loc='upper left', ncol=2)
ax0b.set_xlim(_XLIM)

# ── Fig 1 — Fuente 1 ────────────────────────────────────────────────────────
fig1a, ax1a = _mk_fig('Fig 1a — Fuente 1: Forzamiento regenerativo  [velocidad]')
_shade_regions(ax1a, _T_START, _T_GT, _T_END)
ax1a.plot(t,  v_filt, color=color_azul,  lw=0.4, alpha=0.45,
          label=rf'$\dot{{x}}$ filtrada  [{_F_N - _BW_FILT:.0f}–{_F_N + _BW_FILT:.0f} Hz]')
ax1a.plot(t,  A_full, color=color_verde, lw=1.5,
          label=r'$A_{\dot{x}}(t)$ — envolvente Hilbert')
ax1a.plot(t, -A_full, color=color_verde, lw=1.5)
ax1a.axhline(0, color='gray', lw=0.5, ls=':')
_draw_tgt(ax1a, _T_GT)
ax1a.set_xlabel('Tiempo [s]')
ax1a.set_ylabel(r'$\dot{x}\ [\mathrm{m/s}]$')
ax1a.set_title(
    r'Fig 1a — Fuente 1 (velocidad): $K_c a_p x(t-T)\neq 0$'
    r'  $\Rightarrow$ $A_{\dot{x}}(t)\neq 0$ en zona estable',
    fontsize=12)
ax1a.legend(loc='upper left', ncol=2)
ax1a.set_xlim(_XLIM)

fig1b, ax1b = _mk_fig('Fig 1b — Fuente 1: Forzamiento regenerativo  [desplazamiento]')
_shade_regions(ax1b, _T_START, _T_GT, _T_END)
ax1b.plot(t,  x_filt,   color=color_red,    lw=0.4, alpha=0.45,
          label=rf'$x$ filtrada  [{_F_N - _BW_FILT:.0f}–{_F_N + _BW_FILT:.0f} Hz]')
ax1b.plot(t,  A_full_x, color=color_orange, lw=1.5,
          label=r'$A_x(t)$ — envolvente Hilbert')
ax1b.plot(t, -A_full_x, color=color_orange, lw=1.5)
ax1b.axhline(0, color='gray', lw=0.5, ls=':')
_draw_tgt(ax1b, _T_GT)
ax1b.set_xlabel('Tiempo [s]')
ax1b.set_ylabel(r'$x\ [\mathrm{m}]$')
ax1b.set_title(
    r'Fig 1b — Fuente 1 (desplazamiento): $K_c a_p x(t-T)\neq 0$'
    r'  $\Rightarrow$ $A_x(t)\neq 0$ en zona estable',
    fontsize=12)
ax1b.legend(loc='upper left', ncol=2)
ax1b.set_xlim(_XLIM)

# ── Fig 2 — Fuente 2 ────────────────────────────────────────────────────────
fig2a, ax2a = _mk_fig('Fig 2a — Fuente 2: Susceptibilidad dinamica  [velocidad]')
_shade_regions(ax2a, _T_START, _T_GT, _T_END)
ax2a.plot(t, rms_local, color=color_purple, lw=1.3,
          label=rf'RMS$_{{\dot{{x}}}}$  ($T_w={rms_window * 1e3:.0f}$ ms)')
t_trend = np.array([_t_fit_start, _T_GT])
ax2a.plot(t_trend, np.polyval(poly_rms, t_trend),
          color=color_red, lw=1.8, ls='--',
          label=f'tendencia lineal  (pend.={poly_rms[0] * 1e3:.3e} [m/s]/ms)')
_draw_tgt(ax2a, _T_GT)
ax2a.set_xlabel('Tiempo [s]')
ax2a.set_ylabel(r'RMS$_{\dot{x}}$ $[\mathrm{m/s}]$')
ax2a.set_title(
    r'Fig 2a — Fuente 2 (velocidad): $|H(\omega_n)|\propto 1/\zeta_\mathrm{eff}(a_p)$'
    r'  $\Rightarrow$ RMS$_{\dot{x}}$ crece antes de $t_{gt}$',
    fontsize=12)
ax2a.legend(loc='upper left', ncol=2)
ax2a.set_xlim(_XLIM)

fig2b, ax2b = _mk_fig('Fig 2b — Fuente 2: Susceptibilidad dinamica  [desplazamiento]')
_shade_regions(ax2b, _T_START, _T_GT, _T_END)
ax2b.plot(t, rms_local_x, color=color_orange, lw=1.3,
          label=rf'RMS$_x$  ($T_w={rms_window * 1e3:.0f}$ ms)')
ax2b.plot(t_trend, np.polyval(poly_rms_x, t_trend),
          color='saddlebrown', lw=1.8, ls='--',
          label=f'tendencia lineal  (pend.={poly_rms_x[0] * 1e6:.3e} [m]/ms)')
_draw_tgt(ax2b, _T_GT)
ax2b.set_xlabel('Tiempo [s]')
ax2b.set_ylabel(r'RMS$_x$ $[\mathrm{m}]$')
ax2b.set_title(
    r'Fig 2b — Fuente 2 (desplazamiento): $|H(\omega_n)|\propto 1/\zeta_\mathrm{eff}(a_p)$'
    r'  $\Rightarrow$ RMS$_x$ crece antes de $t_{gt}$',
    fontsize=12)
ax2b.legend(loc='upper left', ncol=2)
ax2b.set_xlim(_XLIM)

# ── Fig 3 — Fuente 3 ────────────────────────────────────────────────────────
fig3a, ax3a = _mk_fig(
    f'Fig 3a — Fuente 3: Batido |n*f_r - f_n|  [velocidad]  (f_beat={abs(_F_R - _F_N):.0f} Hz)'
)
m_f = f_psd <= f_max_plot
ax3a.semilogy(f_psd[m_f], Pxx[m_f], color=color_azul, lw=1.2,
              label=rf'PSD$_{{A_{{\dot{{x}}}}}}$  ({_FC_WIDE-_BW_WIDE:.0f}–{_FC_WIDE+_BW_WIDE:.0f} Hz, Welch)')
f_beats_exp = [fb for fb in f_beats if fb <= f_max_plot]
for i, fb in enumerate(f_beats_exp):
    n_harm = i + 1
    col = _BEAT_COLORS[i % len(_BEAT_COLORS)]
    ax3a.axvline(fb, color=col, ls='--', lw=1.3)
    ax3a.text(fb, 0.95, f'  $|{n_harm}f_r-f_n|$\n  ={fb:.0f} Hz',
              rotation=90, va='top', ha='left', fontsize=9,
              color=col, transform=ax3a.get_xaxis_transform())
for fmark, lbl in [(_F_N, f'$f_n={_F_N:.0f}$ Hz'), (_F_R, f'$f_r={_F_R:.0f}$ Hz')]:
    ax3a.axvline(fmark, color='black', ls=':', lw=1.0)
    ax3a.text(fmark, 0.02, f'  {lbl}',
              rotation=90, va='bottom', ha='left', fontsize=9,
              color='black', transform=ax3a.get_xaxis_transform())
ax3a.set_xlabel('Frecuencia [Hz]')
ax3a.set_ylabel(r'PSD $[(\mathrm{m/s})^2/\mathrm{Hz}]$')
ax3a.set_title(
    rf'Fig 3a — Fuente 3 (velocidad): Batido $|n\cdot f_r - f_n|$'
    rf'  $\Rightarrow$ $f_{{beat}}={abs(_F_R - _F_N):.0f}$ Hz',
    fontsize=12)
ax3a.legend(loc='upper right')
ax3a.set_xlim(0, f_max_plot)

fig3b, ax3b = _mk_fig(
    f'Fig 3b — Fuente 3: Batido |n*f_r - f_n|  [desplazamiento]  (f_beat={abs(_F_R - _F_N):.0f} Hz)'
)
m_fx = f_psd_x <= f_max_plot
ax3b.semilogy(f_psd_x[m_fx], Pxx_x[m_fx], color=color_orange, lw=1.2,
              label=rf'PSD$_{{A_x}}$  ({_FC_WIDE-_BW_WIDE:.0f}–{_FC_WIDE+_BW_WIDE:.0f} Hz, Welch)')
for i, fb in enumerate(f_beats_exp):
    n_harm = i + 1
    col = _BEAT_COLORS[i % len(_BEAT_COLORS)]
    ax3b.axvline(fb, color=col, ls='--', lw=1.3)
    ax3b.text(fb, 0.95, f'  $|{n_harm}f_r-f_n|$\n  ={fb:.0f} Hz',
              rotation=90, va='top', ha='left', fontsize=9,
              color=col, transform=ax3b.get_xaxis_transform())
for fmark, lbl in [(_F_N, f'$f_n={_F_N:.0f}$ Hz'), (_F_R, f'$f_r={_F_R:.0f}$ Hz')]:
    ax3b.axvline(fmark, color='black', ls=':', lw=1.0)
    ax3b.text(fmark, 0.02, f'  {lbl}',
              rotation=90, va='bottom', ha='left', fontsize=9,
              color='black', transform=ax3b.get_xaxis_transform())
ax3b.set_xlabel('Frecuencia [Hz]')
ax3b.set_ylabel(r'PSD $[\mathrm{m}^2/\mathrm{Hz}]$')
ax3b.set_title(
    rf'Fig 3b — Fuente 3 (desplazamiento): Batido $|n\cdot f_r - f_n|$'
    rf'  $\Rightarrow$ $f_{{beat}}={abs(_F_R - _F_N):.0f}$ Hz',
    fontsize=12)
ax3b.legend(loc='upper right')
ax3b.set_xlim(0, f_max_plot)

# ══════════════════════════════════════════════════════════════════════════════
# Figuras espacio de fase — retrato (x_modo, ẋ_modo) y sección de Poincaré
# ══════════════════════════════════════════════════════════════════════════════

# ── Señales filtradas por modo ───────────────────────────────────────────────
x_n = _bandpass(x, _fs, _F_N, _BW_FILT)   # modo chatter  (desplaz.)
v_n = _bandpass(v, _fs, _F_N, _BW_FILT)   # modo chatter  (velocidad)
x_r = _bandpass(x, _fs, _F_R, _BW_FILT)   # modo forzado  (desplaz.)
v_r = _bandpass(v, _fs, _F_R, _BW_FILT)   # modo forzado  (velocidad)

m_st = t <= _T_GT   # máscara zona estable
m_ch = t >  _T_GT   # máscara zona chatter

# Submuestreo para reducir densidad de trazado (1 de cada 10 muestras)
# 40000/10 = 4000 Hz → ~27 puntos por ciclo a 150 Hz → órbita suave
_ds = 10

# ── Fig P1 — Retrato de fase: modo chatter  (x_n, ẋ_n) — estable | chatter ──
fig_p1, (ax_p1a, ax_p1b) = plt.subplots(1, 2, figsize=(14, 6))
fig_p1.canvas.manager.set_window_title(
    f'Fig P1 — Retrato de fase: modo chatter  f_n={_F_N:.0f} Hz')

ax_p1a.plot(x_n[m_st][::_ds] * 1e6, v_n[m_st][::_ds] * 1e3,
            color=color_azul, lw=0.6, alpha=0.8)
ax_p1a.set_xlabel(r'$x_n\ [\mu\mathrm{m}]$')
ax_p1a.set_ylabel(r'$\dot{x}_n\ [\mathrm{mm/s}]$')
ax_p1a.set_title(f'Estable  ($t < {_T_GT:.2f}$ s)', fontsize=11)
ax_p1a.axhline(0, color='gray', lw=0.4, ls=':')
ax_p1a.axvline(0, color='gray', lw=0.4, ls=':')

ax_p1b.plot(x_n[m_ch][::_ds] * 1e6, v_n[m_ch][::_ds] * 1e3,
            color=color_orange, lw=0.6, alpha=0.8)
ax_p1b.set_xlabel(r'$x_n\ [\mu\mathrm{m}]$')
ax_p1b.set_ylabel(r'$\dot{x}_n\ [\mathrm{mm/s}]$')
ax_p1b.set_title(f'Chatter  ($t > {_T_GT:.2f}$ s)', fontsize=11)
ax_p1b.axhline(0, color='gray', lw=0.4, ls=':')
ax_p1b.axvline(0, color='gray', lw=0.4, ls=':')

fig_p1.suptitle(
    rf'Fig P1 — Retrato de fase (modo chatter $f_n={_F_N:.0f}$ Hz)'
    r'  —  órbita pequeña estable / espiral chatter',
    fontsize=12)
fig_p1.tight_layout()

# ── Fig P2 — Retrato de fase: modo forzado  (x_r, ẋ_r) — estable | chatter ──
fig_p2, (ax_p2a, ax_p2b) = plt.subplots(1, 2, figsize=(14, 6))
fig_p2.canvas.manager.set_window_title(
    f'Fig P2 — Retrato de fase: modo forzado   f_r={_F_R:.0f} Hz')

ax_p2a.plot(x_r[m_st][::_ds] * 1e6, v_r[m_st][::_ds] * 1e3,
            color=color_azul, lw=0.6, alpha=0.8)
ax_p2a.set_xlabel(r'$x_r\ [\mu\mathrm{m}]$')
ax_p2a.set_ylabel(r'$\dot{x}_r\ [\mathrm{mm/s}]$')
ax_p2a.set_title(f'Estable  ($t < {_T_GT:.2f}$ s)', fontsize=11)
ax_p2a.axhline(0, color='gray', lw=0.4, ls=':')
ax_p2a.axvline(0, color='gray', lw=0.4, ls=':')

ax_p2b.plot(x_r[m_ch][::_ds] * 1e6, v_r[m_ch][::_ds] * 1e3,
            color=color_orange, lw=0.6, alpha=0.8)
ax_p2b.set_xlabel(r'$x_r\ [\mu\mathrm{m}]$')
ax_p2b.set_ylabel(r'$\dot{x}_r\ [\mathrm{mm/s}]$')
ax_p2b.set_title(f'Chatter  ($t > {_T_GT:.2f}$ s)', fontsize=11)
ax_p2b.axhline(0, color='gray', lw=0.4, ls=':')
ax_p2b.axvline(0, color='gray', lw=0.4, ls=':')

fig_p2.suptitle(
    rf'Fig P2 — Retrato de fase (modo forzado $f_r={_F_R:.0f}$ Hz)'
    r'  —  órbita cuasi-estacionaria en zona estable',
    fontsize=12)
fig_p2.tight_layout()

# ── Fig P3 — Sección de Poincaré (muestreo estroboscópico a f_r) ─────────────
_T_R   = 1.0 / _F_R                          # periodo de revolución [s]
_t0_p  = t[0]
_k_max = int((t[-1] - _t0_p) / _T_R)
_t_poinc = _t0_p + np.arange(_k_max) * _T_R  # tiempos de muestreo

# índices más cercanos en t al tiempo de muestreo
_idx_p = np.searchsorted(t, _t_poinc)
_idx_p = np.clip(_idx_p, 0, len(t) - 1)

xp = x[_idx_p]
vp = v[_idx_p]
tp = t[_idx_p]

m_p_st = tp <= _T_GT
m_p_ch = tp >  _T_GT

fig_p3, (ax_p3a, ax_p3b) = plt.subplots(1, 2, figsize=(14, 6))
fig_p3.canvas.manager.set_window_title(
    f'Fig P3 — Seccion de Poincare  (muestreo cada T_r = 1/f_r = {_T_R*1e3:.2f} ms)')

ax_p3a.plot(xp[m_p_st] * 1e6, vp[m_p_st] * 1e3,
            color=color_azul, lw=0.4, alpha=0.4, zorder=1)
ax_p3a.scatter(xp[m_p_st] * 1e6, vp[m_p_st] * 1e3,
               color=color_azul, s=12, alpha=0.7, zorder=2)
ax_p3a.set_xlabel(r'$x(t_k)\ [\mu\mathrm{m}]$')
ax_p3a.set_ylabel(r'$\dot{x}(t_k)\ [\mathrm{mm/s}]$')
ax_p3a.set_title(f'Estable  ($t < {_T_GT:.2f}$ s)  —  nube compacta', fontsize=11)
ax_p3a.axhline(0, color='gray', lw=0.4, ls=':')
ax_p3a.axvline(0, color='gray', lw=0.4, ls=':')

ax_p3b.plot(xp[m_p_ch] * 1e6, vp[m_p_ch] * 1e3,
            color=color_orange, lw=0.4, alpha=0.4, zorder=1)
ax_p3b.scatter(xp[m_p_ch] * 1e6, vp[m_p_ch] * 1e3,
               color=color_orange, s=12, alpha=0.7, zorder=2)
ax_p3b.set_xlabel(r'$x(t_k)\ [\mu\mathrm{m}]$')
ax_p3b.set_ylabel(r'$\dot{x}(t_k)\ [\mathrm{mm/s}]$')
ax_p3b.set_title(f'Chatter  ($t > {_T_GT:.2f}$ s)  —  nube dispersa', fontsize=11)
ax_p3b.axhline(0, color='gray', lw=0.4, ls=':')
ax_p3b.axvline(0, color='gray', lw=0.4, ls=':')

fig_p3.suptitle(
    rf'Fig P3 — Sección de Poincaré  ($t_k = k \cdot T_r$,  $T_r = 1/f_r = {_T_R*1e3:.2f}$ ms)'
    r'  —  nube compacta (estable) → dispersa (chatter)',
    fontsize=12)
fig_p3.tight_layout()

plt.show()

