"""example_hmm_cone.py — HMM chatter detector applied to the cone signal.

Pipeline
--------
    1. Load cone HDF5 signal
    2. Compute phase-space areas via run_fixed_window()
    3. Run 2-state HMM forward filter on log10(areas)
    4. Print t_d and emission parameters
    5. Save figure: areas (log scale) + p_chatter vs time

Usage
-----
    cd CAMP10_Chatter_detection_Methodes/HMM
    python example_hmm_cone.py

    # with Agg backend (headless / no display):
    set MPLBACKEND=Agg && python example_hmm_cone.py
"""

import os
import sys
import pathlib

import colorsys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm as _scipy_norm

# ── Path hack: locate green_integral src/ ───────────────────────────────────
_gi_src = (
    pathlib.Path(__file__).resolve().parent.parent
    / "indicators" / "green_integral" / "src"
)
if str(_gi_src) not in sys.path:
    sys.path.insert(0, str(_gi_src))

# ── Path hack: this folder — makes hmm_chatter importable ───────────────────
_here = pathlib.Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

# ── Imports ──────────────────────────────────────────────────────────────────
from green_integral import HDF5Reader, SignalData, run_fixed_window
from hmm_chatter import HMMConfig, run_hmm_detector

# ── Ground-truth chatter onset (annotated) ──────────────────────────────────
_T_GT = 5.36577   # [s] — cone signal

# ── Signal path ──────────────────────────────────────────────────────────────
work_space_5mm   = 'D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Chatter-Criteria/CAMP8-Ventanna_Glisante/Nessy2m_Case_Test_Explicit/1DOF_150Hz_5mm/1DOF_150Hz'

_HDF5_PATH = os.path.join(
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz",
    # r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP8-Ventanna_Glisante\Nessy2m_Case_Test_Explicit\1DOF_150Hz_20mm_7.5k-12kSpdS_100_F-0_05_L-50mm_Statico\1DOF_150Hz",
    # work_space_5mm,
    "out.hdf5",
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cut(t: np.ndarray, x: np.ndarray, t0: float, t1: float):
    m = (t >= t0) & (t <= t1)
    return t[m], x[m]


# ===========================================================================
# 1. Load signal
# ===========================================================================
print("=" * 60)
print("HMM Chatter Detector — cone signal example")
print("=" * 60)

data = HDF5Reader(_HDF5_PATH)
tool_dyn = data.get_element("tool_dyn/data")
t_raw    = tool_dyn[:, 0]
x_raw    = tool_dyn[:, 1]
v_raw    = data.get_element("tool_dyn_o/data")[:, 1]

t_cut, v_cut = _cut(t_raw, v_raw, 0.05, 16.0)
_,     x_cut = _cut(t_raw, x_raw, 0.05, 16.0)


#============================================================================
x_mirror = x_cut[::-1]
v_mirror = -v_cut[::-1]

dt = t_cut[1] - t_cut[0]
t_mirror = t_cut[-1] + dt + np.arange(len(t_cut)) * dt

x_cut = np.concatenate([x_cut, x_mirror])
v_cut = np.concatenate([v_cut, v_mirror])
t_cut = np.concatenate([t_cut, t_mirror])
#============================================================================

sig = SignalData(t=t_cut, displacement=x_cut, velocity=v_cut, name="cono")
print(f"\nSignal loaded : {len(t_cut)} samples  "
      f"[{t_cut[0]:.3f} s → {t_cut[-1]:.3f} s]")

# ===========================================================================
# 2. Compute phase-space areas via Green Integral Fixed Window
# ===========================================================================
F_MODAL = 150.0   # Hz (cono modal frequency)
F_REV  = 200.0   # Hz (revolutions per second, for window size reference)
config_fw = {
    "func": "FixedWindow",
    "params": {
        "f_modal":            F_REV,
        "num_T":               4,
        "dt":                  1./F_REV,  # step = 1 × T_modal (overlapping windows)
        "data_filtrated":      True,
        "lambda_ewma":         None,
        "accumulate":          False,
        "G_memory":            None,
        "sigma_method":        "ratio",
        "area_noise_eps":      1e-30,
        "use_area_threshold":  False,   # threshold handled by HMM
        "debug_level":         0,
    },
}

result_fw = run_fixed_window(sig, config_fw)
areas  = result_fw.areas
t_wins = result_fw.t_wins

print(f"Windows computed : {len(areas)}")
print(f"Area range       : [{areas.min():.3e},  {areas.max():.3e}]")

# ===========================================================================
# 3. Run HMM detector
# ===========================================================================
config_hmm = HMMConfig(
    training_intervals=[
        (0.05,   _T_GT, "stable"),
        (_T_GT,  10,  "chatter"),
    ],
    rho=0.9,
    m_consecutive=1,
    # eps debe coincidir con area_noise_eps del Fixed-Window para que las
    # ventanas bajo el piso de ruido sean NaN y no contaminen el entrenamiento.
    eps=1e-30,
    y_clip_n_sigma=3.0,   # clipea y desde abajo en μ_S − 4·σ_S
    transition_matrix=np.array([[0.95, 0.05],   # P(S→S), P(S→C)
                                [0.05, 0.95]]),  # P(C→S), P(C→C)
    mode = "2state" #2state, 1class
)

result_hmm = run_hmm_detector(areas, t_wins, config_hmm)

# ===========================================================================
# 4. Print results
# ===========================================================================
print("\n─── Emission parameters ───────────────────────────────────")
print(f"  Modo usado : {result_hmm.mode_used}  (config.mode='{config_hmm.mode}')")
print(f"  Stable  : μ_S = {result_hmm.mu_S:.4f}   σ_S = {result_hmm.sigma_S:.4f}")
print(f"  Chatter : μ_C = {result_hmm.mu_C:.4f}   σ_C = {result_hmm.sigma_C:.4f}")
print(f"  Separation (μ_C − μ_S) / σ_S = {result_hmm.info['separation_z']:.2f} σ")
print(f"  Stable windows  : {result_hmm.info['n_stable']}")
print(f"  Chatter windows : {result_hmm.info['n_chatter']}")

print("\n─── Detection ─────────────────────────────────────────────")
if result_hmm.t_d is not None:
    delay = result_hmm.t_d - _T_GT
    print(f"  t_d  (ρ={config_hmm.rho}, m={config_hmm.m_consecutive}) : "
          f"{result_hmm.t_d:.4f} s")
    print(f"  t_gt                          : {_T_GT:.5f} s")
    print(f"  Delay  t_d − t_gt             : {delay:+.4f} s")
else:
    print("  t_d : NOT detected (p_chatter never reached threshold)")

# ===========================================================================
# 5. Plot — 7 individual figures (CAMP10 canonical style)
# ===========================================================================

# ── Style helpers ────────────────────────────────────────────────────────────
def _configurar_estilo():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.titlesize': 25, 'axes.labelsize': 25,
        'xtick.labelsize': 23, 'ytick.labelsize': 23, 'legend.fontsize': 23,
        'lines.linewidth': 1.25, 'lines.markersize': 6,
        'axes.linewidth': 0.8,  'grid.linewidth': 0.5,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 4,  'ytick.major.size': 4,
        'xtick.minor.size': 2.5,'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6,'ytick.minor.width': 0.6,
        'mathtext.fontset': 'stix', 'axes.formatter.use_mathtext': True,
        'legend.frameon': False, 'legend.loc': 'best',
        'legend.handlelength': 2.0, 'legend.borderaxespad': 0.5,
        'figure.dpi': 100, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'savefig.transparent': True,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

_configurar_estilo()

def _fig_size(scale=1.0, ncols=1, base_width=3.4):
    w = base_width * ncols * scale
    return (w, w * 0.70)

# ── Color palette (CAMP10 canonical) ─────────────────────────────────────────
_r, _g, _b = colorsys.hls_to_rgb(346/360, 0.45, 0.99); color_red    = (_r, _g, _b)
_r, _g, _b = colorsys.hls_to_rgb( 36/360, 0.45, 0.99); color_orange = (_r, _g, _b)
_r, _g, _b = colorsys.hls_to_rgb(279/360, 0.36, 0.99); color_purple = (_r, _g, _b)
_r, _g, _b = colorsys.hls_to_rgb( 98/360, 0.36, 0.99); color_verde  = (_r, _g, _b)
_r, _g, _b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603); color_azul = (_r, _g, _b)

# ── Plot helpers ──────────────────────────────────────────────────────────────
def _draw_vlines(ax, vlines, default_color="black", default_ls="--"):
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
            ax.text(vx, 0.97, f"  {label}",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color, transform=ax.get_xaxis_transform())

def _find_td(p_chatter, t_wins, rho, m):
    N = len(p_chatter)
    for k in range(N - m + 1):
        if np.all(p_chatter[k : k + m] > rho):
            return float(t_wins[k])
    return None

def _show(fig, ax, window_name, axes_title):
    ax.set_title(axes_title)
    try:
        fig.canvas.manager.set_window_title(window_name)
    except Exception:
        pass
    print(f"  Figura: {window_name}")

# ── Convenience aliases ───────────────────────────────────────────────────────
_y   = result_hmm.y_obs
_pc  = result_hmm.p_chatter
_pp  = result_hmm.p_chatter_predict
_tw  = result_hmm.t_wins
_mS  = result_hmm.mu_S;  _sS = result_hmm.sigma_S
_mC  = result_hmm.mu_C;  _sC = result_hmm.sigma_C
_sm  = result_hmm.info["stable_mask"]
_cm  = result_hmm.info["chatter_mask"]

_vlines_base = [(_T_GT, f"$t_{{gt}}={_T_GT:.3f}$s", "black")]
if result_hmm.t_d is not None:
    _vlines_base.append((result_hmm.t_d, f"$t_d={result_hmm.t_d:.3f}$s", color_orange))

print("\n─── Figures ────────────────────────────────────────────────")

# ---------------------------------------------------------------------------
# Fig 1 — Raw phase-space areas (log scale)
# ---------------------------------------------------------------------------
fig1, ax1_areas = plt.subplots(figsize=_fig_size(scale=3.0))
ax1_areas.semilogy(_tw, areas, color=color_azul, lw=0.9, label="Área (shoelace)")
ax1_areas.axhline(10 ** _mS, color=color_verde,  ls=":", lw=1.0, label=f"$\\mu_S$ ({_mS:.2f})")
ax1_areas.axhline(10 ** _mC, color=color_orange, ls=":", lw=1.0, label=f"$\\mu_C$ ({_mC:.2f})")
_draw_vlines(ax1_areas, _vlines_base)
ax1_areas.set_xlabel("Tiempo [s]")
ax1_areas.set_ylabel("Área en espacio fase [m·m/s]")
ax1_areas.legend()
ax1_areas.grid(True, which="both", alpha=0.3)
_show(fig1, ax1_areas, "Fig 1 — Áreas en espacio fase", "Áreas en espacio fase (escala log)")

# ---------------------------------------------------------------------------
# Fig 2 — Log10-area observations (what the HMM sees)
# ---------------------------------------------------------------------------
fig2, ax2_yobs = plt.subplots(figsize=_fig_size(scale=3.0))
ax2_yobs.plot(_tw, _y, color=color_azul, lw=0.8, label="$y_k = \\log_{10}(A_k)$")
ax2_yobs.axhline(_mS,         color=color_verde,  ls="-",  lw=1.2, label=f"$\\mu_S={_mS:.2f}$")
ax2_yobs.axhline(_mS + 3*_sS, color=color_verde,  ls="--", lw=0.9, label=f"$\\mu_S\\pm3\\sigma_S$")
ax2_yobs.axhline(_mS - 3*_sS, color=color_verde,  ls="--", lw=0.9)
if config_hmm.y_clip_n_sigma is not None:
    _y_clip_floor = _mS - config_hmm.y_clip_n_sigma * _sS
    ax2_yobs.axhline(_y_clip_floor, color=color_red, ls=":", lw=1.0,
                     label=f"piso clip $\\mu_S - {config_hmm.y_clip_n_sigma:.0f}\\sigma_S={_y_clip_floor:.2f}$")
ax2_yobs.axhline(_mC - 3*_sC, color=color_orange, ls="--", lw=0.9)
_draw_vlines(ax2_yobs, _vlines_base)
ax2_yobs.set_xlabel("Tiempo [s]")
ax2_yobs.set_ylabel("$y_k = \\log_{10}(A_k)$")
ax2_yobs.legend()
ax2_yobs.grid(True, alpha=0.3)
_show(fig2, ax2_yobs, "Fig 2 — Observaciones log10(Área)", "Observaciones $y_k = \\log_{10}(A_k)$ con umbrales $\\mu\\pm3\\sigma$")

# ---------------------------------------------------------------------------
# Fig 3 — Posterior P(C | data)
# ---------------------------------------------------------------------------
fig3, ax3_pc = plt.subplots(figsize=_fig_size(scale=3.0))
ax3_pc.plot(_tw, _pc, color=color_purple, lw=1.0, label="$P(C\\,|\\,y_{1:k})$")
ax3_pc.axhline(config_hmm.rho, color=color_red, ls="--", lw=1.0,
               label=f"$\\rho={config_hmm.rho}$")
ax3_pc.axhline(0.5, color="gray", ls=":", lw=0.8)
_draw_vlines(ax3_pc, _vlines_base)
ax3_pc.set_ylim(-0.05, 1.05)
ax3_pc.set_xlabel("Tiempo [s]")
ax3_pc.set_ylabel("$P(C \\mid y_{1:k})$")
ax3_pc.legend()
ax3_pc.grid(True, alpha=0.3)
_show(fig3, ax3_pc, "Fig 3 — Posterior P(Chatter | datos)", "Probabilidad posterior de chatter $P(C\\mid y_{1:k})$")

# ---------------------------------------------------------------------------
# Fig 4 — Posterior P(S | data)  =  1 - P(C | data)
# ---------------------------------------------------------------------------
fig4, ax4_ps = plt.subplots(figsize=_fig_size(scale=3.0))
ax4_ps.plot(_tw, 1.0 - _pc, color=color_azul, lw=1.0, label="$P(S\\,|\\,y_{1:k})$")
ax4_ps.axhline(0.5, color="gray", ls=":", lw=0.8)
_draw_vlines(ax4_ps, _vlines_base)
ax4_ps.set_ylim(-0.05, 1.05)
ax4_ps.set_xlabel("Tiempo [s]")
ax4_ps.set_ylabel("$P(S \\mid y_{1:k})$")
ax4_ps.legend()
ax4_ps.grid(True, alpha=0.3)
_show(fig4, ax4_ps, "Fig 4 — Posterior P(Estable | datos)", "Probabilidad posterior de estado estable $P(S\\mid y_{1:k})$")

# ---------------------------------------------------------------------------
# Fig 5 — Prediction α̂_k vs posterior α_k  (chatter component)
# ---------------------------------------------------------------------------
fig5, ax5_pred = plt.subplots(figsize=_fig_size(scale=3.0))
ax5_pred.plot(_tw, _pp, color=color_azul,   lw=0.9,
              label="Predicción $\\hat{\\alpha}_k(C)$")
ax5_pred.plot(_tw, _pc, color=color_orange, lw=1.0,
              label="Posterior $\\alpha_k(C)$")
ax5_pred.axhline(0.5, color="gray", ls=":", lw=0.8)
_draw_vlines(ax5_pred, _vlines_base)
ax5_pred.set_ylim(-0.05, 1.05)
ax5_pred.set_xlabel("Tiempo [s]")
ax5_pred.set_ylabel("Probabilidad de chatter")
ax5_pred.legend()
ax5_pred.grid(True, alpha=0.3)
_show(fig5, ax5_pred, "Fig 5 — Predicción vs Posterior", "Predicción $\\hat{\\alpha}_k(C)$ vs posterior $\\alpha_k(C)$")

# ---------------------------------------------------------------------------
# Fig 6 — Emission distributions  (histograms + normal curves)
# ---------------------------------------------------------------------------
fig6, ax6_emis = plt.subplots(figsize=_fig_size(scale=3.0))
_sm_fin = _sm & np.isfinite(_y)
_cm_fin = _cm & np.isfinite(_y)
# x-range: centrado en las emisiones ±5σ para no aplastar los histogramas
_x_lo = min(_mS, _mC) - 3.0 * max(_sS, _sC)
_x_hi = max(_mS, _mC) + 3.0 * max(_sS, _sC)
_x_lin = np.linspace(_x_lo, _x_hi, 600)
ax6_emis.hist(_y[_sm_fin], bins=300, density=True, color=color_azul,   alpha=0.45,
              label="Estable", range=(_x_lo, _x_hi))
ax6_emis.hist(_y[_cm_fin], bins=300, density=True, color=color_orange, alpha=0.45,
              label="Chatter", range=(_x_lo, _x_hi))
ax6_emis.plot(_x_lin, _scipy_norm.pdf(_x_lin, _mS, _sS),
              color=color_verde, lw=1.8, ls="-",
              label=f"$\\mathcal{{N}}(\\mu_S={_mS:.3g},\\,\\sigma_S={_sS:.3g})$")
ax6_emis.plot(_x_lin, _scipy_norm.pdf(_x_lin, _mC, _sC),
              color=color_red, lw=1.8, ls="--",
              label=f"$\\mathcal{{N}}(\\mu_C={_mC:.3g},\\,\\sigma_C={_sC:.3g})$")
ax6_emis.set_xlim(_x_lo, _x_hi)
ax6_emis.set_xlabel("$y = \\log_{10}(A)$")
ax6_emis.set_ylabel("Densidad")
ax6_emis.legend()
ax6_emis.grid(True, alpha=0.3)
_show(fig6, ax6_emis, "Fig 6 — Distribuciones de emisión",
      "Distribuciones de emisión: $\\mathcal{N}(\\mu_z,\\,\\sigma_z^2)$")

# ---------------------------------------------------------------------------
# Fig 7 — Sensitivity to ρ  (same p_chatter, three thresholds)
# ---------------------------------------------------------------------------
_rho_list   = [0.5, 0.8, 0.95]
_rho_colors = [color_verde, color_orange, color_red]

fig7, ax7_rho = plt.subplots(figsize=_fig_size(scale=3.0))
ax7_rho.plot(_tw, _pc, color=color_purple, lw=1.0, label="$P(C\\,|\\,y_{1:k})$", zorder=3)
for _rho_val, _rho_col in zip(_rho_list, _rho_colors):
    _td_rho = _find_td(_pc, _tw, _rho_val, config_hmm.m_consecutive)
    ax7_rho.axhline(_rho_val, color=_rho_col, ls="--", lw=1.0,
                    label=f"$\\rho={_rho_val}$")
    if _td_rho is not None:
        ax7_rho.axvline(_td_rho, color=_rho_col, ls="-", lw=1.2)
        ax7_rho.text(_td_rho, 0.97, f"  $t_d={_td_rho:.2f}$s",
                     rotation=90, va="top", ha="right", fontsize=16,
                     color=_rho_col, transform=ax7_rho.get_xaxis_transform())
ax7_rho.axvline(_T_GT, color="black", ls="--", lw=1.2)
ax7_rho.text(_T_GT, 0.97, f"  $t_{{gt}}={_T_GT:.3f}$s",
             rotation=90, va="top", ha="right", fontsize=16,
             color="black", transform=ax7_rho.get_xaxis_transform())
ax7_rho.set_ylim(-0.05, 1.05)
ax7_rho.set_xlabel("Tiempo [s]")
ax7_rho.set_ylabel("$P(C \\mid y_{1:k})$")
ax7_rho.legend()
ax7_rho.grid(True, alpha=0.3)
_show(fig7, ax7_rho, "Fig 7 — Sensibilidad al umbral \u03c1", "Sensibilidad de $t_d$ al umbral $\\rho \\in \\{0.5,\\,0.8,\\,0.95\\}$")

plt.show()
