"""
MaxEnt_Theory_Demo.py
=====================
Five theoretical demonstration figures for the MaxEnt-SPRT indicator:

  D1 -- Lambda_k vs S_k  (why accumulation beats single-point testing)
  D2 -- Reset vs No-reset  (effect of CUSUM-style floor on t_d)
  D3 -- SPRT boundaries a, b  (probabilistic meaning of alpha/beta)
  D4 -- Classic mean+/-3sigma vs MaxEnt-SPRT  (side-by-side comparison)
  D5 -- Equivalence: what n*sigma corresponds to threshold b  (analytical)
"""
from __future__ import annotations

import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

# -- path setup ---------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MaxEnt_SPRT import SignalData, HDF5Reader, run_maxent_sprt
from MaxEnt_SPRT.logging_setup import configure_logging
from MaxEnt_SPRT.viz.maxent_sprt_plots import configurar_estilo_global, fig_size
from MaxEnt_SPRT.viz.maxent_sprt_plots import (
    color_azul   as C_AZUL,
    color_orange as C_ORA,
    color_verde  as C_VER,
    color_red    as C_RED,
)

configure_logging(level=logging.WARNING)

# =============================================================================
# DATA  (same file used in all other examples)
# =============================================================================
_DIR  = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"
_CUT  = (0.1, 16.0)
_T_GT = 5.365770208787228
_ALPHA = 0.00135          # norm.sf(3) ≈ z = 3σ
# _ALPHA = 0.0228        # norm.sf(2) ≈ z = 2σ

# ── β mode ───────────────────────────────────────────────────────────────────
# "symmetric"  →  α = β = 0.00135   (SPRT diseño equilibrado)
# "classical"  →  α = 0.00135, β = P(H < μ₀+3σ₀ | P₁)  (umbral clásico)
BETA_MODE = "symmetric"   # <-- cambia aquí

_RPM   = 12_000.0
_T_REV = 60.0 / _RPM
_F_MOD = 150.0
_T_MOD = 1.0 / _F_MOD

data  = HDF5Reader(os.path.join(_DIR, "out.hdf5"))
t_raw = data.get_element("tool_dyn/data")[:, 0]
v_raw = data.get_element("tool_dyn_o/data")[:, 1]
fs    = 1.0 / (t_raw[1] - t_raw[0])

mask         = (t_raw >= _CUT[0]) & (t_raw <= _CUT[1])
t_cut, v_cut = t_raw[mask], v_raw[mask]

sig = SignalData(
    t_analysis=t_cut,
    signal_analysis=v_cut,
    fs=fs,
    path=os.path.join(_DIR, "out.hdf5"),
    meta={"RPM": _RPM},
)

# =============================================================================
# BASE CONFIG  (by_revolution, raw, 4 rev/seg  -- same as MaxEnt_Detection_NEW)
# =============================================================================
_BASE_CFG = {
    "id":         "MaxEnt_SPRT",
    "func":       "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev":         _T_REV,
        "N_rev_per_seg": 4,
        "step_rev":      1,
        "segmentation":  "raw",
        "t_stable_total":     _T_GT,
        "training_intervals": [
            (_CUT[0], _T_GT, "stable"),
            # (_CUT[0], 3.3, "stable"),
            # (3.3,  4.4, "chatter"),
            # (4.4,  _T_GT, "stable"),
            (_T_GT,   10.0,  "chatter"),
        ],
        "alpha":          _ALPHA,
        "beta":           _ALPHA,
        "reset_on_H0":    True,       # default -- CUSUM
        "cut_start_time": _CUT[0],
        "cut_end_time":   10.0,
    },
}

# Config with reset disabled -- for D2
import copy
_CFG_NO_RESET = copy.deepcopy(_BASE_CFG)
_CFG_NO_RESET["params_physical"]["reset_on_H0"] = False

# Ground-truth label function — derived from training_intervals so that ALL
# figures classify segments consistently with what P0/P1 were fitted on.
_TRAIN_INTERVALS = _BASE_CFG["params_physical"]["training_intervals"]

def _label_mask(t_arr, label):
    """Return boolean mask: True where t_arr falls in a training interval
    whose label matches `label` ('stable' or 'chatter')."""
    m = np.zeros(len(t_arr), dtype=bool)
    for t_lo, t_hi, lbl in _TRAIN_INTERVALS:
        if lbl == label:
            m |= (t_arr >= t_lo) & (t_arr < t_hi)
    return m

# =============================================================================
# RUN  (two variants needed: reset=True and reset=False)
# =============================================================================
print("Running with reset=True  (pre-run) ...")
res_reset = run_maxent_sprt(sig, _BASE_CFG)

# ── Resolve β according to BETA_MODE ─────────────────────────────────────────
if BETA_MODE == "classical":
    _meta_pre = res_reset.meta or {}
    _P0_mu  = _meta_pre["P0_mu"]
    _P0_sig = _meta_pre["P0_sigma"]
    _P1_mu  = _meta_pre["P1_mu"]
    _P1_sig = _meta_pre["P1_sigma"]
    _n_sig  = norm.isf(_ALPHA)                              # ≈ 3.0
    _H_thr  = _P0_mu + _n_sig * _P0_sig
    _BETA   = float(norm.cdf(_H_thr, _P1_mu, _P1_sig))    # β_cl
    print(f"  BETA_MODE='classical': β_cl = {_BETA:.6f}  (α = {_ALPHA:.6f})")
    import copy as _copy
    _BASE_CFG["params_physical"]["beta"] = _BETA
    _CFG_NO_RESET = _copy.deepcopy(_BASE_CFG)
    _CFG_NO_RESET["params_physical"]["reset_on_H0"] = False
    print("Re-running with reset=True  (classical β) ...")
    res_reset = run_maxent_sprt(sig, _BASE_CFG)
elif BETA_MODE == "symmetric":
    _BETA = _ALPHA
    print(f"  BETA_MODE='symmetric': α = β = {_ALPHA:.6f}")
else:
    raise ValueError(f"Unknown BETA_MODE={BETA_MODE!r}. Use 'symmetric' or 'classical'.")

print("Running with reset=False ...")
res_noreset = run_maxent_sprt(sig, _CFG_NO_RESET)


# =============================================================================
# UNPACK
# =============================================================================
def _unpack(res):
    meta   = res.meta or {}
    H      = np.asarray(meta.get("H_seq_online", []))
    P0_mu  = meta.get("P0_mu",    np.nan)
    P0_sig = meta.get("P0_sigma", np.nan)
    P1_mu  = meta.get("P1_mu",    np.nan)
    P1_sig = meta.get("P1_sigma", np.nan)
    if H.size > 0 and not np.isnan(P0_mu):
        log_p0 = -0.5 * ((H - P0_mu) / P0_sig) ** 2 - np.log(P0_sig)
        log_p1 = -0.5 * ((H - P1_mu) / P1_sig) ** 2 - np.log(P1_sig)
        Lambda = log_p1 - log_p0
    else:
        Lambda = np.diff(np.asarray(res.I_t), prepend=0.0)
    t_d = np.asarray(res.t_d) if res.t_d is not None else np.array([])
    return dict(
        t      = np.asarray(res.t),
        H      = H,
        S      = np.asarray(res.I_t),
        Lambda = Lambda,
        t_d    = t_d,
        b      = meta["sprt_result"].b,
        a      = meta["sprt_result"].a,
        P0_mu  = P0_mu,
        P0_sig = P0_sig,
        P1_mu  = P1_mu,
        P1_sig = P1_sig,
    )


def _td_after(t_d, t_gt):
    mask = t_d > t_gt
    return float(t_d[mask][0]) if np.any(mask) else np.nan


d_reset   = _unpack(res_reset)
d_noreset = _unpack(res_noreset)

# Global ground-truth masks — all figures MUST use these instead of t < _T_GT
mask_stable  = _label_mask(d_reset["t"], "stable")
mask_chatter = _label_mask(d_reset["t"], "chatter")

td_reset   = _td_after(d_reset["t_d"],   _T_GT)
td_noreset = _td_after(d_noreset["t_d"], _T_GT)

b_val = d_reset["b"]
a_val = d_reset["a"]
P0_mu, P0_sig = d_reset["P0_mu"], d_reset["P0_sig"]
P1_mu, P1_sig = d_reset["P1_mu"], d_reset["P1_sig"]

# First time Lambda_k >= b after t_gt: single-point (punctual) detection with threshold b
_mask_lp          = (d_reset["t"] > _T_GT) & (d_reset["Lambda"] >= b_val)
td_lambda_puntual = float(d_reset["t"][_mask_lp][0]) if _mask_lp.any() else np.nan



print(f"\n  b = {b_val:.4f}   a = {a_val:.4f}")
print(f"  P0: mu={P0_mu:.4f}  sig={P0_sig:.4f}")
print(f"  P1: mu={P1_mu:.4f}  sig={P1_sig:.4f}")
print(f"  t_d (reset)    = {td_reset:.4f} s   (Δ = {(td_reset-_T_GT)*1e3:+.1f} ms)")
print(f"  t_d (no reset) = {td_noreset:.4f} s   (Δ = {(td_noreset-_T_GT)*1e3:+.1f} ms)\n")

# =============================================================================
# STYLE
# =============================================================================
configurar_estilo_global()
_SC  = 5.0
C_GT = "black"


def _shade_intervals(ax, alpha=0.06):
    """Tint the time-axis background with the training-interval colours
    (blue = stable, orange = chatter).  Call once per axis before tight_layout."""
    for t_lo, t_hi, lbl in _TRAIN_INTERVALS:
        c = C_AZUL if lbl == "stable" else C_ORA
        ax.axvspan(t_lo, t_hi, alpha=alpha, color=c, zorder=0)


def _vline(ax, x, label, color=C_GT, ls="--", ypos=0.70):
    ax.axvline(x, color=color, linestyle=ls)
    ylim = ax.get_ylim()
    ax.annotate(label, xy=(x, ylim[0] + ypos * (ylim[1] - ylim[0])),
                xytext=(4, 0), textcoords="offset points",
                color=color, ha="left", va="center", rotation=90)


def _hline_annot(ax, y, label, color, ls="--", xfrac=0.01, above=True):
    """Horizontal line with inline value label near the left edge."""
    ax.axhline(y, color=color, linestyle=ls)
    xlim = ax.get_xlim()
    dy_offset = 4 if above else -10
    va = "bottom" if above else "top"
    ax.annotate(label,
                xy=(xlim[0] + xfrac * (xlim[1] - xlim[0]), y),
                xytext=(0, dy_offset), textcoords="offset points",
                color=color, fontsize=7, ha="left", va=va)


# =============================================================================
# D1 -- Lambda_k  vs  S_k  (why accumulation is needed)
# =============================================================================
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=fig_size(_SC, ncols=1), sharex=True)
fig1.suptitle("D1 — Log-rapport $\\Lambda_k$ vs statistique accumulée $S_k$")

ax1a.plot(d_reset["t"], d_reset["Lambda"], color=C_AZUL, marker=".", label=r"$\Lambda_k$")
ax1a.axhline(b_val, color=C_RED, linestyle="--", label=rf"$b$ = {b_val:.2f}")
ax1a.axhline(a_val, color=C_VER, linestyle="--", label=rf"$a$ = {a_val:.2f}")
ax1a.axhline(0, color="gray", linestyle=":", label=r"$\Lambda=0$ (neutre)")
ax1a.set_ylabel(r"$\Lambda_k = \log\,p_1/p_0$")
ax1a.set_title(r"$\Lambda_k$ ponctuel — un seul vote positif ne franchit jamais $b$")
_vline(ax1a, _T_GT, f"$t_{{gt}}$ = {_T_GT:.3f} s", C_GT, ls="--")
if not np.isnan(td_lambda_puntual):
    _vline(ax1a, td_lambda_puntual, f"$t_d^{{\\Lambda}}$ = {td_lambda_puntual:.3f}s", C_RED, ls="-.")
ax1a.legend(fontsize=8)

ax1b.plot(d_reset["t"], d_reset["S"], color=C_ORA, marker=".", label=r"$S_k$ (SPRT)")
ax1b.axhline(b_val, color=C_RED, linestyle="--", label=rf"$b$ = {b_val:.2f}")
ax1b.axhline(a_val, color=C_VER, linestyle="--", label=rf"$a$ = {a_val:.2f}")
ax1b.axhline(0, color="gray", linestyle=":")
ax1b.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax1b.set_ylabel(r"$S_k = \sum \Lambda_i$")
ax1b.set_xlabel("Temps (s)")
ax1b.set_title(r"$S_k$ accumulé — preuve soutenue nécessaire pour franchir $b$")
_vline(ax1b, _T_GT, f"$t_{{gt}}$ = {_T_GT:.3f} s", C_GT, ls="--")
if not np.isnan(td_reset):
    _vline(ax1b, td_reset, f"$t_d$ = {td_reset:.3f}s", C_RED, ls="-.")
ax1b.legend(fontsize=8)
_xlim1b = ax1b.get_xlim()
ax1b.annotate(rf"$b = {b_val:.2f}$",
              xy=(_xlim1b[0] + 0.01 * (_xlim1b[1] - _xlim1b[0]), b_val),
              xytext=(0, 4), textcoords="offset points",
              color=C_RED, fontsize=7, ha="left", va="bottom")
ax1b.annotate(rf"$a = {a_val:.2f}$",
              xy=(_xlim1b[0] + 0.01 * (_xlim1b[1] - _xlim1b[0]), a_val),
              xytext=(0, -4), textcoords="offset points",
              color=C_VER, fontsize=7, ha="left", va="top")
fig1.tight_layout()

fig_test, ax_test = plt.subplots(figsize=fig_size(_SC))
ax_test.plot(d_reset["t"], d_reset["Lambda"], color=C_AZUL, marker=".", label=r"$\Lambda_k$")
ax_test.plot(d_reset["t"], d_reset["S"], color=C_ORA, marker=".", label=r"$S_k$ (SPRT)")
ax_test.axhline(b_val, color=C_RED, linestyle="--", label=rf"$b$ = {b_val:.2f}")
ax_test.axhline(a_val, color=C_VER, linestyle="--", label=rf"$a$ = {a_val:.2f}")
ax_test.axhline(0, color="gray", linestyle=":", label=r"$\Lambda=0$ (neutre)")

# =============================================================================
# D2 -- Reset=True  vs  Reset=False
# =============================================================================
fig2, ax2 = plt.subplots(figsize=fig_size(_SC))
fig2.suptitle("D2 — Effet du reset sur $S_k$  (CUSUM vs Wald SPRT)")

ax2.plot(d_reset["t"],   d_reset["S"],   color=C_AZUL, marker=".",
         label=(r"reset=True  (CUSUM) : $S_k = \max(0,\,S_{k-1}+\Lambda_k)$"
                f"  —  $t_d$={td_reset:.3f} s"))
ax2.plot(d_noreset["t"], d_noreset["S"], color=C_ORA,  marker=".", alpha=0.80,
         label=(r"reset=False (Wald)  : $S_k = S_{k-1}+\Lambda_k$"
                f"  —  $t_d$={td_noreset:.3f} s"))
ax2.axhline(b_val, color=C_RED,  linestyle="--", label=rf"$b$ = {b_val:.2f}")
ax2.axhline(a_val, color=C_VER,  linestyle="--", label=rf"$a$ = {a_val:.2f}")
ax2.axhline(0, color="gray", linestyle=":")
ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax2.set_xlabel("Temps (s)")
ax2.set_ylabel(r"$S_k$")
_vline(ax2, _T_GT, f"$t_{{gt}}$ = {_T_GT:.3f} s", C_GT, ls=":")
if not np.isnan(td_reset):
    _vline(ax2, td_reset,   f"$t_d$ CUSUM = {td_reset:.3f} s",   C_AZUL, ls="-.")
if not np.isnan(td_noreset):
    _vline(ax2, td_noreset, f"$t_d$ Wald = {td_noreset:.3f} s",  C_ORA,  ls="-.")
ax2.legend(fontsize=8, loc="upper left")
fig2.tight_layout()

# =============================================================================
# D3 -- P₀ / P₁ (panneaux séparés) : α, 1-α, β, 1-β et frontières SPRT a, b
# =============================================================================
n_sigma   = norm.isf(_ALPHA)                         # ≈ 3.0  (also used in D5)
H_thr3    = P0_mu + n_sigma * P0_sig                 # unique classic threshold μ₀ + nσ₀
beta_cl3  = float(norm.cdf(H_thr3, P1_mu, P1_sig))  # P(H < H_thr | P₁) = true β

_FS_LBL  = 14   # area labels (α, 1-α…)
_FS_TICK = 12   # tick labels
_FS_LEG  = 11   # legend
_FS_AX   = 13   # axis labels & title
_LW_PDF  = 2.5

# Per-panel x-limits: centred on each Gaussian's mean (±4.2σ), H_thr always visible
_xlim_L = (P0_mu - 4.2*P0_sig,  H_thr3 + 1.5*P0_sig)
# Right panel: symmetric ±4.2σ around μ₁; extend left if H_thr falls outside
_xlim_R = (min(P1_mu - 4.2*P1_sig, H_thr3 - 0.5*P1_sig), P1_mu + 4.2*P1_sig)

# Grids include H_thr and μ ± 3σ exactly
def _grid3(lo, hi, mu, sig):
    pts = [mu - 3*sig, mu - sig, mu, mu + sig, mu + 3*sig, H_thr3]
    return np.sort(np.unique(np.concatenate([np.linspace(lo, hi, 700), pts])))

H_grid_L = _grid3(*_xlim_L, P0_mu, P0_sig)
H_grid_R = _grid3(*_xlim_R, P1_mu, P1_sig)
pdf_L = norm.pdf(H_grid_L, P0_mu, P0_sig)
pdf_R = norm.pdf(H_grid_R, P1_mu, P1_sig)
y_peak3 = max(pdf_L.max(), pdf_R.max())

fig3, (ax3L, ax3R) = plt.subplots(1, 2,
    figsize=(fig_size(_SC)[0] * 1, fig_size(_SC)[1] * 1),
    sharey=True)
fig3.suptitle(
    rf"D3 — Erreurs classiques $\alpha$, $\beta$ et frontières SPRT"
    rf"  $b=\ln\!\frac{{1-\beta}}{{\alpha}}={b_val:.2f}$,"
    rf"  $a=\ln\!\frac{{\beta}}{{1-\alpha}}={a_val:.2f}$",
    fontsize=_FS_AX + 1)

for ax, H_g, pdf, mu, sig, C_body, C_tail, xlim, greek, mu_lbl, sub in [
    (ax3L, H_grid_L, pdf_L, P0_mu, P0_sig, C_AZUL, C_RED,
     _xlim_L, "alpha", r"$\mu_0$", "0"),
    (ax3R, H_grid_R, pdf_R, P1_mu, P1_sig, C_ORA,  C_VER,
     _xlim_R, "beta",  r"$\mu_1$", "1"),
]:
    if greek == "alpha":
        ax.fill_between(H_g, pdf, where=(H_g <= H_thr3), alpha=0.22, color=C_body)
        ax.fill_between(H_g, pdf, where=(H_g >= H_thr3), alpha=0.70, color=C_tail)
        # α label: above the tail area, horizontally centred between H_thr and +4σ
        _x_tail = (H_thr3 + P0_mu + 3.5*P0_sig) / 2
        _y_tail = norm.pdf(_x_tail, mu, sig)
        ax.text(_x_tail, _y_tail + 0.06*y_peak3,
                rf"$\alpha = {_ALPHA:.5f}$" + "\n(fausse alarme)",
                ha="center", va="bottom", fontsize=_FS_LBL, color=C_tail, fontweight="bold")
        # 1-α label: above the body, at μ
        ax.text(mu, norm.pdf(mu, mu, sig) + 0.06*y_peak3,
                rf"$1-\alpha$",
                ha="center", va="bottom", fontsize=_FS_LBL + 2, color=C_body, fontweight="bold")
        leg_handles = [
            mpatches.Patch(color=C_body, alpha=0.55,
                label=rf"$P_0$ (stable) $\mathcal{{N}}(\mu_0={P0_mu:.3f},\,\sigma_0={P0_sig:.3f})$"),
            mpatches.Patch(color=C_tail, alpha=0.80,
                label=rf"$\alpha={_ALPHA:.5f}$ — fausse alarme (queue droite $P_0$)"),
            mpatches.Patch(color=C_body, alpha=0.22,
                label=rf"$1-\alpha={1-_ALPHA:.5f}$ — décision correcte"),
            plt.Line2D([0],[0], color="black", ls="--", lw=2.0,
                label=rf"$H_{{thr}}=\mu_0+{n_sigma:.1f}\sigma_0={H_thr3:.3f}$ nat"),
            plt.Line2D([0],[0], color=C_body, ls=":", lw=1.5,
                label=rf"$\mu_0\pm{n_sigma:.1f}\sigma_0$"),
        ]
    else:
        ax.fill_between(H_g, pdf, where=(H_g >= H_thr3), alpha=0.22, color=C_body)
        ax.fill_between(H_g, pdf, where=(H_g <= H_thr3), alpha=0.70, color=C_tail)
        # β label: above the tail, between -4σ and H_thr
        _x_tail = (H_thr3 + P1_mu - 3.5*P1_sig) / 2
        _y_tail = norm.pdf(_x_tail, mu, sig)
        ax.text(_x_tail, _y_tail + 0.06*y_peak3,
                rf"$\beta = {beta_cl3:.5f}$" + "\n(non-détection)",
                ha="center", va="bottom", fontsize=_FS_LBL, color=C_tail, fontweight="bold")
        # 1-β label: above the body, at μ
        ax.text(mu, norm.pdf(mu, mu, sig) + 0.06*y_peak3,
                rf"$1-\beta$",
                ha="center", va="bottom", fontsize=_FS_LBL + 2, color=C_body, fontweight="bold")
        leg_handles = [
            mpatches.Patch(color=C_body, alpha=0.55,
                label=rf"$P_1$ (chatter) $\mathcal{{N}}(\mu_1={P1_mu:.3f},\,\sigma_1={P1_sig:.3f})$"),
            mpatches.Patch(color=C_tail, alpha=0.80,
                label=rf"$\beta={beta_cl3:.5f}$ — non-détection (queue gauche $P_1$)"),
            mpatches.Patch(color=C_body, alpha=0.22,
                label=rf"$1-\beta={1-beta_cl3:.5f}$ — détection correcte"),
            plt.Line2D([0],[0], color="black", ls="--", lw=2.0,
                label=rf"$H_{{thr}}=\mu_0+{n_sigma:.1f}\sigma_0={H_thr3:.3f}$ nat"),
            plt.Line2D([0],[0], color=C_body, ls=":", lw=1.5,
                label=rf"$\mu_1\pm{n_sigma:.1f}\sigma_1$"),
        ]

    # PDF curve
    ax.plot(H_g, pdf, color=C_body, lw=_LW_PDF)

    # H_thr vertical line (solid black dashed)
    ax.axvline(H_thr3, color="black", linestyle="--", lw=2.0)

    # μ centre line + μ ± 3σ dotted lines
    ax.axvline(mu,          color=C_body, linestyle="-",  lw=1.2, alpha=0.60)
    ax.axvline(mu - 3*sig,  color=C_body, linestyle=":",  lw=1.5, alpha=0.80)
    ax.axvline(mu + 3*sig,  color=C_body, linestyle=":",  lw=1.5, alpha=0.80)

    # x-axis labels below the lines
    _ya = -0.025 * y_peak3
    ax.text(mu,         _ya, mu_lbl,
            ha="center", va="top", fontsize=_FS_LBL, color=C_body,
            fontweight="bold", clip_on=False)
    ax.text(mu - 3*sig, _ya, rf"$\mu_{sub}-3\sigma_{sub}$",
            ha="center", va="top", fontsize=_FS_LBL - 1, color=C_body, clip_on=False)
    ax.text(mu + 3*sig, _ya, rf"$\mu_{sub}+3\sigma_{sub}$",
            ha="center", va="top", fontsize=_FS_LBL - 1, color=C_body, clip_on=False)
    ax.text(H_thr3 + 0.2,     _ya + 0.25, rf"$H_{{thr}}$",
            ha="center", va="top", fontsize=_FS_LBL - 1, color="black", clip_on=False)

    ax.set_xlabel("Entropie $H$ (nat)", fontsize=_FS_AX)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=-0.10 * y_peak3, top=y_peak3 * 1.28)
    ax.legend(handles=leg_handles, fontsize=_FS_LEG, loc="upper left",
              framealpha=0.93, edgecolor="gray")
    ax.tick_params(labelsize=_FS_TICK)

ax3L.set_ylabel("Densité de probabilité")
fig3.tight_layout()

# =============================================================================
# D4 -- Classic mean+/-3sigma  vs  MaxEnt-SPRT  (side-by-side)
# =============================================================================
H_all  = d_reset["H"]
t_all  = d_reset["t"]

# Use global training-interval masks (consistent with P0/P1 fit)
H_stable    = H_all[mask_stable]
mu_classic  = H_stable.mean()
sig_classic = H_stable.std()
thr_classic = mu_classic + 3 * sig_classic
thr_lower   = mu_classic - 3 * sig_classic

# False alarms: stable training segments crossing the threshold
fa_mask = mask_stable & (H_all > thr_classic)

# First classic detection: chatter training segments crossing the threshold
classic_post = mask_chatter & (H_all > thr_classic)
td_classic   = float(t_all[classic_post][0]) if classic_post.any() else np.nan

fig4, (ax4a, ax4b) = plt.subplots(1, 2,
                                   figsize=(fig_size(_SC)[0] * 1.0, fig_size(_SC)[1]))
fig4.suptitle("D4 — Classique $\\mu \\pm 3\\sigma$ vs MaxEnt-SPRT")

# -- Left: classic threshold on H(t) -----------------------------------------
ax4a.plot(t_all, H_all, color=C_AZUL, marker=".", label="$H(t)$")
ax4a.axhline(thr_classic, color=C_RED, linestyle="--",
             label=rf"$\mu_0 + 3\sigma_0 = {thr_classic:.4f}$")
ax4a.axhline(thr_lower,   color=C_RED, linestyle=":",
             label=rf"$\mu_0 - 3\sigma_0 = {thr_lower:.4f}$")
ax4a.axhline(mu_classic,  color=C_VER, linestyle="--",
             label=rf"$\mu_0 = {mu_classic:.4f}$")
ax4a.fill_between(t_all, thr_lower, thr_classic,
                  alpha=0.07, color=C_VER, label="$\\pm 3\\sigma$ bande")
if fa_mask.any():
    ax4a.scatter(t_all[fa_mask], H_all[fa_mask], color=C_RED, s=30, zorder=5,
                 label=f"Fausses alarmes ({fa_mask.sum()})")
ax4a.set_title("Classique : seuil ponctuel $\\mu_0 \\pm 3\\sigma_0$ sur $H$")
ax4a.set_xlabel("Temps (s)")
ax4a.set_ylabel("Entropie $H$")
_vline(ax4a, _T_GT, f"$t_{{gt}}$ = {_T_GT:.3f} s", C_GT, ls=":")
if not np.isnan(td_classic):
    _vline(ax4a, td_classic, f"$t_d^{{\\rm cl}}$ = {td_classic:.3f} s", C_RED, ls="-.")
_xlim4a = ax4a.get_xlim()
ax4a.annotate(rf"${thr_classic:.4f}$",
              xy=(_xlim4a[0] + 0.01 * (_xlim4a[1] - _xlim4a[0]), thr_classic),
              xytext=(0, 4), textcoords="offset points",
              color=C_RED, fontsize=14, ha="left", va="bottom")
ax4a.annotate(rf"${thr_lower:.4f}$",
              xy=(_xlim4a[0] + 0.01 * (_xlim4a[1] - _xlim4a[0]), thr_lower),
              xytext=(0, +14), textcoords="offset points",
              color=C_RED, fontsize=14, ha="left", va="top")
ax4a.legend(fontsize=8)
_shade_intervals(ax4a)

# -- Right: SPRT on same H(t) ------------------------------------------------
ax4b.plot(t_all, d_reset["S"], color=C_ORA, marker=".", label=r"$S_k$ (SPRT)")
ax4b.axhline(b_val, color=C_RED, linestyle="--", label=rf"$b = {b_val:.2f}$")
ax4b.axhline(a_val, color=C_VER, linestyle="--", label=rf"$a = {a_val:.2f}$")
ax4b.axhline(0, color="gray", linestyle=":")
ax4b.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax4b.set_title(r"MaxEnt-SPRT : $S_k$ accumulé")
ax4b.set_xlabel("Temps (s)")
ax4b.set_ylabel(r"$S_k$")
_vline(ax4b, _T_GT, f"$t_{{gt}}$ = {_T_GT:.3f} s", C_GT, ls=":")
if not np.isnan(td_reset):
    _vline(ax4b, td_reset, f"$t_d$ = {td_reset:.3f}s", C_RED, ls="-.")
ax4b.legend(fontsize=8)
_shade_intervals(ax4b)
_xlim4b = ax4b.get_xlim()
ax4b.annotate(rf"$b = {b_val:.2f}$",
              xy=(_xlim4b[0] + 0.01 * (_xlim4b[1] - _xlim4b[0]), b_val),
              xytext=(0, 4), textcoords="offset points",
              color=C_RED, fontsize=14, ha="left", va="bottom")
ax4b.annotate(rf"$a = {a_val:.2f}$",
              xy=(_xlim4b[0] + 0.01 * (_xlim4b[1] - _xlim4b[0]), a_val),
              xytext=(0, -4), textcoords="offset points",
              color=C_VER, fontsize=14, ha="left", va="top")
fig4.tight_layout()

# =============================================================================
# D5 -- Equivalence: n*sigma <-> SPRT threshold b  (analytical)
# =============================================================================
# For Gaussian P0/P1 with equal variance the single-point threshold
# H_thr = mu0 + n*sigma0  corresponds to Lambda_k > Lambda_thr_n
# We also compute: given b and mean Lambda per segment in chatter,
# what is the effective n_sigma of the single-point test that would
# give the same expected number of segments to detection.

# Lambda when H = mu0 + n*sigma0 (single segment):
n_vals     = np.linspace(0, 6, 300)
H_thr_n    = P0_mu + n_vals * P0_sig

# Lambda at that threshold value
log_p0_n   = -0.5 * ((H_thr_n - P0_mu) / P0_sig)**2 - np.log(P0_sig)
log_p1_n   = -0.5 * ((H_thr_n - P1_mu) / P1_sig)**2 - np.log(P1_sig)
Lambda_n   = log_p1_n - log_p0_n

# Number of consecutive segments needed to reach b (approximate):
# k_detect ~ b / mean_Lambda_chatter
Lambda_chatter_mean = float(
    (-0.5 * ((d_reset["H"][mask_chatter] - P1_mu) / P1_sig)**2 - np.log(P1_sig)
     - (-0.5 * ((d_reset["H"][mask_chatter] - P0_mu) / P0_sig)**2 - np.log(P0_sig))
    ).mean()
) if mask_chatter.any() else np.nan

k_detect_sprt = b_val / Lambda_chatter_mean if not np.isnan(Lambda_chatter_mean) else np.nan
t_detect_sprt = k_detect_sprt * _T_REV   # seconds from onset

# For the classic test: expected false alarm rate as function of n
fa_rate_n = norm.sf(n_vals)   # P(H > mu0 + n*sigma0 | stable)

print(f"\n  [D5] mean Lambda (chatter region) = {Lambda_chatter_mean:.4f}")
print(f"  [D5] k_detect (SPRT, ~segments)   = {k_detect_sprt:.1f}")
print(f"  [D5] t_detect from onset (SPRT)   = {t_detect_sprt*1e3:.1f} ms")

fig5, (ax5a, ax5b) = plt.subplots(1, 2,
                                   figsize=(fig_size(_SC)[0] * 1.0, fig_size(_SC)[1]))
fig5.suptitle("D5 — Equivalence: $n\\sigma$ threshold  $\\leftrightarrow$  SPRT boundary $b$")

# Left: Lambda vs n*sigma threshold
ax5a.plot(n_vals, Lambda_n, color=C_AZUL, lw=1.5)
ax5a.axhline(0, color="gray", linestyle=":", lw=0.8,
             label=r"$\Lambda=0$ (midpoint $P_0$/$P_1$)")
ax5a.axvline(3, color=C_GT, linestyle="--", label="$3\\sigma$ (classic)")
ax5a.fill_between(n_vals, Lambda_n, 0,
                  where=(Lambda_n > 0), alpha=0.12, color=C_ORA,
                  label=r"$\Lambda > 0$  →  votes chatter")
ax5a.fill_between(n_vals, Lambda_n, 0,
                  where=(Lambda_n < 0), alpha=0.12, color=C_AZUL,
                  label=r"$\Lambda < 0$  →  votes stable")
ax5a.set_xlabel(r"Threshold expressed as $n\sigma_0$  ($H_{thr} = \mu_0 + n\sigma_0$)")
ax5a.set_ylabel(r"$\Lambda_k$ at that threshold")
ax5a.set_title(r"$\Lambda$ of a single segment at threshold $\mu_0 + n\sigma_0$")
ax5a.legend(fontsize=8, loc="lower right")

# Secondary Y-axis: p0(H_thr_n) and p1(H_thr_n) evaluated at each threshold
ax5a_pdf = ax5a.twinx()
ax5a_pdf.plot(n_vals, norm.pdf(H_thr_n, P0_mu, P0_sig),
              color=C_AZUL, lw=1.2, ls="--", alpha=0.55, label=r"$p_0(H_{thr})$")
ax5a_pdf.plot(n_vals, norm.pdf(H_thr_n, P1_mu, P1_sig),
              color=C_ORA,  lw=1.2, ls="--", alpha=0.55, label=r"$p_1(H_{thr})$")
ax5a_pdf.set_ylabel(r"$p(H_{thr})$  (PDF at threshold)", alpha=0.7)
ax5a_pdf.tick_params()
ax5a_pdf.legend(fontsize=7, loc="upper right")

# Right: false alarm rate vs n (classic) vs SPRT alpha
ax5b.semilogy(n_vals, fa_rate_n, color=C_AZUL, lw=1.5, label="Classic: $P(H > \\mu_0+n\\sigma_0\\,|\\,\\text{stable})$")
ax5b.axhline(_ALPHA, color=C_RED, linestyle="--",
             label=rf"SPRT $\alpha = {_ALPHA:.5f}$ ($z={n_sigma:.1f}$)")
ax5b.axvline(3, color=C_GT, linestyle="--", alpha=0.6, label="$3\\sigma$")
n_eq = norm.isf(_ALPHA)
ax5b.axvline(n_eq, color=C_RED, linestyle=":", alpha=0.8,
             label=rf"$n$ equiv. to $\alpha$: {n_eq:.2f}$\sigma$")
ax5b.set_xlabel(r"$n$  (number of $\sigma_0$)")
ax5b.set_ylabel("False alarm rate (log scale)")
ax5b.set_title("Classic single-point FAR vs SPRT $\\alpha$")
ax5b.legend(fontsize=8)
fig5.tight_layout()

print("\n  [D5] Analytical equivalence:")
print(f"       Classic 3σ  →  FAR = {norm.sf(3):.5f}")
print(f"       SPRT α      →  FAR = {_ALPHA:.5f}  (equiv. to {n_eq:.2f}σ single-point)")
print(f"       Key: SPRT α controls *decision* error, not single-segment crossing prob.\n")

# =============================================================================
# D6 -- ln p₀(H) and ln p₁(H) vs H  →  anatomy of Λ(H)
# =============================================================================
_H_lo6 = min(P0_mu - 4.5*P0_sig, P1_mu - 4.5*P1_sig)
_H_hi6 = max(P0_mu + 4.5*P0_sig, P1_mu + 4.5*P1_sig)
H_range6  = np.linspace(_H_lo6, _H_hi6, 600)

ln_p0_H   = -0.5 * ((H_range6 - P0_mu) / P0_sig)**2 - np.log(P0_sig * np.sqrt(2*np.pi))
ln_p1_H   = -0.5 * ((H_range6 - P1_mu) / P1_sig)**2 - np.log(P1_sig * np.sqrt(2*np.pi))
Lambda_H  = ln_p1_H - ln_p0_H          # = Λ(H) point-by-point

H_thr6    = P0_mu + n_sigma * P0_sig   # classical threshold (3σ)
H_cross   = H_range6[np.argmin(np.abs(Lambda_H))]  # approximate crossing Λ=0

fig6, (ax6a, ax6b) = plt.subplots(1, 2,
    figsize=(fig_size(_SC)[0] * 1, fig_size(_SC)[1] * 1.0))
fig6.suptitle(
    r"D6 — Log-likelihoods $\ln p_0(H)$,  $\ln p_1(H)$  and  $\Lambda(H) = \ln\,p_1/p_0$",
    fontsize=13)

# ── Left panel: ln p0 and ln p1 ──────────────────────────────────────────────
ax6a.plot(H_range6, ln_p0_H, color=C_AZUL, lw=2.2, label=r"$\ln p_0(H)$  (stable)")
ax6a.plot(H_range6, ln_p1_H, color=C_ORA,  lw=2.2, label=r"$\ln p_1(H)$  (chatter)")
ax6a.fill_between(H_range6, ln_p0_H, ln_p1_H,
                  where=(ln_p1_H > ln_p0_H), alpha=0.10, color=C_ORA)
ax6a.fill_between(H_range6, ln_p0_H, ln_p1_H,
                  where=(ln_p1_H < ln_p0_H), alpha=0.10, color=C_AZUL)

# Vertical lines — no legend, annotate inline at top of axis
_y6a_top = ax6a.get_ylim()[1] if ax6a.get_ylim()[1] != 0 else ln_p0_H.max()
for _xv, _lbl, _col, _ls in [
    (P0_mu,   rf"$\mu_0$",           C_AZUL, ":"),
    (P1_mu,   rf"$\mu_1$",           C_ORA,  ":"),
    (H_thr6,  rf"$H_{{thr}}$",       C_GT,   "--"),
    (H_cross, rf"$\Lambda\!=\!0$",   "gray", ":"),
]:
    ax6a.axvline(_xv, color=_col, ls=_ls, lw=1.0, alpha=0.75)

# Use transform to place text at top of each vline
_y6a_txt = 0.97
for _xv, _lbl, _col, _ha in [
    (P0_mu,   rf"$\mu_0$",         C_AZUL, "right"),
    (P1_mu,   rf"$\mu_1$",         C_ORA,  "left"),
    (H_thr6,  rf"$H_{{thr}}$",     C_GT,   "right"),
    (H_cross, rf"$\Lambda\!=\!0$", "gray", "left"),
]:
    ax6a.text(_xv, _y6a_txt, _lbl, color=_col, fontsize=8,
              ha=_ha, va="top", transform=ax6a.get_xaxis_transform())

ax6a.set_xlabel(r"$H$  (entropy of segment)")
ax6a.set_ylabel(r"$\ln p(H)$")
ax6a.set_title(r"Log-likelihoods: parabolas centred on $\mu_0$, $\mu_1$")
ax6a.legend(fontsize=9, loc="lower center")

# ── Right panel: Λ(H) = ln p1(H) − ln p0(H) ────────────────────────────────
ax6b.plot(H_range6, Lambda_H, color=C_VER, lw=2.2,
          label=r"$\Lambda(H) = \ln\frac{p_1(H)}{p_0(H)}$")
ax6b.fill_between(H_range6, Lambda_H, 0,
                  where=(Lambda_H > 0), alpha=0.10, color=C_ORA,
                  label=r"$\Lambda>0$  →  votes chatter")
ax6b.fill_between(H_range6, Lambda_H, 0,
                  where=(Lambda_H < 0), alpha=0.10, color=C_AZUL,
                  label=r"$\Lambda<0$  →  votes stable")

# Horizontal lines — annotate on the right edge
_x6b_r = _H_hi6
for _yh, _lbl, _col, _ls in [
    (0,     r"$\Lambda=0$",          "gray",  ":"),
    (b_val, rf"$b={b_val:.2f}$",     C_RED,   "--"),
    (a_val, rf"$a={a_val:.2f}$",     C_AZUL,  "--"),
]:
    ax6b.axhline(_yh, color=_col, ls=_ls, lw=1.2, alpha=0.8)
    ax6b.text(0.99, _yh, f"  {_lbl}", color=_col, fontsize=8,
              ha="right", va="bottom", transform=ax6b.get_yaxis_transform())

# Vertical line for H_thr
ax6b.axvline(H_thr6,  color=C_GT, ls="--", lw=1.0, alpha=0.75)
ax6b.text(H_thr6, 0.97, rf"$H_{{thr}}$", color=C_GT, fontsize=8,
          ha="right", va="top", transform=ax6b.get_xaxis_transform())
ax6b.axvline(H_cross, color="gray", ls=":", lw=0.9, alpha=0.6)

ax6b.set_xlabel(r"$H$  (entropy of segment)", fontsize=12)
ax6b.set_ylabel(r"$\Lambda(H) = \ln p_1(H) - \ln p_0(H)$", fontsize=12)
ax6b.set_title(r"Single-segment log-likelihood ratio $\Lambda(H)$", fontsize=11)
ax6b.legend(fontsize=9, loc="upper left")

fig6.tight_layout()

# =============================================================================
# D7 -- p₀(H(t)) and p₁(H(t)) over time  →  which distribution "wins" each segment
# =============================================================================
_t7   = d_reset["t"]
_H7   = d_reset["H"]

# Evaluate both PDFs on the actual entropy sequence
_p0_t = norm.pdf(_H7, P0_mu, P0_sig)   # probability density under P₀
_p1_t = norm.pdf(_H7, P1_mu, P1_sig)   # probability density under P₁

fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=fig_size(_SC, ncols=1),
                                   sharex=True)
fig7.suptitle(r"D7 — $p_0(H(t))$ and $p_1(H(t))$ over time"
              "\n(which distribution is more likely at each segment?)", fontsize=12)

# Top: both PDFs on same axis
ax7a.plot(_t7, _p0_t, color=C_AZUL, lw=1.2, label=r"$p_0(H(t))$ — how stable-like is this segment?")
ax7a.plot(_t7, _p1_t, color=C_ORA,  lw=1.2, label=r"$p_1(H(t))$ — how chatter-like is this segment?")
ax7a.fill_between(_t7, _p0_t, _p1_t,
                  where=(_p1_t > _p0_t), alpha=0.15, color=C_ORA,
                  label=r"$p_1 > p_0$  →  chatter wins")
ax7a.fill_between(_t7, _p0_t, _p1_t,
                  where=(_p1_t < _p0_t), alpha=0.15, color=C_AZUL,
                  label=r"$p_0 > p_1$  →  stable wins")
ax7a.axvline(_T_GT, color=C_GT, ls="--", lw=1.2)
ax7a.text(_T_GT, 0.98, r"  $t_{gt}$", color=C_GT, fontsize=9,
          va="top", transform=ax7a.get_xaxis_transform())
ax7a.set_ylabel(r"$p(H(t))$", fontsize=11)
ax7a.set_title(r"PDF values at each segment's entropy $H(t)$", fontsize=10)
ax7a.legend(fontsize=8, loc="upper left")
_shade_intervals(ax7a)

# Bottom: ratio p1/p0 (likelihood ratio, linear scale)
_ratio7 = np.where(_p0_t > 1e-300, _p1_t / _p0_t, np.nan)
ax7b.plot(_t7, _ratio7, color=C_VER, lw=1.2,
          label=r"$p_1(H(t))\,/\,p_0(H(t))$  (likelihood ratio per segment)")
ax7b.axhline(1.0, color="gray", ls=":", lw=0.9, label="ratio = 1  (indifferent)")
ax7b.axvline(_T_GT, color=C_GT, ls="--", lw=1.2)
ax7b.text(_T_GT, 0.98, r"  $t_{gt}$", color=C_GT, fontsize=9,
          va="top", transform=ax7b.get_xaxis_transform())
ax7b.set_xlabel("Time [s]", fontsize=11)
ax7b.set_ylabel(r"$p_1 / p_0$", fontsize=11)
ax7b.set_title(r"Likelihood ratio per segment  ($> 1$: evidence for chatter)", fontsize=10)
ax7b.legend(fontsize=8, loc="upper left")
_shade_intervals(ax7b)

fig7.tight_layout()

# =============================================================================
# D8 -- ln p₀(H(t)) and ln p₁(H(t)) over time  →  log-likelihoods and Λₖ
# =============================================================================
_lp0_t = np.log(np.maximum(_p0_t, 1e-300))   # ln p₀(H(t))
_lp1_t = np.log(np.maximum(_p1_t, 1e-300))   # ln p₁(H(t))
_Lk_t  = _lp1_t - _lp0_t                     # Λₖ = ln p₁ - ln p₀  (per segment)

fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=fig_size(_SC, ncols=1),
                                   sharex=True)
fig8.suptitle(r"D8 — $\ln p_0(H(t))$ and $\ln p_1(H(t))$ over time"
              "\n(log-likelihoods and their difference = $\Lambda_k$)", fontsize=12)

# Top: ln p0 and ln p1
ax8a.plot(_t7, _lp0_t, color=C_AZUL, lw=1.2, label=r"$\ln p_0(H(t))$  (log-prob stable)")
ax8a.plot(_t7, _lp1_t, color=C_ORA,  lw=1.2, label=r"$\ln p_1(H(t))$  (log-prob chatter)")
ax8a.fill_between(_t7, _lp0_t, _lp1_t,
                  where=(_lp1_t > _lp0_t), alpha=0.12, color=C_ORA,
                  label=r"$\ln p_1 > \ln p_0$  →  chatter more likely")
ax8a.fill_between(_t7, _lp0_t, _lp1_t,
                  where=(_lp1_t < _lp0_t), alpha=0.12, color=C_AZUL,
                  label=r"$\ln p_0 > \ln p_1$  →  stable more likely")
ax8a.axvline(_T_GT, color=C_GT, ls="--", lw=1.2)
ax8a.text(_T_GT, 0.98, r"  $t_{gt}$", color=C_GT, fontsize=9,
          va="top", transform=ax8a.get_xaxis_transform())
ax8a.set_ylabel(r"$\ln p(H(t))$", fontsize=11)
ax8a.set_title(r"Log-likelihoods at each segment  (gap = $\Lambda_k$)", fontsize=10)
ax8a.legend(fontsize=8, loc="lower left")
_shade_intervals(ax8a)

# Bottom: Λₖ(t) = ln p₁ - ln p₀  (this IS the increments fed into Sₖ)
ax8b.plot(_t7, _Lk_t, color=C_VER, lw=1.0, alpha=0.8,
          label=r"$\Lambda_k = \ln p_1(H(t)) - \ln p_0(H(t))$  ← fed into $S_k$")
ax8b.axhline(0, color="gray", ls=":", lw=0.9)
ax8b.fill_between(_t7, _Lk_t, 0,
                  where=(_Lk_t > 0), alpha=0.15, color=C_ORA,
                  label=r"$\Lambda_k > 0$: pushes $S_k$ toward chatter")
ax8b.fill_between(_t7, _Lk_t, 0,
                  where=(_Lk_t < 0), alpha=0.15, color=C_AZUL,
                  label=r"$\Lambda_k < 0$: pushes $S_k$ toward stable")
ax8b.axvline(_T_GT, color=C_GT, ls="--", lw=1.2)
ax8b.text(_T_GT, 0.98, r"  $t_{gt}$", color=C_GT, fontsize=9,
          va="top", transform=ax8b.get_xaxis_transform())
ax8b.set_xlabel("Time [s]", fontsize=11)
ax8b.set_ylabel(r"$\Lambda_k$", fontsize=11)
ax8b.set_title(r"$\Lambda_k$ per segment over time  (increments of $S_k$)", fontsize=10)
ax8b.legend(fontsize=8, loc="upper left")
_shade_intervals(ax8b)

fig8.tight_layout()

# =============================================================================
# D9 -- Both PDFs p₀(H) and p₁(H): theory + real histogram
# =============================================================================
# Left  : analytical Gaussians with α / β regions filled and key annotations
# Right : histogram of actual H values (stable vs chatter) + fitted PDFs
#         → validates that P₀/P₁ are good models for this signal
# =============================================================================

_H9_lo = P0_mu - 4.5 * P0_sig
_H9_hi = P1_mu + 4.5 * P1_sig
H_ax9  = np.linspace(_H9_lo, _H9_hi, 800)

_pdf0_ax = norm.pdf(H_ax9, P0_mu, P0_sig)
_pdf1_ax = norm.pdf(H_ax9, P1_mu, P1_sig)

# Threshold and error areas
_H_thr9 = P0_mu + norm.isf(_ALPHA) * P0_sig
_alpha_real  = float(norm.sf(_H_thr9,  P0_mu, P0_sig))   # should equal _ALPHA
_beta_real   = float(norm.cdf(_H_thr9, P1_mu, P1_sig))   # miss rate

# Λ=0 crossing (H where p0=p1)
# For unequal sigmas this is the root of the quadratic Λ(H)=0
# Numerically:
_lam_ax9 = (norm.logpdf(H_ax9, P1_mu, P1_sig)
            - norm.logpdf(H_ax9, P0_mu, P0_sig))
_cross_idx = np.argmin(np.abs(_lam_ax9))   # index closest to 0
_H_cross9  = float(H_ax9[_cross_idx])

# Real H sequences split by training intervals (same regions used to fit P0/P1)
_H9_all   = d_reset["H"]
_t9_all   = d_reset["t"]
_H9_stable  = _H9_all[_label_mask(_t9_all, "stable")]
_H9_chatter = _H9_all[_label_mask(_t9_all, "chatter")]

fig9, (ax9a, ax9b) = plt.subplots(
    1, 2,
    figsize=(fig_size(_SC)[0] * 1, fig_size(_SC)[1] * 1),
)
fig9.suptitle(
    r"D9 — Both distributions $P_0$ (stable) and $P_1$ (chatter): "
    "theory and measured $H$ values",
    fontsize=13,
)

# ── Left panel: analytical PDFs with α / β fills ─────────────────────────────
ax9a.plot(H_ax9, _pdf0_ax, color=C_AZUL, lw=2.2,
          label=rf"$p_0(H)\;\sim\mathcal{{N}}(\mu_0,\sigma_0)$"
                rf"$\quad\mu_0={P0_mu:.3f},\;\sigma_0={P0_sig:.3f}$ nat")
ax9a.plot(H_ax9, _pdf1_ax, color=C_ORA,  lw=2.2,
          label=rf"$p_1(H)\;\sim\mathcal{{N}}(\mu_1,\sigma_1)$"
                rf"$\quad\mu_1={P1_mu:.3f},\;\sigma_1={P1_sig:.3f}$ nat")

# α region: area under p₀ to the right of H_thr (false alarm)
_mask_alpha = H_ax9 >= _H_thr9
ax9a.fill_between(H_ax9, _pdf0_ax, where=_mask_alpha,
                  alpha=0.35, color=C_RED,
                  label=rf"$\alpha$ (false alarm) $= {_alpha_real:.5f}$")

# β region: area under p₁ to the left of H_thr (miss / non-detection)
_mask_beta = H_ax9 <= _H_thr9
ax9a.fill_between(H_ax9, _pdf1_ax, where=_mask_beta,
                  alpha=0.25, color=C_VER,
                  label=rf"$\beta$ (miss rate) $= {_beta_real:.4f}$")

# Overlap region between the two PDFs (where both > 0 — visual aid)
ax9a.fill_between(H_ax9, np.minimum(_pdf0_ax, _pdf1_ax),
                  alpha=0.10, color="gray", label="Overlap region")

# Vertical annotations
_y9a_top = max(_pdf0_ax.max(), _pdf1_ax.max())
for _xv, _lbl, _col, _ha in [
    (P0_mu,    rf"$\mu_0={P0_mu:.2f}$",   C_AZUL, "right"),
    (P1_mu,    rf"$\mu_1={P1_mu:.2f}$",   C_ORA,  "left"),
    (_H_thr9,  rf"$H_{{thr}}={_H_thr9:.2f}$", C_RED, "right"),
    (_H_cross9, r"$\Lambda\!=\!0$",         "gray", "left"),
]:
    ax9a.axvline(_xv, color=_col, ls="--", lw=1.0, alpha=0.75)
    ax9a.text(_xv, 0.97, f"  {_lbl}", color=_col, fontsize=8,
              ha=_ha, va="top", transform=ax9a.get_xaxis_transform())

# ±3σ brackets below baseline
for _mu, _sig, _col in [(P0_mu, P0_sig, C_AZUL), (P1_mu, P1_sig, C_ORA)]:
    ax9a.annotate(
        "", xy=(_mu + 3*_sig, -0.008*_y9a_top),
        xytext=(_mu - 3*_sig, -0.008*_y9a_top),
        arrowprops=dict(arrowstyle="<->", color=_col, lw=1.2),
        annotation_clip=False,
    )
    ax9a.text(_mu, -0.03*_y9a_top, r"$\pm3\sigma$",
              ha="center", va="top", fontsize=7, color=_col,
              clip_on=False)

ax9a.set_xlabel(r"Segment entropy $H$  [nat]", fontsize=11)
ax9a.set_ylabel(r"Probability density  [nat$^{-1}$]", fontsize=11)
ax9a.set_title(
    r"Analytical $P_0$ / $P_1$ with error regions $\alpha$, $\beta$",
    fontsize=10,
)
ax9a.set_xlim(_H9_lo, _H9_hi)
ax9a.set_ylim(bottom=-0.06 * _y9a_top)
ax9a.legend(fontsize=8, loc="upper right", framealpha=0.93)

# ── Right panel: real histogram + fitted PDFs ─────────────────────────────────
_n_bins = 50
ax9b.hist(_H9_stable,  bins=_n_bins, density=True,
          color=C_AZUL, alpha=0.45, label=f"Training stable  ($n={len(_H9_stable)}$ seg.)")
ax9b.hist(_H9_chatter, bins=_n_bins, density=True,
          color=C_ORA,  alpha=0.45, label=f"Training chatter ($n={len(_H9_chatter)}$ seg.)")

ax9b.plot(H_ax9, _pdf0_ax, color=C_AZUL, lw=2.2, ls="-",
          label=r"Fitted $p_0(H)$")
ax9b.plot(H_ax9, _pdf1_ax, color=C_ORA,  lw=2.2, ls="-",
          label=r"Fitted $p_1(H)$")

# Threshold and Λ=0 lines
ax9b.axvline(_H_thr9,  color=C_RED,  ls="--", lw=1.2)
ax9b.axvline(_H_cross9, color="gray", ls=":",  lw=0.9)
ax9b.text(_H_thr9,  0.97, rf"$H_{{thr}}$",   color=C_RED,  fontsize=8,
          ha="right", va="top", transform=ax9b.get_xaxis_transform())
ax9b.text(_H_cross9, 0.97, r"  $\Lambda\!=\!0$", color="gray", fontsize=8,
          ha="left",  va="top", transform=ax9b.get_xaxis_transform())

# Goodness-of-fit statistics in a text box
_mu0_meas,  _sig0_meas  = float(_H9_stable.mean()),  float(_H9_stable.std())
_mu1_meas,  _sig1_meas  = float(_H9_chatter.mean()), float(_H9_chatter.std())
_stats_txt = (
    "Training segs. vs fitted\n"
    rf"$\mu_0$:  {_mu0_meas:.3f}  (fit {P0_mu:.3f})" "\n"
    rf"$\sigma_0$: {_sig0_meas:.3f}  (fit {P0_sig:.3f})" "\n"
    rf"$\mu_1$:  {_mu1_meas:.3f}  (fit {P1_mu:.3f})" "\n"
    rf"$\sigma_1$: {_sig1_meas:.3f}  (fit {P1_sig:.3f})"
)
ax9b.text(0.02, 0.97, _stats_txt, transform=ax9b.transAxes,
          fontsize=7.5, va="top", ha="left",
          bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

ax9b.set_xlabel(r"Segment entropy $H$  [nat]", fontsize=11)
ax9b.set_ylabel("Density (normalised histogram)", fontsize=11)
ax9b.set_title(
    r"Real signal: histogram of $H$ values vs fitted Gaussians",
    fontsize=10,
)
ax9b.set_xlim(_H9_lo, _H9_hi)
ax9b.legend(fontsize=8, loc="upper right", framealpha=0.93)

fig9.tight_layout()

plt.show()
