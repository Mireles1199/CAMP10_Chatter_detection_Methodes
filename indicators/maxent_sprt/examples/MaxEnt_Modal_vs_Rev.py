"""
MaxEnt_Modal_vs_Rev.py
======================
Comparison study: ``by_modal`` vs ``by_revolution`` segmentation in MaxEnt-SPRT.

Both modes use the **same effective window duration** T_win = T_lcm = 20 ms:

    by_modal:       N_modal_per_seg = 3   ->  T_win = 3 x 6.667 ms = 20 ms
                    step_modal      = 1   ->  hop   = T_modal       = 6.667 ms

    by_revolution:  N_rev_per_seg   = 4   ->  T_win = 4 x 5 ms     = 20 ms
                    step_rev        = 1   ->  hop   = T_rev         = 5 ms

Hypothesis
----------
Same T_win -> same entropy sensitivity.
Different hop -> SPRT curves sampled at different rates (rev: 200/s, modal: 150/s).
-> Curves should be correlated but possibly time-offset and/or differently dense.

Figures produced
----------------
  F1 -- Entropy sequences overlaid (H vs t)
  F2 -- SPRT statistic S_k overlaid (S vs t)
  F3 -- Segment grid near T_GT  (visual alignment of windows)
  F4 -- Cross-correlation of H sequences (interpolated to common grid)
  F5 -- Scatter H_modal(t) vs H_rev(t)  [interpolated]
  F6 -- Trained model comparison (P0/P1 histograms + PDFs)
  F7 -- Summary table
"""
from __future__ import annotations

import colorsys
import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -- path setup ---------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MaxEnt_SPRT import SignalData, HDF5Reader, run_maxent_sprt
from MaxEnt_SPRT.logging_setup import configure_logging
from MaxEnt_SPRT.viz.maxent_sprt_plots import configurar_estilo_global, fig_size
from MaxEnt_SPRT.viz.maxent_sprt_plots import (
    color_azul   as _CAZUL,
    color_orange as _CORANGE,
    color_verde  as _CVERDE,
    color_red    as _CRED,
)

configure_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -- palette (same HLS as maxent_sprt_plots.py) --------------------------------
def _hls(h_deg, l, s):
    r, g, b = colorsys.hls_to_rgb(h_deg / 360, l, s)
    return (r, g, b)

C_MODAL = _hls(206.957, 0.40941, 0.55603)   # azul    -- by_modal
C_REV   = _hls(36,      0.45,    0.99)       # naranja -- by_revolution
C_RED   = _hls(346,     0.45,    0.99)       # rojo    -- thresholds
C_GT    = "black"

# =============================================================================
# DATA
# =============================================================================
_DIR  = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"
_CUT  = (0.05, 16.0)
_T_GT = 5.365770208787228

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
    meta={"RPM": 12_000},
)

# =============================================================================
# CONFIGS
# T_lcm(f_modal=150 Hz, f_rev=200 Hz) = 1/gcd(150,200) = 1/50 = 20 ms
# -----------------------------------------------------------------------------
#   by_modal:   N=3 x T_modal=6.667 ms  ->  T_win = 20 ms,  hop = 6.667 ms
#   by_rev:     N=4 x T_rev=5 ms        ->  T_win = 20 ms,  hop = 5 ms
# =============================================================================
_RPM   = 12_000.0
_T_REV = 60.0 / _RPM          # 5.000 ms
_F_MOD = 150.0
_T_MOD = 1.0 / _F_MOD         # 6.667 ms
_ALPHA = 0.00135

_BASE = {
    "t_stable_total":     _T_GT,
    "training_intervals": [
        (_CUT[0], _T_GT, "stable"),
        (_T_GT,   10.0,  "chatter"),
    ],
    "alpha":          _ALPHA,
    "beta":           _ALPHA,
    "reset_on_H0":    True,
    "cut_start_time": _CUT[0],
    "cut_end_time":   10.0,
    "segmentation":   "raw",
}

CFG_MODAL = {
    "id":         "MaxEnt_SPRT",
    "func":       "Default",
    "param_mode": "by_modal",
    "params_physical": {
        "T_rev":           _T_REV,
        "T_modal":         _T_MOD,
        "N_modal_per_seg": 3,       # T_win = 3 x 6.667 ms = 20 ms
        "step_modal":      1.0,       # hop   = T_modal = 6.667 ms
        **_BASE,
    },
}

CFG_REV = {
    "id":         "MaxEnt_SPRT",
    "func":       "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev":         _T_REV,
        "N_rev_per_seg": 4,         # T_win = 4 x 5 ms = 20 ms
        "step_rev":      1.0,         # 1.333 EXACTO
        **_BASE,
    },
}

# =============================================================================
# RUN
# =============================================================================
print("Running by_modal  ...")
res_modal = run_maxent_sprt(sig, CFG_MODAL)
print("Running by_revolution ...")
res_rev   = run_maxent_sprt(sig, CFG_REV)


# -- helpers ------------------------------------------------------------------
def _unpack(res):
    meta   = res.meta or {}
    det    = meta.get("detector", None)
    H      = np.asarray(meta.get("H_seq_online", []))
    P0_mu  = meta.get("P0_mu",    np.nan)
    P0_sig = meta.get("P0_sigma", np.nan)
    P1_mu  = meta.get("P1_mu",    np.nan)
    P1_sig = meta.get("P1_sigma", np.nan)
    # Lambda_k = log p1(H_k) - log p0(H_k)  (Gaussian log-likelihood ratio)
    if H.size > 0 and not np.isnan(P0_mu):
        log_p0 = -0.5 * ((H - P0_mu) / P0_sig) ** 2 - np.log(P0_sig)
        log_p1 = -0.5 * ((H - P1_mu) / P1_sig) ** 2 - np.log(P1_sig)
        Lambda = log_p1 - log_p0
    else:
        Lambda = np.diff(np.asarray(res.I_t), prepend=0.0)   # fallback
    return dict(
        t      = np.asarray(res.t),
        H      = H,
        S      = np.asarray(res.I_t),
        Lambda = Lambda,
        t_d    = np.asarray(res.t_d) if res.t_d is not None else np.array([]),
        b      = meta["sprt_result"].b,
        a      = meta["sprt_result"].a,
        P0_mu  = P0_mu,
        P0_sig = P0_sig,
        P1_mu  = P1_mu,
        P1_sig = P1_sig,
        H_free = det.H_free if det else np.array([]),
        H_chat = det.H_chat if det else np.array([]),
    )


def _td_after(t_d, t_gt):
    mask = t_d > t_gt
    return float(t_d[mask][0]) if np.any(mask) else np.nan


def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if (not isinstance(v, float) or not np.isnan(v)) else "---"


modal = _unpack(res_modal)
rev   = _unpack(res_rev)

td_modal = _td_after(modal["t_d"], _T_GT)
td_rev   = _td_after(rev["t_d"],   _T_GT)

T_WIN_MS = 3 * _T_MOD * 1000    # = 4 * _T_REV * 1000 = 20 ms
HOP_M_MS = _T_MOD * 1000        # 6.667 ms
HOP_R_MS = _T_REV * 1000        # 5.000 ms

print(f"\n{'--'*28}")
print(f"  T_win (both)    = {T_WIN_MS:.3f} ms")
print(f"  hop  modal      = {HOP_M_MS:.3f} ms   ({1000/HOP_M_MS:.1f} seg/s)")
print(f"  hop  rev        = {HOP_R_MS:.3f} ms   ({1000/HOP_R_MS:.1f} seg/s)")
print(f"  N_seg  modal    = {len(modal['t'])}")
print(f"  N_seg  rev      = {len(rev['t'])}")
print(f"  t_d    modal    = {td_modal:.4f} s  (delta = {(td_modal - _T_GT)*1e3:+.1f} ms)")
print(f"  t_d    rev      = {td_rev:.4f} s  (delta = {(td_rev - _T_GT)*1e3:+.1f} ms)")
print(f"  |dt_d|          = {abs(td_modal - td_rev)*1e3:.2f} ms")
print(f"{'--'*28}\n")

# =============================================================================
# ANALYSIS -- cross-correlation (needed for F4, F5, F7)
# =============================================================================
t_com_min = max(modal["t"].min(), rev["t"].min())
t_com_max = min(modal["t"].max(), rev["t"].max())
dt_fine   = min(_T_MOD, _T_REV) / 8
t_com     = np.arange(t_com_min, t_com_max, dt_fine)

H_m_i = np.interp(t_com, modal["t"], modal["H"])
H_r_i = np.interp(t_com, rev["t"],   rev["H"])


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


xcorr     = np.correlate(_norm(H_m_i), _norm(H_r_i), mode="full") / len(t_com)
lags_ms   = (np.arange(len(xcorr)) - len(t_com) + 1) * dt_fine * 1000
lag_pk_ms = lags_ms[np.argmax(xcorr)]
r_zero    = float(np.corrcoef(H_m_i, H_r_i)[0, 1])

print(f"  Pearson r at zero lag : {r_zero:.4f}")
print(f"  Cross-corr peak lag   : {lag_pk_ms:.3f} ms\n")

# =============================================================================
# ANALYSIS -- Option A: Lambda_k cross-correlation & scatter
# =============================================================================
# Interpolate Lambda to the same common grid used for H
L_m_i = np.interp(t_com, modal["t"], modal["Lambda"])
L_r_i = np.interp(t_com, rev["t"],   rev["Lambda"])

xcorr_L    = np.correlate(_norm(L_m_i), _norm(L_r_i), mode="full") / len(t_com)
lag_L_ms   = lags_ms[np.argmax(xcorr_L)]   # reuse same lags_ms array
r_zero_L   = float(np.corrcoef(L_m_i, L_r_i)[0, 1])

print(f"  [Lambda] Pearson r at zero lag : {r_zero_L:.4f}")
print(f"  [Lambda] Cross-corr peak lag   : {lag_L_ms:.3f} ms\n")

# =============================================================================
# ANALYSIS -- Option B: S vs segment index k
# =============================================================================
# Use only the post-onset region (t > t_gt) for the slope comparison
mask_m_post = modal["t"] > _T_GT
mask_r_post = rev["t"]   > _T_GT

k_modal_post = np.arange(np.sum(mask_m_post))   # 0, 1, 2, ...
k_rev_post   = np.arange(np.sum(mask_r_post))
S_modal_post = modal["S"][mask_m_post]
S_rev_post   = rev["S"][mask_r_post]

# Linear fit: slope = average Lambda per segment in chatter region
if k_modal_post.size > 1:
    slope_m = np.polyfit(k_modal_post, S_modal_post, 1)[0]
else:
    slope_m = np.nan
if k_rev_post.size > 1:
    slope_r = np.polyfit(k_rev_post,   S_rev_post,   1)[0]
else:
    slope_r = np.nan

print(f"  [S vs k] slope modal   = {slope_m:.4f} per segment")
print(f"  [S vs k] slope rev     = {slope_r:.4f} per segment")
print(f"  slope ratio (rev/modal)= {slope_r/slope_m:.4f}  (expected ~1 if same Lambda)\n")

# =============================================================================
# PLOTS
# =============================================================================
configurar_estilo_global()
_SC = 5.0


def _annot_vl(ax, x, label, color, ls="--"):
    ax.axvline(x, color=color, linestyle=ls)
    ylim = ax.get_ylim()
    ax.annotate(label,
                xy=(x, ylim[0] + 0.88 * (ylim[1] - ylim[0])),
                xytext=(4, 0), textcoords="offset points",
                color=color, ha="left", va="center", rotation=90)


# -- F1 -- Entropy overlay ----------------------------------------------------
fig1, ax1 = plt.subplots(figsize=fig_size(_SC))
ax1.plot(modal["t"], modal["H"], marker=".", color=C_MODAL, label="by_modal")
ax1.plot(rev["t"],   rev["H"],   marker=".", color=C_REV,
         label="by_revolution", alpha=0.75)
ax1.axvline(_T_GT, color=C_GT, linestyle=":", label=f"$t_{{gt}}$ = {_T_GT:.3f} s")
if not np.isnan(td_modal):
    _annot_vl(ax1, td_modal, f"$t_d$ modal {td_modal:.3f}s", C_MODAL)
if not np.isnan(td_rev):
    _annot_vl(ax1, td_rev,   f"$t_d$ rev   {td_rev:.3f}s",  C_REV)
ax1.set_title("Entropy $H$ -- modal vs revolution")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Entropy $H$")
ax1.legend()

# -- F2 -- SPRT statistic overlay --------------------------------------------
fig2, ax2 = plt.subplots(figsize=fig_size(_SC))
ax2.plot(modal["t"], modal["S"], marker=".", color=C_MODAL, label="by_modal")
ax2.plot(rev["t"],   rev["S"],   marker=".", color=C_REV,
         label="by_revolution", alpha=0.75)
ax2.axhline(modal["b"], color=C_MODAL, linestyle="--", alpha=0.6,
            label=rf"$b$ modal = {modal['b']:.2f}")
ax2.axhline(rev["b"],   color=C_REV,   linestyle="--", alpha=0.6,
            label=rf"$b$ rev = {rev['b']:.2f}")
ax2.axhline(0, color="gray", linestyle=":")
ax2.axvline(_T_GT, color=C_GT, linestyle=":")
if not np.isnan(td_modal):
    _annot_vl(ax2, td_modal, f"$t_d$ modal {td_modal:.3f}s", C_MODAL)
if not np.isnan(td_rev):
    _annot_vl(ax2, td_rev,   f"$t_d$ rev   {td_rev:.3f}s",  C_REV)
ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax2.set_title(r"SPRT statistic $S_k$ -- modal vs revolution")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel(r"$I_{SPRT}$")
ax2.legend()

# -- F3 -- Segment grid near T_GT -------------------------------------------
_WIN_S  = 0.20
t_lo, t_hi = _T_GT - _WIN_S, _T_GT + _WIN_S
T_WIN_S = 3 * _T_MOD     # = 20 ms in seconds

fig3, ax3 = plt.subplots(figsize=(fig_size(_SC)[0] * 2, fig_size(_SC)[1] * 0.85))
for row_y, t_mids, T_w, c, lbl in [
    (1.0, modal["t"], T_WIN_S, C_MODAL,
     f"by_modal       hop = {HOP_M_MS:.2f} ms"),
    (0.0, rev["t"],   T_WIN_S, C_REV,
     f"by_revolution  hop = {HOP_R_MS:.2f} ms"),
]:
    vis = (t_mids >= t_lo - T_WIN_S) & (t_mids <= t_hi + T_WIN_S)
    for tm in t_mids[vis]:
        ax3.barh(row_y, T_w, left=tm - T_w / 2, height=0.30,
                 color=c, alpha=0.35, edgecolor=c, label="_nolegend_")
        ax3.plot(tm, row_y, ".", color=c, markersize=5, label="_nolegend_")
    ax3.plot([], [], color=c, marker="o", markersize=5, label=lbl)

ax3.axvline(_T_GT, color=C_GT, linestyle="--", label=f"$t_{{gt}}$")
ax3.set_xlim(t_lo, t_hi)
ax3.set_yticks([0.0, 1.0])
ax3.set_yticklabels(["by_revolution", "by_modal"])
ax3.set_xlabel("Time (s)")
ax3.set_title(f"Segment grid near $t_{{gt}}$  "
              f"(T_win = {T_WIN_MS:.1f} ms,  zoom +/-{_WIN_S*1000:.0f} ms)")
ax3.legend()

# -- F4 -- Cross-correlation --------------------------------------------------
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=fig_size(_SC, ncols=1))
ax4a.plot(lags_ms, xcorr, color=C_MODAL)
ax4a.axvline(lag_pk_ms, color=C_RED, linestyle="--",
             label=f"peak lag = {lag_pk_ms:.2f} ms")
ax4a.axvline(0, color="gray", linestyle=":")
ax4a.set_xlabel("Lag (ms)")
ax4a.set_ylabel("Norm. cross-correlation")
ax4a.set_title(r"Cross-correlation  $H_{\rm modal} \star H_{\rm rev}$  (full)")
ax4a.legend()

zoom_ms = 60
mask_z  = np.abs(lags_ms) < zoom_ms
ax4b.plot(lags_ms[mask_z], xcorr[mask_z], color=C_MODAL)
ax4b.axvline(lag_pk_ms, color=C_RED, linestyle="--",
             label=f"peak lag = {lag_pk_ms:.2f} ms")
ax4b.axvline(0, color="gray", linestyle=":",
             label=f"Pearson r(0) = {r_zero:.4f}")
ax4b.set_xlabel("Lag (ms)")
ax4b.set_ylabel("Norm. cross-correlation")
ax4b.set_title(f"Zoom +/-{zoom_ms} ms")
ax4b.legend()
fig4.tight_layout()

# -- F5 -- Scatter H_modal vs H_rev ------------------------------------------
fig5, ax5 = plt.subplots(figsize=fig_size(_SC * 0.6))
ax5.scatter(H_m_i, H_r_i, s=2, color=C_MODAL, alpha=0.3)
h_all  = np.concatenate([H_m_i, H_r_i])
h_lim  = (h_all.min(), h_all.max())
ax5.plot(h_lim, h_lim, color=C_GT, linestyle="--", label="$y = x$")
coeffs = np.polyfit(H_m_i, H_r_i, 1)
h_fit  = np.linspace(*h_lim, 200)
ax5.plot(h_fit, np.polyval(coeffs, h_fit), color=C_RED,
         label=f"fit: $y = {coeffs[0]:.3f}x + {coeffs[1]:.4f}$")
ax5.set_xlabel(r"$H_{\rm modal}(t)$")
ax5.set_ylabel(r"$H_{\rm rev}(t)$")
ax5.set_title(f"$H_{{\\rm modal}}$ vs $H_{{\\rm rev}}$  (r = {r_zero:.4f})")
ax5.legend()
ax5.set_aspect("equal", "box")

# -- F6 -- Trained model comparison ------------------------------------------
def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


fig6, axes6 = plt.subplots(1, 2,
                            figsize=(fig_size(_SC)[0] * 2, fig_size(_SC)[1]))
fig6.suptitle("Trained models  $P_0$ / $P_1$ -- modal vs revolution")

for ax_h, d, title_s in [
    (axes6[0], modal, "by_modal"),
    (axes6[1], rev,   "by_revolution"),
]:
    H_all  = np.concatenate([d["H_free"], d["H_chat"]])
    margin = 0.15 * (H_all.max() - H_all.min())
    x_pdf  = np.linspace(H_all.min() - margin, H_all.max() + margin, 600)

    ax_h.hist(d["H_free"], density=True, alpha=0.4, color=_CAZUL,
              bins="auto", label="$H$ stable")
    ax_h.hist(d["H_chat"], density=True, alpha=0.4, color=_CORANGE,
              bins="auto", label="$H$ chatter")
    ax_h.plot(x_pdf, _gauss(x_pdf, d["P0_mu"], d["P0_sig"]),
              color=_CVERDE,
              label=rf"$P_0$: $\mu$={d['P0_mu']:.4f}, $\sigma$={d['P0_sig']:.4f}")
    ax_h.plot(x_pdf, _gauss(x_pdf, d["P1_mu"], d["P1_sig"]),
              color=_CRED,
              label=rf"$P_1$: $\mu$={d['P1_mu']:.4f}, $\sigma$={d['P1_sig']:.4f}")
    ax_h.axvline(d["P0_mu"], color=_CVERDE, linestyle="--", alpha=0.6)
    ax_h.axvline(d["P1_mu"], color=_CRED,   linestyle="--", alpha=0.6)
    ax_h.set_title(title_s)
    ax_h.set_xlabel("Entropy $H$")
    ax_h.set_ylabel("Probability Density")
    ax_h.legend()

fig6.tight_layout()

# -- F7 -- Summary table ------------------------------------------------------
rows = {
    "T_win (ms)":           [f"{T_WIN_MS:.3f}", f"{T_WIN_MS:.3f}"],
    "hop (ms)":             [f"{HOP_M_MS:.3f}", f"{HOP_R_MS:.3f}"],
    "Seg. rate (seg/s)":    [f"{1000/HOP_M_MS:.1f}", f"{1000/HOP_R_MS:.1f}"],
    "N_seg total":          [str(len(modal["t"])), str(len(rev["t"]))],
    "P0  mu":               [_fmt(modal["P0_mu"]),  _fmt(rev["P0_mu"])],
    "P0  sigma":            [_fmt(modal["P0_sig"]), _fmt(rev["P0_sig"])],
    "P1  mu":               [_fmt(modal["P1_mu"]),  _fmt(rev["P1_mu"])],
    "P1  sigma":            [_fmt(modal["P1_sig"]), _fmt(rev["P1_sig"])],
    "b (SPRT threshold)":   [_fmt(modal["b"]), _fmt(rev["b"])],
    "t_d (s)":              [_fmt(td_modal), _fmt(td_rev)],
    "Delta_t_d (ms)":       [_fmt((td_modal - _T_GT) * 1e3, ".1f"),
                              _fmt((td_rev   - _T_GT) * 1e3, ".1f")],
    "Pearson r (H, lag=0)": [f"{r_zero:.4f}", "---"],
    "xcorr peak lag (ms)":  [f"{lag_pk_ms:.3f}", "---"],
}
df = pd.DataFrame(rows, index=["by_modal", "by_revolution"]).T
print(df.to_string())

fig7, ax7 = plt.subplots(figsize=(fig_size(_SC * 0.9)[0], fig_size(_SC * 0.5)[1]))
ax7.axis("off")
tbl = ax7.table(
    cellText=df.values,
    rowLabels=df.index,
    colLabels=df.columns,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(True)
tbl.auto_set_column_width(col=list(range(len(df.columns))))
ax7.set_title(
    f"Summary -- Modal vs Revolution  (T_win = {T_WIN_MS:.0f} ms,  T_lcm = 20 ms)",
    pad=12,
)
fig7.tight_layout()

# -- F8 (Option A) -- Lambda_k cross-correlation ------------------------------
fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=fig_size(_SC, ncols=1))
ax8a.plot(lags_ms, xcorr_L, color=C_MODAL)
ax8a.axvline(lag_L_ms, color=C_RED, linestyle="--",
             label=f"peak lag = {lag_L_ms:.2f} ms")
ax8a.axvline(0, color="gray", linestyle=":")
ax8a.set_xlabel("Lag (ms)")
ax8a.set_ylabel("Norm. cross-correlation")
ax8a.set_title(r"Cross-correlation  $\Lambda_{\rm modal} \star \Lambda_{\rm rev}$  (full)")
ax8a.legend()

zoom_ms = 60
mask_z8 = np.abs(lags_ms) < zoom_ms
ax8b.plot(lags_ms[mask_z8], xcorr_L[mask_z8], color=C_MODAL)
ax8b.axvline(lag_L_ms, color=C_RED, linestyle="--",
             label=f"peak lag = {lag_L_ms:.2f} ms")
ax8b.axvline(0, color="gray", linestyle=":")
ax8b.set_xlabel("Lag (ms)")
ax8b.set_ylabel("Norm. cross-correlation")
ax8b.set_title(f"Zoom +/-{zoom_ms} ms   (Pearson r(0) = {r_zero_L:.4f})")
ax8b.legend()
fig8.suptitle(r"Option A -- Cross-correlation of $\Lambda_k = \log\,p_1/p_0$")
fig8.tight_layout()

# -- F9 (Option A) -- Scatter Lambda_modal vs Lambda_rev ----------------------
fig9, ax9 = plt.subplots(figsize=fig_size(_SC * 0.6))
ax9.scatter(L_m_i, L_r_i, s=2, color=C_MODAL, alpha=0.3)
l_all  = np.concatenate([L_m_i, L_r_i])
l_lim  = (l_all.min(), l_all.max())
ax9.plot(l_lim, l_lim, color=C_GT, linestyle="--", label="$y = x$")
coeffs_L = np.polyfit(L_m_i, L_r_i, 1)
l_fit    = np.linspace(*l_lim, 200)
ax9.plot(l_fit, np.polyval(coeffs_L, l_fit), color=C_RED,
         label=f"fit: $y = {coeffs_L[0]:.3f}x + {coeffs_L[1]:.4f}$")
ax9.set_xlabel(r"$\Lambda_{\rm modal}$")
ax9.set_ylabel(r"$\Lambda_{\rm rev}$")
ax9.set_title(
    f"Option A -- $\\Lambda_{{\\rm modal}}$ vs $\\Lambda_{{\\rm rev}}$  (r = {r_zero_L:.4f})"
)
ax9.legend()
ax9.set_aspect("equal", "box")

# -- F10 (Option B) -- S vs segment index k (post-onset) ---------------------
fig10, ax10 = plt.subplots(figsize=fig_size(_SC))
ax10.plot(k_modal_post, S_modal_post, marker=".", color=C_MODAL,
          label=f"by_modal  slope = {slope_m:.4f}/seg")
ax10.plot(k_rev_post,   S_rev_post,   marker=".", color=C_REV,
          label=f"by_revolution  slope = {slope_r:.4f}/seg", alpha=0.75)
# linear fits
if not np.isnan(slope_m):
    ax10.plot(k_modal_post,
              np.polyval(np.polyfit(k_modal_post, S_modal_post, 1), k_modal_post),
              color=C_MODAL, linestyle="--", alpha=0.5)
if not np.isnan(slope_r):
    ax10.plot(k_rev_post,
              np.polyval(np.polyfit(k_rev_post, S_rev_post, 1), k_rev_post),
              color=C_REV,   linestyle="--", alpha=0.5)
ax10.axhline(modal["b"], color=C_MODAL, linestyle=":", alpha=0.6,
             label=rf"$b$ modal = {modal['b']:.2f}")
ax10.axhline(rev["b"],   color=C_REV,   linestyle=":", alpha=0.6,
             label=rf"$b$ rev = {rev['b']:.2f}")
ax10.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax10.set_title(
    r"Option B -- $S_k$ vs segment index $k$  (post-onset region $t > t_{gt}$)"
)
ax10.set_xlabel("Segment index $k$ (after $t_{gt}$)")
ax10.set_ylabel(r"$S_k$")
ax10.legend()

plt.show()
