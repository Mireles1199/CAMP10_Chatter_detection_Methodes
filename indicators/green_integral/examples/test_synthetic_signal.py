"""Synthetic signal test for FixedWindow indicator.

Signal profile (4 phases):
  0 – 2 s   →  amortiguación   (σ = -1.5)   amplitude decays
  2 – 5 s   →  chatter         (σ = +1.2)   amplitude grows
  5 – 8 s   →  estable         (σ = -0.8)   amplitude decays
  8 – 11 s  →  chatter         (σ = +1.0)   amplitude grows again
"""

from __future__ import annotations
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral.logging_setup import configure_logging, LOGGING_LEVELS
configure_logging(level=LOGGING_LEVELS["warning"])   # quiet — only warnings

from green_integral import SignalData, run_fixed_window, FixedWindowConfig
from green_integral import plots_fixed_window

# ── Parameters ──────────────────────────────────────────────────────────────
fs       = 5000.0        # sampling frequency [Hz]
f_modal  = 150.0         # modal frequency [Hz]
T_modal  = 1.0 / f_modal

# ── Build synthetic signal ───────────────────────────────────────────────────
dt_s = 1.0 / fs
t    = np.arange(0.0, 11.0, dt_s)

# Piecewise σ(t)
sigma_true = np.piecewise(
    t,
    [t < 2.0,
     (t >= 2.0) & (t < 5.0),
     (t >= 5.0) & (t < 8.0),
     t >= 8.0],
    [-1.5, +1.2, -0.8, +1.0],
)

# Instantaneous amplitude: A(t) = exp( ∫₀ᵗ σ(τ) dτ )
A = np.exp(np.cumsum(sigma_true) * dt_s)

# Displacement: sinusoid modulated by A(t), plus small white noise
rng = np.random.default_rng(42)
noise_level = 1e-4
x = A * np.sin(2.0 * np.pi * f_modal * t) + noise_level * rng.standard_normal(len(t))
v = np.gradient(x, t)

sig = SignalData(t=t, displacement=x, velocity=v, name="synthetic_4phases")

# ── Indicator config ─────────────────────────────────────────────────────────
cfg = FixedWindowConfig(
    f_modal       = f_modal,
    num_T         = 6,
    dt            = T_modal,        # step = one period → good time resolution
    data_filtrated= True,
    lambda_ewma   = 0.2,            # moderate smoothing
    accumulate    = True,           # Ĝ desde t=0  (detector de evento)
    G_memory      = 1.5,            # Ĝ deslizante últimos 1.5 s (detector de estado)
    sigma_method  = "ratio",
)

# ── Run ──────────────────────────────────────────────────────────────────────
res = run_fixed_window(sig, cfg)

# ── Console summary ──────────────────────────────────────────────────────────
print("\n── Synthetic signal: 4-phase test ───────────────────────────────")
print(f"  Windows computed : {len(res.areas)}")
print(f"  σ̂ mean (valid)  : {float(np.nanmean(res.sigma)):.3f} 1/s")
print(f"  Ĝ final (accum) : {float(res.G_hat[-1]) if res.G_hat.size else float('nan'):.4f}")
print(f"  Ĝ final (slide) : {float(res.G_hat_sliding[-1]) if res.G_hat_sliding.size else float('nan'):.4f}")
print("─────────────────────────────────────────────────────────────────")

# ── Plot reference: true σ(t) vs estimated ───────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Synthetic test — 4 phases: damp / chatter / stable / chatter", fontsize=12)

# --- row 1: signal amplitude -------------------------------------------------
axes[0].plot(t, x, color="steelblue", linewidth=0.4, alpha=0.7)
axes[0].set_ylabel("x(t)  [m]")
axes[0].set_title("Displacement signal")

# shade phases
_phases = [(0, 2, "lightgreen", "damp"), (2, 5, "lightsalmon", "chatter"),
           (5, 8, "lightgreen",  "stable"), (8, 11, "lightsalmon", "chatter")]
for x0, x1, col, lbl in _phases:
    axes[0].axvspan(x0, x1, color=col, alpha=0.25, label=lbl)
axes[0].legend(fontsize=8, loc="upper right")

# --- row 2: σ̂ raw vs EWMA vs true ------------------------------------------
t_w = res.t_wins
axes[1].plot(t, sigma_true, color="black", linewidth=1.5,
             linestyle="--", label="σ true")
valid = np.isfinite(res.sigma)
axes[1].plot(t_w[valid], res.sigma[valid], color="gray",
             linewidth=0.6, alpha=0.6, label="σ̂ raw")
valid_e = np.isfinite(res.sigma_ewma)
axes[1].plot(t_w[valid_e], res.sigma_ewma[valid_e], color="tomato",
             linewidth=1.5, label="σ̂ EWMA")
axes[1].axhline(0, color="black", linewidth=0.8, linestyle=":")
axes[1].set_ylabel("σ̂  [1/s]")
axes[1].set_title("Lyapunov exponent estimate")
axes[1].legend(fontsize=8)

# --- row 3: Ĝ accumulated vs Ĝ sliding --------------------------------------
if res.G_hat.size > 0:
    axes[2].plot(t_w[:len(res.G_hat)], res.G_hat,
                 color="darkorange", linewidth=1.5, label="Ĝ accumulated (from t=0)")
if res.G_hat_sliding.size > 0:
    axes[2].plot(t_w[:len(res.G_hat_sliding)], res.G_hat_sliding,
                 color="mediumpurple", linewidth=1.5,
                 label=f"Ĝ sliding (T_mem={cfg.G_memory} s)")
axes[2].axhline(0, color="black", linewidth=0.8, linestyle=":")
axes[2].fill_between(t_w[:len(res.G_hat_sliding)], res.G_hat_sliding, 0,
                     where=(res.G_hat_sliding > 0), alpha=0.15,
                     color="red", label="chatter")
axes[2].fill_between(t_w[:len(res.G_hat_sliding)], res.G_hat_sliding, 0,
                     where=(res.G_hat_sliding <= 0), alpha=0.10,
                     color="green", label="stable")
axes[2].set_ylabel("Ĝ")
axes[2].set_xlabel("Time [s]")
axes[2].set_title("Accumulated vs Sliding Ĝ")
axes[2].legend(fontsize=8)

for ax in axes:
    for x0, x1, col, _ in _phases:
        ax.axvspan(x0, x1, color=col, alpha=0.08)

fig.tight_layout()

# Standard indicator plots (areas + σ̂ + Ĝ figures)
plots_fixed_window(signal=sig, result=res, show=False)

plt.show()
