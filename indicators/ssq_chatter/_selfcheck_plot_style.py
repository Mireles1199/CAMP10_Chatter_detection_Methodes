"""Ad-hoc self-check for the article-plot-style migration (not committed).

Verifies: plots render without exception, and every figure's real size matches
the new FIGSIZE_SIMPLE/FIGSIZE_WIDE-derived presets (not the old ~10in scale=3.0
sizes) -- single-panel figures get FIGSIZE_SIMPLE, the two 2-stacked-row figures
(C4, C5) get double the height.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from ssq_chatter import SignalData, run_sst_svd
from ssq_chatter.viz.sst_svd_plots import plots_sst_svd
from ssq_chatter.viz.plot_style import FIGSIZE_SIMPLE, figsize_grid, figsize_from_scale, SCALE

FS = 2000.0


def _make_signal(duration_s: float, amp: float, seed: int) -> SignalData:
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    x = amp * np.sin(2 * np.pi * 120.0 * t) + 0.01 * rng.standard_normal(n)
    return SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=FS)


NATIVE_PARAMS = dict(
    n_fft_power=0, win_length_ms=50.0, hop_ms=25.0, Ai_length=4,
    mode="causal_inclusive", sigma=5.0, frac_stable=0.3,
    alpha=0.05, z=3.0, fallback_mad=True,
)

signal = _make_signal(1.0, amp=0.5, seed=0)
res = run_sst_svd(signal, {"func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS)})

SIMPLE = figsize_from_scale(FIGSIZE_SIMPLE, SCALE)
STACKED = figsize_from_scale(figsize_grid(1, 2), SCALE)

# 1) no exception, including the heavy on-demand spectrogram/waterfall figures.
plt.close("all")
plots_sst_svd(signal=signal, result=res, show_spectrograms=True)
fig_labels = {plt.figure(n).get_label(): plt.figure(n) for n in plt.get_fignums()}
assert fig_labels, "no figures were created"

# 2) single-panel figures -> FIGSIZE_SIMPLE (not the old ~10in scale=3.0 size).
for prefix in ("F1 ", "F1b ", "F2 ", "F2b ", "C1 ", "C2 ", "C3 "):
    lbl = next((l for l in fig_labels if l.startswith(prefix)), None)
    assert lbl is not None, f"missing figure {prefix!r}: {list(fig_labels)}"
    size = fig_labels[lbl].get_size_inches()
    assert np.allclose(size, SIMPLE, atol=0.01), (lbl, size, SIMPLE)

# 3) 2-stacked-row figures (C4, C5) -> double height, same width.
for prefix in ("C4 ",):
    lbl = next((l for l in fig_labels if l.startswith(prefix)), None)
    assert lbl is not None, f"missing figure {prefix!r}: {list(fig_labels)}"
    size = fig_labels[lbl].get_size_inches()
    assert np.allclose(size, STACKED, atol=0.01), (lbl, size, STACKED)
plt.close("all")

# 4) C5 (training signal) is only produced when a training population is
# available together with a per-piece reference -- exercise it directly.
piece = _make_signal(0.6, amp=0.05, seed=1)
res_ref = run_sst_svd(signal, {
    "func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS),
    "reference_signal": [piece],
})
plt.close("all")
plots_sst_svd(signal=signal, result=res_ref, reference_signal=[piece])
fig_labels2 = {plt.figure(n).get_label(): plt.figure(n) for n in plt.get_fignums()}
lbl_c5 = next(l for l in fig_labels2 if l.startswith("C5 "))
size_c5 = fig_labels2[lbl_c5].get_size_inches()
assert np.allclose(size_c5, STACKED, atol=0.01), (lbl_c5, size_c5, STACKED)
plt.close("all")

print("OK: article-plot-style figsize migration (no exceptions, sizes match FIGSIZE_SIMPLE/2-row-stacked presets).")
