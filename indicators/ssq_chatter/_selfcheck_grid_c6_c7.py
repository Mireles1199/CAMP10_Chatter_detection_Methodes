"""Ad-hoc self-check for the C6/C7 per-case grid figures (not committed).

Verifies: (1) with >=2 external reference_signal pieces, C6 (stable training
signal) and C7 (stable decision variable d1) each render as an N-subplot grid,
one per piece, no exception; (2) with plain internal training (no
reference_signal, no multi-label training_intervals), both fall back to a
single panel -- same per-case-grid philosophy as MaxEnt's F0a/F1.
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


def _fig(label_prefix):
    return next(plt.figure(n) for n in plt.get_fignums()
                if plt.figure(n).get_label().startswith(label_prefix))


# 1) >=2 reference pieces -> C6/C7 render as an N-subplot grid.
pieces = [_make_signal(0.6, amp=0.05, seed=s) for s in (1, 2, 3)]
res_multi = run_sst_svd(signal, {
    "func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS),
    "reference_signal": pieces,
})
plt.close("all")
plots_sst_svd(signal=signal, result=res_multi, reference_signal=pieces)
fig_c6 = _fig("C6 ")
fig_c7 = _fig("C7 ")
assert len(fig_c6.axes) == 3, f"C6: expected 3 subplots, got {len(fig_c6.axes)}"
assert len(fig_c7.axes) == 3, f"C7: expected 3 subplots, got {len(fig_c7.axes)}"
for ax in fig_c6.axes:
    assert len(ax.get_lines()) > 0 and ax.get_lines()[0].get_xdata().size > 0, "C6 subplot has no data"
for ax in fig_c7.axes:
    assert len(ax.get_lines()) > 0 and ax.get_lines()[0].get_xdata().size > 0, "C7 subplot has no data"
plt.close("all")

# 2) plain internal training (no reference_signal) -> single-panel fallback.
res_internal = run_sst_svd(signal, {"func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS)})
plt.close("all")
plots_sst_svd(signal=signal, result=res_internal)
fig_c6b = _fig("C6 ")
fig_c7b = _fig("C7 ")
assert len(fig_c6b.axes) == 1, f"C6 (internal, single case): expected 1 panel, got {len(fig_c6b.axes)}"
assert len(fig_c7b.axes) == 1, f"C7 (internal, single case): expected 1 panel, got {len(fig_c7b.axes)}"
plt.close("all")

# 3) grid_scale independently controls C6/C7 size vs. scale.
plt.close("all")
plots_sst_svd(signal=signal, result=res_multi, reference_signal=pieces, scale=1.0, grid_scale=1.0)
size_1x = _fig("C6 ").get_size_inches()
plt.close("all")
plots_sst_svd(signal=signal, result=res_multi, reference_signal=pieces, scale=1.0, grid_scale=2.0)
size_2x = _fig("C6 ").get_size_inches()
assert np.allclose(size_2x, size_1x * 2.0, atol=0.02), (size_1x, size_2x)
plt.close("all")

print("OK: C6/C7 per-case grid figures (multi-piece grid, single-panel fallback, grid_scale) behave as specified.")
