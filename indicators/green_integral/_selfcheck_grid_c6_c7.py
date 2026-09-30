"""Ad-hoc self-check for plots_lyapunov's C6/C7 per-case grid figures (not
committed).

Verifies: (1) with >=2 external reference_signal pieces, C6 (stable training
signal) and C7 (stable decision variable / area) each render as an N-subplot
grid, one per piece; (2) with a single stable training_intervals range (no
reference_signal), both fall back to a single panel -- same per-case-grid
philosophy as MaxEnt's F0a/F1 and SST's C6/C7; (3) grid_scale independently
controls C6/C7 size vs. scale.
"""
import sys
import pathlib
import numpy as np

_here = pathlib.Path(__file__).resolve().parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral.logging_setup import configure_logging, LOGGING_LEVELS
configure_logging(level=LOGGING_LEVELS["warning"])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from green_integral import StdSignalData, run_green_std, plots_lyapunov

fs = 5000.0
f_modal = 150.0


def _make_signal(t: np.ndarray, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * np.sin(2.0 * np.pi * f_modal * t) + 1e-4 * rng.standard_normal(len(t))


BASE = dict(f_modal=f_modal, num_T=4, dt=0.01, data_filtrated=True,
            while_loop_extend=True, use_area_threshold=True, z_sigma=3.0,
            sigma_method="ratio")

t_an = np.arange(0.0, 1.0, 1.0 / fs)
an_std = StdSignalData(t_analysis=t_an, signal_analysis=_make_signal(t_an, 1.0, 0),
                        path="analysis", fs=fs)


def _fig(label_prefix):
    for n in plt.get_fignums():
        fig = plt.figure(n)
        if fig.get_label().startswith(label_prefix):
            return fig
    raise AssertionError(f"no figure with label prefix {label_prefix!r}: "
                          f"{[plt.figure(n).get_label() for n in plt.get_fignums()]}")


# 1) >=2 reference pieces -> C6/C7 render as an N-subplot grid.
t_piece = np.arange(0.0, 0.6, 1.0 / fs)
pieces = [
    StdSignalData(t_analysis=t_piece, signal_analysis=_make_signal(t_piece, 1.0, s),
                  path=f"piece{s}", fs=fs, meta={"name": f"piece{s}"})
    for s in (1, 2, 3)
]
cfg_multi = {"func": "Lyapunov", "param_mode": "native", "params": dict(BASE),
             "reference_signal": pieces}
res_multi = run_green_std(an_std, cfg_multi)
raw_multi = res_multi.meta["raw_result"]
sig_multi = res_multi.meta["signal"]

plt.close("all")
plots_lyapunov(signal=sig_multi, result=raw_multi, reference_signal=pieces, show=False)
fig_c6 = _fig("C6")
fig_c7 = _fig("C7")
assert len(fig_c6.axes) == 3, f"C6: expected 3 subplots, got {len(fig_c6.axes)}"
assert len(fig_c7.axes) == 3, f"C7: expected 3 subplots, got {len(fig_c7.axes)}"
for ax in fig_c6.axes:
    assert len(ax.get_lines()) > 0 and ax.get_lines()[0].get_xdata().size > 0, "C6 subplot has no data"
for ax in fig_c7.axes:
    assert len(ax.get_lines()) > 0 and ax.get_lines()[0].get_xdata().size > 0, "C7 subplot has no data"
plt.close("all")

# 2) single stable training_intervals range (no reference_signal) -> fallback.
cfg_internal = {"func": "Lyapunov", "param_mode": "native",
                 "params": dict(BASE, training_intervals=[(0.0, 0.3, "stable")])}
res_internal = run_green_std(an_std, cfg_internal)
raw_internal = res_internal.meta["raw_result"]
sig_internal = res_internal.meta["signal"]

plt.close("all")
plots_lyapunov(signal=sig_internal, result=raw_internal,
               training_intervals=[(0.0, 0.3, "stable")], show=False)
fig_c6b = _fig("C6")
fig_c7b = _fig("C7")
assert len(fig_c6b.axes) == 1, f"C6 (internal, single case): expected 1 panel, got {len(fig_c6b.axes)}"
assert len(fig_c7b.axes) == 1, f"C7 (internal, single case): expected 1 panel, got {len(fig_c7b.axes)}"
plt.close("all")

# 3) grid_scale independently controls C6/C7 size vs. scale.
plt.close("all")
plots_lyapunov(signal=sig_multi, result=raw_multi, reference_signal=pieces,
               show=False, scale=1.0, grid_scale=1.0)
size_1x = _fig("C6").get_size_inches()
plt.close("all")
plots_lyapunov(signal=sig_multi, result=raw_multi, reference_signal=pieces,
               show=False, scale=1.0, grid_scale=2.0)
size_2x = _fig("C6").get_size_inches()
assert np.allclose(size_2x, size_1x * 2.0, atol=0.02), (size_1x, size_2x)
plt.close("all")

print("OK: C6/C7 per-case grid figures (multi-piece grid, single-panel fallback, grid_scale) behave as specified.")
