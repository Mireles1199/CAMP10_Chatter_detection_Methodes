"""Self-check for the article-plot-style migration (plot_style.py).

Headless (Agg), no example script executed. Verifies:
1. Every figure plots.py produces uses plot_style.FIGSIZE_WIDE scaled by SCALE
   (not the old fig_size(scale=5.0)/(scale=3.0) formulas).
2. plots_lyapunov's C2-C3/D1b/D4b/G-hat/G-hat-sliding panels plus the
   training-distribution pair share the exact same size (_LYAP_FIGSIZE) --
   same invariant test_plot_reliability.py's check #6 already enforces,
   re-checked here against the new preset-based sizes. C1 and the C6/C7
   per-case grids are sized off FIGSIZE_SIMPLE instead (checked separately).
3. plots_signal_diagnostics' 3 figures use the new composed sizes.
4. rcParams from plot_style.ARTICLE_RCPARAMS are actually applied (spot-check
   axes.titlesize / legend.frameon).
"""

from __future__ import annotations
import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral.logging_setup import configure_logging, LOGGING_LEVELS
configure_logging(level=LOGGING_LEVELS["warning"])

from green_integral import (
    StdSignalData, run_green_std, plots_signal_diagnostics, plots_lyapunov,
    plots_green_integral,
)
from green_integral.viz.plots import (
    plot_windows_local, plot_windows_duration, plot_indicator_local,
)
from green_integral.viz.plot_style import FIGSIZE_SIMPLE, FIGSIZE_WIDE, SCALE, figsize_from_scale
from green_integral.viz.green_integral_plots import _LYAP_FIGSIZE, _diag_figsize

fs = 5000.0
f_modal = 150.0


def _make_signal(t: np.ndarray, sigma_of_t) -> np.ndarray:
    dt = 1.0 / fs
    A = np.exp(np.cumsum(sigma_of_t(t)) * dt)
    rng = np.random.default_rng(0)
    return A * np.sin(2.0 * np.pi * f_modal * t) + 1e-4 * rng.standard_normal(len(t))


t_an = np.arange(0.0, 3.0, 1.0 / fs)
x_an = _make_signal(t_an, lambda t: np.where(t < 0.3, 0.0, 2.5))
an_std = StdSignalData(t_analysis=t_an, signal_analysis=x_an, path="analysis", fs=fs)

BASE = dict(f_modal=f_modal, num_T=4, dt=0.01, data_filtrated=True,
            while_loop_extend=True, use_area_threshold=True,
            training_intervals=[(0.0, 0.3, "stable")], z_sigma=3.0)


def _run(func: str, params: dict):
    return run_green_std(an_std, {"func": func, "param_mode": "native", "params": dict(params)})


_expected_wide_scaled = tuple(round(v, 6) for v in figsize_from_scale(FIGSIZE_WIDE, SCALE))


def _is_c6c7(title: str) -> bool:
    # C1 (single-panel signal) and the C6/C7 per-case grid figures are sized
    # off FIGSIZE_SIMPLE, not _LYAP_FIGSIZE like the rest of the set -- same
    # reason MaxEnt/SST's own single-panel/grid figures are excluded from
    # their WIDE-composite size checks.
    return title.startswith(("C1 — Signal", "Stable Training Signal", "Stable Decision Variable"))

# ── 1. plots.py figures use FIGSIZE_WIDE scaled by SCALE ───────────────────
res = _run("Default", BASE)
raw = res.meta["raw_result"]
result_dict = {
    "data_window": raw.data_window, "agrupamiento": raw.agrupamiento,
    "global_data": raw.global_data, "t_d": raw.t_d,
}

plt.close("all")
for fig in [
    plot_windows_local(result_dict, name="analysis"),
    plot_indicator_local(result_dict, name="analysis"),
    *plot_windows_duration(result_dict, name="analysis"),
]:
    size = tuple(round(v, 6) for v in fig.get_size_inches())
    assert size == _expected_wide_scaled, f"{fig!r}: size {size} != FIGSIZE_WIDE*SCALE {_expected_wide_scaled}"
plt.close("all")

# ── 2. plots_lyapunov set + training distribution share _LYAP_FIGSIZE ──────
lyap_res = _run("Lyapunov", {**BASE, "sigma_method": "ratio"})
raw_lyap = lyap_res.meta["raw_result"]
sig = lyap_res.meta["signal"]

_expected_lyap = tuple(round(v, 6) for v in _LYAP_FIGSIZE)
plt.close("all")
plots_lyapunov(signal=sig, result=raw_lyap,
               training_intervals=[(0.0, 0.3, "stable")], show=False)
sizes = {}
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    title = fig.axes[0].get_title() if fig.axes else ""
    sizes[title or f"fig{fig_num}"] = tuple(round(v, 6) for v in fig.get_size_inches())
assert sizes, "plots_lyapunov produced no figures"
for title, size in sizes.items():
    if _is_c6c7(title):
        continue
    assert size == _expected_lyap, f"{title!r}: size {size} != _LYAP_FIGSIZE {_expected_lyap}"
# C6/C7: single stable range (training_intervals has only 1 entry) -> both
# fall back to a single FIGSIZE_SIMPLE-based panel, same fallback rule as
# MaxEnt/SST's own per-case grids.
_expected_simple = tuple(round(v, 6) for v in figsize_from_scale(FIGSIZE_SIMPLE, SCALE))
for title, size in sizes.items():
    if _is_c6c7(title):
        assert size == _expected_simple, f"{title!r}: size {size} != FIGSIZE_SIMPLE*SCALE {_expected_simple}"
plt.close("all")

# ── 3. plots_signal_diagnostics composed sizes ──────────────────────────────
plt.close("all")
plots_signal_diagnostics(signal=sig, result=raw_lyap, stable_range=(0.0, 0.3),
                          zoom_range=(0.05, 0.1), eq_smooth_s=0.02, show=False)
diag_figs = [plt.figure(n) for n in plt.get_fignums()]
assert len(diag_figs) == 3, f"expected 3 diagnostic figures, got {len(diag_figs)}"
expected_diag = [
    tuple(round(v, 6) for v in _diag_figsize(nrows=2)),
    tuple(round(v, 6) for v in _diag_figsize(nrows=3)),
    tuple(round(v, 6) for v in _diag_figsize(ncols=3)),
]
actual_diag = [tuple(round(v, 6) for v in f.get_size_inches()) for f in diag_figs]
assert actual_diag == expected_diag, f"{actual_diag} != {expected_diag}"
plt.close("all")

# ── 4. ARTICLE_RCPARAMS actually applied ────────────────────────────────────
assert plt.rcParams["axes.titlesize"] == 16, plt.rcParams["axes.titlesize"]
assert plt.rcParams["legend.frameon"] is False, plt.rcParams["legend.frameon"]
assert plt.rcParams["font.family"] == ["serif"], plt.rcParams["font.family"]

# ── 5. per-call `scale` actually resizes figures (independent of plot_style.SCALE) ──
# The user asked to control figsize/figscale from the example script, per call --
# not just from the shared plot_style.py default. scale=2.0 must double every figure.
_SCALE2 = 2.0
_expected_wide_2x = tuple(round(v, 6) for v in figsize_from_scale(FIGSIZE_WIDE, _SCALE2))
assert _expected_wide_2x == tuple(round(v * 2, 6) for v in FIGSIZE_WIDE)

plt.close("all")
plots_green_integral(signal=res.meta["signal"], result=raw, show=False, scale=_SCALE2)
for fig_num in plt.get_fignums():
    size = tuple(round(v, 6) for v in plt.figure(fig_num).get_size_inches())
    assert size == _expected_wide_2x, f"plots_green_integral scale=2.0: size {size} != {_expected_wide_2x}"
plt.close("all")

_expected_lyap_2x = tuple(round(v, 6) for v in figsize_from_scale(FIGSIZE_WIDE, _SCALE2))
plots_lyapunov(signal=sig, result=raw_lyap,
               training_intervals=[(0.0, 0.3, "stable")], show=False, scale=_SCALE2)
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    title = fig.axes[0].get_title() if fig.axes else ""
    if _is_c6c7(title):
        continue
    size = tuple(round(v, 6) for v in fig.get_size_inches())
    assert size == _expected_lyap_2x, f"plots_lyapunov scale=2.0: size {size} != {_expected_lyap_2x}"
plt.close("all")

plots_signal_diagnostics(signal=sig, result=raw_lyap, stable_range=(0.0, 0.3),
                          zoom_range=(0.05, 0.1), eq_smooth_s=0.02, show=False, scale=_SCALE2)
diag_figs_2x = [plt.figure(n) for n in plt.get_fignums()]
expected_diag_2x = [
    tuple(round(v, 6) for v in _diag_figsize(nrows=2, scale=_SCALE2)),
    tuple(round(v, 6) for v in _diag_figsize(nrows=3, scale=_SCALE2)),
    tuple(round(v, 6) for v in _diag_figsize(ncols=3, scale=_SCALE2)),
]
actual_diag_2x = [tuple(round(v, 6) for v in f.get_size_inches()) for f in diag_figs_2x]
assert actual_diag_2x == expected_diag_2x, f"{actual_diag_2x} != {expected_diag_2x}"
# and confirm it's actually double the true scale=1.0 (unscaled) sizes, not just
# coincidentally equal to `expected_diag` above (which uses the package's default
# SCALE, not necessarily 1.0)
expected_diag_1x = [
    tuple(round(v, 6) for v in _diag_figsize(nrows=2, scale=1.0)),
    tuple(round(v, 6) for v in _diag_figsize(nrows=3, scale=1.0)),
    tuple(round(v, 6) for v in _diag_figsize(ncols=3, scale=1.0)),
]
assert all(a == tuple(round(v * 2, 6) for v in e) for a, e in zip(actual_diag_2x, expected_diag_1x)), (
    "scale=2.0 diagnostics figures aren't 2x the scale=1.0 sizes"
)
plt.close("all")

# ── 6. per-call `figsize_wide` actually resizes figures (not just `scale`) ──
# The user asked to control the base FIGSIZE preset itself from the example
# script, not just a multiplier on top of the fixed plot_style.py default.
_CUSTOM_WIDE = (4.0, 3.0)

plt.close("all")
plots_green_integral(signal=res.meta["signal"], result=raw, show=False, scale=1.0, figsize_wide=_CUSTOM_WIDE)
for fig_num in plt.get_fignums():
    size = tuple(round(v, 6) for v in plt.figure(fig_num).get_size_inches())
    assert size == _CUSTOM_WIDE, f"plots_green_integral figsize_wide override: size {size} != {_CUSTOM_WIDE}"
plt.close("all")

plots_lyapunov(signal=sig, result=raw_lyap,
               training_intervals=[(0.0, 0.3, "stable")], show=False, scale=1.0, figsize_wide=_CUSTOM_WIDE)
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    title = fig.axes[0].get_title() if fig.axes else ""
    if _is_c6c7(title):
        continue  # sized off FIGSIZE_SIMPLE, not figsize_wide
    size = tuple(round(v, 6) for v in fig.get_size_inches())
    assert size == _CUSTOM_WIDE, f"plots_lyapunov figsize_wide override: size {size} != {_CUSTOM_WIDE}"
plt.close("all")

print("OK — article-plot-style self-checks passed.")
