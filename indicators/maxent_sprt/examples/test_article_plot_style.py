"""
test_article_plot_style.py
===========================
Self-check for the article-plot-style migration of maxent_sprt_plots.py:
figures must render without exception and use the new FIGSIZE_SIMPLE /
figsize_grid presets (SCALE=1.0), not the old inflated ad-hoc sizes
(scale=3.0/5.0 * base_width=3.4 -> ~10in-wide figures).

Headless (Agg backend), synthetic signal, no real dataset needed.
Run directly: asserts only.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MaxEnt_SPRT import SignalData, run_maxent_sprt, plots_maxent_sprt
from MaxEnt_SPRT.viz.plot_style import FIGSIZE_SIMPLE, FIGSIZE_WIDE, figsize_grid, figsize_from_scale, SCALE


def _make_signal(fs: float, t_total: float, seed: int, chatter_from: float) -> SignalData:
    rng = np.random.default_rng(seed)
    n = int(t_total * fs)
    t = np.arange(n) / fs
    x = rng.normal(0.0, 1.0, size=n)
    x[t >= chatter_from] += rng.normal(0.0, 4.0, size=(t >= chatter_from).sum())
    return SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=fs)


def _fig(label: str) -> plt.Figure:
    # Prefix match, not exact -- every figure's num= now carries a
    # " — {case_id}" suffix (see maxent_sprt_plots.py's per-call case
    # tagging), so an exact plt.figure(label) lookup would silently create
    # a new empty figure instead of finding the existing one.
    matches = [plt.figure(n) for n in plt.get_fignums() if plt.figure(n).get_label().startswith(label)]
    assert matches, \
        f"no open figure with label prefix {label!r} (got {[plt.figure(n).get_label() for n in plt.get_fignums()]})"
    assert len(matches) == 1, f"ambiguous label prefix {label!r}: {[m.get_label() for m in matches]}"
    return matches[0]


def _assert_size(label: str, expected_base: tuple[float, float]) -> None:
    expected = figsize_from_scale(expected_base, SCALE)
    got = tuple(_fig(label).get_size_inches())
    assert np.allclose(got, expected, atol=1e-6), f"{label}: expected figsize {expected}, got {got}"


def main() -> None:
    fs = 5000.0
    signal = _make_signal(fs=fs, t_total=4.0, seed=0, chatter_from=2.0)

    cfg = {
        "id": "MaxEnt_SPRT",
        "func": "Default",
        "param_mode": "native",
        "params": {
            "rpm": 6000.0, "N_seg": 200, "t_stable_total": 2.0,
            "alpha": 0.01, "beta": 0.01, "reset_on_H0": True,
            "cut_start_time": 0.0, "cut_end_time": 4.0,
            "segmentation": "raw", "N_samples_per_seg": 200,
        },
    }
    result = run_maxent_sprt(signal, cfg)
    plots_maxent_sprt(signal=signal, result=result, show_signal=True, show=True, t_gt=2.0)

    # Single-panel figures -> FIGSIZE_SIMPLE
    _assert_size("F1 — Training Entropy: Stable Segments", FIGSIZE_SIMPLE)
    _assert_size("F7 — Training PDF: Stable vs Chatter", FIGSIZE_SIMPLE)
    _assert_size("S1 — MaxEnt-SPRT Detection Statistic Sk", FIGSIZE_SIMPLE)

    # Two side-by-side panels -> figsize_grid(ncols=2)
    _assert_size("D3 — Error Regions alpha, beta and SPRT Boundaries", figsize_grid(2, 1))
    _assert_size("D6 — Log-Likelihoods and Lambda(H) Static", figsize_grid(2, 1))

    # F0a/F0b/F6a/F6b: per-case grid, falls back to a single FIGSIZE_SIMPLE
    # panel here since this test's signal has exactly 1 piece per side
    # (no reference_signal / training_intervals pieces) -- nothing to grid.
    _assert_size("F0a — Training Signal Verification (Stable, per case)", FIGSIZE_SIMPLE)
    _assert_size("F0b — Training Signal Verification (Chatter, per case)", FIGSIZE_SIMPLE)
    _assert_size("F6a — Signal + OPR Sampling (Stable, per case)", FIGSIZE_SIMPLE)
    _assert_size("F6b — Signal + OPR Sampling (Chatter, per case)", FIGSIZE_SIMPLE)

    plt.close("all")
    print("test_article_plot_style: OK")


if __name__ == "__main__":
    main()
