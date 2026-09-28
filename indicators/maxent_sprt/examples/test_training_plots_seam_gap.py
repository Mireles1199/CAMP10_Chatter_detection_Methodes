"""
test_training_plots_seam_gap.py
================================
Self-check for the training-figure seam fix (F0/F1/F2/F3/F6): when the
stable/chatter training data comes from several physically-disjoint pieces
(reference_signal pieces, or multiple training_intervals of the same label),
these figures must break their line at each piece boundary instead of
drawing a straight line across the seam.

Uses a non-interactive matplotlib backend (Agg) so it never opens a window
and never blocks -- safe to run headless. Synthetic signal, no real dataset
needed. Run directly: asserts only.
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


def _make_signal(fs: float, t_total: float, seed: int, chatter_from: float | None = None) -> SignalData:
    rng = np.random.default_rng(seed)
    n = int(t_total * fs)
    t = np.arange(n) / fs
    x = rng.normal(0.0, 1.0, size=n)
    if chatter_from is not None:
        x[t >= chatter_from] += rng.normal(0.0, 4.0, size=(t >= chatter_from).sum())
    return SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=fs)


def _piece(sigma: float, n_samples: int, fs: float, seed: int) -> SignalData:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    x = rng.normal(0.0, sigma, size=n_samples)
    return SignalData(t_analysis=t, signal_analysis=x, path=f"piece_{seed}", fs=fs,
                       meta={"signal_id": f"piece_{seed}"})


def _get_fig(label: str) -> plt.Figure:
    assert label in [f.get_label() for f in map(plt.figure, plt.get_fignums())], \
        f"no open figure with label {label!r} (got {[plt.figure(n).get_label() for n in plt.get_fignums()]})"
    return plt.figure(label)


def main() -> None:
    fs = 5000.0
    n_stable_pieces = 3
    pieces = [_piece(sigma=1.0, n_samples=900 + 37 * i, fs=fs, seed=100 + i) for i in range(n_stable_pieces)]
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
        "reference_signal": pieces,  # 3 pieces -> chatter side keeps its single legacy-cut piece
    }
    result = run_maxent_sprt(signal, cfg)
    assert result.meta["reference_n_pieces"] == n_stable_pieces

    plots_maxent_sprt(signal=signal, result=result, show_signal=True, show=True)

    # ── F0: raw stable-signal line must have exactly n_pieces-1 NaN gaps ──
    fig_F0 = _get_fig("F0 — Training Signal Verification (P0/P1 source)")
    ax_stable_F0 = fig_F0.axes[0]
    y_F0 = ax_stable_F0.lines[0].get_ydata()
    assert np.sum(np.isnan(y_F0)) == n_stable_pieces - 1, (
        f"F0 stable line: expected {n_stable_pieces - 1} NaN gap(s), got {np.sum(np.isnan(y_F0))}"
    )

    # ── F6: same stable line, side-by-side signal+OPR figure ──
    fig_F6 = _get_fig("F6 — Signal + OPR Sampling (Training)")
    ax_stable_F6 = fig_F6.axes[0]
    y_F6 = ax_stable_F6.lines[0].get_ydata()
    assert np.sum(np.isnan(y_F6)) == n_stable_pieces - 1, (
        f"F6 stable line: expected {n_stable_pieces - 1} NaN gap(s), got {np.sum(np.isnan(y_F6))}"
    )

    # ── F1: stable training entropy vs time must also break at each piece ──
    fig_F1 = _get_fig("F1 — Training Entropy: Stable Segments")
    y_F1 = fig_F1.axes[0].lines[0].get_ydata()
    assert np.sum(np.isnan(y_F1)) == n_stable_pieces - 1, (
        f"F1 entropy line: expected {n_stable_pieces - 1} NaN gap(s), got {np.sum(np.isnan(y_F1))}"
    )

    # ── F3: combined stable+chatter entropy axis -- stable curve is the first line ──
    fig_F3 = _get_fig("F3 — Training Entropy: All Labels")
    y_F3_stable = fig_F3.axes[0].lines[0].get_ydata()
    assert np.sum(np.isnan(y_F3_stable)) == n_stable_pieces - 1, (
        f"F3 stable line: expected {n_stable_pieces - 1} NaN gap(s), got {np.sum(np.isnan(y_F3_stable))}"
    )

    # ── F4/F5/F7 histograms must stay NaN-free -- they read detector.H_free/H_chat
    # directly (never spliced), so the p0/p1 fit/histogram is untouched. ──
    detector = result.meta["detector"]
    assert not np.any(np.isnan(detector.H_free)), "detector.H_free must never contain NaN (fit/histogram source)"
    assert not np.any(np.isnan(detector.H_chat)), "detector.H_chat must never contain NaN (fit/histogram source)"

    # ── Chatter side: single legacy-cut piece -> zero gaps (regression: no piece, no NaN) ──
    fig_F2 = _get_fig("F2 — Training Entropy: Chatter Segments")
    y_F2 = fig_F2.axes[0].lines[0].get_ydata()
    assert np.sum(np.isnan(y_F2)) == 0, f"F2 (single chatter piece) should have 0 gaps, got {np.sum(np.isnan(y_F2))}"

    plt.close("all")
    print("test_training_plots_seam_gap: OK")


if __name__ == "__main__":
    main()
