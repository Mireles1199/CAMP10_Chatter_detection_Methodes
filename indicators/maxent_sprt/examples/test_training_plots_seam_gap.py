"""
test_training_plots_seam_gap.py
================================
Self-check for the training-figure seam fix (F1/F2/F3) and the per-case grid
(F0a/F0b/F6a/F6b): when the stable/chatter training data comes from several
physically-disjoint pieces (reference_signal pieces, or multiple
training_intervals of the same label), F1/F2/F3 (single concatenated axis)
must break their line at each piece boundary instead of drawing a straight
line across the seam, and F0a/F0b/F6a/F6b (per-case grid) must instead give
each piece its own clean subplot -- one per piece, never concatenated, so
there's nothing to break.

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
    # Prefix match, not exact -- every figure's num= now carries a
    # " — {case_id}" suffix (see maxent_sprt_plots.py's per-call case
    # tagging), so an exact plt.figure(label) lookup would silently create
    # a new empty figure instead of finding the existing one.
    matches = [plt.figure(n) for n in plt.get_fignums() if plt.figure(n).get_label().startswith(label)]
    assert matches, \
        f"no open figure with label prefix {label!r} (got {[plt.figure(n).get_label() for n in plt.get_fignums()]})"
    assert len(matches) == 1, f"ambiguous label prefix {label!r}: {[m.get_label() for m in matches]}"
    return matches[0]


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

    # ── F0a: stable per-case grid -- one clean (NaN-free) subplot per piece ──
    fig_F0a = _get_fig("F0a — Training Signal Verification (Stable, per case)")
    assert len(fig_F0a.axes) == n_stable_pieces, (
        f"F0a: expected {n_stable_pieces} subplots (one per piece), got {len(fig_F0a.axes)}"
    )
    for i, ax in enumerate(fig_F0a.axes):
        y = ax.lines[0].get_ydata()
        assert not np.any(np.isnan(y)), f"F0a piece {i}: subplot line should never contain NaN"

    # ── F6a: same stable pieces, signal+OPR per-case grid ──
    fig_F6a = _get_fig("F6a — Signal + OPR Sampling (Stable, per case)")
    assert len(fig_F6a.axes) == n_stable_pieces, (
        f"F6a: expected {n_stable_pieces} subplots (one per piece), got {len(fig_F6a.axes)}"
    )
    for i, ax in enumerate(fig_F6a.axes):
        y = ax.lines[0].get_ydata()
        assert not np.any(np.isnan(y)), f"F6a piece {i}: subplot line should never contain NaN"

    # ── F1: stable training entropy, per-case grid -- one clean subplot per piece ──
    fig_F1 = _get_fig("F1 — Training Entropy: Stable Segments")
    assert len(fig_F1.axes) == n_stable_pieces, (
        f"F1: expected {n_stable_pieces} subplots (one per piece), got {len(fig_F1.axes)}"
    )
    for i, ax in enumerate(fig_F1.axes):
        y = ax.lines[0].get_ydata()
        assert not np.any(np.isnan(y)), f"F1 piece {i}: subplot line should never contain NaN"

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

    # ── F0b/F6b: chatter side has 1 piece -> grid falls back to a single panel ──
    fig_F0b = _get_fig("F0b — Training Signal Verification (Chatter, per case)")
    assert len(fig_F0b.axes) == 1, f"F0b (single chatter piece) should fall back to 1 panel, got {len(fig_F0b.axes)}"
    fig_F6b = _get_fig("F6b — Signal + OPR Sampling (Chatter, per case)")
    assert len(fig_F6b.axes) == 1, f"F6b (single chatter piece) should fall back to 1 panel, got {len(fig_F6b.axes)}"

    plt.close("all")
    print("test_training_plots_seam_gap: OK")


if __name__ == "__main__":
    main()
