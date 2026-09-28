"""Assert-based self-check for C8's piece-boundary NaN-gap split.

C8 (CV Training Curve) plots the pooled external-reference training
population. When that pool comes from several unrelated pieces concatenated
end to end (see runner.py's per-piece reference_signal fix), a plain line
plot draws a straight segment connecting the end of one piece to the start
of the next -- representing nothing real. split_with_nan_gaps() breaks the
line at each piece boundary instead.

Pure numpy logic, no matplotlib figures needed -- run directly:
    python test_cv_training_curve_gaps.py
"""
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from rms_cv.viz.rms_cv_plots import split_with_nan_gaps


def test_inserts_one_nan_per_boundary():
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float) * 10
    x_plot, y_plot, bounds = split_with_nan_gaps(x, y, [3, 4, 3])

    # 10 original points + 2 gaps (boundaries between 3 pieces).
    assert x_plot.size == 12
    assert y_plot.size == 12
    nan_idx = np.flatnonzero(np.isnan(x_plot))
    assert np.array_equal(nan_idx, [3, 8]), nan_idx
    assert np.array_equal(np.flatnonzero(np.isnan(y_plot)), nan_idx)

    # non-NaN values are untouched and in original order.
    assert np.array_equal(x_plot[~np.isnan(x_plot)], x)
    assert np.array_equal(y_plot[~np.isnan(y_plot)], y)

    # piece_bounds index into the ORIGINAL (ungapped) arrays.
    assert np.array_equal(bounds, [[0, 3], [3, 7], [7, 10]])
    print("test_inserts_one_nan_per_boundary: OK")


def test_zero_size_pieces_are_dropped_not_gapped():
    # A skipped (too-short) piece contributes size 0 -- must not create an
    # extra empty gap or shift the boundary of real pieces.
    x = np.arange(6, dtype=float)
    y = np.arange(6, dtype=float)
    x_plot, y_plot, bounds = split_with_nan_gaps(x, y, [3, 0, 3])

    assert x_plot.size == 7  # 6 points + 1 gap (only 2 non-empty pieces)
    assert np.array_equal(bounds, [[0, 3], [3, 6]])
    print("test_zero_size_pieces_are_dropped_not_gapped: OK")


def test_mismatched_sizes_fall_back_unchanged():
    # piece_sizes that don't actually partition the data (bug elsewhere, or
    # an internal-mode caller that shouldn't be gapping at all) must not
    # silently corrupt the plot -- fall back to the plain curve.
    x = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)

    for bad_sizes in ([3, 3], [], None):
        x_plot, y_plot, bounds = split_with_nan_gaps(x, y, bad_sizes)
        assert np.array_equal(x_plot, x)
        assert np.array_equal(y_plot, y)
        assert bounds.shape == (0, 2)
    print("test_mismatched_sizes_fall_back_unchanged: OK")


def test_single_piece_no_gap():
    x = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)
    x_plot, y_plot, bounds = split_with_nan_gaps(x, y, [5])
    assert np.array_equal(x_plot, x)
    assert np.array_equal(y_plot, y)
    assert np.array_equal(bounds, [[0, 5]])
    print("test_single_piece_no_gap: OK")


if __name__ == "__main__":
    test_inserts_one_nan_per_boundary()
    test_zero_size_pieces_are_dropped_not_gapped()
    test_mismatched_sizes_fall_back_unchanged()
    test_single_piece_no_gap()
