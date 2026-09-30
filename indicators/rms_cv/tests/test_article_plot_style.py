"""Headless smoke test for the article-plot-style migration of rms_cv_plots.py.

Checks: plots_rms_cv() runs without exception on synthetic data (both the
internal and the external_reference training-source branches, so C3/C8 get
exercised too), and every Figure it creates is sized from FIGSIZE_SIMPLE/
FIGSIZE_WIDE via plot_style.SCALE -- not the old fig_size(scale=3.0) sizes.

Agg backend, no window, no plt.show() block. Run directly:
    python test_article_plot_style.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from rms_cv.utils.types import IndicatorResult, SignalData
from rms_cv.viz.plot_style import FIGSIZE_SIMPLE, FIGSIZE_WIDE, SCALE, figsize_grid, figsize_from_scale
from rms_cv.viz.rms_cv_plots import plots_rms_cv

_EXPECTED_SIZES = {
    tuple(round(v, 6) for v in figsize_from_scale(FIGSIZE_SIMPLE, SCALE)),
    tuple(round(v, 6) for v in figsize_from_scale(figsize_grid(1, 2), SCALE)),
}


def _make_signal(n=2000, fs=2000.0):
    t = np.arange(n) / fs
    x = 0.01 * np.sin(2 * np.pi * 50 * t)
    return SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=fs)


def _make_result(signal, training_source="internal"):
    n_rms = 40
    t_rms = np.linspace(0, signal.t_analysis[-1], n_rms)
    rms_values = np.abs(np.sin(t_rms)) * 0.01 + 0.002
    cv_time = t_rms
    cv_values = np.abs(np.cos(t_rms)) * 0.05 + 0.01
    mu = np.linspace(0.01, 0.012, n_rms)
    sigma = np.linspace(0.001, 0.0012, n_rms)

    meta = {
        "t_rms": t_rms, "rms_values": rms_values,
        "cv_time": cv_time, "cv_values": cv_values,
        "cv_threshold_used": 0.03, "mu_stable": 0.011, "sigma_stable": 0.001,
        "rms_threshold": 0.009, "window_sec": 0.02,
        "idx_rms_windows": np.stack([np.arange(n_rms) * 10, np.arange(n_rms) * 10 + 10], axis=1),
        "n_max": 5,
        "training_source": training_source,
        "cv_training_values": cv_values.copy(),
        "cv_training_time": cv_time.copy(),
        "mu": mu, "sigma": sigma,
    }
    if training_source == "external_reference":
        # 3 pooled pieces -> exercises C8's NaN-gap split too.
        meta["reference_frames_per_piece"] = [15, 12, 13]

    return IndicatorResult(name="RMS_CV", t=cv_time, I_t=cv_values,
                            t_d=np.array([1.0]), meta=meta)


def _run_and_collect_figsizes(signal, result):
    plt.close("all")
    plots_rms_cv(signal, result, t_gt=1.0, show=True)
    sizes = [tuple(round(v, 6) for v in f.get_size_inches()) for f in map(plt.figure, plt.get_fignums())]
    plt.close("all")
    return sizes


def test_internal_training_source_renders_without_exception():
    signal = _make_signal()
    result = _make_result(signal, "internal")
    sizes = _run_and_collect_figsizes(signal, result)
    assert len(sizes) > 0, "no figures were created"
    bad = [s for s in sizes if s not in _EXPECTED_SIZES]
    assert not bad, f"figures with unexpected (non article-plot-style) size: {bad}"
    print(f"test_internal_training_source_renders_without_exception: OK ({len(sizes)} figures)")


def test_external_reference_training_source_renders_without_exception():
    signal = _make_signal()
    result = _make_result(signal, "external_reference")
    sizes = _run_and_collect_figsizes(signal, result)
    assert len(sizes) > 0, "no figures were created"
    bad = [s for s in sizes if s not in _EXPECTED_SIZES]
    assert not bad, f"figures with unexpected (non article-plot-style) size: {bad}"
    print(f"test_external_reference_training_source_renders_without_exception: OK ({len(sizes)} figures)")


if __name__ == "__main__":
    test_internal_training_source_renders_without_exception()
    test_external_reference_training_source_renders_without_exception()
