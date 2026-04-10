"""Legacy visualisation utilities for the RMS-CV indicator.

.. deprecated::
    These wrappers provide quick, single-purpose figures useful during
    exploratory development.  New code should use
    :func:`~rms_cv.viz.rms_cv_plots.plots_rms_cv` instead, which produces
    publication-quality composite figures.
"""

from __future__ import annotations
from typing import Dict, Any, Sequence
import numpy as np
import matplotlib.pyplot as plt


def plot_signal(t: "np.ndarray", x: "np.ndarray", *, title: str = "Signal") -> None:
    """Plot a time-domain signal.

    .. deprecated::
        Use :func:`~rms_cv.viz.rms_cv_plots.plots_rms_cv` for
        publication-quality output.

    Args:
        t (np.ndarray): Time vector [s], shape ``(T,)``.
        x (np.ndarray): Signal amplitude, shape ``(T,)``.
        title (str, optional): Axes title.  Defaults to ``"Signal"``.

    Note:
        A new :class:`~matplotlib.figure.Figure` is created and left open.
        Call :func:`matplotlib.pyplot.show` or
        :func:`matplotlib.pyplot.savefig` explicitly.
    """
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("time (s)")
    plt.ylabel("amplitude")
    plt.title(title)
    plt.grid(True)

def plot_rms(times: "np.ndarray", rms: "np.ndarray", *, title: str = "RMS") -> None:
    """Plot a windowed RMS sequence.

    .. deprecated::
        Use :func:`~rms_cv.viz.rms_cv_plots.plots_rms_cv` for
        publication-quality output.

    Args:
        times (np.ndarray): Centre-of-frame timestamps [s], shape ``(F,)``.
        rms (np.ndarray): Corresponding RMS values, shape ``(F,)``.
        title (str, optional): Axes title.  Defaults to ``"RMS"``.
    """
    # Traza secuencia RMS
    plt.figure()
    plt.plot(times, rms, marker="o")
    plt.xlabel("time (s)")
    plt.ylabel("rms")
    plt.title(title)
    plt.grid(True)

def plot_cv(time_seq: Sequence[float], cv_seq: Sequence[float], cv_threshold: float, *, title: str = "CV") -> None:
    """Plot the online Coefficient of Variation (CV) sequence with its threshold.

    .. deprecated::
        Use :func:`~rms_cv.viz.rms_cv_plots.plots_rms_cv` for
        publication-quality output.

    Args:
        time_seq (Sequence[float]): Frame timestamps [s], length *F*.
        cv_seq (Sequence[float]): CV values per frame, length *F*.
        cv_threshold (float): Alert threshold drawn as a horizontal dashed
            red line.
        title (str, optional): Axes title.  Defaults to ``"CV"``.
    """
    # Traza CV con su umbral de alerta
    plt.figure()
    plt.scatter(time_seq, cv_seq)
    plt.axhline(y=cv_threshold, color="r", linestyle="--", label="CV threshold")
    plt.xlabel("time (s)")
    plt.ylabel("cv")
    plt.title(title)
    plt.grid(True)
