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

from .plot_style import FIGSIZE_SIMPLE, SCALE, figsize_from_scale, apply_sci_yaxis
from .plot_style import apply as _apply_style

_apply_style()   # article-plot-style, at the default scale (was an import-time side effect of plot_style.py)


def plot_signal(t: "np.ndarray", x: "np.ndarray", *, title: str = "Tool Velocity Signal") -> None:
    """Plot a time-domain signal.

    .. deprecated::
        Use :func:`~rms_cv.viz.rms_cv_plots.plots_rms_cv` for
        publication-quality output.

    Args:
        t (np.ndarray): Time vector [s], shape ``(T,)``.
        x (np.ndarray): Signal amplitude, shape ``(T,)``.
        title (str, optional): Axes title.  Defaults to
            ``"Tool Velocity Signal"``.

    Note:
        A new :class:`~matplotlib.figure.Figure` is created and left open.
        Call :func:`matplotlib.pyplot.show` or
        :func:`matplotlib.pyplot.savefig` explicitly.
    """
    fig = plt.figure(figsize=figsize_from_scale(FIGSIZE_SIMPLE, SCALE), constrained_layout=True)
    ax = fig.gca()
    ax.plot(t, x)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    apply_sci_yaxis(ax)
    ax.set_title(title)
    ax.grid(True)

def plot_rms(times: "np.ndarray", rms: "np.ndarray", *, title: str = "RMS Envelope of the Signal") -> None:
    """Plot a windowed RMS sequence.

    .. deprecated::
        Use :func:`~rms_cv.viz.rms_cv_plots.plots_rms_cv` for
        publication-quality output.

    Args:
        times (np.ndarray): Centre-of-frame timestamps [s], shape ``(F,)``.
        rms (np.ndarray): Corresponding RMS values, shape ``(F,)``.
        title (str, optional): Axes title.  Defaults to
            ``"RMS Envelope of the Signal"``.
    """
    # Traza secuencia RMS
    fig = plt.figure(figsize=figsize_from_scale(FIGSIZE_SIMPLE, SCALE), constrained_layout=True)
    ax = fig.gca()
    ax.plot(times, rms, marker="o")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rms")
    apply_sci_yaxis(ax)
    ax.set_title(title)
    ax.grid(True)

def plot_cv(time_seq: Sequence[float], cv_seq: Sequence[float], cv_threshold: float, *, title: str = "Coefficient of Variation (CV) Sequence") -> None:
    """Plot the online Coefficient of Variation (CV) sequence with its threshold.

    .. deprecated::
        Use :func:`~rms_cv.viz.rms_cv_plots.plots_rms_cv` for
        publication-quality output.

    Args:
        time_seq (Sequence[float]): Frame timestamps [s], length *F*.
        cv_seq (Sequence[float]): CV values per frame, length *F*.
        cv_threshold (float): Alert threshold drawn as a horizontal dashed
            red line.
        title (str, optional): Axes title.  Defaults to
            ``"Coefficient of Variation (CV) Sequence"``.
    """
    # Traza CV con su umbral de alerta
    fig = plt.figure(figsize=figsize_from_scale(FIGSIZE_SIMPLE, SCALE), constrained_layout=True)
    ax = fig.gca()
    ax.scatter(time_seq, cv_seq)
    ax.axhline(y=cv_threshold, color="r", linestyle="--", label="CV threshold")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cv")
    apply_sci_yaxis(ax)
    ax.set_title(title)
    ax.grid(True)
