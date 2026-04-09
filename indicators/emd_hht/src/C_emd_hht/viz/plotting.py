from __future__ import annotations
import numpy as np
from typing import Any, Optional, Tuple, Union, Sequence, Mapping
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  # type: ignore
from matplotlib.ticker import AutoMinorLocator, MultipleLocator  # noqa: F401  # type: ignore
from scipy.signal import savgol_filter

import numpy.polynomial.chebyshev as cheb
from scipy.interpolate import PchipInterpolator

from scipy.optimize import curve_fit

from ..lib.misc import _time_axis, _smoothstep
from ..utils.signal_chatter import amplitude_spectrum


def plot_imfs_separados(imfs: np.ndarray, fs: Optional[float] = None, max_to_plot: Optional[int] = None, show: bool = True) -> list[Any]:
    """Creates one independent figure for each IMF (no subplots).

    Args:
        imfs (np.ndarray): (K, N) array of IMFs.
        fs (Optional[float]): Sampling frequency or None.
        max_to_plot (Optional[int]): Maximum number of IMFs to plot. If None, plots all.
        show (bool): If True, calls `plt.show()` at the end.

    Returns:
        list: List of created Figure objects.
    """
    if plt is None:
        raise ImportError("Matplotlib is required. Install it with: pip install matplotlib")
    if imfs.ndim != 2:
        raise ValueError("imfs must have shape (K, N)")
    K, N = imfs.shape
    if max_to_plot is None or max_to_plot > K:
        max_to_plot = K
    t, xlabel = _time_axis(N, fs)
    figs: list[Any] = []
    for k in range(max_to_plot):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(t, imfs[k])
        ax.set_title(f"IMF {k+1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Amplitude")
        figs.append(fig)
    if show:
        plt.show()
    return figs


def plot_imf_seleccionado(
    selected_imf: np.ndarray,
    fs: Optional[float] = None,
    f_max: Optional[float] = None,
    plot_spectrum: bool = False,
    A: Optional[np.ndarray] = None,
    f_inst: Optional[np.ndarray] = None,
    show: bool = True
) -> list[Any]:
    """Plots the selected IMF and optionally A(t) and f_inst(t) in separate figures.

    Args:
        selected_imf (np.ndarray): 1D IMF to plot.
        fs (Optional[float]): Sampling frequency (Hz).
        plot_spectrum (bool): If True, plots the IMF spectrum in an additional figure.
        A (Optional[np.ndarray]): Instantaneous amplitude to plot (optional).
        f_inst (Optional[np.ndarray]): Instantaneous frequency to plot (optional).
        f_max (Optional[float]): Maximum frequency to display in the spectrum plot (optional).
        show (bool): If True, calls `plt.show()` at the end.

    Returns:
        list: List of created figures.
    """
    if plt is None:
        raise ImportError("Matplotlib is required. Install it with: pip install matplotlib")
    N: int = len(selected_imf)
    t, xlabel = _time_axis(N, fs)

    figs: list[Any] = []

    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.plot(t, selected_imf)
    ax1.set_title("Selected IMF - time domain")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Amplitude")
    figs.append(fig1)

    if A is not None and len(A) == N:
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.plot(t, A)
        ax2.set_title("Instantaneous amplitude A(t)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("A")
        figs.append(fig2)

    if f_inst is not None and len(f_inst) == N:
        fig3 = plt.figure()
        ax3 = fig3.gca()
        ax3.plot(t, f_inst)
        ax3.set_title("Instantaneous frequency f_inst(t)")
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel("Hz")
        figs.append(fig3)

    if plot_spectrum:
        fig2, axes_2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        axes_2 = np.array(axes_2).reshape(-1)
        ax = axes_2[0]
        f, Pxx = amplitude_spectrum(selected_imf, fs=fs, normalize_to=0.1)  # type: ignore
        ax.plot(f, Pxx)
        ax.set_title("Spectrum of selected IMF")
        ax.set_xlabel("Frequency (Hz)" if fs is not None else "Frequency (samples)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, f_max if fs is not None else None)
        # ax.xaxis.set_major_locator(MultipleLocator(100))  # type: ignore
        fig2.tight_layout()

    if show:
        plt.show()
    return figs


def plot_imfs(
    imfs: np.ndarray,
    fs: Optional[float] = None,
    f_max: Optional[float] = None,
    plot_spectrum: bool = False,
    max_to_plot: Optional[int] = None,
    ncols: int = 1,
    show: bool = True
) -> Tuple[Any, np.ndarray]:
    """Plots all IMFs in a single figure using subplots.

    Args:
        imfs (np.ndarray): (K, N) array with IMFs.
        fs (Optional[float]): Sampling frequency. If None, x-axis is in samples.
        plot_spectrum (bool): If True, creates a second figure with IMF spectra.
        max_to_plot (Optional[int]): Upper limit of IMFs to plot. If None, plots all.
        ncols (int): Number of columns in the subplot grid.
        show (bool): If True, calls plt.show() at the end.

    Returns:
        Tuple[object, np.ndarray]: Figure and array of axes (subplots).
    """
    if plt is None:
        raise ImportError("Matplotlib is required. Install it with: pip install matplotlib")
    if imfs.ndim != 2:
        raise ValueError("imfs must have shape (K, N)")
    K, N = imfs.shape
    if max_to_plot is None or max_to_plot > K:
        max_to_plot = K
    if ncols < 1:
        ncols = 1
    nrows: int = int(np.ceil(max_to_plot / ncols))

    t, xlabel = _time_axis(N, fs)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(10, max(3, 2*nrows)))
    axes = np.array(axes).reshape(-1)

    for i in range(max_to_plot):
        ax = axes[i]
        ax.plot(t, imfs[i])
        ax.set_title(f"IMF {i+1}")
        ax.set_ylabel("Amplitude")
    for j in range(max_to_plot, len(axes)):
        axes[j].set_visible(False)

    axes[min(max_to_plot-1, len(axes)-1)].set_xlabel(xlabel)

    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    fig.tight_layout()

    if plot_spectrum:
        fig2, axes_2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(10, max(3, 2*nrows)))
        axes_2 = np.array(axes_2).reshape(-1)
        for i in range(max_to_plot):
            ax = axes_2[i]
            f, Pxx = amplitude_spectrum(imfs[i], fs=fs, normalize_to=0.1)  # type: ignore
            ax.plot(f, Pxx)
            ax.set_title(f"Spectrum IMF {i+1}")
            ax.set_ylabel("Amplitude")
            ax.set_xlim(0, f_max if fs is not None else None)
            ax.tick_params(axis='x', )
            ax.tick_params(axis='y', )
            # ax.xaxis.set_major_locator(MultipleLocator(100))  # type: ignore

        for j in range(max_to_plot, len(axes_2)):
            axes_2[j].set_visible(False)

        axes_2[min(max_to_plot-1, len(axes_2)-1)].set_xlabel("Frequency (Hz)" if fs is not None else "Frequency (samples)")
        fig2.tight_layout()
        for ax in axes_2:
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    if show:
        plt.show()
    return fig, axes


def plot_tendencia(
    t: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
    tipo: str = "lineal",
    grado: int = 3,
    ventana: int = 7,
    ls: str = '--',
    lw: float = 3,
    alpha: float = 1.0,
    color: str = 'k',
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    trend_kwargs: Optional[Mapping[str, Any]] = None,
    devolver: bool = False,
    n_bins: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Plots points and a trend line according to the selected method.

    Args:
        t (array-like): x-axis (time or index).
        y (array-like): Series to smooth/fit.
        tipo (str): 'lineal' | 'polinomial' | 'suavizada' | 'pchip' | 'sigmoide' | 'step_suave'.
        grado (int): Polynomial degree (if applicable).
        ventana (int): Window length for Savitzky-Golay (if 'suavizada').
        ls (str): Line style ('--' by default).
        lw (float): Line width.
        alpha (float): Line transparency.
        color (str): Line color.
        scatter_kwargs (dict): Extra kwargs for `plt.scatter`.
        trend_kwargs (dict): Extra kwargs for `plt.plot` of the trend.
        devolver (bool): If True, returns (t, trend).
        n_bins (int): Number of bins for aggregation in PCHIP.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (t, trend) if `devolver=True`.

    Notes:
        - The 'step_suave' branch fits a smoothed step (logistic/smoothstep) robustly.
        - Original behavior is preserved; only comments and docstrings are added/translated.
    """
    if plt is None:
        raise ImportError("Matplotlib is required. Install it with: pip install matplotlib")

    t = np.asarray(t)
    y = np.asarray(y)

    if tipo == "lineal":
        coef = np.polyfit(t, y, 1)
        trend = np.polyval(coef, t)
    elif tipo == "polinomial":
        t_scaled = np.linspace(-1, 1, len(t))
        coef = cheb.chebfit(t_scaled, y, grado)
        trend = cheb.chebval(t_scaled, coef)
    elif tipo == "suavizada":
        if ventana % 2 == 0:
            ventana += 1
        ventana = max(3, min(ventana, len(y) - (len(y)+1)%2))
        if ventana >= 3:
            trend = savgol_filter(y, window_length=ventana, polyorder=min(2, grado))
        else:
            trend = y.copy()
    elif tipo == "pchip":
        edges = np.linspace(t.min(), t.max(), n_bins+1)
        idx = np.digitize(t, edges) - 1
        tc: list[float] = []
        yc: list[float] = []
        for i in range(n_bins):
            m = idx == i
            if np.any(m):
                tc.append(float(t[m].mean()))
                yc.append(float(np.median(y[m])))
        p = PchipInterpolator(np.asarray(tc), np.asarray(yc))
        trend = p(t)
    elif tipo == "sigmoide":
        def logistic(x, L, k, x0, b):  # b + L/(1+e^{-k(x-x0)})
            return b + L/(1.0 + np.exp(-k*(x - x0)))
        L0 = np.percentile(y, 95) - np.percentile(y, 5)
        k0 = 2.0/(np.ptp(t)+1e-12); x0 = np.median(t); b0 = np.percentile(y, 5)
        popt, _ = curve_fit(logistic, t, y, p0=[L0, k0, x0, b0], maxfev=10000)
        trend = logistic(t, *popt)

    elif tipo == "step_suave":
        # Model: y = y_lo + (y_hi - y_lo) * smoothstep((t - t1)/w)
        def model(x, y_lo, y_hi, t1, w):
            return y_lo + (y_hi - y_lo) * _smoothstep((x - t1)/w)

        # Robust initializations
        y_lo0 = np.median(y[:max(5, len(y)//10)])
        y_hi0 = np.median(y[-max(5, len(y))//10:])
        t1_0  = t[np.argmax(np.gradient(y))]  # approximate onset
        w0    = max((t.max()-t.min())/10, 1e-3)

        bounds = (
            [y.min()-abs(np.ptp(y)), y.min(), t.min(), 1e-6],   # lower
            [y.max(),               y.max()+abs(np.ptp(y)), t.max(), np.ptp(t)]  # upper
        )
        try:
            popt, _ = curve_fit(
                model,
                t,
                y,
                p0=[y_lo0, y_hi0, t1_0, w0],
                bounds=bounds,
                maxfev=20000
            )
            trend = model(t, *popt)
        except Exception:
            # Fallback if it does not converge: PCHIP
            p = PchipInterpolator(t, y)
            trend = p(t)

    else:
        raise ValueError("tipo must be 'lineal', 'polinomial' or 'suavizada'")

    # ---- plotting ----
    if scatter_kwargs is None:
        scatter_kwargs = dict(s=20, marker='o', color='b')  # blue points
    if trend_kwargs is None:
        # Nicer dashes (segments 6, spaces 3) and rounded caps
        trend_kwargs = dict(dashes=(6, 3), dash_capstyle='round')

    plt.figure()
    plt.scatter(t, y, **scatter_kwargs)
    plt.plot(t, trend, ls=ls, lw=lw, alpha=alpha, color=color, **trend_kwargs)
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.title("Band energy counts per window")
    plt.grid(True)

    if devolver:
        return t, trend


def plot_HHS(
    t: Union[Sequence[float], np.ndarray],
    fgrid: Union[Sequence[float], np.ndarray],
    HHS: np.ndarray,
    cmap: Optional[ListedColormap] = None,
    fs: Optional[float] = None,
    fmax: Optional[float] = None,
    show: bool = False
) -> Any:
    """Plots the Hilbert-Huang Spectrum (HHS) as a heatmap.

    Args:
        t (array-like): Time axis.
        fgrid (array-like): Frequency axis.
        HHS (np.ndarray): 2D HHS matrix (binary or continuous).
        cmap (Optional[ListedColormap]): Colormap for the HHS plot.
        fs (Optional[float]): Sampling frequency (Hz).
        fmax (Optional[float]): Upper frequency limit to display (Hz).
        show (bool): If True, calls plt.show() at the end.

    Returns:
        object: Created Figure object.
    """
    if plt is None:
        raise ImportError("Matplotlib is required. Install it with: pip install matplotlib")

    HHS = np.where(HHS > 0, 1, 0)
    if cmap is None:
        cmap = ListedColormap(['blue', 'yellow'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.pcolormesh(t, fgrid, HHS, shading='nearest', cmap=cmap, vmin=None, vmax=None)
    ax.set_title("Hilbert-Huang Spectrum")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, fmax if fmax is not None else fgrid.max())
    plt.show()

    if show:
        plt.show()
    return fig, ax
