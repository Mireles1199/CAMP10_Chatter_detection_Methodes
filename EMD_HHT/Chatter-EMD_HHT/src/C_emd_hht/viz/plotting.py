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


def plot_imfs_separados(imfs: np.ndarray , fs: Optional[float] = None, max_to_plot: Optional[int] = None, show: bool = True) -> list[Any]:
    """Crea una figura independiente por cada IMF (sin subplots).

    Args:
        imfs (np.ndarray): Matriz (K, N) de IMFs.
        fs (Optional[float]): Frecuencia de muestreo o None.
        max_to_plot (Optional[int]): Máximo de IMFs a graficar. Si None, grafica todos.
        show (bool): Si True, realiza `plt.show()` al final.

    Returns:
        list: Lista de objetos Figure creados.
    """
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
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
        ax.set_ylabel("Amplitud")
        figs.append(fig)
    if show:
        plt.show()
    return figs



def plot_imf_seleccionado(
    selected_imf: np.ndarray ,
    fs: Optional[float] = None,
    plot_spectrum: bool = False,
    A: Optional[np.ndarray ] = None,
    f_inst: Optional[np.ndarray ] = None,
    show: bool = True
) -> list[Any]:
    """Grafica el IMF seleccionado y opcionalmente A(t) y f_inst(t), en figuras separadas.

    Args:
        selected_imf (np.ndarray): IMF 1D a graficar.
        fs (Optional[float]): Frecuencia de muestreo (Hz).
        plot_spectrum (bool): Si True, grafica espectro del IMF en figura adicional.
        A (Optional[np.ndarray]): Amplitud instantánea para trazar (opcional).
        f_inst (Optional[np.ndarray]): Frecuencia instantánea para trazar (opcional).
        show (bool): Si True, hace `plt.show()` al final.

    Returns:
        list: Lista de figuras creadas.
    """
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    N: int = len(selected_imf)
    t, xlabel = _time_axis(N, fs)

    figs: list[Any] = []

    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.plot(t, selected_imf)
    ax1.set_title("IMF seleccionado - dominio temporal")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Amplitud")
    figs.append(fig1)

    if A is not None and len(A) == N:
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.plot(t, A)
        ax2.set_title("Amplitud instantánea A(t)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("A")
        figs.append(fig2)

    if f_inst is not None and len(f_inst) == N:
        fig3 = plt.figure()
        ax3 = fig3.gca()
        ax3.plot(t, f_inst)
        ax3.set_title("Frecuencia instantánea f_inst(t)")
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel("Hz")
        figs.append(fig3)

    if plot_spectrum:
        fig2, axes_2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        axes_2 = np.array(axes_2).reshape(-1)
        ax = axes_2[0]
        f, Pxx = amplitude_spectrum(selected_imf, fs=fs, normalize_to=0.1)  # type: ignore
        ax.plot(f, Pxx)
        ax.set_title("Espectro IMF seleccionado")
        ax.set_xlabel("Frecuencia (Hz)" if fs is not None else "Frecuencia (muestras)")
        ax.set_ylabel("Amplitud")
        ax.set_xlim(0, 1000 if fs is not None else None)
        ax.xaxis.set_major_locator(MultipleLocator(100))  # type: ignore
        fig2.tight_layout()

    if show:
        plt.show()
    return figs



def plot_imfs(
    imfs: np.ndarray ,
    fs: Optional[float] = None,
    plot_spectrum: bool = False,
    max_to_plot: Optional[int] = None,
    ncols: int = 1,
    show: bool = True
) -> Tuple[Any, np.ndarray]:
    """Dibuja *todos* los IMFs en **una sola figura** usando subplots.

    Args:
        imfs (np.ndarray): Matriz (K, N) con los IMFs.
        fs (Optional[float]): Frecuencia de muestreo. Si None, el eje x será en muestras.
        plot_spectrum (bool): Si True, crea una segunda figura con espectros de IMFs.
        max_to_plot (Optional[int]): Límite superior de IMFs a graficar. Si None, grafica todos.
        ncols (int): Número de columnas en el mosaico de subplots.
        show (bool): Si True, hace plt.show() al final.

    Returns:
        Tuple[object, np.ndarray]: Figura y arreglo de ejes (subplots).
    """
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
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
        ax.set_ylabel("Amplitud")
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
            ax.set_title(f"Espectro IMF {i+1}")
            ax.set_ylabel("Amplitud")
            ax.set_xlim(0, 1000 if fs is not None else None)
            ax.tick_params(axis='x', )
            ax.tick_params(axis='y', )
            ax.xaxis.set_major_locator(MultipleLocator(100))  # type: ignore

        for j in range(max_to_plot, len(axes_2)):
            axes_2[j].set_visible(False)

        axes_2[min(max_to_plot-1, len(axes_2)-1)].set_xlabel("Frecuencia (Hz)" if fs is not None else "Frecuencia (muestras)")
        fig2.tight_layout()
        for ax in axes_2:
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    if show:
        plt.show()
    return fig, axes



def plot_tendencia(
    t: Union[Sequence[float], np.ndarray ],
    y: Union[Sequence[float], np.ndarray ],
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
) -> Optional[Tuple[np.ndarray , np.ndarray ]]:
    """Dibuja puntos y línea de tendencia según método elegido.

    Args:
        t (array-like): Eje x (tiempo o índice).
        y (array-like): Serie a suavizar/ajustar.
        tipo (str): 'lineal' | 'polinomial' | 'suavizada' | 'pchip' | 'sigmoide' | 'step_suave'.
        grado (int): Grado polinómico (si aplica).
        ventana (int): Ventana para Savitzky-Golay (si 'suavizada').
        ls (str): Estilo de línea ('--' por defecto).
        lw (float): Grosor de línea.
        alpha (float): Transparencia de la línea.
        color (str): Color de línea.
        scatter_kwargs (dict): kwargs extra para `plt.scatter`.
        trend_kwargs (dict): kwargs extra para `plt.plot` de la tendencia.
        devolver (bool): Si True, devuelve (t, trend).
        n_bins (int): Número de bins para agregación en PCHIP.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (t, trend) si `devolver=True`.

    Notas:
        - La rama 'step_suave' ajusta un escalón suavizado (logística/smoothstep) robusto.
        - Se preserva el comportamiento original; sólo se añaden comentarios y docstrings.
    """
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")

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
        L0 = np.percentile(y,95) - np.percentile(y,5)
        k0 = 2.0/(np.ptp(t)+1e-12); x0 = np.median(t); b0 = np.percentile(y,5)
        popt,_ = curve_fit(logistic, t, y, p0=[L0,k0,x0,b0], maxfev=10000)
        trend = logistic(t, *popt)

    elif tipo == "step_suave":
        # Modelo: y = y_lo + (y_hi - y_lo) * smoothstep( (t - t1)/w )
        def model(x, y_lo, y_hi, t1, w):
            return y_lo + (y_hi - y_lo) * _smoothstep((x - t1)/w)

        # Inicializaciones robustas
        y_lo0 = np.median(y[:max(5, len(y)//10)])
        y_hi0 = np.median(y[-max(5, len(y))//10:])
        t1_0  = t[np.argmax(np.gradient(y))]  # aprox del arranque
        w0    = max( (t.max()-t.min())/10, 1e-3 )

        bounds = (
            [y.min()-abs(np.ptp(y)), y.min(), t.min(), 1e-6],   # bajos
            [y.max(),             y.max()+abs(np.ptp(y)), t.max(), np.ptp(t)]  # altos
        )
        try:
            popt,_ = curve_fit(model, t, y, p0=[y_lo0, y_hi0, t1_0, w0],
                               bounds=bounds, maxfev=20000)
            trend = model(t, *popt)
        except Exception:
            # fallback por si no converge: PCHIP
            p = PchipInterpolator(t, y)
            trend = p(t)


    else:
        raise ValueError("tipo must be 'lineal', 'polinomial' or 'suavizada'")

    # ---- trazar
    if scatter_kwargs is None:
        scatter_kwargs = dict(s=20, marker='o', color='b')  # puntos azules
    if trend_kwargs is None:
        # Dashes más 'bonitos' (segmentos 6, espacios 3) y puntas redondeadas
        trend_kwargs = dict(dashes=(6, 3), dash_capstyle='round')

    plt.figure()
    plt.scatter(t, y, **scatter_kwargs)
    plt.plot(t, trend, ls=ls, lw=lw, alpha=alpha, color=color, **trend_kwargs)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Conteo")
    plt.title("Conteo de energía en banda por ventana")
    plt.grid(True)


    if devolver:
        return t, trend
    
    

def plot_HHS(
    t: Union[Sequence[float], np.ndarray ],
    fgrid: Union[Sequence[float], np.ndarray ],
    HHS: np.ndarray ,
    cmap: Optional[ListedColormap] = None,
    fs: Optional[float] = None,
    fmax: Optional[float] = 1000.0,
    show: bool = False
) -> Any:
    """Grafica el Hilbert-Huang Spectrum (HHS) como un mapa de calor.

    Args:
        t (array-like): Eje temporal.
        fgrid (array-like): Eje de frecuencia.
        HHS (np.ndarray): Matriz 2D del HHS (binario o continuo).
        fs (Optional[float]): Frecuencia de muestreo (Hz).
        fmax (Optional[float]): Límite superior de frecuencia a mostrar (Hz).
        show (bool): Si True, hace plt.show() al final.

    Returns:
        object: Objeto Figure creado.
    """
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")

    
    
    HHS =np.where(HHS > 0, 1, 0)
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
