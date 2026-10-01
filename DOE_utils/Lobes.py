from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
from typing import Callable, Iterable, List, Protocol, Sequence, Tuple, Optional
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.collections as mpathcoll
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import os


# ─────────────────────────────────────────────────────────────
# Decoradores utilitarios
# ─────────────────────────────────────────────────────────────
def ensure_1d_numpy(fn: Callable) -> Callable:
    """Asegura que los arrays sean 1D sin tocar 'self' si es método."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            self = args[0]
            rest = args[1:]
        else:
            self = None
            rest = tuple()

        def _to_1d(x):
            if isinstance(x, np.ndarray):
                return x.ravel()
            if callable(x) or hasattr(x, "__dict__"):
                return x
            try:
                return np.array(x).ravel()
            except Exception:
                return x

        new_rest   = tuple(_to_1d(a) for a in rest)
        new_kwargs = {k: _to_1d(v) for k, v in kwargs.items()}

        if self is None:
            return fn(*new_rest, **new_kwargs)
        return fn(self, *new_rest, **new_kwargs)
    return wrapper


def timeit(fn: Callable) -> Callable:
    """Decorador simple para medir tiempo de ejecución (desactivado por defecto)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        enable = False  # pon True si quieres medir tiempos
        if not enable:
            return fn(*args, **kwargs)
        import time
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"[timeit] {fn.__name__}: {1000*(t1 - t0):.2f} ms")
        return out
    return wrapper


# ─────────────────────────────────────────────────────────────
# Abstracción de FRF: interfaz por Protocolo (typing) opcional
# ─────────────────────────────────────────────────────────────
class FRFLike(Protocol):
    """Cualquier objeto que exponga fG y fH con la misma firma."""
    def fG(self, w: np.ndarray) -> np.ndarray: ...
    def fH(self, w: np.ndarray) -> np.ndarray: ...


# ─────────────────────────────────────────────────────────────
# FRFModel por composición: N modos (1-DOF, 2-DOF, …)
# ─────────────────────────────────────────────────────────────
class FRFModel:
    """Modelo FRF con número arbitrario de modos. Sin banderas; suma de contribuciones."""

    class Mode:
        """Un modo individual del FRF: (k, zeta, theta_A, r(w))."""
        def __init__(
            self,
            k: float,
            zeta: float,
            theta_A: float,
            r: Callable[[np.ndarray], np.ndarray],
        ) -> None:
            # Parámetros del modo (encapsulados)
            self.k = float(k)
            self.zeta = float(zeta)
            self.theta_A = float(theta_A)
            self.r = r

        def contrib_G(self, w: np.ndarray) -> np.ndarray:
            """Contribución del modo a la parte real G(ω)."""
            rw = self.r(w)
            c  = np.cos(self.theta_A)**2
            den = self.k * ((1.0 - rw**2)**2 + (2.0 * self.zeta * rw)**2)
            return c * ((1.0 - rw**2) / den)

        def contrib_H(self, w: np.ndarray) -> np.ndarray:
            """Contribución del modo a la parte imaginaria H(ω)."""
            rw = self.r(w)
            c  = np.cos(self.theta_A)**2
            den = self.k * ((1.0 - rw**2)**2 + (2.0 * self.zeta * rw)**2)
            return c * (-2.0 * self.zeta * rw / den)

    def __init__(self, modes: Iterable[Tuple[float, float, float, Callable[[np.ndarray], np.ndarray]]]) -> None:
        """
        Construye el FRF con una lista de modos:
        - Cada modo es (k, zeta, theta_A, r_func) con r_func(w) = w / w_natural.
        - Si pasas 1 modo → 1-DOF; 2 modos → 2-DOF; etc.
        """
        self._modes: List[FRFModel.Mode] = [
            FRFModel.Mode(k, zeta, theta_A, r) for (k, zeta, theta_A, r) in modes
        ]
        if len(self._modes) == 0:
            raise ValueError("At least one mode is required to build FRFModel.")

    @classmethod
    def one_dof(
        cls,
        k1: float, zeta1: float, theta1_A: float,
        r1: Callable[[np.ndarray], np.ndarray],
    ) -> "FRFModel":
        """Conveniencia: FRF de 1-DOF."""
        return cls(modes=[(k1, zeta1, theta1_A, r1)])

    @classmethod
    def two_dof(
        cls,
        k1: float, zeta1: float, theta1_A: float, r1: Callable[[np.ndarray], np.ndarray],
        k2: float, zeta2: float, theta2_A: float, r2: Callable[[np.ndarray], np.ndarray],
    ) -> "FRFModel":
        """Conveniencia: FRF de 2-DOF."""
        return cls(modes=[(k1, zeta1, theta1_A, r1),
                          (k2, zeta2, theta2_A, r2)])

    def add_mode(self, k: float, zeta: float, theta_A: float, r: Callable[[np.ndarray], np.ndarray]) -> None:
        """Añade un modo (útil para pasar de 1→2 DOF sin cambiar la API)."""
        self._modes.append(FRFModel.Mode(k, zeta, theta_A, r))

    def set_modes(self, modes: Iterable[Tuple[float, float, float, Callable[[np.ndarray], np.ndarray]]]) -> None:
        """Reemplaza todos los modos activos."""
        self._modes = [FRFModel.Mode(k, zeta, theta_A, r) for (k, zeta, theta_A, r) in modes]
        if len(self._modes) == 0:
            raise ValueError("At least one mode is required to build FRFModel.")

    @ensure_1d_numpy
    def fG(self, w: np.ndarray) -> np.ndarray:
        """Parte real G(ω) como suma de aportes de todos los modos."""
        total = np.zeros_like(w, dtype=float)
        for m in self._modes:
            total = total + m.contrib_G(w)
        return total

    @ensure_1d_numpy
    def fH(self, w: np.ndarray) -> np.ndarray:
        """Parte imaginaria H(ω) como suma de aportes de todos los modos."""
        total = np.zeros_like(w, dtype=float)
        for m in self._modes:
            total = total + m.contrib_H(w)
        return total


# ─────────────────────────────────────────────────────────────
# Estrategias de fase (polimorfismo)
# ─────────────────────────────────────────────────────────────
class PhaseStrategy(Protocol):
    """Estrategia abstracta para calcular epsilon a partir de FG y FH."""
    def compute_epsilon(self, FG: np.ndarray, FH: np.ndarray) -> np.ndarray: ...


class AltintasPhaseStrategy:
    """Estrategia estilo Altintas: ψ ≤ 0 y ε = 3π + 2ψ."""
    @ensure_1d_numpy
    def compute_epsilon(self, FG: np.ndarray, FH: np.ndarray) -> np.ndarray:
        phi = np.arctan2(FH, FG)
        psi = np.where(phi <= 0.0, phi, phi - 2.0 * np.pi)
        e = 3.0 * np.pi + 2.0 * psi
        return e


class PhiPhaseStrategy:
    """Estrategia por φ directo: ε' = π + 2φ (equivalente a tu mode='phi')."""
    @ensure_1d_numpy
    def compute_epsilon(self, FG: np.ndarray, FH: np.ndarray) -> np.ndarray:
        phi = np.arctan2(FH, FG)
        e = np.pi + 2.0 * phi
        return e


# ─────────────────────────────────────────────────────────────
# Utilidades de arrays (segmentación por regiones negativas)
# ─────────────────────────────────────────────────────────────
class ArrayUtils:
    """Funciones auxiliares para máscaras y segmentación de índices consecutivos."""

    @staticmethod
    @ensure_1d_numpy
    def negative_region(
        w: np.ndarray, data: np.ndarray, target: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (y_neg, w_neg, idx_starts) para data < target."""
        mask = data < target
        y = data[mask]
        w_neg = w[mask]
        pos = np.where(mask)[0]
        cuts = ArrayUtils._segment_cuts(pos)
        idx_starts = np.insert(cuts, 0, 0).astype(int)
        return y, w_neg, idx_starts

    @staticmethod
    def _segment_cuts(indexes: np.ndarray) -> np.ndarray:
        """Devuelve posiciones donde se rompe la consecutividad."""
        cuts: List[int] = []
        for i in range(1, len(indexes)):
            if indexes[i] != indexes[i - 1] + 1:
                cuts.append(i)
        return np.array(cuts, dtype=int)


# ─────────────────────────────────────────────────────────────
# Cálculo de lóbulos (usa FRFLike + estrategia de fase)
# ─────────────────────────────────────────────────────────────
class LobeCalculator:
    """Calcula lóbulos de estabilidad usando un FRF y una estrategia de fase."""

    def __init__(self, frf: FRFLike, phase_strategy: PhaseStrategy) -> None:
        # Inyección de dependencias (DIP) contra abstracciones
        self._frf = frf
        self._phase = phase_strategy

    @timeit
    def compute_lobes(
        self,
        w: np.ndarray,
        target: float,
        k_list: Sequence[int],
        Kf: float,
    ) -> np.ndarray:
        """
        Calcula lóbulos.
        Salida shape = [segment, len(k), len(w_neg), 2]
        donde [:,:,:,0] = n [rpm] y [:,:,:,1] = a_lim [mm].
        """
        # 1) regiones donde G < target
        FG_full = self._frf.fG(w)
        # FH_full no es necesario para segmentar, pero lo calculamos por consistencia
        FH_full = self._frf.fH(w)

        _, w_neg, seg_starts = ArrayUtils.negative_region(w, FG_full, target)
        # Si no hay regiones negativas, devolvemos vacío coherente
        if len(seg_starts) == 0:
            return np.empty((0, len(k_list), 0, 2), dtype=float)
        
        f_peaks = self.frecuencias_resonancia_multimodal(w, FG_full, FH_full,
                                    n_peaks=len(seg_starts),
                                    prominence_frac=0.03,
                                    real_lt=None)  # quita real_lt si no quieres filtrar

        f_peaks = np.array(f_peaks, dtype=float)
        f_peaks = np.round(f_peaks, decimals=0)  # evitar problemas numéricos
        

        # 2) proyectar FRF a w_neg
        FG = self._frf.fG(w_neg)
        FH = self._frf.fH(w_neg)

        # 3) epsilon por estrategia (polimorfismo)
        e = self._phase.compute_epsilon(FG, FH)

        # 4) profundidad crítica (proteger división)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_lim = (-1.0 / (2.0 * Kf * FG)) * 1000.0
        a_lim = np.where(np.isfinite(a_lim), a_lim, np.nan)

        # 5) reservar salida
        n_segments = len(seg_starts)
        lobes = np.full((n_segments, len(k_list), len(w_neg), 2), np.nan, dtype=float)

        # 6) recorrer segmentos (fase continua ya manejada por la estrategia escogida)
        for j in range(n_segments):
            start = seg_starts[j]
            stop  = seg_starts[j + 1] if j < n_segments - 1 else len(w_neg)
            idx   = np.arange(start, stop)

            w_seg = w_neg[idx]
            e_seg = e[idx]

            denom = (w_seg * (2.0 * np.pi))
            safe = np.abs(denom) > 1e-12

            for i, ki in enumerate(k_list):
                T = np.full_like(w_seg, np.nan, dtype=float)
                T[safe] = (2.0 * np.pi * ki + e_seg[safe]) / denom[safe]

                n_rpm = np.full_like(T, np.nan, dtype=float)
                safe_T = np.abs(T) > 1e-12
                n_rpm[safe_T] = 60.0 / T[safe_T]

                lobes[j, i, idx, 0] = n_rpm
                lobes[j, i, idx, 1] = a_lim[idx]

        return lobes, f_peaks
     
    def _local_maxima(self, y: np.ndarray) -> np.ndarray:
        if y.size < 3:
            return np.array([], dtype=int)
        return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

    def _quad_refine(self, x: np.ndarray, y: np.ndarray, i: int) -> float:
        if i <= 0 or i >= len(y)-1:
            return float(x[i])
        y1, y2, y3 = y[i-1], y[i], y[i+1]
        denom = (y1 - 2.0*y2 + y3)
        if denom == 0:
            return float(x[i])
        delta = 0.5 * (y1 - y3) / denom      # desplazamiento en bins
        dx = 0.5 * ((x[i] - x[i-1]) + (x[i+1] - x[i]))  # paso medio (vale para malla no uniforme)
        return float(x[i] + delta*dx)

    def frecuencias_resonancia_multimodal(
        self,
        f: np.ndarray,
        reH: np.ndarray,
        imH: np.ndarray,
        n_peaks: int = 6,
        prominence_frac: float = 0.05,
        real_lt: Optional[float] = None  # p.ej. 0.0 para “solo donde Re{H} < 0”
    ) -> np.ndarray:
        """
        Devuelve hasta n_peaks frecuencias de resonancia (refinadas) ordenadas por frecuencia.
        - f: vector de frecuencias (Hz o rad/s, creciente)
        - reH, imH: partes real e imaginaria
        - prominence_frac: filtra picos débiles según el rango dinámico de |H|
        - real_lt: si no es None, solo conserva picos donde Re{H} < real_lt
        """
        f = np.asarray(f, float)
        M = np.hypot(reH, imH)  # |H|

        # 1) Candidatos: máximos locales del módulo
        idx = self._local_maxima(M)
        if idx.size == 0:
            # Sin máximos internos: devuelve el máximo global si existe
            return np.array([float(f[int(np.nanargmax(M))])])

        # 2) Filtro por “prominencia” simple respecto a rango dinámico
        dyn = np.nanmax(M) - np.nanmin(M)
        thr = np.nanmin(M) + prominence_frac * dyn
        idx = idx[M[idx] > thr]
        if idx.size == 0:
            return np.array([], dtype=float)

        # 3) (Opcional) limitar a región con Re{H} < real_lt
        if real_lt is not None:
            idx = idx[reH[idx] < real_lt]
            if idx.size == 0:
                return np.array([], dtype=float)

        # 4) Quedarse con los picos más altos
        idx = idx[np.argsort(M[idx])[::-1][:n_peaks]]

        # 5) Refinamiento parabólico sub-bin
        f_peaks = np.array([self._quad_refine(f, M, i) for i in idx], dtype=float)

        # 6) Ordenar por frecuencia ascendente para lectura cómoda
        f_peaks.sort()
        return f_peaks 


# ─────────────────────────────────────────────────────────────
# Plotter (estilo local + gráficos GH y lóbulos)
# ─────────────────────────────────────────────────────────────
class PlotStyle:
    """Context manager para aplicar rcParams locales."""
    def __init__(self, params: dict) -> None:
        self._params = dict(params)

    def __enter__(self):
        self._ctx = plt.rc_context(self._params)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


class Plotter:
    """Encapsula el graficado con un estilo consistente (similar a tu local_style)."""

    def __init__(self, style_params: Optional[dict] = None) -> None:
        default = {
            # Tipografía
            'font.family': 'serif',
            'font.size': 8,
            'axes.titlesize': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            # Estética
            'lines.linewidth': 0.75,
            'lines.markersize': 1,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 3.5,
            'ytick.major.size': 3.5,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'mathtext.fontset': 'stix',
            'axes.formatter.use_mathtext': True,
            'legend.frameon': False,
            'legend.loc': 'best',
            'legend.handlelength': 2.0,
            'legend.borderaxespad': 0.5,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'savefig.transparent': True,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
        }
        self._style = style_params or default

    def plot_GH(self, w: np.ndarray, G: np.ndarray, H: np.ndarray, target: float = 0.0) -> None:
        """Gráfica G y H con referencia horizontal en target."""
        with PlotStyle(self._style):
            rescale = 1.0
            figsize = (85 / 25.4 * rescale, 64 / 25.4 * rescale)
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(w, G, label='Real part', linestyle='-', linewidth=1.5, alpha=1.0)
            ax.axhline(target, color='black', linewidth=0.8)
            ax.set_xlim([0, 300])
            ax.set_title('Real part G')
            ax.set_xlabel('w')
            ax.set_ylabel('G')
            ax.legend()
            fig.tight_layout()



            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(w, H, label='Imaginary part', linestyle='-', linewidth=1.5, alpha=1.0)
            ax.axhline(target, color='black', linewidth=0.8)
            ax.set_xlim([0, 300])
            ax.set_title('Imaginary part H')
            ax.set_xlabel('w')
            ax.set_ylabel('H')
            ax.legend()
            fig.tight_layout()

    def plot_lobes(
        self,
        lobes: np.ndarray,
        frequency_peaks: np.ndarray,
        mode_index: int,
        title: str = "Lobes",
        error_percent: float = 0.0,
        k_count: int = None,
        k_index: int = None,
        full_color: bool = True,
        intersections: bool = False,
        A_intersection: Optional[Tuple[float, float]] = None,
        B_intersection: Optional[Tuple[float, float]] = None,
        flag_points: bool = False,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[List[str]] = None,
        show_min: bool = False,
        save_flag: bool = False,
        save_path: str = "SLD_cases_AD.png",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Grafica los lóbulos manteniendo tu estética y lógica."""
        with PlotStyle(self._style):
            # Paletas dependientes de j (segmento) para consistencia visual
            linestyles = ['-', '--', ':', '-.']
            n_j = lobes.shape[0]
            linestyle_by_j = [linestyles[j % len(linestyles)] for j in range(n_j)]

            cmap = plt.get_cmap('Accent')
            den = max(1, n_j - 1)
            color_by_j = [cmap(j / den) for j in range(n_j)]

            rescale = 1
            figsize = (85 / 25.4 * rescale, 64 / 25.4 * rescale)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylim(0, 60)
            ax.set_xlim(7000, 15000)
            ax.set_xlabel(r"Spindle Speed - $\Omega \: [rpm]$")
            ax.set_ylabel(r"$h_0 \: [mm]$")
            fig.tight_layout()

            notes: List[Tuple[str, Tuple[float, float, float, float]]] = []
            plotted_cont = False
            plotted_dash = False

            if mode_index < 0:
                # Modo “todos los k”
                for j in range(lobes.shape[0]):      # segmentos
                    for i in range(lobes.shape[1]):  # k
                        x = lobes[j, i, :, 0]
                        y = lobes[j, i, :, 1]
                        mask = ~np.isnan(x) & ~np.isnan(y)
                        x = x[mask]; y = y[mask]
                        color = color_by_j[j]
                        linestyle = linestyle_by_j[j]

                    
                        if i == 0:
                            label = f' Mode {frequency_peaks[j]:.0f} Hz'
                        else:
                            label = None 

                        ax.plot(x, y, label=label, linestyle=linestyle, linewidth=1.0, color=color)

                        if show_min and x.size:
                            idx_min = np.nanargmin(y)
                            x_min, y_min = x[idx_min], y[idx_min]
                            ax.scatter(x_min, y_min, color=color, marker="x", s=25)
                            if i == 0:
                                notes.append((f"Min: {y_min:.4f} mm, $\Omega: {x_min:.0f}$ rpm", color))
                                ax.hlines(y_min, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
                                          colors=color, linestyles='--')

                        if error_percent > 0.0:
                            e = y * (error_percent / 100.0)
                            ax.fill_between(x, y - e, y + e, alpha=0.3, color=color)


                if flag_points and points is not None:
                    x_pts = points[:, 0]
                    y_pts = points[:, 1]
                    mask = ~np.isnan(x_pts) & ~np.isnan(y_pts)
                    x_pts = x_pts[mask]
                    y_pts = y_pts[mask]
                    ax.scatter(x_pts, y_pts, color='red', marker='o', s=10)

                    if point_labels is not None and len(point_labels) == len(x_pts):
                        for (xp, yp, lbl) in zip(x_pts, y_pts, point_labels):
                            ax.annotate(lbl, (xp, yp),
                                        textcoords="offset points",
                                        xytext=(10, 0), ha='center', color='red')

            else:
                # Modo “un k”
                if lobes.shape[0] > 1:
                    j = mode_index
                else:
                    j = 0
                for i in range(lobes.shape[1]):
                    x = lobes[j, i, :, 0]
                    y = lobes[j, i, :, 1]
                    mask = ~np.isnan(x) & ~np.isnan(y)
                    x = x[mask]; y = y[mask]
                    color = color_by_j[j]
                    linestyle = linestyle_by_j[j]

                    
                    if i == 0:
                        label = f' Mode {frequency_peaks[j]:.0f} Hz'
                    else:
                        label = None

                    ax.plot(x, y, label=label, linestyle=linestyle, linewidth=1.0, color=color)

                    if show_min and x.size:
                        idx_min = np.nanargmin(y)
                        x_min, y_min = x[idx_min], y[idx_min]
                        ax.scatter(x_min, y_min, color=color, marker="x", s=25)
                        if i ==0 :
                            notes.append((f"Min: {y_min:.4f} mm, $\Omega: {x_min:.2f}$ rpm", color))
                        ax.hlines(y_min, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
                                  colors=color, linestyles='--')

                    if error_percent > 0.0:
                        e = y * (error_percent / 100.0)
                        ax.fill_between(x, y - e, y + e, alpha=0.3, color=color)

                    # Intersecciones con recta AB si aplica
                    if intersections and A_intersection is not None and B_intersection is not None and x.size:
                        print("Calculating intersections...")
                        def line(xv: float | np.ndarray) -> float | np.ndarray:
                            m = (B_intersection[1] - A_intersection[1]) / (B_intersection[0] - A_intersection[0])
                            return m * (xv - A_intersection[0]) + A_intersection[1]

                        curve = interp1d(x, y, kind='linear', fill_value="extrapolate")
                        def diff(xv: float) -> float:
                            return float(curve(xv) - line(xv))

                        x_est = x
                        for idx_pair in range(len(x_est) - 1):
                            x0, x1 = x_est[idx_pair], x_est[idx_pair + 1]
                            if diff(x0) * diff(x1) <= 0:
                                try:
                                    print(f"Finding intersection between {x0:.2f} and {x1:.2f}")
                                    root = fsolve(diff, (x0 + x1) / 2.0)[0]
                                    y_root = float(curve(root))
                                    ax.plot(root, y_root, 'o')
                                    ax.axvline(root, linestyle='--', alpha=0.5)
                                    ax.annotate(f"({root:.2f}, {y_root:.2f})",
                                                (root, y_root),
                                                textcoords="offset points",
                                                xytext=(5, 10), ha='left')
                                except Exception:
                                    pass

                if flag_points and points is not None:  
                    x_pts = points[:, 0]
                    y_pts = points[:, 1]
                    mask = ~np.isnan(x_pts) & ~np.isnan(y_pts)
                    x_pts = x_pts[mask]
                    y_pts = y_pts[mask]
                    ax.scatter(x_pts, y_pts, color='red', marker='o', s=10)

                    if point_labels is not None and len(point_labels) == len(x_pts):
                        for (xp, yp, lbl) in zip(x_pts, y_pts, point_labels):
                            ax.annotate(lbl, (xp, yp),
                                        textcoords="offset points",
                                        xytext=(7.5, 0), ha='center', fontsize=6, color='red')

            # fig.tight_layout()
            # Notas inferiores
            ypos, dy = 0.02, 0.04
            for k, (txt, col) in enumerate(notes):
                fig.text(0.55, ypos + k * dy, txt, ha='center', va='bottom', color=col)
            plt.subplots_adjust(bottom=0.15 + len(notes) * 0.05)
            plt.subplots_adjust(top=0.90)

            ax.set_title(title)
            if save_flag:
                ax.legend()
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            ax.legend()

            return fig, ax


# ─────────────────────────────────────────────────────────────
# Ejemplo de uso (main) — MISMAS SALIDAS que tu código original
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":


    def unir_figuras(figs, xlim=None, ylim=None, estilos=None, labels=None, linewidth=0.75,
                    scatter_size_factor=0.5, flag_minimun=True,
                    copiar_notas=True,
                    apilar_notas=True,
                    dy_stack=0.08) -> Tuple[plt.Figure, plt.Axes]:
        """
        Une todas las curvas de varias figuras en una sola.
        
        - Cada figura original recibe un estilo distinto (color/linestyle).
        - Si `labels` se pasa, cada figura usará ese label en lugar de los originales.
        - Si no hay labels definidos, se mantienen los labels originales de cada línea.
        - Se pueden limitar ejes x e y.
        
        Parámetros:
        figs: lista de objetos Figure
        xlim: tupla (xmin, xmax) o None
        ylim: tupla (ymin, ymax) o None
        estilos: lista de kwargs (ej. {"color":"red","linestyle":"--"})
        labels: lista de strings o None (uno por figura)
        
        Devuelve:
        fig, ax -> nueva figura con todas las curvas
        """

        def _copiar_notas_en_top(ax, figs, colores, dy=0.055, y0=1.02):
            """
            Copia las notas de cada figura y las coloca debajo del label X.
            Cada grupo se colorea con el color asignado.
            
            Parámetros:
            ax       -> eje destino
            figs     -> lista de figuras originales
            colores  -> lista de colores (uno por figura)
            dy       -> separación vertical entre notas
            y0       -> posición inicial (en coords de eje, <0 es debajo del gráfico)
            """
            fila = 0
            for i, f in enumerate(figs):
                col = colores[i % len(colores)]
                for t in getattr(f, "texts", []):
                    ax.text(
                        0.5,                      # centrado horizontal
                        y0 - fila*dy,             # apiladas hacia abajo
                        t.get_text(),
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=t.get_fontsize(),
                        fontstyle=t.get_fontstyle(),
                        fontweight=t.get_fontweight(),
                        color=col,
                        clip_on=False
                    )
                    fila += 1

            # Asegurar espacio suficiente debajo
            ax.figure.subplots_adjust(bottom=0.25 + fila*dy/5)

        rescale = 1.0
        figsize = (85 / 25.4 * rescale, 64 / 25.4 * rescale)
        fig, ax = plt.subplots(figsize=figsize)

        # Colores de tab10
        tab10 = plt.get_cmap("tab10").colors  # 10 colores

        # Si hay más de 10 figuras: recicla colores y cambia el linestyle por “bloques” de 10
        linestyles = ["-", "--", "-.", ":"]
        colores_usados = []

        for i, f in enumerate(figs):
            color = tab10[i % 10]
            colores_usados.append(color)
            ls = linestyles[(i // 10) % len(linestyles)]  # cambia estilo cada 10 figs
            custom_label = None
            if labels is not None and i < len(labels):
                custom_label = labels[i]

            # 1) detectar label "representativo" de la figura si no se pasó uno
            fig_label = custom_label
            if fig_label is None:
                # primer label no automático encontrado en cualquiera de sus ejes
                for old_ax in f.axes:
                    for line in old_ax.get_lines():
                        lbl = line.get_label()
                        if lbl and not lbl.startswith("_"):
                            fig_label = lbl
                            break
                    if fig_label:
                        break
                # si no encontramos, intenta usar el título del primer eje
                if not fig_label and f.axes:
                    fig_label = f.axes[0].get_title() or None

            # 2) copiar líneas, pero solo la PRIMERA línea lleva label (las otras van sin leyenda)
            first_line = True
            for old_ax in f.axes:
                for line in old_ax.get_lines():
                    x = line.get_xdata()
                    y = line.get_ydata()
                    if first_line:
                        ax.plot(x, y, color=color, linestyle=ls, linewidth=linewidth,
                                label=fig_label)
                        first_line = False
                    else:
                        ax.plot(x, y, color=color, linestyle=ls, linewidth=linewidth,
                                label="_nolegend_")


                if flag_minimun:    
                 # Copiar solo LineCollection (ej. hlines, vlines)
                    for coll in old_ax.collections:
                        if isinstance(coll, mcoll.LineCollection):
                            for seg in coll.get_segments():
                                xs = [p[0] for p in seg]
                                ys = [p[1] for p in seg]
                                if np.allclose(ys[0], ys[1]):  # hline
                                    ax.hlines(ys[0], min(xs), max(xs),
                                            colors=color, linestyles='--', linewidth=linewidth*0.5)
                                elif np.allclose(xs[0], xs[1]):  # vline
                                    ax.vlines(xs[0], min(ys), max(ys),
                                            colors=color, linestyles='--', linewidth=linewidth*0.5)
                                    
                        #  3) Copiar scatter (PathCollection)
                        elif isinstance(coll, mpathcoll.PathCollection):
                            offsets = coll.get_offsets()
                            fc = coll.get_facecolor()
                            ec = coll.get_edgecolor()
                            sizes = coll.get_sizes() * scatter_size_factor
                            for (x, y) in offsets:
                                ax.scatter(x, y, 
                                        s=sizes[0] if len(sizes) > 0 else 20,
                                        facecolors= color,
                                        # edgecolors=ec if len(ec) > 0 else "none",
                                        marker=coll.get_paths()[0] if coll.get_paths() else "o")


        if copiar_notas:
            _copiar_notas_en_top(ax, figs, colores_usados, dy=dy_stack, y0=-0.2)
            

        # Limitar ejes si corresponde
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel("Spindle Speed [rpm]")
        ax.set_ylabel("Ap [mm]")
        fig.tight_layout()

        ax.legend()
        return fig, ax


    style = {
            # Tipografía
            'font.family': 'serif',
            'font.size': 8,
            'axes.titlesize': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            # Estética
            'lines.linewidth': 0.75,
            'lines.markersize': 1,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 3.5,
            'ytick.major.size': 3.5,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'mathtext.fontset': 'stix',
            'axes.formatter.use_mathtext': True,
            'legend.frameon': False,
            'legend.loc': 'best',
            'legend.handlelength': 2.0,
            'legend.borderaxespad': 0.5,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'savefig.transparent': True,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
        }
    
    # —— Parámetros
    w1 = 250.0
    w2 = 150.0
    w_delta = 0.5
    k1 = 2.26e8
    k2 = 2.13e8
    Kf = 1000e6
    zeta1 = 0.012
    zeta2 = 0.01
    theta1_A = 30.0 * np.pi / 180.0
    theta2_A = -45.0 * np.pi / 180.0
    target = 0.0
    num_lobes = 4
    k_list = np.arange(0, num_lobes)
    num_points = 50000

    r1 = lambda w: w / w1
    r2 = lambda w: w / w2

    all_axes = []
    all_figs = []

    # ─────────────────────────────────────────────────────────────
    #                          2-DOF
    # ─────────────────────────────────────────────────────────────

    # —— FRF (2-DOF) 
    frf_2DOF = FRFModel.two_dof(
        k1, zeta1, theta1_A, r1,
        k2, zeta2, theta2_A, r2
    )

    # —— Mallado y FRF
    w_min, w_max = np.min([w1, w2]) , np.max([w1, w2]) 
    w = np.linspace(0, w_max*(1 + w_delta), num_points)
    G = frf_2DOF.fG(w)
    H = frf_2DOF.fH(w)

    # —— Estrategia de fase: usa 'PhiPhaseStrategy' para reproducir tu mode="phi"
    phase = AltintasPhaseStrategy()
    # Si deseas Altintas, cambia a: phase = AltintasPhaseStrategy()

    # —— Cálculo de lóbulos (misma fórmula de tu implementación)
    calculator = LobeCalculator(frf=frf_2DOF, phase_strategy=phase)
    lobes, f_peaks = calculator.compute_lobes(w=w, target=target, k_list=k_list, Kf=Kf)

    # —— Gráficas (mismo estilo que tu local_style)
    plotter = Plotter(style_params=None)
    # plotter.plot_GH(w, G, H, target)

    A_intersection = (12093.99536, 5.)
    B_intersection = (12093.99536, 15.)
    fig_2DOF, ax_2DOF = plotter.plot_lobes(
        lobes=lobes,
        frequency_peaks=f_peaks,
        mode_index = -1,
        title= "SLD - Analitycal ",
        error_percent=0.0,
        intersections=False,
        # A_intersection=A_intersection, B_intersection=B_intersection,
        # flag_points=True,
        # points=np.array([[12000, 15], [12000, 5]]),
        # point_labels=["B", "A"],
        show_min=True,
        save_flag=False,
        save_path=r"C:\Users\quiqu\OneDrive-ensam.eu\Desktop\Thesis\04-Articles\Manufacturing_21\Artiuclo_Manuf21_2026\Images\SLD.png",
    )
    all_axes.append(ax_2DOF)
    all_figs.append(fig_2DOF)


    # ─────────────────────────────────────────────────────────────
    #                      1-DOF - 150HZ
    # ─────────────────────────────────────────────────────────────

    frf_1DOF_150Hz = FRFModel.one_dof(
        k2, zeta2, theta2_A, r2
    )

    # —— Mallado y FRF
    w_min, w_max = np.min([w1, w2]) , np.max([w1, w2]) 
    w = np.linspace(0, w_max*(1 + w_delta), num_points)
    G = frf_1DOF_150Hz.fG(w)
    H = frf_1DOF_150Hz.fH(w)

    # —— Estrategia de fase: usa 'PhiPhaseStrategy' para reproducir tu mode="phi"
    phase = AltintasPhaseStrategy()
    # Si deseas Altintas, cambia a: phase = AltintasPhaseStrategy()

    # —— Cálculo de lóbulos (misma fórmula de tu implementación)
    calculator = LobeCalculator(frf=frf_1DOF_150Hz, phase_strategy=phase)
    lobes, f_peaks = calculator.compute_lobes(w=w, target=target, k_list=k_list, Kf=Kf)

    # —— Gráficas (mismo estilo que tu local_style)
    plotter = Plotter(style_params=None)
    # plotter.plot_GH(w, G, H, target)

    A_intersection = (8000.0, 25.0)
    B_intersection = (17000.0, 25.0)
    fig_1DOF_150Hz, ax_1DOF_150Hz = plotter.plot_lobes(
        lobes=lobes,
        frequency_peaks=f_peaks,
        mode_index = 1,  
        title = "1-DOF - 150Hz",
        intersections=False,
        A_intersection=A_intersection, B_intersection=B_intersection,
        show_min=True,
        save_flag=False,
        save_path="SLD_cases_AD.png",
    )
    all_axes.append(ax_1DOF_150Hz)
    all_figs.append(fig_1DOF_150Hz)

    # ─────────────────────────────────────────────────────────────
    #                      1-DOF - 250HZ
    # ─────────────────────────────────────────────────────────────

    frf_1DOF_250Hz = FRFModel.one_dof(
        k1, zeta1, theta1_A, r1
    )

    # —— Mallado y FRF
    w_min, w_max = np.min([w1, w2]) , np.max([w1, w2]) 
    w = np.linspace(0, w_max*(1 + w_delta), num_points)
    G = frf_1DOF_250Hz.fG(w)
    H = frf_1DOF_250Hz.fH(w)

    # —— Estrategia de fase: usa 'PhiPhaseStrategy' para reproducir tu mode="phi"
    phase = AltintasPhaseStrategy()
    # Si deseas Altintas, cambia a: phase = AltintasPhaseStrategy()

    # —— Cálculo de lóbulos (misma fórmula de tu implementación)
    calculator = LobeCalculator(frf=frf_1DOF_250Hz, phase_strategy=phase)
    lobes, f_peaks = calculator.compute_lobes(w=w, target=target, k_list=k_list, Kf=Kf)

    # —— Gráficas (mismo estilo que tu local_style)
    plotter = Plotter(style_params=None)
    # plotter.plot_GH(w, G, H, target)

    A_intersection = (8000.0, 25.0)
    B_intersection = (17000.0, 25.0)
    fig_1DOF_250Hz, ax_1DOF_250Hz = plotter.plot_lobes(
        lobes=lobes,
        frequency_peaks=f_peaks,
        mode_index = 1,
        title = "1-DOF - 250Hz",
        error_percent=0.0,
        intersections=False,
        A_intersection=A_intersection, 
        B_intersection=B_intersection,
        # flag_points=True,
        # points=np.array([[12000, 20], [12000, 5]]),
        # point_labels=["A", "B"],
        show_min=True,
        save_flag=False,
        save_path="SLD_cases_AD.png",
    )
    all_axes.append(ax_1DOF_250Hz)
    all_figs.append(fig_1DOF_250Hz)

    with PlotStyle(style):
        fig_all, ax_all = unir_figuras(
        [fig_2DOF, fig_1DOF_150Hz],
        labels=["2DOF - 150Hz", "1-DOF - 150Hz"],
        xlim=(0, 18750),
        ylim=(0, 60),
        flag_minimun=True,
        copiar_notas=True,
        apilar_notas=True,
        dy_stack=0.07,
            )

    


    plt.show()
