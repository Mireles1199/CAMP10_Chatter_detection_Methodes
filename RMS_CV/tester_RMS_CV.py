#%%#
import numpy as np
from matplotlib import pyplot as plt
from collections import deque, defaultdict

from typing import Deque, Sequence, Tuple, Dict, Optional, Callable, Mapping, List, Any, Union

from dataclasses import dataclass 


#%%
def five_senos(
    fs: float,
    duracion: float,
    ruido_std: float = 0.0,
    fase_aleatoria: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (t, x) with five sinusoids plus optional white noise.
    """
    # Comentario: generador de números aleatorios
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # Comentario: vector de tiempo
    N = int(np.round(fs * duracion))
    t = np.arange(N, dtype=float) / fs

    # Comentario: componentes seno (amplitud, frecuencia[Hz])
    comps = [
        (1.5, 80.0),
        (2.0, 120.0),
        (1.5, 160.0),
        (1.5, 240.0),
        (2.0, 320.0),
    ]

    x = np.zeros_like(t)
    for A, f in comps:
        phi = rng.uniform(0, 2 * np.pi) if fase_aleatoria else 0.0
        x += A * np.sin(2 * np.pi * f * t + phi)

    if ruido_std > 0.0:
        x += rng.normal(0.0, ruido_std, size=t.shape)

    return t, x


from typing import Optional, Tuple, Union, Dict, Any
import numpy as np


def rms_sequence(
    signal: Union[np.ndarray, list],
    fs: float,
    *,
    # Ventaneo en tiempo
    window_sec: Optional[float] = None,
    step_sec: Optional[float] = None,
    overlap_pct: Optional[float] = None,  # si se da, tiene prioridad sobre step_sec
    # Ventaneo en muestras (tiene prioridad sobre *_sec)
    N: Optional[int] = None,
    hop: Optional[int] = None,
    # Pretratamientos
    detrend: bool = False,
    bandpass: Optional[Tuple] = None,  # Hook para filtrado no implementado
    clip: Optional[Tuple[float, float]] = None,  # (vmin, vmax)
    # Salidas / bordes
    return_times: bool = True,
    return_indices: bool = False,
    pad_mode: str = "none",  # "none" | "reflect" | "constant"
    # Aliases por errores de tecleo comunes
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calcula la secuencia RMS por ventanas para monitoreo en línea.

    Parámetros
    ----------
    signal : array_like, shape (T,) o (T, C)
        Señal temporal (por ejemplo, aceleración). Tiempo está en el eje 0.
    fs : float
        Frecuencia de muestreo en Hz.

    Ventaneo (elige en tiempo o en muestras, prevalece el de muestras):
    - En tiempo: `window_sec > 0` y uno entre: `step_sec` o `overlap_pct` (0 <= x < 1)
    - En muestras: `N >= 1` y opcionalmente `hop >= 1`

    detrend : bool
        Si True, resta la media en cada ventana (por canal) antes del cálculo de RMS.
    bandpass : None o tupla
        Hook para filtrado (no implementado aquí).
    clip : None o (vmin, vmax)
        Recorte de valores antes del RMS, útil para limitar valores extremos.

    return_times : bool
        Si True, devuelve los tiempos (segundos) del centro de cada ventana.
    return_indices : bool
        Si True, devuelve los índices (inicio, fin) de cada ventana en la señal.
    pad_mode : {"none", "reflect", "constant"}
        Modo de relleno al final de la señal para asegurar ventanas completas.

    Retorna
    -------
    results : dict
        Diccionario con al menos:
            - "rms": ndarray con valores RMS por ventana (shape (F,) o (F, C))
        Puede incluir:
            - "times": tiempos centrales de cada ventana
            - "indices": tuplas (start, end) de cada ventana
            - "pad_mode": eco del parámetro de entrada
    """

    # --- Corrección de errores de tecleo
    if "derend" in kwargs:
        detrend = kwargs["derend"]
    if "bandpas" in kwargs:
        bandpass = kwargs["bandpas"]

    # Convertir señal a arreglo numpy
    x = np.asarray(signal)
    if x.ndim == 1:
        x = x[:, None]  # (T, 1) - señal mono canal
    elif x.ndim != 2:
        raise ValueError("`signal` debe ser 1D (T,) o 2D (T, C) con tiempo en eje 0.")

    # Validación de frecuencia de muestreo
    if not (isinstance(fs, (int, float)) and fs > 0):
        raise ValueError("`fs` debe ser > 0.")

    T, C = x.shape  # T = muestras, C = canales

    # --- Determinar ventana en muestras
    if N is not None:
        if not (isinstance(N, (int, np.integer)) and N >= 1):
            raise ValueError("`N` debe ser entero >= 1.")
        win = int(N)
    else:
        if window_sec is None or window_sec <= 0:
            raise ValueError("Debe especificar `window_sec > 0` si no se da `N`.")
        win = int(round(window_sec * fs))
        if win < 1:
            raise ValueError("`window_sec * fs` < 1: ventana demasiado pequeña.")

    # --- Determinar salto entre ventanas
    if hop is not None:
        if not (isinstance(hop, (int, np.integer)) and hop >= 1):
            raise ValueError("`hop` debe ser entero >= 1.")
        step = int(hop)
    else:
        if overlap_pct is not None:
            if not (0 <= overlap_pct < 1):
                raise ValueError("`overlap_pct` debe estar en [0, 1).")
            step_sec_eff = (1.0 - overlap_pct) * (win / fs if N is not None else window_sec)
            step = max(1, int(round(step_sec_eff * fs)))
        elif step_sec is not None:
            if step_sec <= 0:
                raise ValueError("`step_sec` debe ser > 0.")
            step = max(1, int(round(step_sec * fs)))
        else:
            step = win  # ventanas contiguas

    # --- Verificación de bandpass (no implementado)
    if bandpass is not None:
        raise NotImplementedError(
            "El parámetro `bandpass` está definido como hook, pero no se implementa "
            "en esta función sin dependencias externas."
        )

    # --- Preparar padding al final de la señal si es necesario
    pad_mode = str(pad_mode).lower()
    if pad_mode not in ("none", "reflect", "constant"):
        raise ValueError("`pad_mode` debe ser 'none', 'reflect' o 'constant'.")

    if T < win:
        if pad_mode == "none":
            return {"rms": np.empty((0, C)), "pad_mode": pad_mode}
        else:
            pad_needed = win - T
    else:
        if pad_mode == "none":
            pad_needed = 0
        else:
            frames = int(np.ceil((T - win) / step)) + 1
            total_needed = (frames - 1) * step + win
            pad_needed = max(0, total_needed - T)

    if pad_needed > 0:
        if pad_mode == "reflect":
            x = np.pad(x, ((0, pad_needed), (0, 0)), mode="reflect")
        elif pad_mode == "constant":
            x = np.pad(x, ((0, pad_needed), (0, 0)), mode="constant", constant_values=0)

    T_eff = x.shape[0]

    # --- Crear ventanas deslizantes
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except Exception as e:
        raise RuntimeError("Se requiere NumPy >= 1.20 para sliding_window_view.") from e

    if pad_mode == "none":
        starts = np.arange(0, max(0, T - win + 1), step, dtype=int)
    else:
        starts = np.arange(0, T_eff - win + 1, step, dtype=int)

    if starts.size == 0:
        out = {"rms": np.empty((0, C)), "pad_mode": pad_mode}
        if return_times:
            out["times"] = np.empty((0,), dtype=float)
        if return_indices:
            out["indices"] = np.empty((0, 2), dtype=int)
        return out

    sw = sliding_window_view(x, window_shape=win, axis=0)  # (T_eff - win + 1, win, C)
    windows = sw[starts]  # (F, win, C)

    # --- Eliminar tendencia (media) por ventana
    if detrend:
        mean_win = windows.mean(axis=1, keepdims=True)
        windows = windows - mean_win

    # --- Recorte de valores extremos
    if clip is not None:
        vmin, vmax = clip
        windows = np.clip(windows, vmin, vmax)

    # --- Calcular RMS: sqrt(mean(x^2)) en cada ventana y canal
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))  # (F, C)

    if signal is not None and np.asarray(signal).ndim == 1:
        rms = rms[:, 0]  # Devuelve como vector si señal original era 1D

    results = {"rms": rms, "pad_mode": pad_mode}

    if return_indices:
        idx = np.stack([starts, starts + win], axis=1)
        results["indices"] = idx

    if return_times:
        centers = starts + (win / 2.0)
        times = centers / fs
        results["times"] = times

    return results


@dataclass
class CVOnlineConfig:
    """
    Configuración para el monitor CVOnlineMonitor.

    Atributos:
        n_max: Tamaño máximo de la ventana deslizante (en muestras).
        use_unbiased_std: Si True, usa desviación estándar muestral insesgada (n-1).
        eps: Pequeño valor para evitar divisiones por cero en el cálculo del CV.

        cv_threshold: Umbral para activar alerta basado en el coeficiente de variación (CV).
        rms_threshold: Umbral para activar alerta basado en el valor RMS.

        n_min_cv: Número mínimo de muestras requeridas para que CV sea válido.
        warmup_ignore_alerts: Si es True, ignora alertas durante fase de calentamiento.

        fs_rms: Frecuencia de muestreo (Hz) del RMS (alternativa a dt_rms).
        dt_rms: Intervalo temporal entre muestras RMS.
        start_time: Tiempo inicial para el seguimiento temporal.
    """
    n_max: int
    use_unbiased_std: bool = True
    eps: float = 1e-12

    cv_threshold: Optional[float] = None
    rms_threshold: Optional[float] = None

    n_min_cv: int = 2
    warmup_ignore_alerts: bool = False

    fs_rms: Optional[float] = None
    dt_rms: Optional[float] = None
    start_time: float = 0.0


@dataclass
class CVOnlineState:
    """
    Estado mutable del monitor CVOnlineMonitor.

    Se actualiza en cada paso con estadísticas acumuladas.
    """
    n: int = 0                    # Número de muestras actuales
    sum1: float = 0.0             # Suma de valores: Σx
    sum2: float = 0.0             # Suma de cuadrados: Σx²
    mu: float = 0.0               # Media actual
    sigma: float = 0.0            # Desviación estándar actual
    cv: float = 0.0               # Coeficiente de variación actual
    t_last: Optional[float] = None  # Último tiempo procesado
    idx: int = 0                  # Índice actual (número de actualizaciones)


class CVOnlineMonitor:
    """
    Monitor en línea de RMS para cálculo eficiente del Coeficiente de Variación (CV).

    Características:
    - Mantiene una ventana deslizante sobre valores RMS entrantes.
    - Calcula media (μ), desviación estándar (σ) y CV = σ / μ en tiempo constante O(1).
    - Permite detección de condiciones anómalas mediante umbrales configurables.

    Uso típico: detección de chatter (vibración) en procesos mecánicos.
    """

    def __init__(self, config: CVOnlineConfig) -> None:
        """
        Inicializa el monitor con configuración dada.
        
        Args:
            config: Objeto de configuración CVOnlineConfig.
        """
        if config.n_max < 1:
            raise ValueError("`n_max` debe ser >= 1.")
        if config.n_min_cv < 1:
            raise ValueError("`n_min_cv` debe ser >= 1.")

        # Si no se especificó dt_rms, lo calculamos desde fs_rms
        if config.dt_rms is None and config.fs_rms:
            config.dt_rms = 1.0 / float(config.fs_rms)

        self.config: CVOnlineConfig = config
        self.state: CVOnlineState = CVOnlineState()
        self.window: Deque[float] = deque(maxlen=config.n_max)

    def reset(self) -> None:
        """
        Reinicia el monitor completamente, incluyendo ventana y estadísticas.
        """
        self.window.clear()
        self.state = CVOnlineState()

    def update(self, rms_value: float) -> Dict[str, Any]:
        """
        Actualiza el estado del monitor con un nuevo valor RMS.

        Args:
            rms_value: Valor flotante de RMS a procesar.

        Returns:
            Diccionario con:
                - n: número de muestras en ventana
                - mu: media actual
                - sigma: desviación estándar actual
                - cv: coeficiente de variación actual
                - alert: True si se dispara una alerta
                - reason: 'cv' | 'rms' | None
                - idx: índice interno
                - time: tiempo estimado (si disponible)
        """
        cfg = self.config
        st = self.state

        if rms_value is None:
            return self._result(alert=False, reason=None)

        x = float(rms_value)

        # Estimación temporal (si está habilitado)
        if cfg.dt_rms is not None:
            st.t_last = cfg.start_time + st.idx * cfg.dt_rms

        # Ventana inicial aún sin llenar
        if st.n < cfg.n_max:
            self.window.append(x)
            st.n += 1
            st.sum1 += x
            st.sum2 += x * x
        else:
            # Ventana llena: eliminar el más viejo y agregar el nuevo
            oldest = self.window[0]
            self.window.append(x)
            st.sum1 += x - oldest
            st.sum2 += (x * x) - (oldest * oldest)

        # Actualización de estadísticas en O(1)
        n = st.n
        st.mu = st.sum1 / n
        if n >= 2:
            var_num = st.sum2 - (st.sum1 * st.sum1) / n
            denom = (n - 1) if cfg.use_unbiased_std else n
            var_val = max(var_num / denom, 0.0)
            st.sigma = var_val ** 0.5
        else:
            st.sigma = 0.0

        # CV = σ / μ (evitamos división por cero)
        denom_mu = st.mu if abs(st.mu) > cfg.eps else cfg.eps
        st.cv = st.sigma / denom_mu

        # Detección de alerta
        alert = False
        reason: Optional[str] = None

        if n < cfg.n_min_cv:
            if cfg.rms_threshold is not None and x > cfg.rms_threshold and not cfg.warmup_ignore_alerts:
                alert, reason = True, "rms"
        else:
            if cfg.cv_threshold is not None and st.cv >= cfg.cv_threshold:
                if not (cfg.warmup_ignore_alerts and n < cfg.n_min_cv):
                    alert, reason = True, "cv"

        st.idx += 1  # Avanza el índice (para tiempo)

        return self._result(alert=alert, reason=reason)

    def current_state(self) -> CVOnlineState:
        """
        Retorna el estado actual del monitor (estadísticas y tiempo).
        """
        return self.state

    def _result(self, alert: bool, reason: Optional[str]) -> Dict[str, Any]:
        """
        Empaqueta los resultados de un paso de actualización.

        Args:
            alert: True si se dispara alerta.
            reason: Causa de alerta ('cv', 'rms', o None).

        Returns:
            Diccionario con valores actuales del monitoreo.
        """
        st = self.state
        cfg = self.config
        time_val: Optional[float] = None

        if cfg.dt_rms is not None:
            time_val = cfg.start_time + (st.idx - 1) * cfg.dt_rms

        return {
            "n": st.n,
            "mu": st.mu,
            "sigma": st.sigma,
            "cv": st.cv,
            "alert": alert,
            "reason": reason,
            "idx": st.idx - 1,
            "time": time_val,
        }
        


def signal_1(
    fs: float,
    T: float,
    tpf: float,
    chatter_freqs: List[float],
    t_chatter_start: float,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (t, x) with TPF harmonics and chatter frequencies after a start time.
    """
    # Comentario: tiempo total y vector de tiempo
    t = np.linspace(0.0, T, int(fs * T), endpoint=False)

    # Comentario: armónicos de TPF
    harmonics = [tpf * i for i in range(1, 6)]
    x_base = np.zeros_like(t)
    for f in harmonics:
        x_base += 3.5 * np.sin(2 * np.pi * f * t)

    # Comentario: envolventes
    envelope_base = 0.1 + 0.6 * (t / T)
    mask_chatter = t > t_chatter_start
    envelope_chatter = np.zeros_like(t)
    if np.any(mask_chatter):
        envelope_chatter[mask_chatter] = 0.1 + 0.6 * (
            (t[mask_chatter] - t_chatter_start) / (T - t_chatter_start)
        )

    # Comentario: componentes chatter (sin modulación para simplicidad)
    x_chatter = np.zeros_like(t)
    for f in chatter_freqs:
        x_chatter += 5.0 * np.sin(2 * np.pi * f * t)
    x_chatter *= envelope_chatter

    # Comentario: ruido blanco
    noise = noise_std * np.random.randn(len(t))

    # Comentario: combinación final
    x = envelope_base * x_base + x_chatter + noise
    return t, x



#%%
fs = 5000.0
duracion = 10.0
t, sig_s = five_senos(fs, duracion=duracion, ruido_std=0.2 , fase_aleatoria=True, seed=42)


t, sig_chatter = signal_1(
    fs=fs,
    T=duracion,
    tpf=200.0,
    chatter_freqs=[150.0, 300.0],
    t_chatter_start=5.0,
    noise_std=0.2,  
)

sig = np.concatenate([sig_s[:len(t)//2], sig_chatter[len(t)//2:]])

# t = np.arange(0.0, duracion, 1/fs)
# signal_stable = 0.5 + 0.02 * np.random.randn(t.size // 2)
# signal_chatter = 0.5 + 0.10 * np.random.randn(t.size - signal_stable.size)
# sig = np.concatenate([signal_stable, signal_chatter])


# t2, sig_s2 = five_senos(fs, duracion=duracion, ruido_std=0.2 , fase_aleatoria=True, seed=42)
# sig_s2*= 2.0
# sig = np.concatenate([sig_s[:len(t)//2], sig_s2[len(t)//2:]])



plt.plot(t, sig)
 
#%% Calculo RMS

window_sec = 0.05
overlap_pct = 0.0

dt_rms = window_sec * (1.0 - overlap_pct)



# RMS cada 0.1 s, sin solape (tasa 10 Hz)
out = rms_sequence(sig, fs, window_sec=dt_rms, overlap_pct=overlap_pct, detrend=False, pad_mode="none")
print("RMS shape:", out["rms"].shape)
print("Times head:", out["times"][:5])

rms_vals = out["rms"]      

plt.figure()
plt.plot(out["times"], out["rms"], marker="o")




cfg = CVOnlineConfig(
    n_max=20,
    use_unbiased_std=True,
    eps=1e-12,
    cv_threshold=0.15,
    rms_threshold=0.9,
    n_min_cv=2,
    warmup_ignore_alerts=False,
    dt_rms=dt_rms,   
    start_time=0.0,
)

mon = CVOnlineMonitor(cfg)

results = defaultdict(list)
for r in rms_vals:
    res = mon.update(float(r))
    for k, v in res.items():
        results.setdefault(k, []).append(v)


print(f"Processed {len(sig)} RMS values.")

print(f"alert { results['alert']}")

figure = plt.figure()
plt.plot(t, sig, label="RMS Signal")

plt.figure()
plt.scatter(results['time'], results['cv'], label="CV Signal", color='red')





plt.show()

