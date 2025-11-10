from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple, Any
from matplotlib import pyplot as plt

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


# =========================
# Ejemplo de uso
# =========================
if __name__ == "__main__":
    import numpy as np

    dt_rms = 0.2  # cada muestra cada 0.2 segundos

    t = np.arange(0.0, 60.0, dt_rms)
    rms_stable = 0.5 + 0.02 * np.random.randn(t.size // 2)
    rms_chatter = 0.5 + 0.10 * np.random.randn(t.size - rms_stable.size)
    rms_series = np.concatenate([rms_stable, rms_chatter])

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
    for r in rms_series:
        res = mon.update(float(r))
        for k, v in res.items():
            results.setdefault(k, []).append(v)

    print(f"Processed {len(rms_series)} RMS values.")
    
    print(f"alert { results['alert']}")

    plt.plot(t, rms_series, label="RMS Signal")
    plt.scatter(results['time'], results['cv'], label="CV Signal", color='red')
    plt.legend()
    plt.show()
