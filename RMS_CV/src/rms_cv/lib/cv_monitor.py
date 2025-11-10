# Comentario: monitor en línea de CV sobre secuencia RMS
from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Dict, Any, Optional
from collections import deque

@dataclass
class CVOnlineConfig:
    """
    Configuración del monitor CV en línea.
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
    Estado mutable del monitor (estadísticas acumuladas).
    """
    n: int = 0
    sum1: float = 0.0
    sum2: float = 0.0
    mu: float = 0.0
    sigma: float = 0.0
    cv: float = 0.0
    t_last: Optional[float] = None
    idx: int = 0


class CVOnlineMonitor:
    """
    Monitor en línea de RMS para cálculo eficiente del Coeficiente de Variación (CV).
    """
    def __init__(self, config: CVOnlineConfig) -> None:
        if config.n_max < 1:
            raise ValueError("`n_max` debe ser >= 1.")
        if config.n_min_cv < 1:
            raise ValueError("`n_min_cv` debe ser >= 1.")
        if config.dt_rms is None and config.fs_rms:
            config.dt_rms = 1.0 / float(config.fs_rms)

        self.config: CVOnlineConfig = config
        self.state: CVOnlineState = CVOnlineState()
        self.window: Deque[float] = deque(maxlen=config.n_max)

    def reset(self) -> None:
        """Reinicia ventana y estado."""
        self.window.clear()
        self.state = CVOnlineState()

    def update(self, rms_value: float) -> Dict[str, Any]:
        """
        Actualiza el estado con un valor RMS y evalúa alertas.
        """
        cfg = self.config
        st = self.state

        if rms_value is None:
            return self._result(alert=False, reason=None)

        x = float(rms_value)

        if cfg.dt_rms is not None:
            st.t_last = cfg.start_time + st.idx * cfg.dt_rms

        if st.n < cfg.n_max:
            self.window.append(x)
            st.n += 1
            st.sum1 += x
            st.sum2 += x * x
        else:
            oldest = self.window[0]
            self.window.append(x)
            st.sum1 += x - oldest
            st.sum2 += (x * x) - (oldest * oldest)

        n = st.n
        st.mu = st.sum1 / n
        if n >= 2:
            var_num = st.sum2 - (st.sum1 * st.sum1) / n
            denom = (n - 1) if cfg.use_unbiased_std else n
            var_val = max(var_num / denom, 0.0)
            st.sigma = var_val ** 0.5
        else:
            st.sigma = 0.0

        denom_mu = st.mu if abs(st.mu) > cfg.eps else cfg.eps
        st.cv = st.sigma / denom_mu

        alert = False
        reason: Optional[str] = None

        if n < cfg.n_min_cv:
            if cfg.rms_threshold is not None and x > cfg.rms_threshold and not cfg.warmup_ignore_alerts:
                alert, reason = True, "rms"
        else:
            if cfg.cv_threshold is not None and st.cv >= cfg.cv_threshold:
                if not (cfg.warmup_ignore_alerts and n < cfg.n_min_cv):
                    alert, reason = True, "cv"

        st.idx += 1
        return self._result(alert=alert, reason=reason)

    def current_state(self) -> CVOnlineState:
        """Devuelve el estado actual del monitor."""
        return self.state

    def _result(self, alert: bool, reason: Optional[str]) -> Dict[str, Any]:
        """Empaqueta un diccionario de resultados del paso actual."""
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
