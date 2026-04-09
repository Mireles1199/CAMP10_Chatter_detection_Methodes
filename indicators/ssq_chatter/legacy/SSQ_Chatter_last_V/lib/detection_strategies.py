from __future__ import annotations
# Comentario: regla con selección flexible del tramo estable (por índices o por tiempo)
from typing import Dict, Sequence, Optional, Tuple, Union
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from abc import ABC, abstractmethod

IndexRange = Union[int, Tuple[int, int]]          # Comentario: N  ó  (i0, i1)
TimeRange  = Union[float, Tuple[float, float]]    # Comentario: t_end  ó  (t0, t1)

class DetectionRule(ABC):
    # Comentario: interfaz para reglas de detección
    @abstractmethod
    def detect(
        self,
        d1: np.ndarray,
        idx_stable: Optional[Sequence[int]] = None,
        *,
        t: Optional[np.ndarray] = None,
        stable_index: Optional[IndexRange] = None,
        stable_time: Optional[TimeRange] = None,
    ) -> Dict[str, object]:
        raise NotImplementedError

class ThreeSigmaWithLilliefors(DetectionRule):
    # Comentario: implementación 3σ con verificación de normalidad y fallback MAD
    def __init__(self, frac_stable: float = 0.25, alpha: float = 0.05, z: float = 3.0, fallback_mad: bool = True,
                 stable_time: Optional[TimeRange] = None, stable_index: Optional[IndexRange] = None):
        self.frac_stable = float(frac_stable)
        self.alpha = float(alpha)
        self.z = float(z)
        self.fallback_mad = bool(fallback_mad)
        self.stable_index: Optional[IndexRange] = stable_index
        self.stable_time: Optional[TimeRange] = stable_time

        if stable_time is not None and stable_index is not None:
            self.stable_index = stable_index  # Comentario: prioridad a stable_index
            self.stable_time = None
            print("Warning: both stable_time and stable_index provided; using stable_index only.")
            

    def _build_idx_from_ranges(
        self,
        n: int,
        *,
        t: Optional[np.ndarray],
        stable_index: Optional[IndexRange],
        stable_time: Optional[TimeRange],
    ) -> Optional[np.ndarray]:
        # Comentario: prioridad: stable_index > stable_time; ambos opcionales
        if stable_index is not None:
            if isinstance(stable_index, int):
                i0, i1 = 0, int(stable_index)        # Comentario: [0, N)
            else:
                i0, i1 = int(stable_index[0]), int(stable_index[1])
            i0 = max(0, i0)
            i1 = min(n - 1, i1)
            if i1 < i0:
                raise ValueError("stable_index produce rango vacío")
            return np.arange(i0, i1 + 1, dtype=int)

        if stable_time is not None:
            if t is None:
                raise ValueError("stable_time requiere vector t")
            tt = np.asarray(t, dtype=float)
            if isinstance(stable_time, (float, int)):
                t0, t1 = float(np.min(tt)), float(stable_time)
            else:
                t0, t1 = float(stable_time[0]), float(stable_time[1])
            if t1 < t0:
                raise ValueError("stable_time inválido (t1 < t0)")
            mask = (tt >= t0) & (tt <= t1)
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                raise ValueError("stable_time produce rango vacío")
            return idx.astype(int)
        return None

    def detect(
        self,
        d1: np.ndarray,
        idx_stable: Optional[Sequence[int]] = None,
        *,
        t: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        d1 = np.asarray(d1, dtype=float)
        if d1.ndim != 1:
            raise ValueError("d1 must be 1D")
        n = d1.size

        # Comentario: construir idx_estable: prioridad explícitos > tiempo > fracción
        if idx_stable is not None:
            idx_est = np.asarray(list(idx_stable), dtype=int)
        else:
            built = self._build_idx_from_ranges(n, t=t, stable_index=self.stable_index, stable_time=self.stable_time)
            if built is not None:
                idx_est = built
            else:
                m = max(1, int(self.frac_stable * n))
                idx_est = np.arange(0, m, dtype=int)

        # Comentario: recorte defensivo y unicidad
        idx_est = idx_est[(idx_est >= 0) & (idx_est < n)]
        if idx_est.size == 0:
            raise ValueError("idx_estable vacío tras validación")

        d_est = d1[idx_est]

        mu = float(np.mean(d_est))
        sigma = float(np.std(d_est, ddof=1)) if d_est.size > 1 else 0.0

        # Comentario: prueba de normalidad Lilliefors
        try:
            _, p_value = lilliefors(d_est, dist="norm")
            normal_ok = bool(p_value >= self.alpha)
        except Exception:
            p_value = 0.0
            normal_ok = False

        z = self.z
        if sigma == 0.0:
            eps = 1e-12 if mu == 0 else 1e-6 * abs(mu)
            lim_inf, lim_sup = mu - z * eps, mu + z * eps
        else:
            lim_inf, lim_sup = mu - z * sigma, mu + z * sigma

        metodo = "sigma"
        if self.fallback_mad and not normal_ok:
            med = float(np.median(d_est))
            mad = float(np.median(np.abs(d_est - med)))
            sigma_rob = 1.4826 * mad
            if sigma_rob == 0.0:
                eps = 1e-12 if med == 0 else 1e-6 * abs(med)
                lim_inf, lim_sup = med - z * eps, med + z * eps
            else:
                lim_inf, lim_sup = med - z * sigma_rob, med + z * sigma_rob
            mu, sigma = med, sigma_rob
            metodo = "MAD"

        mask = ((d1 < lim_inf) | (d1 > lim_sup)).astype(int)
        return {
            "mask": mask,
            "mu": mu,
            "sigma": sigma,
            "lim_inf": lim_inf,
            "lim_sup": lim_sup,
            "normal_ok": normal_ok,
            "p_value": float(p_value),
            "metodo_umbral": metodo,
            "idx_estable_usados": idx_est.tolist(),
        }
