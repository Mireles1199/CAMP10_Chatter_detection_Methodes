# Comentario: detección de chatter vía 3σ con verificación de normalidad (Lilliefors)
from typing import Dict, Sequence, Optional
import numpy as np
from statsmodels.stats.diagnostic import lilliefors


def detectar_chatter_3sigma(
    d1: np.ndarray,
    idx_estable: Optional[Sequence[int]] = None,
    fraccion_estable: float = 0.2,
    alpha: float = 0.05,
    z: float = 3.0,
    fallback_mad: bool = True,
) -> Dict[str, object]:
    """
    Chatter detection via ±z·σ (default 3σ) with Lilliefors normality check.
    """
    # Comentario: sanitizar d1
    d1 = np.asarray(d1, dtype=float)
    if d1.ndim != 1 or d1.size == 0:
        raise ValueError("d1 must be non-empty 1D array")
    if np.any(~np.isfinite(d1)):
        raise ValueError("d1 contains NaN/Inf")

    n = d1.size

    # Comentario: determinar tramo estable
    if idx_estable is None:
        n_estable = max(3, int(np.ceil(n * fraccion_estable)))
        idx_est = np.arange(n_estable)
    else:
        idx_est = np.array(idx_estable, dtype=int)
        if idx_est.size < 3:
            raise ValueError("Need ≥3 stable indices")
        if np.any((idx_est < 0) | (idx_est >= n)):
            raise ValueError("idx_estable out of range")

    d1_est = d1[idx_est]

    # Comentario: estimadores clásicos
    mu = float(np.mean(d1_est))
    sigma = float(np.std(d1_est, ddof=1))

    stat, p_value = lilliefors(d1_est)
    normal_ok = bool(p_value > alpha)

    metodo = "3sigma"
    if sigma == 0.0:
        eps = 1e-12 if mu == 0 else 1e-6 * abs(mu)
        lim_inf = mu - z * eps
        lim_sup = mu + z * eps
    else:
        lim_inf = mu - z * sigma
        lim_sup = mu + z * sigma

    # Comentario: fallback robusto si no hay normalidad
    if fallback_mad and not normal_ok:
        med = float(np.median(d1_est))
        mad = float(np.median(np.abs(d1_est - med)))
        sigma_rob = 1.4826 * mad
        if sigma_rob == 0.0:
            eps = 1e-12 if med == 0 else 1e-6 * abs(med)
            lim_inf = med - z * eps
            lim_sup = med + z * eps
        else:
            lim_inf = med - z * sigma_rob
            lim_sup = med + z * sigma_rob
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
        "idx_estable_usados": idx_est,
    }
