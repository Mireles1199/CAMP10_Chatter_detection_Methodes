from __future__ import annotations
# Comentario: compatibilidad para la función detectar_chatter_3sigma con la estrategia nueva
from typing import Dict, Sequence, Optional
import numpy as np
from ..lib.detection_strategies import ThreeSigmaWithLilliefors

def detectar_chatter_3sigma(
    d1: np.ndarray,
    idx_estable: Optional[Sequence[int]] = None,
    fraccion_estable: float = 0.25,
    alpha: float = 0.05,
    z: float = 3.0,
    fallback_mad: bool = True,
) -> Dict[str, object]:
    # Comentario: wrapper que llama a la nueva implementación para mantener el mismo dict de salida
    rule = ThreeSigmaWithLilliefors(frac_stable=fraccion_estable, alpha=alpha, z=z, fallback_mad=fallback_mad)
    res = rule.detect(d1=d1, idx_stable=idx_estable)
    # Comentario: clave alias para compatibilidad exacta
    res["idx_estable_usados"] = res.get("idx_estable_usados", [])
    return res
