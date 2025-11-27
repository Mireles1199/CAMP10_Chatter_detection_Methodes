
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..models.maxent import MaxEntModels

class LLRModel(ABC):
    """
    Estrategia abstracta para calcular el LLR de una observación H.

    Esto desacopla el motor SPRT de los detalles del modelo (gaussiano,
    kernel, histogramas, etc.) → DIP (Dependency Inversion Principle).
    """

    @abstractmethod
    def llr(self, h_obs: float) -> float:
        """
        Devuelve log( p1(h_obs) / p0(h_obs) ).
        """

@dataclass(frozen=True)
class GaussianIndicatorLLR(LLRModel):
    """
    Implementación concreta de LLR usando dos gaussianas sobre el indicador H.
    """
    models: MaxEntModels

    def llr(self, h_obs: float) -> float:
        return self.models.p1.logpdf(h_obs) - self.models.p0.logpdf(h_obs)