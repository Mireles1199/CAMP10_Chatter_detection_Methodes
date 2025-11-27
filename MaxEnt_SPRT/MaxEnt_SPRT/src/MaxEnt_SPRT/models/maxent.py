
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .prob import GaussianPDF

@dataclass
class MaxEntModels:
    """
    Modelos globales de la pdf del indicador MaxEnt (entropía) bajo:
    - H0: chatter-free  -> p0(H)
    - H1: early chatter -> p1(H)
    """
    p0: GaussianPDF
    p1: GaussianPDF


def fit_maxent_gaussians(
    samples_H0: Iterable[float],
    samples_H1: Iterable[float],
    min_sigma: float = 1e-12,
) -> MaxEntModels:
    """
    Ajusta dos gaussianas a:
    - samples_H0: valores MaxEnt en estado chatter-free
    - samples_H1: valores MaxEnt en estado chatter

    Devuelve modelos p0(H) y p1(H) sobre el indicador H.
    """
    g0 = GaussianPDF.from_samples(samples_H0, eps=min_sigma)
    g1 = GaussianPDF.from_samples(samples_H1, eps=min_sigma)
    return MaxEntModels(p0=g0, p1=g1)