
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .prob import GaussianPDF

@dataclass
class MaxEntModels:
    """
    Pair of Gaussian models used by the MaxEnt detector.

    ``p0`` models the stable regime and ``p1`` models the chatter regime.
    The pair is treated as a coherent statistical state and passed as a single
    object across detector components (LLR, SPRT wrapper, plotting metadata).
    """

    p0: GaussianPDF
    """Gaussian density fitted on samples belonging to the stable or chatter-free regime."""
    p1: GaussianPDF
    """Gaussian density fitted on samples belonging to the chatter regime."""


def fit_maxent_gaussians(
    samples_H0: Iterable[float],
    samples_H1: Iterable[float],
    min_sigma: float = 1e-12,
) -> MaxEntModels:
    """
    Fit maximum entropy Gaussian models to two sets of samples.

    Given indicator samples from stable and chatter regimes, this function fits
    one Gaussian per hypothesis under the maximum-entropy rationale (mean and
    variance constraints). The output is ready to be consumed by the LLR model.

    :param samples_H0: Indicator samples representing the stable reference regime used to fit ``p0``.
    :param samples_H1: Indicator samples representing the chatter reference regime used to fit ``p1``.
    :param min_sigma: Lower bound imposed on the estimated standard deviation to avoid degenerate or numerically unstable Gaussian models.

    Returns:
        MaxEntModels: Container with ``p0`` fitted from ``samples_H0`` and
        ``p1`` fitted from ``samples_H1``.

    Notes
    -----
    The minimum sigma parameter prevents zero or near-zero standard deviations
    that could cause numerical instability in likelihood calculations.

    In practice, this regularization avoids overconfident LLR jumps when a
    training segment has very low variance.
    """
    g0 = GaussianPDF.from_samples(samples_H0, eps=min_sigma)
    g1 = GaussianPDF.from_samples(samples_H1, eps=min_sigma)
    return MaxEntModels(p0=g0, p1=g1)