
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .prob import GaussianPDF

@dataclass
class MaxEntModels:
    """
    Maximum Entropy-based Statistical Models.
    A class for managing maximum entropy models using Gaussian probability density
    functions for two-class classification or hypothesis testing scenarios.
    Attributes:
        p0 (GaussianPDF): Gaussian probability density function for the null hypothesis
            or primary class distribution.
        p1 (GaussianPDF): Gaussian probability density function for the alternative
            hypothesis or secondary class distribution.
    """

    p0: GaussianPDF
    p1: GaussianPDF


def fit_maxent_gaussians(
    samples_H0: Iterable[float],
    samples_H1: Iterable[float],
    min_sigma: float = 1e-12,
) -> MaxEntModels:
    """
    Fit maximum entropy Gaussian models to two sets of samples.
    This function creates Gaussian probability density functions for two hypothesis
    classes (H0 and H1) using the maximum entropy principle, where each Gaussian is
    characterized by the mean and standard deviation of its respective samples.
    Parameters
    ----------
    samples_H0 : Iterable[float]
        Samples from the null hypothesis (H0) distribution.
    samples_H1 : Iterable[float]
        Samples from the alternative hypothesis (H1) distribution.
    min_sigma : float, optional
        Minimum standard deviation threshold to ensure numerical stability.
        Default is 1e-12.
    Returns
    -------
    MaxEntModels
        A MaxEntModels object containing two fitted Gaussian PDFs:
        - p0: Gaussian model fitted to samples_H0
        - p1: Gaussian model fitted to samples_H1
    Notes
    -----
    The minimum sigma parameter prevents zero or near-zero standard deviations
    that could cause numerical instability in likelihood calculations.
    """
    g0 = GaussianPDF.from_samples(samples_H0, eps=min_sigma)
    g1 = GaussianPDF.from_samples(samples_H1, eps=min_sigma)
    return MaxEntModels(p0=g0, p1=g1)