"""
sweep/metrics.py
================
Compute performance metrics for a single indicator run.

Metrics
-------
delta_t_d : float
    Detection latency relative to ground-truth onset:
    ``t_d[0] - t_gt`` [s].  NaN if no detection occurred.
N_fa : int
    Number of false alarms: count of detection timestamps strictly before
    ``t_gt``.
P_det : int
    Binary detection flag: 1 if at least one detection occurred (including
    false alarms), 0 otherwise.
score : float
    Composite score: ``delta_t_d + lam * N_fa * T_unit``.
    NaN if no detection (delta_t_d is NaN).
lower_bound_delta_td : float
    Theoretical minimum latency for this configuration:
    ``K_total * T_unit`` [s].
    This is a floor imposed by the minimum time required to accumulate
    K_total physical cycles before the first decision.
score_lb : float
    Score lower bound: ``lower_bound_delta_td`` (zero false-alarm optimum).

Usage
-----
    from sweep.metrics import evaluate

    metric = evaluate(
        t_d=result.t_d,      # np.ndarray (possibly empty)
        t_gt=5.365,
        T_unit=1/150.0,
        K_total=8,
        lam=1.0,
    )
    print(metric.delta_t_d, metric.N_fa, metric.score)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

__all__ = ["MetricResult", "evaluate"]


@dataclass
class MetricResult:
    """Performance metrics for one indicator run.

    Attributes
    ----------
    delta_t_d : float
        First detection latency relative to ground-truth onset [s].
        ``t_d[0] - t_gt``.  NaN when no detection.
    N_fa : int
        False-alarm count: number of detections before ``t_gt``.
    P_det : int
        1 if at least one detection was raised, else 0.
    score : float
        ``delta_t_d + lam * N_fa * T_unit``.  NaN when no detection.
    lower_bound_delta_td : float
        Theoretical minimum latency = ``K_total * T_unit`` [s].
    score_lb : float
        Lower bound on score = ``lower_bound_delta_td`` (zero false alarms).
    """

    delta_t_d:            float
    N_fa:                 int
    P_det:                int
    score:                float
    lower_bound_delta_td: float
    score_lb:             float


def evaluate(
    t_d: Optional[Union[np.ndarray, float]],
    t_gt: float,
    T_unit: float,
    K_total: int,
    lam: float,
) -> MetricResult:
    """
    Compute performance metrics from indicator detection timestamps.

    Parameters
    ----------
    t_d : np.ndarray or float or None
        Detection timestamps [s].  May be:
        - ``np.ndarray`` (possibly empty),
        - a scalar float (treated as single detection),
        - ``None`` (no detection).
    t_gt : float
        Ground-truth chatter onset time [s].
    T_unit : float
        Physical time unit for this basis [s].  Used to compute score penalty.
    K_total : int
        Total physical cycles for this combo.  Used to compute lower bound.
    lam : float
        False-alarm penalty coefficient (≥ 0).

    Returns
    -------
    MetricResult
    """
    # ── normalise t_d to a 1-D numpy array ───────────────────────────────────
    if t_d is None:
        arr = np.array([], dtype=float)
    elif np.isscalar(t_d):
        arr = np.atleast_1d(float(t_d))
    else:
        arr = np.asarray(t_d, dtype=float).ravel()

    # ── false alarms: detections strictly before ground-truth ────────────────
    N_fa = int(np.sum(arr < t_gt))

    # ── detection flag ───────────────────────────────────────────────────────
    P_det = 1 if arr.size > 0 else 0

    # ── first detection latency ───────────────────────────────────────────────
    if arr.size > 0:
        t_d_first   = float(arr[0])
        delta_t_d   = t_d_first - t_gt
        score       = delta_t_d + lam * N_fa * T_unit
    else:
        delta_t_d   = math.nan
        score       = math.nan

    # ── lower bound ──────────────────────────────────────────────────────────
    lower_bound_delta_td = K_total * T_unit
    score_lb             = lower_bound_delta_td  # optimistic: N_fa = 0

    return MetricResult(
        delta_t_d=delta_t_d,
        N_fa=N_fa,
        P_det=P_det,
        score=score,
        lower_bound_delta_td=lower_bound_delta_td,
        score_lb=score_lb,
    )
