
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, math
from typing import Iterable

from .llr import LLRModel

@dataclass
class SPRTConfig:
    """
    Configuration of the Sequential Probability Ratio Test.

    This object stores the statistical error tolerances that define the two
    decision thresholds used by the sequential test. It does not contain any
    signal-processing logic; it only describes how strict the test should be
    when deciding between the stable and chatter hypotheses.
    """
    alpha: float = 0.01
    """Maximum Type I error probability (false-alarm rate): probability of declaring chatter when the process is actually stable."""
    beta: float = 0.01
    """Maximum Type II error probability (missed-detection rate): probability of failing to declare chatter when chatter is present."""
    reset_on_H0: bool = True
    """If ``True``, the cumulative SPRT statistic is reset to zero after crossing the lower threshold and accepting the stable hypothesis."""

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError("alpha y beta deben estar en (0, 1).")

    @property
    def a(self) -> float:
        """
        Lower SPRT decision threshold.

        Returns:
            float: Log-threshold associated with accepting hypothesis H0
            (stable, chatter-free regime).
        """
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def b(self) -> float:
        """
        Upper SPRT decision threshold.

        Returns:
            float: Log-threshold associated with accepting hypothesis H1
            (chatter regime).
        """
        return math.log((1.0 - self.beta) / self.alpha)

@dataclass
class SPRTResult:
    """
    Result of running SPRT on a sequence of indicator values.

    This dataclass captures both the final decision and the full cumulative
    statistic history so downstream code can inspect not only whether chatter
    was detected, but also how the evidence evolved over time.
    """
    final_state: str           # "free", "chatter", "indeterminado"
    """Final categorical decision: ``"free"``, ``"chatter"``, or ``"indeterminado"`` when no bound was crossed."""
    decision_index: int        # índice del segmento donde decide (-1 si no decide)
    """Index of the segment where the latest decisive threshold crossing occurred. Remains ``-1`` if no threshold is ever crossed."""
    S_history: np.ndarray      # trayectoria de S_n
    """One-dimensional array of the running cumulative log-likelihood ratio statistic, one entry per processed segment."""
    a: float                   # umbral inferior
    """Lower decision threshold used for stable-state acceptance."""
    b: float
    """Upper decision threshold used for chatter-state acceptance."""

@dataclass
class SequentialProbabilityRatioTest:
    """
    Stateful sequential hypothesis-testing engine.

    The engine is intentionally minimal: it only knows how to accumulate an LLR
    model and how to compare the cumulative statistic against thresholds defined
    by :class:`SPRTConfig`. It is agnostic to how features were extracted,
    which probability model produced the LLR, or whether the inputs come from
    vibration, force, or any other measured signal.
    """
    llr_model: LLRModel
    """Strategy object that converts one scalar observation into an incremental log-likelihood ratio contribution."""
    config: SPRTConfig
    """Statistical configuration determining the lower/upper decision thresholds and the reset policy."""

    def run(self, H_seq: Iterable[float]) -> SPRTResult:
        """
        Run the sequential test over an ordered indicator sequence.

        :param H_seq: Ordered scalar observations, usually one entropy-like indicator value per signal segment, consumed in time order.

        Returns:
            SPRTResult: Final decision summary containing the final state,
            decision index, cumulative statistic history, and the thresholds
            used during the test.

        Decision behavior:
        When the cumulative statistic falls below ``a``, the test favors the
        stable regime. When it rises above ``b``, the test favors chatter. If
        ``reset_on_H0`` is enabled, the statistic is reset to zero after a lower
        threshold crossing so evidence can accumulate again on later segments.
        """
        H_list = list(H_seq)
        S_hist = np.zeros(len(H_list), dtype=float)
        S = 0.0
        state = "indeterminado"
        idx_decision = -1

        a = self.config.a
        b = self.config.b

        for i, h_obs in enumerate(H_list):
            S += self.llr_model.llr(h_obs)
            S_hist[i] = S

            if S <= a:
                state = "free"
                idx_decision = i
                if self.config.reset_on_H0:
                    S = 0.0

            if S >= b:
                state = "chatter"
                idx_decision = i
                # Se podría parar aquí con break si se quisiera detección temprana.

        return SPRTResult(
            final_state=state,
            decision_index=idx_decision,
            S_history=S_hist,
            a=a,
            b=b,
        )