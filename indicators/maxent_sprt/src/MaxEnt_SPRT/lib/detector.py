
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Optional
import numpy as np

from ..models.maxent import MaxEntModels
from .llr import GaussianIndicatorLLR, LLRModel
from .sprt import SPRTConfig, SPRTResult
from .entropy import entropy_from_segments, EntropyEstimator, GaussianMaxEntEstimator
from ..utils.opr import sample_opr, segment_opr
from ..lib.offline import offline_train_maxent_sprt



@dataclass
class SequentialProbabilityRatioTest:
    """
    Stateful Sequential Probability Ratio Test engine used by the detector.

    The object consumes an indicator sequence one sample at a time through the
    cumulative statistic:

    ``S_k = sum_i llr(h_i)``.

    Two decision bounds are used: a lower bound ``a`` for the stable regime and
    an upper bound ``b`` for chatter. The returned history allows post-analysis
    of how close the process was to each decision boundary over time.
    """

    llr_model: LLRModel
    """Model that maps one scalar indicator value to one incremental log-likelihood contribution."""
    config: SPRTConfig
    """Object containing the statistical thresholds and reset policy used during the sequential decision process."""

    def run(self, H_seq: Iterable[float]) -> SPRTResult:
        """
        Execute the Sequential Probability Ratio Test (SPRT) on a sequence of observations.

        This method processes a sequence of observation values and performs sequential
        hypothesis testing to detect chatter conditions. The test maintains a cumulative
        log-likelihood ratio (S) and terminates when it crosses defined thresholds.

        :param H_seq: Ordered sequence of scalar indicator values, typically one entropy value per processed segment.

        Returns:
            SPRTResult: Final decision state, decision index, cumulative statistic
            history, and threshold values.

        Notes:
            The test starts in ``"indeterminado"`` state. Crossing the lower
            threshold favors the stable regime, while crossing the upper
            threshold favors chatter. The current implementation keeps scanning
            the full sequence instead of exiting early on the first detection.
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

@dataclass
class MaxEntSPRTConfig:
    """
    High-level detector configuration shared by offline and online stages.

    ``alpha`` and ``beta`` control SPRT error probabilities, while
    ``reset_on_H0`` defines whether the cumulative statistic is reset after a
    stable decision. Keeping this policy explicit makes detector behavior easier
    to compare across experiments.
    """

    alpha: float = 0.01
    """Target false-alarm probability for chatter detection."""
    beta: float = 0.01
    """Target missed-detection probability for chatter detection."""
    reset_on_H0: bool = True
    """Whether the cumulative SPRT statistic is reset after a lower-threshold crossing."""

@dataclass
class MaxEntSPRTDetector:
    """
    End-to-end detector that combines MaxEnt modeling and SPRT decision logic.

    The detector exposes offline fitting methods, online detection methods, and
    diagnostic arrays useful for plotting, reproducibility, and threshold
    troubleshooting. The same estimator and configuration are reused across
    stages so the modeling assumptions remain consistent from training to
    deployment.

    """

    models: MaxEntModels | None = None
    """Trained Gaussian pair for the stable and chatter regimes. Remains ``None`` until an offline fit method runs."""
    config: MaxEntSPRTConfig = field(default_factory=MaxEntSPRTConfig)
    """High-level detector settings controlling the SPRT behavior."""
    estimator: EntropyEstimator = field(default_factory=GaussianMaxEntEstimator)
    """Segment-to-entropy transformation used consistently in offline fitting and online inference."""

    # Variables de diagnóstico offline (opcionales)
    H_free: np.ndarray | None = field(default=None, init=False)
    """Offline entropy sequence computed from the stable training data."""
    H_chat: np.ndarray | None = field(default=None, init=False)
    """Offline entropy sequence computed from the chatter training data."""
    t_mid_free: np.ndarray | None = field(default=None, init=False)
    """Time midpoints of the stable training segments."""
    t_mid_chat: np.ndarray | None = field(default=None, init=False)
    """Time midpoints of the chatter training segments."""

    def _build_sprt_config(self) -> SPRTConfig:
        """
        Build the low-level SPRT configuration from the detector settings.

        Returns:
            SPRTConfig: Configuration object with alpha, beta, and reset policy.
        """

        return SPRTConfig(
            alpha=self.config.alpha,
            beta=self.config.beta,
            reset_on_H0=self.config.reset_on_H0,
        )

    def _check_models(self) -> MaxEntModels:
        """
        Return the trained models or raise if training has not happened yet.

        Returns:
            MaxEntModels: The trained probability models.

        Raises:
            RuntimeError: If ``fit_offline_from_opr`` or ``fit_offline_from_signals``
            has not been called yet.
        """

        if self.models is None:
            raise RuntimeError("MaxEnt models are not trained. Call fit_offline_* first.")
        return self.models

    # ------------------ OFFLINE ------------------

    def fit_offline_from_opr(
        self,
        opr_free: np.ndarray,
        opr_t_free: np.ndarray,
        opr_chat: np.ndarray,
        opr_t_chat: np.ndarray,
        N_seg: int,
    ) -> "MaxEntSPRTDetector":
        """
        Fit the MaxEnt SPRT detector offline using operational deflection shape (OPR) data.

        This method trains the detector models on provided free and chatter operational data,
        segmented into N_seg parts. It extracts entropy characteristics and time midpoints
        for both state conditions.

        :param opr_free: OPR-resampled signal representing the stable reference condition.
        :param opr_t_free: Time vector aligned with ``opr_free``.
        :param opr_chat: OPR-resampled signal representing the chatter reference condition.
        :param opr_t_chat: Time vector aligned with ``opr_chat``.
        :param N_seg: Number of OPR samples per segment used to build the offline training windows.

        Returns:
            MaxEntSPRTDetector: ``self`` with trained models and diagnostic
            arrays populated.
        """

        models, H_free, H_chat, t_mid_free, t_mid_chat = offline_train_maxent_sprt(
            opr_free=opr_free,
            opr_chat=opr_chat,
            opr_t_free=opr_t_free,
            opr_t_chat=opr_t_chat,
            N_seg=N_seg,
            estimator=self.estimator,
        )
        self.models = models
        self.H_free = H_free
        self.H_chat = H_chat
        self.t_mid_free = t_mid_free
        self.t_mid_chat = t_mid_chat
        return self

    def fit_offline_from_signals(
        self,
        y_free: np.ndarray,
        t_free: np.ndarray,
        y_chat: np.ndarray,
        t_chat: np.ndarray,
        rpm: float,
        ratio_sampling: float,
        N_seg: int,
    ) -> "MaxEntSPRTDetector":
        """
        Fit the detector offline using raw signal data from free and chatter conditions.

        This method processes raw vibration signals by resampling them according to
        the spindle rotation frequency and then fits the detector using the resampled
        operational parameter data.

        :param y_free: Raw signal samples measured under stable cutting conditions.
        :param t_free: Time vector aligned with ``y_free``.
        :param y_chat: Raw signal samples measured under chatter conditions.
        :param t_chat: Time vector aligned with ``y_chat``.
        :param rpm: Spindle speed in revolutions per minute used to derive the fundamental rotation frequency.
        :param ratio_sampling: Number of analysis samples per revolution, used to derive the OPR sampling frequency.
        :param N_seg: Number of OPR samples per entropy segment.

        Returns:
            MaxEntSPRTDetector: ``self`` after OPR resampling and offline model
            fitting.

        Notes:
            This method is a convenience wrapper around ``sample_opr`` followed
            by ``fit_offline_from_opr``.
        """
        fr = rpm / 60.0
        fs = ratio_sampling * fr

        opr_free, opr_t_free = sample_opr(y_free, t_free, fs=fs, fr=fr)
        opr_chat, opr_t_chat = sample_opr(y_chat, t_chat, fs=fs, fr=fr)

        return self.fit_offline_from_opr(
            opr_free=opr_free,
            opr_t_free=opr_t_free,
            opr_chat=opr_chat,
            opr_t_chat=opr_t_chat,
            N_seg=N_seg,
        )

    # ------------------ ONLINE / DETECCIÓN ------------------

    def detect_from_H_seq( self,
        H_seq: Iterable[float],
    ) -> SPRTResult:
        """
        Classify a precomputed entropy sequence using SPRT.

        This path is useful when entropy values are produced externally and only
        the sequential decision logic is needed.

        :param H_seq: Precomputed scalar indicator sequence, typically one entropy value per segment.

        Returns:
            SPRTResult: The result of the Sequential Probability Ratio Test containing the
                        detection decision and test statistics.

        Raises:
            RuntimeError: If MaxEnt models are not fitted yet.
        """
        models = self._check_models()
        llr_model = GaussianIndicatorLLR(models=models)
        sprt_config = self._build_sprt_config()
        sprt = SequentialProbabilityRatioTest(llr_model=llr_model, config=sprt_config)
        return sprt.run(H_seq=H_seq)

    def detect_online_from_signal(
        self,
        y_online: np.ndarray,
        t_online: np.ndarray,
        rpm: float,
        ratio_sampling: float,
        N_seg: int,
        fs: Optional[float] = None,
    ) -> Tuple[SPRTResult, np.ndarray, np.ndarray]:
        """
        Detect chatter from an online signal using sequential probability ratio test (SPRT).

        Processes a real-time signal by computing its order-preserving representation (OPR),
        segmenting it, calculating entropy for each segment, and applying SPRT for chatter detection.

        :param y_online: Raw online signal samples to classify.
        :param t_online: Time vector aligned with ``y_online``.
        :param rpm: Spindle speed in revolutions per minute used to derive the rotation frequency.
        :param ratio_sampling: Sampling ratio used to derive the target OPR sampling frequency when ``fs`` is not explicitly provided.
        :param N_seg: Number of OPR samples grouped into each entropy segment.
        :param fs: Explicit OPR sampling frequency in Hz. If ``None``, it is computed from ``ratio_sampling`` and ``rpm``.

        Returns:
            Tuple[SPRTResult, np.ndarray, np.ndarray]: Detection result, entropy
            sequence, and segment midpoint times.

        Raises:
            ValueError: If insufficient segments are generated from the online signal.
        """
        models = self._check_models()

        fr = rpm / 60.0

        if fs is None:
            fs = ratio_sampling * fr
        else:
            fs = fs

        # 1) OPR
        opr_online, opr_t_online = sample_opr(y_online, t_online, fs=fs, fr=fr)

        # 2) Segmentation
        segments_online, segments_t_online = segment_opr(opr_online, opr_t_online, N_seg=N_seg)
        if len(segments_online) == 0:
            raise ValueError("Insufficient segments generated from the online signal.")

        # 3) Entropy per segment
        H_seq = entropy_from_segments(segments_online, estimator=self.estimator)
        # t_mid_segments = np.array([np.mean(seg_t) for seg_t in segments_t_online])
        # Last time in segment
        t_mid_segments = np.array([seg_t[-1] for seg_t in segments_t_online])

        # 4) SPRT with OO design
        result = self.detect_from_H_seq(H_seq=H_seq)
        return result, H_seq, t_mid_segments