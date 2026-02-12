
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Tuple
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
    Sequential Probability Ratio Test (SPRT) detector for chatter identification.

    This class implements the SPRT algorithm to sequentially analyze a stream of
    likelihood ratio indicators and make binary decisions (chatter vs. free state)
    based on cumulative log-likelihood ratios.

    Attributes:
        llr_model (LLRModel): Model that computes log-likelihood ratios from observations.
        config (SPRTConfig): Configuration parameters including decision thresholds (a, b)
                             and reset behavior.

    Methods:
        run(H_seq: Iterable[float]) -> SPRTResult:
            Executes the SPRT algorithm on a sequence of H_n indicators.

            Args:
                H_seq (Iterable[float]): Sequence of H_n indicator values to analyze.

            Returns:
                SPRTResult: Contains the final decision state ("free", "chatter", or
                           "indeterminado"), the index where decision was made,
                           cumulative likelihood ratio history, and threshold values.

            Algorithm:
                - Sequentially accumulates log-likelihood ratios from observations
                - Compares cumulative sum S against thresholds a (lower) and b (upper)
                - When S <= a: declares "free" state, optionally resets S if configured
                - When S >= b: declares "chatter" state
                - Continues processing entire sequence; early stopping can be enabled
    """

    llr_model: LLRModel
    config: SPRTConfig

    def run(self, H_seq: Iterable[float]) -> SPRTResult:
        """
        Execute the Sequential Probability Ratio Test (SPRT) on a sequence of observations.
        This method processes a sequence of observation values and performs sequential
        hypothesis testing to detect chatter conditions. The test maintains a cumulative
        log-likelihood ratio (S) and terminates when it crosses defined thresholds.
        Args:
            H_seq (Iterable[float]): An iterable sequence of observation values to be tested.
        Returns:
            SPRTResult: An object containing:
                - final_state (str): The final decision state ("free", "chatter", or "indeterminado").
                - decision_index (int): The index in the sequence where the decision was made (-1 if indeterminate).
                - S_history (np.ndarray): The complete history of cumulative log-likelihood ratio values.
                - a (float): The lower threshold for rejecting the alternative hypothesis.
                - b (float): The upper threshold for accepting the alternative hypothesis.
        Notes:
            - The test starts in "indeterminado" state and continues until a decision boundary is crossed.
            - If S <= a, the state transitions to "free" and may reset S if config.reset_on_H0 is True.
            - If S >= b, the state transitions to "chatter".
            - Early termination with break can be implemented if early detection is desired.
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
    Configuration class for Maximum Entropy Sequential Probability Ratio Test (MaxEnt SPRT) detector.
    This class defines the hyperparameters used to control the behavior of the MaxEnt SPRT
    detection algorithm.
    Attributes:
        alpha (float): Type I error rate (false positive rate). Default is 0.01.
            Represents the probability of rejecting the null hypothesis when it is true.
        beta (float): Type II error rate (false negative rate). Default is 0.01.
            Represents the probability of failing to reject the null hypothesis when it is false.
        reset_on_H0 (bool): Flag to reset the detector state when null hypothesis is accepted.
            Default is True. When True, the detector resets its internal state after accepting H0,
            allowing for fresh detection cycles.
    """

    alpha: float = 0.01
    beta: float = 0.01
    reset_on_H0: bool = True

@dataclass
class MaxEntSPRTDetector:
    """
    MaxEnt-SPRT Detector for chatter identification using entropy-based features.
    This class implements a Sequential Probability Ratio Test (SPRT) detector that uses
    Maximum Entropy (MaxEnt) models trained on entropy features extracted from vibration
    signals. The detector supports both offline training and online detection workflows.

    Attributes:
        models (MaxEntModels | None):
            Trained MaxEnt models containing p0(H) and p1(H) distributions.
            Initialized to None and set during fit_offline_* methods.
        config (MaxEntSPRTConfig):
            High-level configuration parameters for SPRT including alpha, beta,
            and reset behavior. Defaults to MaxEntSPRTConfig().
        estimator (EntropyEstimator):
            Entropy estimation method used during training and detection.
            Defaults to GaussianMaxEntEstimator.
        H_free (np.ndarray | None):
            Diagnostic variable storing entropy values computed from free-chatter signals
            during offline training. Used for post-training analysis.
        H_chat (np.ndarray | None):
            Diagnostic variable storing entropy values computed from chatter signals
            during offline training. Used for post-training analysis.
        t_mid_free (np.ndarray | None):
            Diagnostic variable storing mean segment times from free-chatter training signals.
        t_mid_chat (np.ndarray | None):
            Diagnostic variable storing mean segment times from chatter training signals.

    Methods:
        _build_sprt_config() -> SPRTConfig:
            Constructs an SPRTConfig object from high-level configuration parameters.
        _check_models() -> MaxEntModels:
            Validates that models are trained before detection operations.
        fit_offline_from_opr(opr_free, opr_t_free, opr_chat, opr_t_chat, N_seg) -> MaxEntSPRTDetector:
            Trains MaxEnt models directly from pre-computed Order-Preserving
            Representation (OPR) features labeled as either free-chatter or chatter.
            Returns self for method chaining.
        fit_offline_from_signals(y_free, t_free, y_chat, t_chat, rpm, ratio_sampling, N_seg) -> MaxEntSPRTDetector:
            High-level training wrapper that extracts OPR features from raw vibration signals
            and trains MaxEnt models. Handles automatic sampling rate computation.
            Returns self for method chaining.
        detect_from_H_seq(H_seq) -> SPRTResult:
            Executes SPRT detection on a pre-computed entropy sequence using trained models.
            Returns SPRTResult with final state and likelihood ratio history.
        detect_online_from_signal(y_online, t_online, rpm, ratio_sampling, N_seg) -> Tuple[SPRTResult, np.ndarray, np.ndarray]:
            Executes the complete online detection pipeline on new vibration signals,
            including OPR extraction, segmentation, entropy computation, and SPRT execution.
            Returns detection result, entropy sequence, and segment timings.

    Design Pattern:
        This class uses a combination of high-level and low-level APIs:
        - High-level: fit_offline_from_signals, detect_online_from_signal
        - Low-level: fit_offline_from_opr, detect_from_H_seq
        Users can either use the high-level methods for end-to-end pipelines or
        use low-level methods for more granular control over intermediate computations.
    Notes:
        x
    """

    models: MaxEntModels | None = None
    config: MaxEntSPRTConfig = field(default_factory=MaxEntSPRTConfig)
    estimator: EntropyEstimator = field(default_factory=GaussianMaxEntEstimator)

    # Variables de diagnóstico offline (opcionales)
    H_free: np.ndarray | None = field(default=None, init=False)
    H_chat: np.ndarray | None = field(default=None, init=False)
    t_mid_free: np.ndarray | None = field(default=None, init=False)
    t_mid_chat: np.ndarray | None = field(default=None, init=False)

    def _build_sprt_config(self) -> SPRTConfig:
        """
        Build and return a SPRT (Sequential Probability Ratio Test) configuration object.
        Constructs an SPRTConfig instance by extracting relevant parameters from the
        detector's configuration settings. This configuration is used to initialize
        or configure the Sequential Probability Ratio Test with the appropriate
        alpha, beta, and reset behavior parameters.
        Returns:
            SPRTConfig: A configuration object for SPRT containing:
                - alpha: The significance level (Type I error rate)
                - beta: The power parameter (Type II error rate)
                - reset_on_H0: Boolean flag indicating whether to reset the test upon accepting the null hypothesis
        """

        return SPRTConfig(
            alpha=self.config.alpha,
            beta=self.config.beta,
            reset_on_H0=self.config.reset_on_H0,
        )

    def _check_models(self) -> MaxEntModels:
        """
        Verifies that the MaxEnt models have been trained.

        Validates that the models attribute is not None, ensuring that
        the MaxEnt models have been initialized and trained previously.

            MaxEntModels: The trained MaxEnt models.

            RuntimeError: If the MaxEnt models have not been trained.
                          fit_offline_* must be called first.
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

        Parameters
        ----------
        opr_free : np.ndarray
            Operational deflection shape data for the free (non-chatter) state.
        opr_t_free : np.ndarray
            Time values corresponding to the free operational deflection shape data.
        opr_chat : np.ndarray
            Operational deflection shape data for the chatter state.
        opr_t_chat : np.ndarray
            Time values corresponding to the chatter operational deflection shape data.
        N_seg : int
            Number of segments to divide the operational data into for training.

        Returns
        -------
        MaxEntSPRTDetector
            Returns self with fitted models and entropy characteristics stored as instance attributes.
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
        Parameters
        ----------
        y_free : np.ndarray
            Raw vibration signal samples during free (non-chatter) operation.
        t_free : np.ndarray
            Time vector corresponding to y_free samples.
        y_chat : np.ndarray
            Raw vibration signal samples during chatter operation.
        t_chat : np.ndarray
            Time vector corresponding to y_chat samples.
        rpm : float
            Spindle rotation speed in revolutions per minute.
        ratio_sampling : float
            Sampling frequency ratio relative to the spindle rotation frequency.
            Determines the resampling rate: fs = ratio_sampling * (rpm / 60).
        N_seg : int
            Number of segments to divide the data into for analysis.
        Returns
        -------
        MaxEntSPRTDetector
            Returns self with fitted parameters for chatter detection.
        Notes
        -----
        The method internally resamples the input signals using the sample_opr function
        before fitting the detector.
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
        Detect chatter from a sequence of Hurst exponents using Sequential Probability Ratio Test.

        Args:
            H_seq (Iterable[float]): An iterable sequence of Hurst exponent values to analyze.

        Returns:
            SPRTResult: The result of the Sequential Probability Ratio Test containing the
                        detection decision and test statistics.

        Raises:
            ValueError: If models are not properly configured or initialized.
            TypeError: If H_seq is not iterable or contains non-float values.
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
    ) -> Tuple[SPRTResult, np.ndarray, np.ndarray]:
        """
        Detect chatter from an online signal using sequential probability ratio test (SPRT).
        Processes a real-time signal by computing its order-preserving representation (OPR),
        segmenting it, calculating entropy for each segment, and applying SPRT for chatter detection.
        Args:
            y_online (np.ndarray): Online signal samples (amplitude values).
            t_online (np.ndarray): Time vector corresponding to signal samples.
            rpm (float): Spindle speed in revolutions per minute.
            ratio_sampling (float): Sampling ratio relative to the fundamental frequency.
            N_seg (int): Number of segments to divide the OPR signal into.
        Returns:
            Tuple[SPRTResult, np.ndarray, np.ndarray]: A tuple containing:
                - SPRTResult: Sequential probability ratio test result with decision and likelihood ratio.
                - np.ndarray: Entropy values computed for each segment.
                - np.ndarray: Time midpoints of each segment.
        Raises:
            ValueError: If insufficient segments are generated from the online signal.
        """
        models = self._check_models()

        fr = rpm / 60.0
        fs = ratio_sampling * fr

        # 1) OPR
        opr_online, opr_t_online = sample_opr(y_online, t_online, fs=fs, fr=fr)

        # 2) Segmentation
        segments_online, segments_t_online = segment_opr(opr_online, opr_t_online, N_seg=N_seg)
        if len(segments_online) == 0:
            raise ValueError("Insufficient segments generated from the online signal.")

        # 3) Entropy per segment
        H_seq = entropy_from_segments(segments_online, estimator=self.estimator)
        t_mid_segments = np.array([np.mean(seg_t) for seg_t in segments_t_online])

        # 4) SPRT with OO design
        result = self.detect_from_H_seq(H_seq=H_seq)
        return result, H_seq, t_mid_segments