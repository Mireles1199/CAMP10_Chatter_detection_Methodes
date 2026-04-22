from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Tuple, Dict, Any, Optional
import numpy as np

from .tf_transformers import TimeFrequencyTransform
from .detection_strategies import DetectionRule
from ..utils.tf_windows import WindowExtractor
from ..utils.decorators import ensure_1d_array, timeit

@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration class for the chatter detection pipeline.
    Attributes:
        fs (float): Sampling frequency in Hz.
        win_length_ms (float): Window length in milliseconds for STFT analysis.
        hop_ms (float): Hop length in milliseconds between consecutive frames.
        n_fft (int): Number of FFT points for frequency resolution.
        Ai_length (int): Length of the analysis interval in seconds. Defaults to 4.
        mode (str): Processing mode for the pipeline. Defaults to "causal_inclusive".
    """
    fs: float
    win_length_ms: float
    hop_ms: float
    n_fft: int
    Ai_length: int = 4
    mode: str = "causal_inclusive"

class ChatterPipeline:
    """
    ChatterPipeline: Time-Frequency Analysis and Chatter Detection Pipeline
    A comprehensive signal processing pipeline that combines time-frequency transformations
    with chatter detection rules. This class implements a strategy pattern to support
    multiple transformation methods (STFT, SSQ_STFT) and detection algorithms.
    Attributes:
        transformer (TimeFrequencyTransform): Strategy for time-frequency transformation.
        detector (DetectionRule): Strategy for chatter detection rule application.
        config (PipelineConfig): Configuration parameters including sampling frequency,
                                window lengths, and extraction modes.
    Methods:
        run(x, *, return_TF=True) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray,
                                            np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]
            Executes the complete chatter detection pipeline on input signal.
            Args:
                x (np.ndarray): 1D input signal array.
                return_TF (bool, optional): Flag to return time-frequency representation.
                                           Defaults to True.
            Returns:
                Tsx (np.ndarray or None): Synchrosqueezing transform (None if using STFT).
                Sx (np.ndarray): STFT magnitude spectrogram.
                fs (float): Sampling frequency in Hz.
                t (np.ndarray): Time vector.
                A_i (np.ndarray): Local window submatrices extracted from TF representation.
                t_i (np.ndarray): Time vector for local windows.
                D (np.ndarray): Singular values matrix from SVD decomposition.
                d1 (np.ndarray): First singular values (primary detection feature).
                res (Dict[str, Any]): Detection results from detector strategy.
                w (np.ndarray): Phase velocity (SSQ_STFT only).
                dWx (np.ndarray): Phase derivative (SSQ_STFT only).
            Raises:
                ValueError: If window length < 3 samples or hop length < 1 sample.
            Process:
                1. Convert millisecond parameters to sample counts
                2. Apply time-frequency transformation (STFT or SSQ_STFT)
                3. Extract local windows using specified windowing strategy
                4. Compute SVD decomposition per window
                5. Detect chatter using first singular value vector
                6. Return comprehensive results for analysis and visualization
    """

    def __init__(self, transformer: TimeFrequencyTransform, detector: DetectionRule, config: PipelineConfig):
        self._transformer = transformer
        self._detector = detector
        self._config = config

    @property
    def transformer(self) -> TimeFrequencyTransform:
        """
        Get the time-frequency transformer instance.
        Returns
        -------
        TimeFrequencyTransform
            The time-frequency transformer object used for converting time-domain
            signals into time-frequency representations during chatter detection analysis.
        """
        return self._transformer

    @property
    def detector(self) -> DetectionRule:
        """
        Retrieve the detection rule used for chatter identification.
        Returns
        -------
        DetectionRule
            The detection rule object that defines the criteria and methods
            for identifying chatter in the milling process data.
        """

        return self._detector

    @property
    def config(self) -> PipelineConfig:
        """
        Retrieve the pipeline configuration object.
        Returns
        -------
        PipelineConfig
            The configuration object containing settings and parameters
            for the chatter detection pipeline.
        """
        return self._config

    @timeit
    @ensure_1d_array
    def run(self, x: np.ndarray, signal_time: np.ndarray, *, return_TF: bool = True
            ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute the chatter detection pipeline on input signal.
        This method applies a time-frequency transformation (STFT or SSQ_STFT),
        extracts local windows from the resulting representation, computes Singular
        Value Decomposition (SVD) for each window, and performs chatter detection
        based on the first singular value.
        Parameters
        ----------
        x : np.ndarray
            Input signal to analyze for chatter detection.
        signal_time : np.ndarray
            Time vector corresponding to the input signal samples.
        return_TF : bool, optional
            Whether to return time-frequency representations (default: True).
            Currently not used in implementation.
        Returns
        -------
        Tsx : np.ndarray or None
            Synchrosqueezed STFT matrix if SSQ_STFT transformer is used, None otherwise.
        Sx : np.ndarray
            STFT magnitude spectrogram (time x frequency).
        fs : float
            Sampling frequency in Hz.
        t : np.ndarray
            Time vector corresponding to STFT frames.
        A_i : np.ndarray
            Extracted local windows from the time-frequency representation.
        t_i : np.ndarray
            Time indices corresponding to each local window.
        D : np.ndarray
            Singular values matrix from SVD decomposition (windows x singular values).
        d1 : np.ndarray
            First singular value for each window, used for chatter detection.
        res : Dict[str, Any]
            Detection results containing chatter indicators and metrics.
        w : np.ndarray or None
            Instantaneous frequencies (returned by transformer if available).
        dWx : np.ndarray or None
            Instantaneous frequency derivatives (returned by transformer if available).
        Raises
        ------
        ValueError
            If window length is less than 3 samples or hop length is less than 1 sample.
        Notes
        -----
        The transformer strategy (STFT vs SSQ_STFT) determines the primary representation
        used for window extraction and analysis.
        """

        fs = float(self._config.fs)
        win_length = int((self._config.win_length_ms * 1e-3 * fs))
        hop_length = int((self._config.hop_ms * 1e-3 * fs))
        if win_length < 3 or hop_length < 1:
            raise ValueError(": window length must be at least 3 samples and hop length must be at least 1 sample")


        # a TF (Strategy)
        if self._transformer.__class__.__name__ == "STFT":
            Sx, t, f = self._transformer.transform(x, fs=fs)
            Tsx = None
            S1 = Sx

        if self._transformer.__class__.__name__ == "SSQ_STFT":
            Tsx, Sx, t, f, w, dWx = self._transformer.transform(x, fs=fs)
            S1 = Tsx


        # Extract local windows from time-frequency representation
        A_i, t_i = WindowExtractor.extract_local_windows(S1, K=self._config.Ai_length, time_vector=t, mode=self._config.mode)
        t_i = np.asarray(t_i)
        t_i = t_i + signal_time[0]  # Adjust window time indices to match original signal time
        # SVD per window and first singular value
        U, D, Vh = WindowExtractor.compute_svd(A_i, ensure_real=True)
        d1 = D[:, 0]

        # Detection (Strategy)
        res = self._detector.detect(d1=d1, t=t, idx_stable=None)

        return Tsx, Sx, fs, t, A_i, t_i, D, d1, res, w, dWx

