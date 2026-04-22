from __future__ import annotations
# Comentario: abstracciones TF (DIP + Strategy para STFT/SSQ-STFT)
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np




try:
    from ..utils.ssq_core import ssq_stft_T  
except Exception as _e:  # noqa: N816
    ssq_stft_T = None

from scipy.signal import stft, get_window

class TimeFrequencyTransform(ABC):
    """
    Apply time-frequency transformation to the input signal.
    Args:
        x (np.ndarray): Input signal in the time domain.
        fs (float): Sampling frequency in Hz.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - S (np.ndarray): Time-frequency representation (spectrogram/scalogram).
            - t (np.ndarray): Time axis values.
            - f (np.ndarray): Frequency axis values.
    Raises:
        NotImplementedError: This is an abstract method and must be implemented by subclasses.
    """

    @abstractmethod
    def transform(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

class STFT(TimeFrequencyTransform):
    """
    Short-Time Fourier Transform (STFT) for time-frequency analysis.
    A time-frequency transform that decomposes a signal into its frequency components
    over time using a sliding window approach.
    Attributes:
        win_length (int): Length of the window in samples.
        hop_length (int): Number of samples between successive windows.
        n_fft (int): Length of the FFT (Fast Fourier Transform).
        window (str | tuple[str, float]): Window function to apply. Can be a string
            (e.g., "hann") or a tuple of (window_name, parameter) for parameterized windows.
    Methods:
        transform(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Computes the STFT of the input signal.
            Args:
                x (np.ndarray): Input signal as a 1D array.
                fs (float): Sampling frequency of the signal in Hz.
            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    - Zxx (np.ndarray): Complex-valued STFT of shape (n_fft/2+1, num_frames).
                    - t (np.ndarray): Time bins corresponding to the center of each window.
                    - f (np.ndarray): Frequency bins in Hz.
    """

    def __init__(self, win_length: int, hop_length: int, n_fft: int, window: str | tuple[str, float] = "hann"):
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.window = window

    def transform(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Short-Time Fourier Transform (STFT) of the input signal.
        Parameters
        ----------
        x : np.ndarray
            Input signal as a 1-D array.
        fs : float
            Sampling frequency of the input signal in Hz.
        Returns
        -------
        Zxx : np.ndarray
            STFT of the input signal. A 2-D array where rows represent frequency bins
            and columns represent time frames.
        t : np.ndarray
            Array of time values corresponding to the STFT frames.
        f : np.ndarray
            Array of frequency values in Hz corresponding to the STFT frequency bins.
        """
        f, t, Zxx = stft(x, fs=fs, window=get_window(self.window, self.win_length),
                         nperseg=self.win_length, noverlap=self.win_length - self.hop_length, nfft=self.n_fft,
                         boundary=None, padded=False)

        return Zxx, t, f

class SSQ_STFT(TimeFrequencyTransform):
    """
    Synchrosqueezed Short-Time Fourier Transform (SSQ-STFT) time-frequency transformer.
    This class implements the SSQ-STFT transform, which provides improved time-frequency
    resolution by applying a reassignment procedure to the STFT. It extends the
    TimeFrequencyTransform base class.
    Attributes:
        win_length (int): Length of the analysis window in samples.
        hop_length (int): Number of samples between successive frames.
        n_fft (int): Length of the FFT.
        sigma (float): Standard deviation parameter for the Gaussian window,
                       used to compute sigma_samples as win_length / sigma.
    Raises:
        ImportError: If ssqueezepy is not installed.
    """
    def __init__(self, win_length: int, hop_length: int, n_fft: int, sigma: float):
        if ssq_stft_T is None:
            raise ImportError("SSQ_STFT not available")
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.sigma = float(sigma)

    def transform(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Synchrosqueezed FFT (SSQ) transform of an input signal.
        Applies a Gaussian-windowed Short-Time Fourier Transform (STFT) with 
        synchrosqueezing to the input signal. Returns the reassigned time-frequency 
        representation along with the standard STFT magnitude and associated parameters.
        Parameters
        ----------
        x : np.ndarray
            Input signal to be transformed. Should be a 1D array of samples.
        fs : float
            Sampling frequency of the input signal in Hz.
        Returns
        -------
        Tsx : np.ndarray
            Reassigned time-frequency representation (synchrosqueezed STFT).
        Sx : np.ndarray
            Magnitude of the Short-Time Fourier Transform (STFT).
        t : np.ndarray
            Time axis vector aligned with hop length increments.
        f : np.ndarray
            Frequency axis vector from 0 to fs/2 Hz.
        w : np.ndarray
            Gaussian window coefficients used in the transform.
        dWx : np.ndarray
            Frequency derivative of the windowed STFT.
        Notes
        -----
        - Uses a Gaussian window with standard deviation proportional to win_length/sigma.
        - Time values correspond to STFT frame positions based on hop_length.
        - Frequency resolution extends up to the Nyquist frequency (fs/2).
        - The reassigned representation (Tsx) is prioritized as the primary output.
        """

        sigma_samples = self.win_length / self.sigma # Convert sigma to samples for window generation
        w = get_window(("gaussian", sigma_samples), self.win_length) # Generate Gaussian window
        Tsx, Sx, _, _, w, dWx = ssq_stft_T(x, window=w, n_fft=self.n_fft, win_len=self.win_length, hop_len=self.hop_length, fs=fs, get_dWx=True, get_w=True)

        # Generate time and frequency vectors based on the STFT output dimensions and sampling frequency
        t = np.arange(Sx.shape[1]) * ((self.win_length - (self.win_length - self.hop_length))/ fs)
        t = t + (self.win_length /fs) # Center time values on the window
        f = np.linspace(0, fs/2, Sx.shape[0], endpoint=True)

        return Tsx, Sx, t, f, w, dWx
