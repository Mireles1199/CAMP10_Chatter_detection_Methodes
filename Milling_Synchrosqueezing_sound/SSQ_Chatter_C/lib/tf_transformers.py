from __future__ import annotations
# Comentario: abstracciones TF (DIP + Strategy para STFT/SSQ-STFT)
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

try:
    from ssqueezepy import ssq_stft  # Comentario: dependencia opcional
except Exception as _e:  # noqa: N816
    ssq_stft = None

from scipy.signal import stft, get_window

class TimeFrequencyTransform(ABC):
    # Comentario: interfaz de transformadas TF
    @abstractmethod
    def transform(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Comentario: devuelve (S1, t, f)
        raise NotImplementedError

class STFT(TimeFrequencyTransform):
    # Comentario: implementación STFT estándar
    def __init__(self, win_length: int, hop_length: int, n_fft: int, window: str | tuple[str, float] = "hann"):
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.window = window

    def transform(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, t, Zxx = stft(x, fs=fs, window=get_window(self.window, self.win_length), nperseg=self.win_length, noverlap=self.win_length - self.hop_length, nfft=self.n_fft, boundary=None, padded=False)
        return Zxx, t, f

class SSQ_STFT(TimeFrequencyTransform):
    # Comentario: implementación SSQ-STFT (si está disponible)
    def __init__(self, win_length: int, hop_length: int, n_fft: int, sigma: float):
        if ssq_stft is None:
            raise ImportError("ssqueezepy no disponible; instale ssqueezepy para SSQ-STFT")
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.sigma = float(sigma)

    def transform(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sigma_samples = self.win_length / self.sigma
        w = get_window(("gaussian", sigma_samples), self.win_length)
        Tsx, Sx, *_ = ssq_stft(x, window=w, n_fft=self.n_fft, win_len=self.win_length, hop_len=self.hop_length, fs=fs, get_dWx=True, get_w=True)
        # Comentario: vector de tiempo acorde al hop
        t = np.arange(Sx.shape[1]) * (self.hop_length / fs)
        f = np.linspace(0, fs/2, Sx.shape[0], endpoint=True)
        # Comentario: devolvemos Tsx como S1 al priorizar la versión reasignada
        return Tsx, Sx, t, f
