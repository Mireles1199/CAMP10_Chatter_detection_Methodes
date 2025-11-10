from __future__ import annotations
# Comentario: tubería de detección con inyección de dependencias (DIP) y SRP
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np

from .tf_transformers import TimeFrequencyTransform
from .detection_strategies import DetectionRule
from ..utils.tf_windows import WindowExtractor
from ..utils.decorators import ensure_1d_array, timeit

@dataclass(frozen=True)
class PipelineConfig:
    # Comentario: parámetros de TF y ventanas
    fs: float
    win_length_ms: float
    hop_ms: float
    n_fft: int
    Ai_length: int = 4
    mode: str = "causal_inclusive"  

class ChatterPipeline:
    # Comentario: aplica SOLID: SRP (una tubería), OCP (estrategias pluggables), DIP (inyecta interfaces)

    def __init__(self, transformer: TimeFrequencyTransform, detector: DetectionRule, config: PipelineConfig):
        self._transformer = transformer
        self._detector = detector
        self._config = config

    # Comentario: encapsulamiento con propiedades de solo lectura
    @property
    def transformer(self) -> TimeFrequencyTransform:
        return self._transformer

    @property
    def detector(self) -> DetectionRule:
        return self._detector

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @timeit
    @ensure_1d_array
    def run(self, x: np.ndarray, *, return_TF: bool = True) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # Comentario: convierte ms->muestras
        fs = float(self._config.fs)
        win_length = int(round(self._config.win_length_ms * 1e-3 * fs))
        hop_length = int(round(self._config.hop_ms * 1e-3 * fs))
        if win_length < 3 or hop_length < 1:
            raise ValueError("ventanas inválidas")
            # Reminder This time-resolved framework

        # Comentario: transformada TF (Strategy)
        if self._transformer.__class__.__name__ == "STFT":
            Sx, t, f = self._transformer.transform(x, fs=fs)
            Tsx = None
            S1 = Sx
            
        if self._transformer.__class__.__name__ == "SSQ_STFT":
            Tsx, Sx, t, f = self._transformer.transform(x, fs=fs)
            S1 = Tsx

        # Comentario: extraer subventanas locales
        A_i, t_i = WindowExtractor.extract_local_windows(S1, K=self._config.Ai_length, time_vector=t, mode=self._config.mode)

        # Comentario: SVD por ventana y primer valor singular
        U, D, Vh = WindowExtractor.compute_svd(A_i, ensure_real=True)
        d1 = D[:, 0]

        # Comentario: detección (Strategy)
        res = self._detector.detect(d1=d1, t=t, idx_stable=None)

        # Comentario: preparar salidas (mantener compatibilidad)
        return Tsx, Sx, fs, t, A_i, t_i, D, d1, res
    
    
    
