
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, math
from typing import Iterable

from .llr import LLRModel

@dataclass
class SPRTConfig:
    """
    Configuración del test SPRT:
    - alpha: prob. de falso positivo máxima.
    - beta: prob. de falso negativo máxima.
    - reset_on_H0: si True, resetea S_n a 0 cuando cae por debajo de a.
    """
    alpha: float = 0.01
    beta: float = 0.01
    reset_on_H0: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError("alpha y beta deben estar en (0, 1).")

    @property
    def a(self) -> float:
        """
        Umbral inferior: región de aceptación de H0 (chatter-free).
        """
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def b(self) -> float:
        """
        Umbral superior: región de aceptación de H1 (chatter).
        """
        return math.log((1.0 - self.beta) / self.alpha)

@dataclass
class SPRTResult:
    """
    Resultado del test secuencial SPRT sobre una secuencia H_n.
    """
    final_state: str           # "free", "chatter", "indeterminado"
    decision_index: int        # índice del segmento donde decide (-1 si no decide)
    S_history: np.ndarray      # trayectoria de S_n
    a: float                   # umbral inferior
    b: float

@dataclass
class SequentialProbabilityRatioTest:
    """
    Motor del SPRT: solo sabe:
    - cómo calcular LLR (objeto LLRModel)
    - qué umbrales usar (SPRTConfig)

    No sabe nada de MaxEnt, ni de señales, ni de PDFs concretas.
    """
    llr_model: LLRModel
    config: SPRTConfig

    def run(self, H_seq: Iterable[float]) -> SPRTResult:
        """
        Ejecuta el SPRT sobre una secuencia H_seq de indicadores H_n.
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