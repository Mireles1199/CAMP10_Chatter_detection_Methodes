# Comentario: generadores de señales sintéticas (senos, TPF + chatter)
from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np

def five_senos(
    fs: float,
    duracion: float,
    ruido_std: float = 0.0,
    fase_aleatoria: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (t, x) con cinco sinusoides y ruido blanco opcional.

    Notas
    -----
    - Mantiene la firma/semántica del script original.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    N = int(np.round(fs * duracion))
    t = np.arange(N, dtype=float) / fs

    comps = [
        (1.5, 80.0),
        (2.0, 120.0),
        (1.5, 160.0),
        (1.5, 240.0),
        (2.0, 320.0),
    ]

    x = np.zeros_like(t)
    for A, f in comps:
        phi = rng.uniform(0, 2 * np.pi) if fase_aleatoria else 0.0
        x += A * np.sin(2 * np.pi * f * t + phi)

    if ruido_std > 0.0:
        x += rng.normal(0.0, ruido_std, size=t.shape)

    return t, x


def signal_1(
    fs: float,
    T: float,
    tpf: float,
    chatter_freqs: List[float],
    t_chatter_start: float,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (t, x) con armónicos TPF y chatter tras un instante de inicio.
    """
    t = np.linspace(0.0, T, int(fs * T), endpoint=False)

    harmonics = [tpf * i for i in range(1, 6)]
    x_base = np.zeros_like(t)
    for f in harmonics:
        x_base += 3.5 * np.sin(2 * np.pi * f * t)

    envelope_base = 0.1 + 0.6 * (t / T)
    mask_chatter = t > t_chatter_start
    envelope_chatter = np.zeros_like(t)
    if np.any(mask_chatter):
        envelope_chatter[mask_chatter] = 0.1 + 0.6 * (
            (t[mask_chatter] - t_chatter_start) / (T - t_chatter_start)
        )

    x_chatter = np.zeros_like(t)
    for f in chatter_freqs:
        x_chatter += 5.0 * np.sin(2 * np.pi * f * t)
    x_chatter *= envelope_chatter

    noise = noise_std * np.random.randn(len(t))

    x = envelope_base * x_base + x_chatter + noise
    return t, x
