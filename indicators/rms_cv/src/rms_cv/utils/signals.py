"""Synthetic vibration signal generators for testing the RMS-CV pipeline.

Two generators are provided:

* :func:`five_senos` — superposition of five fixed-frequency sinusoids with
  optional Gaussian white noise; useful as a stable-cut baseline.
* :func:`signal_1` — tooth-passing-frequency (TPF) harmonics plus a
  chatter burst that grows in amplitude after a configurable onset time;
  simulates a realistic chatter event for validation purposes.
"""

# Generadores de señales sintéticas (senos, TPF + chatter)
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
    """Generate a synthetic vibration signal composed of five sinusoids.

    Superposes five fixed-frequency components (80, 120, 160, 240, 320 Hz)
    with amplitudes (1.5, 2.0, 1.5, 1.5, 2.0) and optional additive
    Gaussian white noise.  This signal represents a typical stable-cut
    vibration with multiple harmonics for testing RMS-CV baseline detection.

    Args:
        fs (float): Sampling frequency [Hz].  Must be > 0.
        duracion (float): Signal duration [s].  Must be > 0.
        ruido_std (float, optional): Standard deviation of the additive
            Gaussian white noise.  Set to ``0.0`` (default) for a
            noise-free, perfectly periodic signal.
        fase_aleatoria (bool, optional): If ``True``, each sinusoid receives
            an independent random initial phase drawn from
            :math:`\\mathcal{U}[0, 2\\pi)`.  Defaults to ``False`` (all
            phases zero).
        seed (Optional[int], optional): Integer seed forwarded to
            :class:`numpy.random.Generator` for reproducible random phases
            and noise.  Defaults to ``None`` (non-deterministic).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Pair ``(t, x)`` where

        * **t** — uniformly spaced time vector of length
          :math:`\\text{round}(f_s \\cdot T_{dur})` [s].
        * **x** — composite signal, same length as *t* [a.u.].

    Note:
        The function signature and component parameters are kept identical
        to the original exploratory script so that existing analysis
        notebooks remain compatible.

    Example:
        >>> t, x = five_senos(fs=20_000.0, duracion=5.0, seed=42)
        >>> len(t)
        100000
        >>> abs(x).max() > 0
        True
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
    """Generate a synthetic machining signal with TPF harmonics and chatter burst.

    Constructs a two-phase time series that mimics real machining vibration:

    * **Stable phase** ``[0, T]``: five harmonics of the tooth-passing
      frequency (TPF) modulated by a linearly growing envelope
      :math:`0.1 + 0.6 \\cdot t/T`.
    * **Chatter phase** ``[t_chatter_start, T]``: additional sinusoids at
      each frequency in *chatter_freqs* (amplitude 5.0) with an independent
      linear envelope that starts at zero and grows to 0.7 by the end of
      the signal.

    Optional Gaussian white noise is added to both phases.

    Args:
        fs (float): Sampling frequency [Hz].
        T (float): Total signal duration [s].
        tpf (float): Fundamental tooth-passing frequency [Hz].  The first
            five harmonics (TPF, 2·TPF, …, 5·TPF) are included with
            amplitude 3.5 each.
        chatter_freqs (List[float]): Chatter frequency components [Hz].  Each
            entry generates one sinusoid in the chatter burst.
        t_chatter_start (float): Chatter onset time [s].  Must satisfy
            ``0 <= t_chatter_start < T``.
        noise_std (float, optional): Standard deviation of the additive
            Gaussian white noise applied to the full signal.  Defaults to
            ``0.0``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Pair ``(t, x)`` where

        * **t** — time vector [s], length :math:`f_s \\cdot T`.
        * **x** — composite signal: envelope-modulated TPF harmonics +
          chatter burst with onset envelope + white noise.

    Note:
        The stable-component envelope grows from 0.1 to 0.7 over ``[0, T]``.
        The chatter envelope grows from 0.1 to 0.7 over
        ``[t_chatter_start, T]`` and is identically zero before the onset.
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
