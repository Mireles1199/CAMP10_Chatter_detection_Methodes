# Comentario: utilidades de visualización
from __future__ import annotations
from typing import Dict, Any, Sequence
import numpy as np
import matplotlib.pyplot as plt

def plot_signal(t: "np.ndarray", x: "np.ndarray", *, title: str = "Signal") -> None:
    # Comentario: traza señal temporal
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("time (s)")
    plt.ylabel("amplitude")
    plt.title(title)
    plt.grid(True)

def plot_rms(times: "np.ndarray", rms: "np.ndarray", *, title: str = "RMS") -> None:
    # Comentario: traza secuencia RMS
    plt.figure()
    plt.plot(times, rms, marker="o")
    plt.xlabel("time (s)")
    plt.ylabel("rms")
    plt.title(title)
    plt.grid(True)

def plot_cv(time_seq: Sequence[float], cv_seq: Sequence[float], cv_threshold: float, *, title: str = "CV") -> None:
    # Comentario: traza CV
    plt.figure()
    plt.scatter(time_seq, cv_seq)
    plt.axhline(y=cv_threshold, color="r", linestyle="--", label="CV threshold")
    plt.xlabel("time (s)")
    plt.ylabel("cv")
    plt.title(title)
    plt.grid(True)
