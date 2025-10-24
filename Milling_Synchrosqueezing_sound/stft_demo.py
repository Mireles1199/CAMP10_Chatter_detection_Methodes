#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STFT paso a paso — script didáctico con gráficos
------------------------------------------------

Qué hace este script:
1) Genera una señal x(t) con eventos en diferentes tiempos y frecuencias.
2) Define y explica las variables de la STFT: ventana w(t), tamaño de ventana, hop, NFFT, etc.
3) Implementa una STFT "desde cero" con NumPy (sin SciPy), para que veas cada paso.
4) Grafica:
   - Señal en el tiempo
   - Ventana en el dominio temporal
   - Un cuadro/segmento ventaneado y su espectro
   - El espectrograma (|STFT|)
5) Guarda las figuras como PNG en la carpeta desde donde ejecutes el script.

Requisitos:
- Python 3.x
- numpy, matplotlib (instalados por defecto en la mayoría de entornos científicos)

Ejecuta:
    python stft_demo.py

Autor: (tu nombre)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

def five_senos(fs: float,
                  duracion: float,
                  ruido_std: float = 0.0,
                  fase_aleatoria: bool = False,
                  seed: int | None = None):
    """
    Devuelve (t, x) para la mezcla sinusoidal dada + ruido N(t).

    Parámetros
    ----------
    fs : float
        Frecuencia de muestreo en Hz.
    duracion : float
        Duración en segundos.
    ruido_std : float, opcional
        Desvío estándar del ruido gaussiano blanco N(t). Por defecto 0 (sin ruido).
    fase_aleatoria : bool, opcional
        Si True, usa fases aleatorias U[0, 2π) para cada seno. Si False, fase 0.
    seed : int | None, opcional
        Semilla para reproducibilidad del ruido y fases.

    Devuelve
    --------
    t : np.ndarray, shape (N,)
        Vector de tiempo en segundos.
    x : np.ndarray, shape (N,)
        Señal muestreada s(t) en los tiempos t.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Vector de tiempo
    N = int(np.round(fs * duracion))
    t = np.arange(N, dtype=float) / fs

    # Componentes (amplitud, frecuencia en Hz)
    comps = [
        (1.5,  80.0),
        (2.0, 120.0),
        (1.5, 160.0),
        (1.5, 240.0),
        (2.0, 320.0),
    ]

    x = np.zeros_like(t)
    for A, f in comps:
        phi = rng.uniform(0, 2*np.pi) if fase_aleatoria else 0.0
        x += A * np.sin(2*np.pi*f*t + phi)

    if ruido_std > 0.0:
        x += rng.normal(0.0, ruido_std, size=t.shape)

    return t, x



def stft_demo(fs: float = 8000, duracion: float = 2.0):
    """
    Genera una señal sintética con distintos componentes temporales y de frecuencia.

    Descripción
    -----------
    La señal se define así:
      - 0 a 1.0 s : tono de 220 Hz
      - 1.0 a 2.0 s : tono de 440 Hz
      - Bursts (ventanas cortas) de 660 Hz alrededor de t = 0.75 s y t = 1.5 s

    Parámetros
    ----------
    fs : float, opcional
        Frecuencia de muestreo en Hz. Por defecto 8000 Hz.
    duracion : float, opcional
        Duración total de la señal en segundos. Por defecto 2.0 s.

    Devuelve
    --------
    t : np.ndarray, shape (N,)
        Vector de tiempo en segundos (0, 1/fs, 2/fs, ...).
    x : np.ndarray, shape (N,)
        Señal generada, normalizada entre [-1, 1].

    Ejemplo
    --------
    >>> t, x = generar_senal_tiempo_frecuencia(fs=8000, duracion=2.0)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, x)
    >>> plt.xlabel("Tiempo [s]")
    >>> plt.ylabel("Amplitud")
    >>> plt.show()
    """

    # Vector de tiempo
    t = np.arange(0, duracion, 1 / fs)

    # Frecuencias de cada componente
    f1 = 220.0
    f2 = 440.0
    f_burst = 660.0

    # Inicialización de la señal
    x = np.zeros_like(t)

    # Tonos largos
    x += (t < 1.0) * np.sin(2 * np.pi * f1 * t)      # primer segundo: 220 Hz
    x += (t >= 1.0) * np.sin(2 * np.pi * f2 * t)     # segundo segundo: 440 Hz

    # Bursts (ventanas cortas multiplicando el seno)
    burst1 = ((t > 0.70) & (t < 0.80)) * np.sin(2 * np.pi * f_burst * t)
    burst2 = ((t > 1.45) & (t < 1.55)) * np.sin(2 * np.pi * f_burst * t)
    x += 0.8 * burst1 + 0.8 * burst2

    # Normalización
    x = x / (np.max(np.abs(x)) + 1e-12)

    return t, x


fs = 8000  # Hz
T = 2.0    # segundos

t_demo, x_demo = stft_demo(fs=fs, duracion=T)
t_five_senos, x_five_senos = five_senos(fs=fs, duracion=T, ruido_std=4, fase_aleatoria=False, seed=120)


t, x = t_five_senos, x_five_senos


# ------------------------------------------------------------
# 2) Parámetros de la STFT (controlan resolución tiempo-frecuencia)
# ------------------------------------------------------------
win_length_ms = 100.0   # duración de la ventana [ms]  -> controla resolución en frecuencia
hop_ms        = 10.0   # salto entre ventanas [ms]    -> controla resolución temporal
n_fft         = 1024   # tamaño de la FFT (potencia de 2 suele ser conveniente)

# Cálculos derivados
win_length = int(round(win_length_ms * 1e-3 * fs))   # muestras por ventana
hop_length = int(round(hop_ms        * 1e-3 * fs))   # muestras por hop

# Ventana de Hann: w[n] con n=0..win_length-1
# (valor alto en el centro, bajo en los bordes -> reduce fugas espectrales)
def hann_window(L: int) -> np.ndarray:
    if L <= 1:
        return np.ones(L)
    n = np.arange(L)
    return 0.5 - 0.5*np.cos(2*np.pi*n/(L-1))

w = hann_window(win_length)

# Frecuencias asociadas a la FFT (solo positivas, espectro unilateral)
freqs = np.fft.rfftfreq(n_fft, d=1/fs)  # [Hz]

# ------------------------------------------------------------
# 3) Implementación simple de la STFT (desde cero)
# ------------------------------------------------------------
def stft_numpy(x: np.ndarray,
               fs: float,
               win: np.ndarray,
               hop_length: int,
               n_fft: int):
    """
    Calcula la STFT "a mano" (sin SciPy), retornando:
        X_stft: matriz compleja de tamaño (num_frames, n_freqs)
        t_frames: vector de tiempos (centro de cada ventana) [s]
        freqs: vector de frecuencias [Hz]
    Donde:
        - La ventana se aplica por multiplicación punto a punto.
        - Cada frame se zero-paddea a n_fft antes de la FFT.
        - Se usa rfft para quedarnos con frecuencias [0..fs/2].
    """
    L = len(win)
    assert hop_length > 0, "hop_length debe ser > 0"
    assert L <= n_fft, "n_fft debe ser >= win_length"

    n_samples = len(x)
    # Número de frames que caben con ese hop y ventana
    num_frames = 1 + max(0, (n_samples - L) // hop_length)

    X_list = []
    t_list = []

    for m in range(num_frames):
        start = m * hop_length
        end = start + L
        frame = x[start:end]
        if len(frame) < L:
            # Zero-pad si la última ventana queda corta (no debería con la fórmula de num_frames)
            frame = np.pad(frame, (0, L - len(frame)))
        # Ventaneado en tiempo: x[n] * w[n]
        frame_win = frame * win

        # Zero-padding hasta n_fft
        frame_pad = np.zeros(n_fft, dtype=float)
        frame_pad[:L] = frame_win

        # FFT real (solo positivas)
        X_f = np.fft.rfft(frame_pad)

        X_list.append(X_f)
        # Tiempo del centro de la ventana (para ubicar el frame)
        t_center = (start + L/2) / fs
        t_list.append(t_center)

    X_stft = np.vstack(X_list)  # shape: (num_frames, n_freqs)
    t_frames = np.array(t_list) # shape: (num_frames,)
    freqs = np.fft.rfftfreq(n_fft, d=1/fs)

    return X_stft, t_frames, freqs

# Ejecutamos la STFT
X, t_frames, freqs = stft_numpy(x, fs, w, hop_length, n_fft)

# Magnitud (amplitud) y/o en dB
mag = np.abs(X) + 1e-12
mag_db = 20*np.log10(mag / np.max(mag))

mag_graph = mag  # para graficar el espectrograma

# ------------------------------------------------------------
# 4) Visualizaciones (cada figura en su propio plot)
# ------------------------------------------------------------
# Ajuste de carpeta de salida (donde se guardarán las figuras)
outdir = Path(".")
# Si prefieres, puedes cambiar a Path("./salidas") y crearla:
# outdir = Path("./salidas"); outdir.mkdir(exist_ok=True, parents=True)

# 4.a) Señal en el tiempo
plt.figure(figsize=(10, 3))
plt.plot(t, x)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal x(t)")
plt.tight_layout()
# plt.show()

# 4.b) Ventana en el tiempo
plt.figure(figsize=(6, 3))
plt.plot(np.arange(win_length)/fs*1000.0, w)
plt.xlabel("Tiempo relativo de ventana [ms]")
plt.ylabel("w[n]")
plt.title(f"Ventana de Hann (L = {win_length} muestras, ~{win_length_ms:.1f} ms)")
plt.tight_layout()
# plt.show()

# 4.c) Un frame ventaneado y su espectro (elige, por ejemplo, el frame del medio)
m_mid = len(t_frames) // 2
start_mid = m_mid * hop_length
end_mid = start_mid + win_length
frame_mid = x[start_mid:end_mid] * w

# Señal del frame en el tiempo (eje local relativo a ese frame)
t_local = (np.arange(win_length) / fs) * 1000.0  # ms

plt.figure(figsize=(8, 3))
plt.plot(t_local, frame_mid)
plt.xlabel("Tiempo dentro del frame [ms]")
plt.ylabel("Amplitud")
plt.title("Un frame ventaneado (ejemplo)")
plt.tight_layout()
# plt.show()

# Espectro del frame (magnitud en dB)
frame_pad = np.zeros(n_fft, dtype=float)
frame_pad[:win_length] = frame_mid
X_frame = np.fft.rfft(frame_pad)
mag_frame = np.abs(X_frame) + 1e-12
mag_frame_db = 20*np.log10(mag_frame / np.max(mag_frame))

plt.figure(figsize=(8, 3))
plt.plot(freqs, mag_frame_db)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB ref. máx. del frame]")
plt.title("Espectro (magnitud) de un frame")
plt.tight_layout()
# plt.show()

# 4.d) Espectrograma (|STFT| en dB)
plt.figure(figsize=(10, 4))
# imshow espera matriz [filas, columnas]; filas->frecuencia, columnas->tiempo
# Transponemos para que el eje Y sea frecuencia (origin='lower' para que 0 Hz esté abajo)
plt.imshow(mag_graph.T,
           origin='lower',
           aspect='auto',
           extent=[t_frames[0], t_frames[-1], freqs[0], freqs[-1]])
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.ylim(0, 500)  # limitar a 500 Hz para mejor visualización
plt.title("Espectrograma (|STFT|")
plt.colorbar(label="")
plt.tight_layout()
# plt.show()

# --- Espectrograma en escala lineal ---
plt.figure(figsize=(7,4))
plt.pcolormesh(t_frames, freqs, mag_graph.T, shading='gouraud', cmap='viridis')
plt.title('Espectrograma (STFT) - Magnitud lineal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.colorbar(label='Magnitud (normalizada)')
plt.ylim(0, 500)


# --- Contornos para hacerlo más “analítico” ---
# levels = np.linspace(0.1, 1.0, 10)  # niveles de magnitud
plt.figure(figsize=(7,4))
cs = plt.contour(t_frames, freqs, mag_graph.T, cmap='viridis')
# plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
plt.title('Contornos de magnitud (escala lineal)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.colorbar(label='Magnitud (normalizada)')
plt.ylim(0, 500)




# ------------------------------------------------------------
# 4.e) Fase de un frame y espectrograma de fase
# ------------------------------------------------------------
# Fase de un frame (el mismo usado antes)
phase_frame = np.angle(X_frame)  # envuelta en [-pi, pi]

plt.figure(figsize=(8, 3))
plt.plot(freqs, phase_frame)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.title("Fase del frame (envuelta)")
plt.tight_layout()

# Fase del STFT completa (envuelta)
phase = np.angle(X)  # shape: (num_frames, n_freqs_positivas)

plt.figure(figsize=(10, 4))
plt.imshow(phase.T,
           origin='lower',
           aspect='auto',
           extent=[t_frames[0], t_frames[-1], freqs[0], freqs[-1]])
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.ylim(0, 500)  # limitar a 500 Hz para mejor visualización
plt.title("Espectrograma de fase (envuelta)")
plt.colorbar(label="Fase [rad]")
plt.tight_layout()

# ------------------------------------------------------------
# 4.f) Fase "desenvuelta" a lo largo del tiempo para una frecuencia concreta
# ------------------------------------------------------------
# Elegimos una frecuencia cercana a 440 Hz para seguir su fase en el tiempo
target_freq = 440.0
k = np.argmin(np.abs(freqs - target_freq))

# Desenvuelve la fase a lo largo del eje temporal (frames) para esa frecuencia
phase_unwrapped = np.unwrap(phase[:, k])

plt.figure(figsize=(8, 3))
plt.plot(t_frames, phase_unwrapped)
plt.xlabel("Tiempo [s]")
plt.ylabel("Fase desenvuelta [rad]")
plt.title(f"Fase desenvuelta en f ≈ {freqs[k]:.1f} Hz (a lo largo del tiempo)")
plt.tight_layout()

# (Opcional) Estimar frecuencia instantánea a partir del gradiente temporal de la fase
# fi[t] ≈ (1/(2π)) * dφ/dt ; aquí dφ/dt ≈ (φ[m]-φ[m-1]) / (hop_length/fs)
dphi = np.diff(phase_unwrapped)
dt_hop = hop_length / fs
fi_est = (1/(2*np.pi)) * (dphi / dt_hop)  # Hz, aproximación
t_inst = 0.5*(t_frames[1:] + t_frames[:-1])

plt.figure(figsize=(8, 3))
plt.plot(t_inst, fi_est)
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia instantánea [Hz]")
plt.title(f"Frecuencia instantánea estimada cerca de {freqs[k]:.1f} Hz")
plt.tight_layout()

# ------------------------------------------------------------
# 5.b) Mensajes extra sobre fase
# ------------------------------------------------------------
print("Fase: arg{X(t,f)} — argumento del coeficiente complejo del STFT.")
print(" - Es relativa a la ventana y al origen temporal del frame.")
print(" - Útil para: reconstrucción de señal (p.ej., Griffin-Lim), phase vocoder,")
print("   time-stretch/pitch-shift, detección de onsets, estimación de frecuencia instantánea,")
print("   y análisis de alineamiento temporal (group delay).")
print("Figuras de fase guardadas:")
print("  - stft_fig_frame_phase.png (fase de un frame)")
print("  - stft_phase_spectrogram.png (fase envuelta)")
print("  - stft_phase_unwrapped_bin.png (fase desenvuelta en ~440 Hz vs tiempo)")
print("  - stft_phase_instant_freq.png (frecuencia instantánea estimada)")


# ------------------------------------------------------------
# 5) Imprimir ayuda/variables clave en consola
# ------------------------------------------------------------
print("=== Parámetros y variables de la STFT ===")
print(f"fs = {fs} Hz (frecuencia de muestreo)")
print(f"Duración T = {T:.3f} s, muestras totales = {len(t)}")
print(f"Ventana: Hann, win_length = {win_length} muestras (~{win_length_ms:.1f} ms)")
print(f"Hop (desplazamiento) = {hop_length} muestras (~{hop_ms:.1f} ms)")
print(f"n_fft = {n_fft} -> resolución de frecuencia ≈ fs/n_fft = {fs/n_fft:.3f} Hz")
print(f"Nº de frames = {len(t_frames)}")
print("t_frames (s) es el centro temporal de cada ventana -> barre el tiempo")
print("freqs (Hz) es el eje de análisis espectral por frame -> barre las frecuencias")
print("Figuras guardadas:")
print("  - stft_fig_signal.png")
print("  - stft_fig_window.png")
print("  - stft_fig_frame_time.png")
print("  - stft_fig_frame_fft.png")
print("  - stft_spectrogram.png")


plt.show()

# Fin del script
