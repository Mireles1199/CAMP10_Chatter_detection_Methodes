# -*- coding: utf-8 -*-
"""
Archivo: synthetic_chatter_signal_chatter.py
Descripción:
    Genera una señal sintética de mecanizado con:
        - Tooth passing frequency (480 Hz) y armónicos visibles desde el inicio.
        - Frecuencias de chatter (400–2500 Hz) que aparecen progresivamente desde 8 s.
        - Amplitud global creciente.
        - Espectrograma tipo STFT similar al observado en experimentos reales.
Autor: Codigo - Asistente GPT-5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib.colors import ListedColormap

# ==============================
# Parámetros generales
# ==============================
fs = 10000       # Frecuencia de muestreo (Hz)
T = 25           # Duración total (s)
t = np.linspace(0, T, int(fs * T))  # Vector de tiempo

# ==============================
# Tooth Passing Frequency (TPF)
# ==============================
tpf = 480  # Frecuencia base
harmonics = [tpf * i for i in range(1, 6)]  # 480, 960, 1440, 1920 Hz

signal_base = np.zeros_like(t)
for f in harmonics:
    signal_base += 3.5*np.sin(2 * np.pi * f * t)

# ==============================
# Frecuencias de chatter (aparecen después de 8 s)
# ==============================
chatter_freqs = [390, 920, 850, 1375, 1300, 2168, 2350]
signal_chatter = np.zeros_like(t)

envelope_base = 0.1 + 0.6 * (t / T)*1


# Envolvente que activa las chatter después de 8 s
mask_chatter = (t > 8)
envelope_chatter = np.zeros_like(t)
# envelope_chatter[mask_chatter] = np.linspace(0, 1, np.sum(mask_chatter)) **2
envelope_chatter[mask_chatter] = 0.1 + 0.6* ((t[mask_chatter] - 8) / (T - 8))*1

for f in chatter_freqs:
    # Pequeña modulación en frecuencia (para hacerlas inestables)
    mod = 0.02 * np.sin(2 * np.pi * 0.3 * t)
    mod = 0
    signal_chatter += 5*np.sin(2 * np.pi * (f + f * mod) * t)

signal_chatter *= envelope_chatter

# ==============================
# Ruido blanco
# ==============================
noise = 0.02 * np.random.randn(len(t))

# ==============================
# Combinación y envolvente global
# ==============================
# Envolvente de amplitud global con crecimiento exponencial suave


# envelope_total = np.ones_like(t) # Alternativa: sin envolvente global   

signal = envelope_base * signal_base + envelope_chatter * signal_chatter + noise

# ==============================
# Gráficas de resultados
# ==============================

# Señal temporal
plt.figure(figsize=(10, 4))
plt.plot(t, signal, color='blue', linewidth=0.8)
plt.title("Synthetic Machining Sound Signal with Chatter Onset")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.xlim(0, 25)
plt.grid(True)
plt.tight_layout()
# plt.show()

# Espectrograma (tipo TF)
f, tt, Sxx = spectrogram(signal, fs=fs, nperseg=1024, noverlap=512, nfft=2048, window='hann', mode='magnitude')

Sxx_dB = 10 * np.log10(Sxx + 1e-12)
vmax = np.percentile(Sxx_dB, 99)   # el valor alto (techo)
vmin = vmax - 20                   # muestra hasta 60 dB debajo del máximo


plt.figure(figsize=(10, 5))
two_colors = ListedColormap(['blue', 'yellow'])
# plt.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap=two_colors)
# plt.pcolormesh(tt, f, Sxx, shading='gouraud', cmap=two_colors)
plt.pcolormesh(tt, f, Sxx_dB, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)

plt.title("Spectrogram - Chatter Appears After 8 s")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.ylim(0, 3000)
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()
