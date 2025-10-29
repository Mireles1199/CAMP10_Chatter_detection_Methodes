

import numpy as np
import matplotlib.pyplot as plt
from ssq_stft_chatter import ssq_stft
import scipy.signal as sig

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






fs = 10240.0          # [Hz] frecuencia de muestreo
duration = 2.0       # [s] duración
t,x_five = five_senos(fs=fs, duracion=duration, ruido_std=4, fase_aleatoria=False, seed=120)

win_length_ms = 200.0   # duración de la ventana [ms]  -> controla resolución en frecuencia
hop_ms        = 10.0   # salto entre ventanas [ms]    -> controla resolución temporal
n_fft         = 1024*2  # tamaño de la FFT (potencia de 2 suele ser conveniente)

win_length = int(round(win_length_ms * 1e-3 * fs))   # muestras por ventana
hop_length = int(round(hop_ms        * 1e-3 * fs))    # muestras por hop

L = win_length             # longitud impar
sigma = L/6.         # en muestras
window = sig.get_window(('gaussian', sigma), win_length)





Tx, Sx, _, _, w, dSx = ssq_stft(
    x_five,
    window=window,
    n_fft=n_fft,
    win_len=win_length,
    hop_len=hop_length,
    fs=fs,
    get_dWx=True,
    get_w=True,
)

print(f"Tx.shape={Tx.shape}, Sx.shape={Sx.shape}")

f = np.linspace(0, fs/2, Sx.shape[0])
t = np.arange(Sx.shape[1]) * hop_length / fs

plt.figure(figsize=(7,4))
plt.pcolormesh(t, f, np.abs(Sx), shading='auto')
plt.title("|S_x(μ, ξ)|  (STFT)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.ylim(0, 500)
plt.colorbar(label="Magnitud")
# plt.show()

plt.figure(figsize=(7,4))
plt.imshow(np.abs(Sx), aspect='auto', cmap='turbo')


# plt.figure(figsize=(7,4))
# plt.pcolormesh(t, f, np.abs(Tx), shading='turbo')
# plt.title("|T_x(μ, ω)| (SSQ STFT)")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Frecuencia [Hz]")
# plt.ylim(0, 500)
# plt.colorbar(label="Magnitud")

plt.figure(figsize=(7,4))
plt.imshow(np.abs(Tx), aspect='auto', cmap='turbo')

plt.show()

