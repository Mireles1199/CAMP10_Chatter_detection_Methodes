import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window


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


# --- Señal de ejemplo ---
fs = 8000
T = 2.0    # segundos

t_five_senos, x_five_senos = five_senos(fs=fs, duracion=T, ruido_std=4, fase_aleatoria=False, seed=120)

t, x = t_five_senos, x_five_senos

# --- Cálculo de la STFT ---
win_length_ms = 100.0   # duración de la ventana [ms] 
nperseg = int(fs * win_length_ms / 1000.0)  # en muestras
noverlap = int(0.9 * nperseg)
window = get_window('hann', nperseg)

f, t_stft, Zxx = stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
S = np.abs(Zxx)

# --- Normalización opcional (para mejor visualización) ---
# S = S / np.max(S)  # normaliza a [0,1]

# --- Espectrograma en escala lineal ---
plt.figure(figsize=(7,4))
plt.pcolormesh(t_stft, f, S, shading='gouraud', cmap='viridis')
plt.title('Espectrograma (STFT) - Magnitud lineal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.ylim(0, 500)
plt.colorbar(label='Magnitud (normalizada)')
plt.show()

# --- Contornos para hacerlo más “analítico” ---
# levels = np.linspace(0.1, 1.0, 10)  # niveles de magnitud
plt.figure(figsize=(7,4))
cs = plt.contour(t_stft, f, S, cmap='viridis')
# plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
plt.title('Contornos de magnitud (escala lineal)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.colorbar(label='Magnitud (normalizada)')
plt.ylim(0, 500)

plt.show()
