# %%
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft
from ssqueezepy.experimental import scale_to_freq
from scipy.signal import  get_window


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


def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', cmap='turbo')
    plt.show()

#%%# Define signal ####################################
N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]  # add self reflected
x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

plt.plot(xo); plt.show()
plt.plot(x);  plt.show()

# %% FSenal 5 senos + ruido
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
window = get_window(('gaussian', sigma), win_length)



N = int(np.round(fs * duration))
t = np.arange(N, dtype=float) / fs
xo_2 = 1.5 * np.sin(2*np.pi*10.0*t + 0.0)

# x_grap  = xo_2
x_grap  = x_five

plt.figure()
plt.plot(t, x_grap, linewidth=1.0)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal de prueba 5 senos + Noise")
plt.grid(True)
plt.tight_layout()

#%%# CWT + SSQ CWT ####################################
# Twxo, Wxo, *_ = ssq_cwt(xo_2)
# viz(xo_2, Twxo, Wxo)

# Twx, Wx, *_ = ssq_cwt(x)
# viz(x, Twx, Wx)

#%%# STFT + SSQ STFT ##################################
Tsx_five, Sx_five, _, _, w, dSx = ssq_stft(x_five, window=window, n_fft=n_fft, win_len=win_length, hop_len=hop_length, fs=fs,
                                 get_dWx=True,
                                 get_w=True,
                                 )
viz(x_five, np.flipud(Tsx_five), np.flipud(Sx_five))

f = np.linspace(0, fs/2, Sx_five.shape[0])
t = np.arange(Sx_five.shape[1]) * hop_length / fs

plt.figure(figsize=(7,4))
plt.pcolormesh(t, f, np.abs(Sx_five), shading='auto')
plt.title("|S_x(μ, ξ)|  (STFT)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.ylim(0, 500)
plt.colorbar(label="Magnitud")
plt.show()

plt.figure(figsize=(7,4))
plt.imshow(np.abs(w), aspect='auto', cmap='turbo')


plt.figure(figsize=(7,4))
cs = plt.contour(t, f, np.abs(Sx_five), cmap='turbo')
# plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
plt.title('Contornos |S_x(μ, ξ)|  (STFT)')
plt.xlabel("Tiempo μ [s]")
plt.ylabel('Frecuencia [Hz]')
plt.colorbar(label='Magnitud')
plt.ylim(0, 500)

plt.figure(figsize=(7,4))
plt.pcolormesh(t, f, np.abs(Tsx_five), shading='turbo')
plt.title("|T_x(μ, ω)| (SSQ STFT)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.ylim(0, 500)
plt.colorbar(label="Magnitud")
plt.show()

plt.figure(figsize=(7,4))
cs = plt.contour(t, f, np.abs(Tsx_five), cmap='turbo')
# plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
plt.title('Contornos |T_x(μ, ω)| (SSQ STFT)')
plt.xlabel("Tiempo μ [s]")
plt.ylabel('Frecuencia [Hz]')
plt.colorbar(label='Magnitud')
plt.ylim(0, 500)







# Tsx, Sx, *_ = ssq_stft(x)
# viz(x, np.flipud(Tsx), np.flipud(Sx))

# %% STFT + SSQ STFT  5 senos ##########################
Sx = stft(x_five)[::-1]
freqs_stft = np.linspace(1, 0, len(Sx)) * fs/2
ikw = dict(abs=1, xticks=t, xlabel="Time [sec]", ylabel="Frequency [Hz]")

imshow(Sx, **ikw, yticks=freqs_stft)



#%%# With units #######################################
from ssqueezepy import Wavelet, cwt, stft, imshow
fs = 400
t = np.linspace(0, N/fs, N)
wavelet = Wavelet()
Wx, scales = cwt(x, wavelet)
Sx = stft(x)[::-1] 

freqs_cwt = scale_to_freq(scales, wavelet, len(x), fs=fs)
freqs_stft = np.linspace(1, 0, len(Sx)) * fs/2

ikw = dict(abs=1, xticks=t, xlabel="Time [sec]", ylabel="Frequency [Hz]")
imshow(Wx, **ikw, yticks=freqs_cwt)
imshow(Sx, **ikw, yticks=freqs_stft)