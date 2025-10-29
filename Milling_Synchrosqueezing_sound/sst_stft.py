
# %%
"""
sst_stft.py
===========
Implementación pedagógica (de extremo a extremo) del **Synchro-Squeezing Transform (SST)**
basado en la STFT, siguiendo las ecuaciones (1)-(5) de tus notas.

- Código 100% NumPy + Matplotlib (sin dependencias externas) para que funcione en entornos mínimos.
- Comentarios y docstrings explican **qué es** cada variable, **por qué** existe y **cómo** se usa.
- Incluye una **demo** reproducible con señales AM-FM + seno puro y **gráficas** para entender
  cada etapa: ventana, STFT, estimador de frecuencia instantánea e imagen SST.

Autor: (Enrique Mireles Hernandez)
Licencia: MIT
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window


# ---------------------------------------------------------------------------
# 0) Utilidades y ventanas
# ---------------------------------------------------------------------------
def gaussian_window(L: int, sigma: float) -> np.ndarray:
    """
    Genera una ventana Gaussiana discreta g[n] de longitud L y parámetro sigma.

    Parámetros
    ----------
    L : int
        Longitud total de la ventana (número de muestras). Debe ser impar idealmente
        para que el centro quede exactamente en (L-1)/2.
    sigma : float
        Desviación típica (en muestras). Controla el "ancho" de la ventana:
        - sigma pequeña -> ventana estrecha (mejor resolución temporal, peor en frecuencia)
        - sigma grande  -> ventana ancha  (mejor resolución en frecuencia, peor en tiempo)

    Devuelve
    --------
    g : np.ndarray de shape (L,)
        Ventana Gaussiana normalizada a energía unitaria (||g||_2 = 1),
        conveniente para operaciones STFT.

    Notas
    -----
    La forma continua es g(t) = exp(-t^2/(2 sigma^2)). Aquí discretizamos alrededor del centro.
    """
    # Eje discreto centrado en 0 para que g sea "simétrica" alrededor del centro
    n = np.arange(L) - (L - 1) / 2.0
    g = np.exp(-0.5 * (n / sigma) ** 2)

    # Normalizamos la energía para que el escalado sea consistente al variar L o sigma
    g = g / np.linalg.norm(g)
    return g


# ---------------------------------------------------------------------------
# 1) STFT discreta (ecuación (1) en forma práctica por marcos)
# ---------------------------------------------------------------------------
def stft_manual(
    x: np.ndarray,
    win: np.ndarray,
    hop: int,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la STFT por marcos:
        S[m, k] = sum_n x[n] * g[n - mH] * e^{-i 2π k (n-mH)/nfft}
    usando FFT de tamaño nfft sobre cada marco ventana-do.

    Parámetros
    ----------
    x : np.ndarray shape (N,)
        Señal 1D en el tiempo.
    win : np.ndarray shape (L,)
        Ventana (p. ej., gaussiana). Debe ser más corta o igual a N.
    hop : int
        Desplazamiento entre marcos en **muestras**. Controla el muestreo temporal μ_m.
        Δμ = hop / fs (si conoces la frecuencia de muestreo).
    nfft : int
        Tamaño de la FFT. Recom. potencia de dos >= L. Define la rejilla de ξ_k.

    Devuelve
    --------
    S : np.ndarray shape (M, K)
        Espectrograma complejo (STFT). M = número de marcos, K = nfft//2 + 1 (frecuencias no-negativas).
    times : np.ndarray shape (M,)
        Vector de tiempos-centrales μ_m en **muestras** (no en segundos).
        Para segundos, dividir por fs en la parte de demo.
    freqs : np.ndarray shape (K,)
        Frecuencias **cíclicas** (no angulares) en ciclos por muestra (∈ [0, 0.5]).
        Para Hz, multiplicar por fs; para angulares ξ=2π f, multiplicar por 2π fs.

    Notas
    -----
    - Usamos sólo la mitad positiva del espectro (rfft) para señales reales.
    - Alineamos cada marco centrando la ventana sobre la muestra m*hop.
    """
    N = len(x)
    L = len(win)

    # Número de marcos M tal que el centro de la ventana no se salga del vector
    # Aquí hacemos *padding* reflectante para simplificar bordes y mantener alineación
    pad = (L - 1) // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    centers = np.arange(0, N, hop) + pad  # índices de centro en x_pad
    M = len(centers)

    # Preparar salida
    K = nfft // 2 + 1
    S = np.zeros((M, K), dtype=np.complex128)

    # Ventana zero-padded a nfft para multiplicar antes de la FFT si se desea
    # (aquí aplicamos ventana en el dominio temporal sobre el trozo L, luego rfft).
    for m, c in enumerate(centers):
        frame = x_pad[c - pad : c + pad + 1]  # L muestras centradas
        frame_win = frame * win
        # FFT real (solo K bins no-negativos)
        spec = np.fft.rfft(frame_win, n=nfft)
        S[m, :] = spec

    # Rejillas de tiempo y frecuencia
    times = np.arange(M) * hop  # en muestras
    freqs = np.fft.rfftfreq(nfft, d=1.0)  # ciclos por "muestra" (sin fs)
    return S, times, freqs


# ---------------------------------------------------------------------------
# 2) Derivada respecto a μ (ecuación (3) discreta) y estimador IF (ecuación (4))
# ---------------------------------------------------------------------------
def partial_mu_central(S: np.ndarray, delta_mu_samples: float) -> np.ndarray:
    """
    Aproxima la derivada parcial respecto a μ (tiempo-centro del marco) usando
    diferencias finitas centrales a lo largo del eje de marcos.

    Parámetros
    ----------
    S : np.ndarray shape (M, K)
        STFT compleja por marcos (M) y frecuencias (K).
    delta_mu_samples : float
        Paso Δμ en **muestras** entre marcos consecutivos. Si hop=128, Δμ=128.

    Devuelve
    --------
    dS_dmu : np.ndarray shape (M, K)
        Derivada aproximada ∂_μ S. En los bordes (m=0, m=M-1) usamos diferencias hacia delante/atrás.

    Notas
    -----
    Para convertir a derivada por **segundos**, dividir por Δμ_s = hop / fs fuera de esta función.
    """
    M, K = S.shape
    dS = np.zeros_like(S, dtype=np.complex128)

    # Interior: central
    dS[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2.0 * delta_mu_samples)

    # Bordes: hacia delante/atrás
    dS[0, :] = (S[1, :] - S[0, :]) / delta_mu_samples
    dS[-1, :] = (S[-1, :] - S[-2, :]) / delta_mu_samples
    return dS


def inst_freq_from_derivative(S: np.ndarray, dS_dmu: np.ndarray, rel: float = 0.15,
                              k_smooth: int = 0) -> np.ndarray:
    """
    Estima la **frecuencia instantánea** (IF) por (μ, ξ) según la ecuación (4):
        \hat{ω}(μ, ξ) = Im{ ∂_μ S(μ, ξ) / S(μ, ξ) }

    Parámetros
    ----------
    S : np.ndarray shape (M, K)
        STFT compleja.
    dS_dmu : np.ndarray shape (M, K)
        Derivada de S respecto a μ (en "por muestra").
    rel : float
        rel: umbral relativo a max(|S|) por frame (0.05-0.25 típico)
    k_smooth:  int
        Tamaño impar para suavizado ligero a lo largo de k (0 = sin suavizado).
    

    Devuelve
    --------
    omega_hat : np.ndarray shape (M, K)
        Estimación de frecuencia **angular** \hat{ω} en radianes por **muestra**.
        Para pasar a Hz: f_hat = omega_hat * fs / (2π).

    Notas
    -----
    - Usamos la forma Im{ ∂_μ S / S } que es equivalente a (1/i)*(∂_μ S)/S.
    - En bins con |S| < eps devolvemos NaN para marcarlos como inválidos.
    """
    # mag = np.abs(S)
    # valid = mag >= eps
    # ratio = np.zeros_like(S, dtype=np.complex128)
    # # Evitar división por cero
    # ratio[valid] = dS_dmu[valid] / S[valid]

    # # Parte imaginaria = velocidad de fase respecto a μ (por muestra)
    # omega_hat = np.full(S.shape, np.nan, dtype=float)
    # omega_hat[valid] = np.imag(ratio[valid])  # radianes / muestra
    # return omega_hat

    M, K = S.shape
    mag = np.abs(S)

    # Umbral relativo por frame (mucho mejor que 'eps' fijo o mediana)
    # max_per_frame = np.max(mag, axis=1, keepdims=True) + 1e-16
    # valid = mag >= (rel * max_per_frame)
    
    mag = np.abs(S)
    mag_db = 20*np.log10(mag + 1e-12)
    max_db = np.max(mag_db, axis=1, keepdims=True)
    valid = mag_db >= (max_db - 25.0) 

    ratio = np.zeros_like(S, dtype=np.complex128)
    ratio[valid] = dS_dmu[valid] / S[valid]   # rad/muestra

    omega_hat = np.full((M, K), np.nan, float)
    omega_hat[valid] = np.imag(ratio[valid])  # rad/muestra

    # (opcional) suavizado en frecuencia para estabilizar
    if k_smooth and k_smooth > 1:
        from scipy.ndimage import uniform_filter1d
        omega_hat = uniform_filter1d(omega_hat, size=k_smooth, axis=1, mode='nearest')

    # Limitar al rango de rfft: [0, π] rad/muestra
    omega_hat[(omega_hat < 0) | (omega_hat > np.pi)] = np.nan
    return omega_hat, valid


# ---------------------------------------------------------------------------
# 3) Synchro-Squeezing (ecuación (5) discreta)
# ---------------------------------------------------------------------------
def synchrosqueeze(
    S: np.ndarray,
    omega_hat: np.ndarray,
    freqs_cps: np.ndarray,
    out_freqs_cps: np.ndarray,
    rel: float = 0.15,
) -> np.ndarray:
    """
    Reasigna horizontalmente la energía de S[m, k] al bin q cuya frecuencia
    (en ciclos por muestra) sea más cercana a \hat{f} = \hat{ω} / (2π).

    Parámetros
    ----------
    S : np.ndarray shape (M, K)
        STFT compleja.
    omega_hat : np.ndarray shape (M, K)
        IF angular estimada (radianes por muestra).
    freqs_cyc_per_sample : np.ndarray shape (K,)
        Frecuencias de análisis de la STFT en **ciclos por muestra**, correspondientes a k.
        (Devueltas por stft(...)[2])
    out_freqs_cyc_per_sample : np.ndarray shape (Q,)
        Rejilla de frecuencias de salida para el SST (ciclos por muestra).
        Puedes usar igual que 'freqs_cyc_per_sample' o una más densa.


    Devuelve
    --------
    T : np.ndarray shape (M, Q)
        Mapa Synchro-Squeezed en la rejilla de salida.

    Notas
    -----
    - Asignación "dura": todo S[m,k] se suma al bin q* más cercano a \hat{f}(m,k).
      Alternativas: *kernels* suaves (p. ej., triangular o gaussiano).
    """
    M, K = S.shape
    Q = out_freqs_cps.size
    T = np.zeros((M, Q), dtype=np.complex128)

    mag = np.abs(S)
    max_per_frame = np.max(mag, axis=1, keepdims=True) + 1e-16
    valid = (mag >= rel * max_per_frame) & np.isfinite(omega_hat)

    # IF angular -> ciclos/muestra
    f_hat = omega_hat / (2*np.pi)        # [ciclos/muestra]
    f_hat[(f_hat < 0) | (f_hat > 0.5)] = np.nan  # Nyquist

    for m in range(M):
        idx = np.where(valid[m])[0]
        if idx.size == 0:
            continue
        f_m = f_hat[m, idx]
        s_m = S[m, idx]
        # índice q más cercano (vectorizado)
        q = np.searchsorted(out_freqs_cps, f_m, side='left')
        q = np.clip(q, 0, Q-1)
        # acumular
        np.add.at(T[m], q, s_m)
    return T


# ---------------------------------------------------------------------------
# 4) Señales de prueba (AM-FM y seno puro) para la demo
# ---------------------------------------------------------------------------
def test_signal(fs: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Crea una señal sintética de demostración:
    - Un seno puro a f0 = 80 Hz.
    - Un chirp lineal de 20 -> 200 Hz.
    - Una portadora AM-FM suave.

    Parámetros
    ----------
    fs : float
        Frecuencia de muestreo en Hz.
    duration : float
        Duración en segundos.

    Devuelve
    --------
    x : np.ndarray shape (N,)
        Señal temporal.
    t : np.ndarray shape (N,)
        Eje temporal en segundos.
    """
    t = np.arange(int(fs * duration)) / fs
    N = len(t)

    # Seno puro
    f0 = 80.0  # Hz
    x1 = 0.7 * np.cos(2.0 * np.pi * f0 * t)

    # Chirp lineal 20 -> 200 Hz
    f_start, f_end = 20.0, 200.0
    k = (f_end - f_start) / duration  # barrido por segundo
    # fase(t) = 2π (f_start t + 0.5 k t^2)
    x2 = 0.5 * np.cos(2.0 * np.pi * (f_start * t + 0.5 * k * t**2))

    # AM-FM: amplitud lenta + modulación de frecuencia suave
    a = 0.4 * (1.0 + 0.5 * np.sin(2.0 * np.pi * 0.5 * t))            # AM a 0.5 Hz
    f_inst = 60 + 15 * np.sin(2.0 * np.pi * 0.3 * t)                 # IF alrededor de 60 Hz
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs                     # integrar frecuencia -> fase
    x3 = a * np.cos(phase)

    # Suma
    x = x1 + x2 + x3
    return x, t

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


# %%
# ---------------------------------------------------------------------------
# 5) Demo completa con gráficas (una figura por gráfico, sin seaborn, sin estilos)
# ---------------------------------------------------------------------------

"""
Ejecuta un flujo completo:
1) Genera señal de prueba.
2) Construye ventana Gaussiana y muestra su forma.
3) Calcula STFT y visualiza |S|.
4) Estima IF por (μ,k) y muestra mapa de \hat{f}.
5) Realiza Synchro-Squeezing y visualiza |T|.
"""
# ------------------ Parámetros globales ------------------
fs = 10240.0          # [Hz] frecuencia de muestreo
duration = 2.0       # [s] duración
x, t = test_signal(fs, duration)
t,x = five_senos(fs=fs, duracion=duration, ruido_std=4, fase_aleatoria=False, seed=120)

N = int(np.round(fs * duration))
t = np.arange(N, dtype=float) / fs 
xo_2 = 1.5 * np.sin(2*np.pi*10.0*t + 0.0)

# 1) Señal temporal
plt.figure()
plt.plot(t, x, linewidth=1.0)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal de prueba 5 senos + Noise")
plt.grid(True)
plt.tight_layout()

plt.plot(t, xo_2, linewidth=1.0)


# %% numpy STFT

win_length_ms = 100.0   # duración de la ventana [ms]  -> controla resolución en frecuencia
hop_ms        = 5.0   # salto entre ventanas [ms]    -> controla resolución temporal
n_fft         = 1024   # tamaño de la FFT (potencia de 2 suele ser conveniente)

win_length = int(round(win_length_ms * 1e-3 * fs))   # muestras por ventana
hop_length = int(round(hop_ms        * 1e-3 * fs))    # muestras por hop

# ------------------------------------------------
# ------------------ STFT manual -----------------
# ------------------------------------------------
if win_length % 2 == 0:
    win_length += 1  # aseguramos longitud impar para centrar ventana
L = win_length             # longitud impar
sigma = 0.25*L         # en muestras
g = gaussian_window(L, sigma)

# 2) Ventana Gaussiana
n = np.arange(L) - (L - 1) / 2.0
plt.figure()
plt.plot(n, g, linewidth=1.0)
plt.xlabel("Muestra respecto al centro")
plt.ylabel("g[n] (normalizada) Numpy")
plt.title("Ventana Gaussiana (Heisenberg-óptima) - Numpy")
plt.grid(True)
plt.tight_layout()



# STFT
hop = hop_length            # [muestras] -> Δμ = hop/fs [s]
nfft = n_fft
S_numpy, times, freqs_cps = stft_manual(x, g, hop=hop, nfft=nfft)
S_numpy_xo2, _, _ = stft_manual(xo_2, g, hop=hop, nfft=nfft)

# # Escalas a unidades físicas
times_s = times / fs                  # μ_m en segundos
freqs_hz = freqs_cps * fs             # f_k en Hz

# 3) Módulo de la STFT |S|
plt.figure()
# espectrograma: tiempo en eje x, frecuencia en eje y
# S_db = 20.0 * np.log10(np.maximum(np.abs(S), 1e-10))
S_graph = np.abs(S_numpy)  # para graficar
extent = [times_s[0], times_s[-1], freqs_hz[0], freqs_hz[-1]]

plt.imshow(
    S_graph.T,
    origin="lower",
    aspect="auto",
    extent=extent,
)
plt.xlabel("Tiempo μ [s]")
plt.ylabel("Frecuencia f [Hz]")
plt.title("|S_x(μ, ξ)|  (STFT) - Numpy")
plt.colorbar(label="Magnitud ")
plt.ylim(0, 500)
plt.tight_layout()


plt.figure(figsize=(7,4))
cs = plt.contour(times_s, freqs_hz, S_graph.T, cmap='viridis')
# plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
plt.title('Contornos |S_x(μ, ξ)|  (STFT) - Numpy')
plt.xlabel("Tiempo μ [s]")
plt.ylabel('Frecuencia [Hz]')
plt.colorbar(label='Magnitud')
plt.ylim(0, 500)





# %% SCIPY STFT

# ---------------------------------
# ---------- SCIPY STFT -----------
# ---------------------------------

nperseg = int(fs * win_length_ms / 1000.0)   # en muestras
window = get_window(('gaussian', sigma), nperseg)
noverlap = int(0.90 * nperseg)

# 2) Ventana Gaussiana
n = np.arange(nperseg)
plt.figure()
plt.plot(n, window, linewidth=1.0)
plt.xlabel("Muestra respecto al centro")
plt.ylabel("g[n] - SCIPY")
plt.title("Ventana Gaussiana - SCIPY")
plt.grid(True)
plt.tight_layout()

freqs_scipy, t_scipy, S_scipy = stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary=None)

# Módulo de la STFT |S|
plt.figure()
# espectrograma: tiempo en eje x, frecuencia en eje y
# S_db = 20.0 * np.log10(np.maximum(np.abs(S), 1e-10))
S_graph = np.abs(S_scipy)  # para graficar
extent = [t_scipy[0], t_scipy[-1], freqs_scipy[0], freqs_scipy[-1]]

plt.imshow(
    S_graph,
    origin="lower",
    aspect="auto",
    extent=extent,
)
plt.xlabel("Tiempo μ [s]")
plt.ylabel("Frecuencia f [Hz]")
plt.title("|S_x(μ, ξ)|  (STFT) - SCIPY")
plt.colorbar(label="Magnitud ")
plt.ylim(0, 500)
plt.tight_layout()

plt.figure(figsize=(7,4))
cs = plt.contour(t_scipy, freqs_scipy, S_graph, cmap='viridis')
# plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
plt.title('Contornos |S_x(μ, ξ)|  (STFT)- SCIPY')
plt.xlabel("Tiempo μ [s]")
plt.ylabel('Frecuencia [Hz]')
plt.colorbar(label='Magnitud')
plt.ylim(0, 500)






#%%

S = S_numpy #  Usamos STFT de Numpy 

# Derivada respecto a μ y frecuencia instantánea (angular por muestra)
dS_dmu = partial_mu_central(S, delta_mu_samples=float(hop))
omega_hat, valid = inst_freq_from_derivative(S, dS_dmu, rel=0.3   , k_smooth=0)  # rad/muestra
# IF en Hz
fhat_hz = (omega_hat * fs) / (2*np.pi)  # Hz
plt.figure()
plt.imshow(fhat_hz.T, origin="lower", aspect="auto",
           extent=[times_s[0], times_s[-1], freqs_hz[0], freqs_hz[-1]],
           interpolation="none")
plt.xlabel("Tiempo μ [s]")
plt.ylabel("Frecuencia [Hz] (rejilla STFT)")
plt.title("IF válida por píxel")
cbar = plt.colorbar(); cbar.set_label("Hz")
plt.ylim(0, 500)

plt.figure()
plt.imshow(valid.T, origin='lower', aspect='auto',
           extent=[times_s[0], times_s[-1], freqs_hz[0], freqs_hz[-1]])
plt.title('Píxeles válidos usados en IF')
plt.ylabel('Hz'); plt.xlabel('s'); plt.ylim(0, 500)
plt.show()




# %%

# Synchro-Squeezing en la misma rejilla de frecuencia
out_freqs_cps = freqs_cps  # podrías hacerla más densa si deseas
T = synchrosqueeze(S, omega_hat, freqs_cps, out_freqs_cps, rel=0.01)


# Módulo del Synchro-Squeezed |T|
# SST: ojo, 'out_freqs_cps' está en ciclos/muestra; conviértelo a Hz para el eje.
out_freqs_hz = out_freqs_cps * fs
plt.figure()
plt.imshow(np.abs(T).T, origin="lower", aspect="auto",
           extent=[times_s[0], times_s[-1], out_freqs_hz[0], out_freqs_hz[-1]],
           interpolation="none") 
plt.xlabel("Tiempo μ [s]")
plt.ylabel("Frecuencia [Hz] (salida SST)")
plt.title("|T_x(μ, ω)|")
plt.colorbar(label="Magnitud")
plt.ylim(0, 500)




# %%

# ------------------ Gráficas pedagógicas ------------------








# plt.figure(figsize=(7,4))
# cs = plt.contour(t_scipy, freqs_scipy, T_graph.T, cmap='viridis')
# # plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
# plt.title('Contornos |T_x(μ, ω)|  (Synchro-Squeezing)')
# plt.xlabel("Tiempo μ [s]")
# plt.ylabel('Frecuencia [Hz]')
# plt.colorbar(label='Magnitud')
# plt.ylim(0, 500)

# 6) Cortes espectrales: comparar |S| vs |T| en un instante
# Elegimos un tiempo intermedio y mostramos el perfil de frecuencia
mid_m = len(t_scipy) // 2
# plt.figure()
# plt.plot(freqs_scipy, np.abs(S[mid_m, :]), linewidth=1.0, label="|S| (STFT)")
# plt.plot(freqs_scipy, np.abs(T[mid_m, :]), linewidth=1.0, label="|T| (SST)")
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Magnitud")
# plt.title("Corte en μ ≈ {:.3f} s: STFT vs SST".format(t_scipy[mid_m]))
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# 7) Señal y crestas esperadas (solo guía visual, sin detección)
plt.figure()
plt.plot(t, x, linewidth=0.8)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal (guía) — usa el mapa SST para leer trayectorias IF")
plt.grid(True)
plt.tight_layout()

plt.show()



