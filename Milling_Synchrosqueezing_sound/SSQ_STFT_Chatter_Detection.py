import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft
from ssqueezepy.experimental import scale_to_freq
from scipy.signal import  get_window
from typing import Tuple, List, Union, Dict, Optional, Sequence

from statsmodels.stats.diagnostic import lilliefors



#%%# Define signal ####################################
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


def extract_local_windows(
    S1: np.ndarray,
    K: int,
    time_vector: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae submatrices A_i de S1 y devuelve los tiempos que representan cada A_i.
    
    Parámetros
    ----------
    S1 : np.ndarray
        Matriz tiempo-frecuencia (MxN).
    K : int
        Longitud de la ventana (en columnas).
    time_vector : np.ndarray | None
        Vector de tiempos (N,). Si no se pasa, se asume np.arange(N).
    
    Retorna
    -------
    A_all : np.ndarray
        Tensor 3D con forma (N-K+1, M, K), cada A_i es una submatriz local.
    t_Ai : np.ndarray
        Vector 1D con los tiempos asociados a cada submatriz (los tiempos actuales).
    """

    if not isinstance(S1, np.ndarray):
        raise TypeError("S1 debe ser un array de NumPy.")
    M, N = S1.shape
    if K <= 0 or K > N:
        raise ValueError("K debe ser mayor que 0 y menor o igual a N.")
    
    # Si no se proporciona vector de tiempos, usamos índices
    if time_vector is None:
        time_vector = np.arange(N)

    num_windows = N - K + 1
    A_all = np.empty((num_windows, M, K), dtype=S1.dtype)

    for i in range(num_windows):
        A_all[i] = S1[:, i:i + K]
    
    # Tiempos asociados → últimos de cada ventana
    t_Ai = time_vector[K-1:]
    
    return A_all, t_Ai



def compute_svd(A: np.ndarray, ensure_real: bool = True
               ) -> Union[
                   Tuple[np.ndarray, np.ndarray, np.ndarray],
                   Tuple[np.ndarray, np.ndarray, np.ndarray]
               ]:
    """
    SVD flexible:
      - A 2D: (M, K) -> U (M,r), S (r,), Vh (r,K)
      - A 3D: (B, M, K) -> U (B,M,r), S (B,r), Vh (B,r,K)
    donde r = min(M, K).

    ensure_real=True convierte a real los resultados si son 'casi reales'.
    """

    if A.ndim == 2:
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        if ensure_real:
            U  = np.real_if_close(U)
            S  = np.real_if_close(S)
            Vh = np.real_if_close(Vh)
        return U, S, Vh

    elif A.ndim == 3:
        # np.linalg.svd soporta batch: (..., M, N)
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        if ensure_real:
            U  = np.real_if_close(U)
            S  = np.real_if_close(S)
            Vh = np.real_if_close(Vh)
        return U, S, Vh

    else:
        raise ValueError("El array A debe ser 2D (M,K) o 3D (B,M,K).")

def detectar_chatter_3sigma(
    d1: np.ndarray,
    idx_estable: Optional[Sequence[int]] = None,
    fraccion_estable: float = 0.2,
    alpha: float = 0.05,
    z: float = 3.0,
    fallback_mad: bool = True,
) -> Dict[str, object]:
    """
    Detección de chatter por criterio ±z·σ (por defecto 3σ) con verificación de normalidad.
    NO genera gráficos.

    Parámetros
    ----------
    d1 : np.ndarray
        Vector de indicadores [d1^1, ..., d1^n].
    idx_estable : Sequence[int], opcional
        Índices que se consideran corte estable para estimar μ y σ.
        Si es None, se usa la fracción inicial 'fraccion_estable'.
    fraccion_estable : float
        Fracción inicial considerada estable cuando idx_estable es None. (default 0.2)
    alpha : float
        Nivel de significancia para Lilliefors (default 0.05).
    z : float
        Multiplicador de sigma (default 3.0).
    fallback_mad : bool
        Si True y falla normalidad, aplica umbral robusto por MAD.

    Retorna
    -------
    dict con las claves:
        mask            : np.ndarray (0=estable, 1=chatter)
        mu              : float (media estable; si robusto, mediana)
        sigma           : float (desv. estándar; si robusto, 1.4826*MAD)
        lim_inf         : float
        lim_sup         : float
        normal_ok       : bool (p>alpha en Lilliefors)
        p_value         : float (Lilliefors)
        metodo_umbral   : str ("3sigma" o "MAD")
        idx_estable_usados : np.ndarray (índices efectivos usados)
    """
    d1 = np.asarray(d1, dtype=float)

    # Sanitizar datos
    if d1.ndim != 1 or d1.size == 0:
        raise ValueError("d1 debe ser un vector 1D no vacío.")
    if np.any(~np.isfinite(d1)):
        raise ValueError("d1 contiene NaN/Inf; limpie los datos antes de continuar.")

    n = d1.size

    # Elegir subconjunto estable
    if idx_estable is None:
        n_estable = max(3, int(np.ceil(n * fraccion_estable)))
        idx_est = np.arange(n_estable)
    else:
        idx_est = np.array(idx_estable, dtype=int)
        if idx_est.size < 3:
            raise ValueError("Se requieren ≥3 índices estables para estimar μ y σ.")
        if np.any((idx_est < 0) | (idx_est >= n)):
            raise ValueError("idx_estable contiene índices fuera de rango.")

    d1_est = d1[idx_est]

    # Estimadores clásicos
    mu = float(np.mean(d1_est))
    # ddof=1 para estimador insesgado; si var=0, evitamos división por cero abajo
    sigma = float(np.std(d1_est, ddof=1))

    # Test de normalidad (solo sobre tramo estable)
    stat, p_value = lilliefors(d1_est)
    normal_ok = bool(p_value > alpha)

    metodo = "3sigma"
    lim_inf = mu - z * sigma
    lim_sup = mu + z * sigma

    # Caso degenerado: sigma=0 (todo plano en estable)
    if sigma == 0.0:
        # Si todo estable es constante, cualquier desviación > 0 marca chatter.
        # Para ser prudentes, usamos una tolerancia relativa al nivel.
        eps = 1e-12 if mu == 0 else 1e-6 * abs(mu)
        lim_inf = mu - z * eps
        lim_sup = mu + z * eps

    # Fallback robusto si falla normalidad
    if fallback_mad and not normal_ok:
        med = float(np.median(d1_est))
        mad = float(np.median(np.abs(d1_est - med)))
        # Consistencia para normal: 1.4826*MAD ≈ σ
        sigma_rob = 1.4826 * mad
        # Si mad=0, aplicar tolerancia similar al caso sigma=0
        if sigma_rob == 0.0:
            eps = 1e-12 if med == 0 else 1e-6 * abs(med)
            lim_inf = med - z * eps
            lim_sup = med + z * eps
        else:
            lim_inf = med - z * sigma_rob
            lim_sup = med + z * sigma_rob
        mu, sigma = med, sigma_rob
        metodo = "MAD"

    # Máscara de chatter
    mask = ( (d1 < lim_inf) | (d1 > lim_sup) ).astype(int)

    return {
        "mask": mask,
        "mu": mu,
        "sigma": sigma,
        "lim_inf": lim_inf,
        "lim_sup": lim_sup,
        "normal_ok": normal_ok,
        "p_value": float(p_value),
        "metodo_umbral": metodo,
        "idx_estable_usados": idx_est,
    }


def signal_1 (fs: float, T: float,
              tpf: float, chatter_freqs: List[float]= [0.0],
              t_chatter_start: float = [0.0],
              noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    
    A = 3.5  # Amplitud de los senos base
    # ==============================
    # Parámetros generales
    # ==============================
    T = 25           # Duración total (s)
    t = np.linspace(0, T, int(fs * T))  # Vector de tiempo

    # ==============================
    # Tooth Passing Frequency (TPF)
    # ==============================
    harmonics = [tpf * i for i in range(1, 6)]  

    signal_base = np.zeros_like(t)
    
    for f in harmonics:
        signal_base += 3.5*np.sin(2 * np.pi * f * t)

    # ==============================
    # Frecuencias de chatter (aparecen después de 8 s)
    # ==============================
    signal_chatter = np.zeros_like(t)

    envelope_base = 0.1 + 0.6 * (t / T)*1


    # Envolvente que activa las chatter después de 8 s
    mask_chatter = (t > 8)
    envelope_chatter = np.zeros_like(t)
    # envelope_chatter[mask_chatter] = np.linspace(0, 1, np.sum(mask_chatter)) **2
    envelope_chatter[mask_chatter] = 0.1 + 0.6* ((t[mask_chatter] - t_chatter_start) / (T - t_chatter_start))*1

    for f in chatter_freqs:
        # Pequeña modulación en frecuencia (para hacerlas inestables)
        mod = 0.02 * np.sin(2 * np.pi * 0.3 * t)
        mod = 0
        signal_chatter += 5*np.sin(2 * np.pi * (f + f * mod) * t)

    signal_chatter *= envelope_chatter

    # ==============================
    # Ruido blanco
    # ==============================
    noise =noise_std * np.random.randn(len(t))

    # ==============================
    # Combinación y envolvente global
    # ==============================
    # Envolvente de amplitud global con crecimiento exponencial suave


    # envelope_total = np.ones_like(t) # Alternativa: sin envolvente global   

    signal = envelope_base * signal_base + envelope_chatter * signal_chatter + noise
    
    return t, signal
    
    
    
    
    
#%%# Generate signal ##################################
fs = 10240.0          # [Hz] frecuencia de muestreo
duration = 2.0       # [s] duración
t,x_five = five_senos(fs=fs, duracion=duration, ruido_std=4, fase_aleatoria=False, seed=120)

tpf = 480  # Frecuencia base
chatter_freqs = [390, 920, 850, 1375, 1300, 2168, 2350] 
# t, x_signal1 = signal_1(fs=fs, T=duration, tpf=tpf,
#                         chatter_freqs=chatter_freqs,
#                         t_chatter_start=8.0,
#                         noise_std=0.02) 

win_length_ms = 100.0   # duración de la ventana [ms]  -> controla resolución en frecuencia
hop_ms        = 10.0   # salto entre ve ntanas [ms]    -> controla resolución temporal
n_fft         = 1024*2  # tamaño de la FFT (potencia de 2 suele ser conveniente)

win_length = int(round(win_length_ms * 1e-3 * fs))   # muestras por ventana
hop_length = int(round(hop_ms        * 1e-3 * fs))    # muestras por hop

L = win_length             # longitud impar
sigma = L/6.         # en muestras
window = get_window(('gaussian', sigma), win_length)

# N = int(np.round(fs * duration)) 
# t = np.arange(N, dtype=float) / fs

x_use = x_five  


plt.figure()
plt.plot(t, x_use, linewidth=1.0)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal de prueba 5 senos + Noise")
plt.grid(True)
plt.tight_layout()

#%%# STFT + SSQ STFT ##################################

Tsx, Sx, _, _, w, dSx = ssq_stft(x_use, window=window, n_fft=n_fft, win_len=win_length, hop_len=hop_length, fs=fs,
                                 get_dWx=True,
                                 get_w=True,
                                 )

# Sxx_dB = 10 * np.log10(abs(Sx) + 1e-12)
# vmax = np.percentile(Sxx_dB, 99)   # el valor alto (techo)
# vmin = vmax - 20      

f = np.linspace(0, fs/2, Sx.shape[0])
t = np.arange(Sx.shape[1]) * hop_length / fs

plt.figure(figsize=(7,4))
plt.pcolormesh(t, f, abs(Sx), shading='auto', cmap='jet', vmin=None, vmax=None)
plt.title("|S_x(μ, ξ)|  (STFT)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")  
plt.ylim(0, 3500)
plt.colorbar(label="Magnitud") 
plt.show()  


# plt.figure(figsize=(7,4))
# plt.contour(t, f, np.abs(Sx), cmap='turbo')
# # plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
# plt.title('Contornos |S_x(μ, ξ)|  (STFT)')
# plt.xlabel("Tiempo μ [s]")
# plt.ylabel('Frecuencia [Hz]')
# plt.colorbar(label='Magnitud')
# plt.ylim(0, 3500)

# Tsx_dB = 10 * np.log10(abs(Tsx) + 1e-12)
# vmax = np.percentile(Tsx_dB, 99)   # el valor alto (techo)
# vmin = vmax - 20      


plt.figure(figsize=(7,4))
plt.pcolormesh(t, f, np.abs(Tsx), shading='auto', cmap='jet', vmin=None, vmax=None)
plt.title("|T_x(μ, ω)| (SSQ STFT)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.ylim(0, 3500)
plt.colorbar(label="Magnitud")
plt.show()

# plt.figure(figsize=(7,4))
# plt.contour(t, f, Tsx_dB, cmap='jet')
# # plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
# plt.title('Contornos |T_x(μ, ω)| (SSQ STFT)')
# plt.xlabel("Tiempo μ [s]")
# plt.ylabel('Frecuencia [Hz]')
# plt.colorbar(label='Magnitud')
# plt.ylim(0, 3500)

#%%# SVD de submatrices A_i extraídas de Tsx ##########
STFT_use = Tsx
M = STFT_use.shape[0]  # número de frecuencias
N = STFT_use.shape[1]  # número de tiempos

print(f"\nM (frecuencias) = {M}, N (tiempos) = {N}")

S1 = STFT_use  # matriz tiempo-frecuencia (M x N)
K = 4     # longitud de la ventana temporal (número de columnas)

A_i, t_i= extract_local_windows(S1, K)

print(f"Total de submatrices extraídas: {A_i.shape[0]}")
print(f"Cada A_i tiene tamaño: {A_i[0].shape}")

U_0, D_0, Vt_0 = compute_svd(A_i[0])

print("\nValores singulares de la primera A_i:")
print(D_0)
  
U, D, Vt = compute_svd(A_i, ensure_real=True)
print(f"\nSVD de todas las submatrices A_i:")
print(f"U tiene forma: {U.shape}")
print(f"D tiene forma: {D.shape}")
print(f"Vt tiene forma: {Vt.shape}")

print("\nValores singulares de la primera A_i:")
print(D[0])

print("\nValores singulares de la segunda A_i:")
print(D[1])

d1 = D[:, 0]



#%%" Chatter Criteria based on SVD ##########################"


res = detectar_chatter_3sigma(d1, fraccion_estable=0.25, alpha=0.05, z=3.0)

print("metodo_umbral :", res["metodo_umbral"])
print("normal_ok     :", res["normal_ok"], f"(p={res['p_value']:.4f})")
print("mu, sigma     :", f"{res['mu']:.4f}", f"{res['sigma']:.4f}")
print("lim_inf/sup   :", f"{res['lim_inf']:.4f}", f"{res['lim_sup']:.4f}")
print("chatter(%)    :", f"{100*res['mask'].mean():.2f}%")


plt.figure(figsize=(7,4))
plt.plot(t_i, d1, marker='o')
plt.title("Primer valor singular de cada A_i a lo largo del tiempo")
plt.hlines([res['lim_inf'], res['lim_sup']], xmin=t_i.min(), xmax=t_i.max(), colors='red', linestyles='dashed', label='Límites de chatter')
