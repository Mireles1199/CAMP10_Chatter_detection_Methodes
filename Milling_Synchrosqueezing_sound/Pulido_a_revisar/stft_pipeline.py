#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STFT Pipeline - version PLUS (autocontenida, mas fiel y robusta)
----------------------------------------------------------------
Agrega utilidades clave respecto a tu stft_pipeline.py:
  - p2up(): padding a la siguiente potencia de 2 (mas fiel al original)
  - window_area() y window_resolution(): metricas de ventana
  - istft(): reconstruccion temporal desde Sx
  - Correccion get_window_2 -> get_window (SciPy)
  - Pequenas mejoras de robustez sin romper tu API

Comentarios en espanol; nombres de variables y texto en ingles.
"""

import numpy as np
import scipy.signal as sig
import logging
from scipy.fft import fft, ifft, rfft, irfft, fftshift, ifftshift
import matplotlib.pyplot as plt

# ===========================================================
# Configuracion y backend opcional GPU
# ===========================================================
logging.basicConfig(format='')
WARN = lambda msg: logging.warning(f"WARNING: {msg}")

try:
    import torch
    USE_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None
    USE_TORCH = False


# ===========================================================
# Utilidades nuevas (PLUS)      
# ===========================================================
def p2up(N: int) -> int:
    """Devuelve la siguiente potencia de 2 >= N (util para padding FFT)."""
    if N <= 1:
        return 1
    return 1 << (int(np.ceil(np.log2(N))))


def window_area(window: np.ndarray, time: bool = True, frequency: bool = False):
    """Area temporal/frecuencial de la ventana (energia integrada)."""
    if not time and not frequency:
        raise ValueError("Debe pedirse `time` o `frequency`.")
    # Area en tiempo
    at = None
    if time:
        t = np.arange(-len(window)/2, len(window)/2, step=1)
        at = np.trapezoid(np.abs(window)**2, t)
    # Area en frecuencia
    aw = None
    if frequency:
        ws = np.fft.fftfreq(len(window), 1.0)  # unidades relativas
        apsih2s = np.abs(fftshift(fft(window)))**2
        aw = np.trapezoid(apsih2s, np.fft.fftshift(ws))
    return (at, aw) if (time and frequency) else (at if time else aw)


def window_resolution(window: np.ndarray):
    """Anchos efectivos de ventana: std_w, std_t, harea (producto de anchos)."""
    # Dominio temporal
    N = len(window)
    t = np.arange(-N/2, N/2, step=1)
    apsi2 = np.abs(window)**2
    # Dominio de frecuencia angular (centrado)
    ws = np.fft.fftfreq(N, d=1.0) * 2*np.pi
    ws = fftshift(ws)
    psihs = fftshift(fft(window))
    apsih2s = np.abs(psihs)**2

    # Varianzas normalizadas
    var_t = np.trapezoid(t**2 * apsi2, t) / np.trapezoid(apsi2, t)
    var_w = np.trapezoid(ws**2 * apsih2s, ws) / np.trapezoid(apsih2s, ws)
    std_t, std_w = np.sqrt(var_t), np.sqrt(var_w)
    harea = std_t * std_w
    return std_w, std_t, harea


# ===========================================================
# Funciones comunes (de tu archivo original, con retoques)
# ===========================================================
def _process_params_dtype(x, dtype='float32', auto_gpu=False):
    """Convierte arrays segun dtype y GPU si esta disponible."""
    if USE_TORCH and auto_gpu:
        return torch.as_tensor(x, dtype=getattr(torch, dtype), device='cuda')
    if USE_TORCH and isinstance(x, torch.Tensor):
        return x.to(dtype=getattr(torch, dtype))
    return np.asarray(x, dtype=dtype)


def padsignal(x, padtype='reflect', padlength=None):
    """Padding simetrico/reflectivo con longitud objetivo `padlength`."""
    N = x.shape[-1]
    if padlength is None:
        n_up = p2up(N)
    else:
        n_up = padlength
    diff = n_up - N
    if diff < 0:
        # Si N ya es mayor que padlength, no recortamos; simplemente devolvemos x
        WARN("`padlength` menor que N; no se aplicara padding.")
        return x

    n1 = diff // 2
    n2 = diff - n1

    if padtype == 'reflect':
        xp = np.pad(x, (n1, n2), mode='reflect')
    elif padtype == 'zero':
        xp = np.pad(x, (n1, n2))
    elif padtype == 'symmetric':
        xp = np.pad(x, (n1, n2), mode='symmetric')
    else:
        raise ValueError(f"padtype desconocido: {padtype}")
    return xp


def _xifn(scale, N):
    """Frecuencias angulares discretas (identico al enfoque del original)."""
    return 2 * np.pi * np.fft.fftfreq(N, 1 / N)


# ===========================================================
# Procesamiento de fs y t
# ===========================================================
def _process_fs_and_t(fs, t, N):
    """Valida y unifica `fs` y `t` con longitud N."""
    if fs is not None and t is not None:
        WARN("`t` sobrescribe `fs` (ambos fueron pasados)")
    if t is not None:
        if len(t) != N:
            raise ValueError(f"`t` debe tener la misma longitud que x ({len(t)} != {N})")
        if not np.mean(np.abs(np.diff(t, 2, axis=0))) < 1e-7:
            raise ValueError("`t` debe estar uniformemente espaciado.")
        fs = 1 / (t[1] - t[0])
    else:
        if fs is None:
            fs = 1
        elif fs <= 0:
            raise ValueError("`fs` debe ser > 0")
    dt = 1 / fs
    return dt, fs, t


# ===========================================================
# Bufferizado y reconstruccion por solapamiento
# ===========================================================
def buffer(x, seg_len, n_overlap, modulated=False):
    """Construye matriz con segmentos solapados (columnas) de longitud seg_len."""
    hop = seg_len - n_overlap
    n_segs = (len(x) - seg_len) // hop + 1
    s20 = int(np.ceil(seg_len / 2))
    s21 = s20 - 1 if seg_len % 2 == 1 else s20
    out = np.zeros((seg_len, n_segs), dtype=x.dtype)

    for i in range(n_segs):
        if not modulated:
            start = hop * i
            end = start + seg_len
            out[:, i] = x[start:end]
        else:
            start0 = hop * i
            end0 = start0 + s21
            start1 = end0
            end1 = start1 + s20
            out[:s20, i] = x[start1:end1]
            out[s20:, i] = x[start0:end0]
    return out


def unbuffer(xbuf, window, hop_len, n_fft, N, win_exp=1):
    """Overlap-add inverso con potencia de ventana `win_exp`."""
    if N is None:
        N = xbuf.shape[1] * hop_len + len(window) - 1
    if len(window) != n_fft:
        raise ValueError(f"len(window) != n_fft ({len(window)} != {n_fft})")
    if win_exp == 0:
        window = 1
    elif win_exp != 1:
        window = window ** win_exp
    x = np.zeros(N + n_fft - 1, dtype=xbuf.dtype)

    for i in range(xbuf.shape[1]):
        n = i * hop_len
        x[n:n + n_fft] += xbuf[:, i] * window
    return x


def window_norm(window, hop_len, n_fft, N, win_exp=1):
    """Factor de normalizacion para overlap-add con potencia de ventana `win_exp`."""
    wn = np.zeros(N + n_fft - 1)
    max_hops = (len(wn) - n_fft) // hop_len + 1
    wpow = window ** (win_exp + 1)
    for i in range(max_hops):
        n = i * hop_len
        wn[n:n + n_fft] += wpow
    return wn


# ===========================================================
# Ventanas
# ===========================================================
def get_window(window, win_len, n_fft=None, derivative=False, dtype='float32'):
    """Obtiene ventana (SciPy) y opcionalmente su derivada en el tiempo."""
    if n_fft is None:
        n_fft = win_len
    if window is None:
        window = sig.windows.dpss(win_len, max(4, win_len // 8), sym=False)
    elif isinstance(window, str) or isinstance(window, tuple):
        window = sig.get_window(window, win_len, fftbins=True)
    elif isinstance(window, np.ndarray):
        if len(window) != win_len:
            WARN(f"len(window) != win_len ({len(window)} != {win_len})")
    else:
        raise ValueError("`window` invalida")

    if len(window) < n_fft:
        pad_left = (n_fft - win_len) // 2
        pad_right = n_fft - win_len - pad_left
        window = np.pad(window, (pad_left, pad_right))
    window = window.astype(dtype)

    if derivative:
        wf = fft(window)
        Nw = len(window)
        xi = _xifn(1, Nw)
        if Nw % 2 == 0:
            xi[Nw // 2] = 0
        diff_window = np.real(ifft(wf * 1j * xi))
        diff_window = diff_window.astype(dtype)
        return window, diff_window
    return window


# ===========================================================
# Verificacion NOLA
# ===========================================================
def _check_NOLA(window, hop_len, dtype='float32'):
    """Comprueba la condicion NOLA para invertibilidad numerica."""
    if hop_len > len(window):
        WARN("`hop_len > len(window)`; STFT no invertible")
    elif not sig.check_NOLA(window, len(window), len(window) - hop_len):
        WARN("La ventana no cumple NOLA; STFT no invertible")
    # Aviso adicional por precision float32 (igual espiritu al original)
    try:
        if dtype == 'float32' and not sig.check_NOLA(window, len(window), len(window) - hop_len, tol=1e-3):
            WARN("Posible imprecision float32 en el ultimo hop de senal.")
    except TypeError:
        # versiones antiguas de SciPy pueden no exponer 'tol' -> ignorar
        pass


# ===========================================================
# FFT helpers
# ===========================================================
def rfft_tensor(x, axis=0):
    """rFFT unificada para NumPy/Torch (devuelve mismo tipo de entrada)."""
    if USE_TORCH and isinstance(x, torch.Tensor):
        return torch.fft.rfft(x, dim=axis)
    return np.fft.rfft(x, axis=axis)


def irfft_tensor(X, n=None, axis=0):
    """irFFT unificada para NumPy/Torch (devuelve mismo tipo de entrada)."""
    if USE_TORCH and isinstance(X, torch.Tensor):
        return torch.fft.irfft(X, n=n, dim=axis)
    return np.fft.irfft(X, n=n, axis=axis)


def ifftshift_tensor(x):
    """ifftshift unificada para NumPy/Torch."""
    if USE_TORCH and isinstance(x, torch.Tensor):
        return torch.fft.ifftshift(x)
    return ifftshift(x)


# ===========================================================
# STFT principal
# ===========================================================
def stft(x, window=None, n_fft=None, win_len=None, hop_len=1, fs=None, t=None,
         padtype='reflect', modulated=True, derivative=False, dtype='float32'):
    """STFT + derivada temporal (opcional). Devuelve (Sx, dSx, meta) si derivative; si no, (Sx, None, meta)."""
    assert x.ndim == 1, "solo 1D implementado"
    N = len(x)
    _, fs, _ = _process_fs_and_t(fs, t, N)
    if n_fft is None:
        n_fft = min(N // max(hop_len, 1), 512)
    win_len = win_len or n_fft

    window_arr, diff_window = get_window(window, win_len, n_fft, derivative=True, dtype=dtype)
    _check_NOLA(window_arr, hop_len, dtype=dtype)
    x = _process_params_dtype(x, dtype=dtype, auto_gpu=False)

    # padlength = p2up(N + n_fft - 1)
    padlength = N
    xp = padsignal(x, padtype, padlength=padlength)

    # Nota: mantenemos calculo en NumPy para buffer; se puede portar a torch si interesa.
    hop = hop_len
    Sx_frames = buffer(np.array(xp), n_fft, n_fft - hop, modulated)
    dSx_frames = buffer(np.array(xp), n_fft, n_fft - hop, modulated)

    if modulated:
        window_eff = ifftshift_tensor(window_arr)
        diff_window_eff = ifftshift_tensor(diff_window) * fs
    else:
        window_eff = window_arr
        diff_window_eff = diff_window * fs

    Sx_frames *= np.asarray(window_eff)[:, None]
    dSx_frames *= np.asarray(diff_window_eff)[:, None]

    Sx = rfft(Sx_frames, axis=0)
    dSx = rfft(dSx_frames, axis=0)

    meta = dict(padlength=padlength, N=N, window=window_arr, hop_len=hop_len, n_fft=n_fft, modulated=modulated)
    if derivative:
        return Sx, dSx, meta
    else:
        return Sx, None, meta


# ===========================================================
# iSTFT (reconstruccion) usando utilidades
# ===========================================================
def istft(Sx, meta, win_power: int = 1):
    """Reconstruye x aproximada a partir de Sx y metadatos devueltos por stft()."""
    padlength = meta["padlength"]
    N = meta["N"]
    window = meta["window"]
    hop_len = meta["hop_len"]
    n_fft = meta["n_fft"]
    modulated = meta["modulated"]

    # Inversa por columnas
    frames = irfft(Sx, n=n_fft, axis=0)

    # Ventana para sintesis (misma que en analisis)
    window_eff = ifftshift(window) if modulated else window

    # Overlap-add y normalizacion
    x_ola = unbuffer(frames, window_eff, hop_len, n_fft, N=None, win_exp=1)
    wn = window_norm(window_eff, hop_len, n_fft, len(frames)*hop_len, win_exp=win_power)
    wn[wn == 0] = 1.0  # evitar division por cero en extremos
    x_ola /= wn[:len(x_ola)]

    # Recortar padding al tamano original
    pad_left = (padlength - N) // 2
    x_rec = x_ola[pad_left:pad_left + N]
    return x_rec


# ===========================================================
# Generador de senal multi-seno (opcional)
# ===========================================================
def five_sines(fs: float,
               duration: float,
               noise_std: float = 0.0,
               random_phase: bool = False,
               seed: int | None = None):
    """Genera mezcla de cinco senos + ruido blanco opcional."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    N = int(np.round(fs * duration))
    t = np.arange(N, dtype=float) / fs

    comps = [
        (1.5,  80.0),
        (2.0, 120.0),
        (1.5, 160.0),
        (1.5, 240.0),
        (2.0, 320.0),
    ]

    x = np.zeros_like(t)
    for A, f in comps:
        phi = rng.uniform(0, 2*np.pi) if random_phase else 0.0
        x += A * np.sin(2*np.pi*f*t + phi)

    if noise_std > 0.0:
        x += rng.normal(0.0, noise_std, size=t.shape)

    return t, x

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


# ===========================================================
# Ejemplo minimo de uso (sin graficos)
# ===========================================================
if __name__ == "__main__":
    fs = 10240.0          # [Hz] frecuencia de muestreo
    duration = 1.0       # [s] duración
    t,x_five = five_senos(fs=fs, duracion=duration, ruido_std=5, fase_aleatoria=False, seed=120)

    win_length_ms = 75.0   # duración de la ventana [ms]  -> controla resolución en frecuencia
    hop_ms        = 2.0   # salto entre ventanas [ms]    -> controla resolución temporal
    n_fft         = 1024*2  # tamaño de la FFT (potencia de 2 suele ser conveniente)

    win_length = int(round(win_length_ms * 1e-3 * fs))   # muestras por ventana
    hop_length = int(round(hop_ms        * 1e-3 * fs))    # muestras por hop

    L = win_length             # longitud impar
    sigma = L/6.         # en muestras
    window = get_window(('gaussian', sigma), win_length)

    t = None
    _, fs_out, _ = _process_fs_and_t(fs, t, x_five.shape[-1])
    Sx, _, meta = stft(x_five, window=window, n_fft=n_fft, win_len=win_length, hop_len=hop_length, fs=fs, derivative=False)

    f = np.linspace(0, fs/2, Sx.shape[0])
    t = np.arange(Sx.shape[1]) * hop_length / fs

    plt.figure(figsize=(7,4))
    plt.pcolormesh(t, f, np.abs(Sx), shading='auto')
    plt.title("|S_x(μ, ξ)|  (STFT)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.ylim(0, 500)
    plt.colorbar(label="Magnitud")

    
    plt.figure(figsize=(7,4))
    cs = plt.contour(t, f, np.abs(Sx), cmap='turbo')
    # plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    plt.title('Contornos |S_x(μ, ξ)|  (STFT)')
    plt.xlabel("Tiempo μ [s]")
    plt.ylabel('Frecuencia [Hz]')
    plt.colorbar(label='Magnitud')
    plt.ylim(0, 500)

    # Reconstruccion iSTFT
    x_rec = istft(Sx, meta, win_power=1)
    print(f"x_rec.shape={x_rec.shape}, N_original={x_five.shape[0]}")
    
    
    plt.show()
