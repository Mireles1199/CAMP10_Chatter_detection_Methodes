from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional

# ------------------------------------------------------------
# Generación de señal sintética con espectro similar al de la figura
# - Picos muy fuertes en muy baja frecuencia
# - Picos pequeños 50–150 Hz
# - Pico notable ~200 Hz
# - Grupito alrededor de ~600 Hz + ruido estrecho 550–650 Hz
# - Piso de ruido bajo global
# ------------------------------------------------------------

def make_chatter_like_signal(
    *,
    fs: float = 10_000.0,        # Hz
    T: float = 6.0, 
    signal_chatter:  bool = False,
    t0_chatter: float = 3.0,    # s
    grow_tau: float = 3,       # s
    grow_gain: float = 20.0,
    f_chatter: float = 560.0,
    am_freqs_chatter: Tuple[float, ...] = (9.0, 17.0, 26.0),
    am_depths_chatter: Tuple[float, ...] = (0.25, 0.15, 0.06),
    base_chatter_amp: float = 0.1,
    
    seed: Optional[int] = 123,  # semilla para reproducibilidad
    low_freqs: Tuple[float, ...] = (8.0, 16.0,20),
    low_amps:  Tuple[float, ...] = (2.2, 0.9),
    mid_freqs: Tuple[float, ...] = (55.0, 72.0, 95.0, 110.0, 135.0, 150.0),
    mid_amps:  Tuple[float, ...] = (0.20, 0.12, 0.15, 0.10, 0.12, 0.10),
    f200_amp: float = 0.45,
    f200_hz: float = 200.0,
    cluster_center: Tuple[float, ...] = (20.0, 600.0, 200.0, 300, 400.0,800.0, 950.0),
    cluster_offsets: np.ndarray = np.array([[-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
                                           [-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0]]),
    
    cluster_amps:    np.ndarray = np.array([[0.24, 0.24, 0.20, 0.30, 0.18, 0.26, 0.10],
                                           [0.14, 0.14, 0.10, 0.15, 0.10, 0.16, 0.10],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1]]),
    narrow_noise_band: Tuple[float, float] = (0.0, 1000.0),
    narrow_noise_sigma: float = 2,
    white_noise_sigma: float = 1,
    return_components: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Devuelve:
        signal: np.ndarray con la señal sintética 1D
        meta:   diccionario con fs, t y (opcionalmente) componentes individuales
    """
    # -- reloj
    N: int = int(T * fs)
    t: np.ndarray = np.arange(N, dtype=float) / fs
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # -- 1) muy baja frecuencia dominante
    low_sig = sum(
        a * np.sin(2*np.pi*f*t + 2*np.pi*rng.random())
        for f, a in zip(low_freqs, low_amps)
    )

    # -- 2) varias líneas débiles 50–150 Hz
    mid_sig = sum(
        a * np.sin(2*np.pi*f*t + 2*np.pi*rng.random())
        for f, a in zip(mid_freqs, mid_amps)
    )

    # -- 3) pico ~200 Hz
    sig200 = f200_amp * np.sin(2*np.pi*f200_hz*t + 2*np.pi*rng.random())

    # -- 4) grupo alrededor de ~600 Hz
    # cluster_sig = sum(
    #     a * np.sin(2*np.pi*(cluster_center+df)*t + 2*np.pi*rng.random())
    #     for df, a in zip(cluster_offsets, cluster_amps)
    # )
    # --- 4a) cluster_sig (componente de clusters sinusoidales)
    cluster_sig = np.zeros_like(t, dtype=float)

    for idx, center in enumerate(cluster_center):
        dfs   = np.asarray(cluster_offsets[idx], dtype=float)
        amps  = np.asarray(cluster_amps[idx], dtype=float)

        phases = 2*np.pi * rng.random(size=dfs.shape)

        # grid (k,t)
        omega  = 2*np.pi*(center + dfs)[:, None] * t[None, :]
        sig_k  = amps[:, None] * np.sin(omega + phases[:, None])

        cluster_sig += sig_k.sum(axis=0)


    # --- 4b) ruido banda angosta 550–650 Hz (filtrado en frecuencia)
    wide_noise = narrow_noise_sigma * rng.standard_normal(N)
    F = np.fft.rfft(wide_noise)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    lo, hi = narrow_noise_band
    mask_band = (freqs >= lo) & (freqs <= hi)

    F_filtered = np.zeros_like(F)
    F_filtered[mask_band] = F[mask_band]

    narrow_noise = np.fft.irfft(F_filtered, n=N)


    # --- 5) piso de ruido blanco bajo
    white_noise = white_noise_sigma * rng.standard_normal(N)


    # --- señal final
    if signal_chatter:
    

        # --- ventana de arranque (suave) para que el chatter sea 0 antes de T1
        ramp_sec =2
        tau_ramp = ramp_sec/6
        w_start = 1.0 / (1.0 + np.exp(-(t - (t0_chatter + ramp_sec/2)) / (tau_ramp + 1e-12)))
        # (si quieres arranque duro: w_start = np.heaviside(t - T1, 1.0))
        w_start = np.heaviside(t - t0_chatter, 1.0)

        # --- envolvente sigmoide de crecimiento: ~1 antes de grow_t0 y -> grow_gain después
        grow_t0   = t0_chatter                     # centro del crecimiento (puede ser >= T1)
        sigmoid   = 1.0 / (1.0 + np.exp(-(t - grow_t0) / (grow_tau + 1e-12)))
        env_grow  = 1.0 + (grow_gain - 1.0) * sigmoid

        # --- chatter con AM (portadora + bandas laterales)

        rng = np.random.default_rng(7)
        phase_c = 2*np.pi*rng.random()
        carrier = base_chatter_amp * np.sin(2*np.pi*f_chatter*t + phase_c)

        am = np.ones_like(t)
        for fm, dm in zip(am_freqs_chatter, am_depths_chatter):
            am *= (1.0 + dm * np.sin(2*np.pi*fm*t + 2*np.pi*rng.random()))

        chatter_raw = carrier * am

        # --- chatter final: solo desde T1 y creciendo luego
        chatter_sig =  env_grow * chatter_raw
        
        
        
        signal = low_sig + mid_sig + sig200 + chatter_sig + narrow_noise + white_noise + cluster_sig
    else:
        signal = low_sig + mid_sig + sig200 + cluster_sig + narrow_noise + white_noise


    meta: Dict[str, Any] = {"fs": fs, "t": t}
    if return_components:
        meta.update({
            "low_sig": low_sig,
            "mid_sig": mid_sig,
            "sig200": sig200,
            "cluster_sig": cluster_sig,
            "narrow_noise": narrow_noise,
            "white_noise": white_noise,
        })
    return signal, meta


# ------------------------------------------------------------
# Utilidad para espectro de amplitud (sin graficar)
# ------------------------------------------------------------
def amplitude_spectrum(
    signal: np.ndarray,
    fs: float,
    *,
    fmax: Optional[float] = None,
    window: str = "hann",
    normalize_to: Optional[float] = None,  # p.ej. 0.1 para escalar el máximo a 0.1
    clip_max: Optional[float] = None,      # p.ej. 0.1 para recortar valores por encima
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula espectro de amplitud (0..Nyquist) con opciones de normalización/recorte.

    Parámetros
    ----------
    normalize_to : float | None
        Si se da, escala todo el espectro para que su máximo sea 'normalize_to'.
    clip_max : float | None
        Si se da, recorta amplitudes por encima de este valor.

    Retorna
    -------
    freqs : np.ndarray
    amp   : np.ndarray
    """
    # -- FFT con ventana
    x = np.asarray(signal, dtype=float)
    N = x.size
    if window == "hann":
        win = np.hanning(N)
    elif window is None:
        win = np.ones(N)
    else:
        raise ValueError("window debe ser 'hann' o None")

    X = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # -- Escala proporcional (útil para comparar picos)
    amp = (2.0 / np.sum(win)) * np.abs(X) * np.sum(win)

    # -- Limitar rango de frecuencias si se pide
    if fmax is not None:
        mask = freqs <= float(fmax)
        freqs = freqs[mask]
        amp = amp[mask]

    # -- Normalización opcional (máximo -> normalize_to)
    if normalize_to is not None:
        max_amp = float(np.max(amp)) if amp.size else 0.0
        if max_amp > 0.0:
            amp = amp * (normalize_to / max_amp)

    # -- Recorte opcional
    if clip_max is not None:
        amp = np.minimum(amp, float(clip_max))

    return freqs, amp
# ------------------------------------------------------------
# Ejemplo de uso (no grafica automáticamente)
# ------------------------------------------------------------
if __name__ == "__main__":
    sig, meta = make_chatter_like_signal(fs=5_000.0, T=6.0, seed=123, signal_chatter=False)
    f, A = amplitude_spectrum(sig, meta["fs"], fmax=1000.0, normalize_to=0.1)

    # Aquí podrías guardar para usar en tu pipeline:
    # np.save("synthetic_chatter_like_signal.npy", sig)

    # Si luego quieres graficar, puedes hacerlo fuera:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(f, A)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1000)
    plt.show()
    
    plt.figure()
    plt.plot(meta["t"], sig)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Amplitude")  
    plt.show()
