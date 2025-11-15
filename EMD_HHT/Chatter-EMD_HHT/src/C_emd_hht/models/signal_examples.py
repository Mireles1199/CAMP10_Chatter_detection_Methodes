
import numpy as np

from C_emd_hht import make_chatter_like_signal, amplitude_spectrum
def signal_chatter_example(fs=2000.0, T=6.0, seed=123):
    # --- 1) Señal de ejemplo (sintética). Sustituye por tu F real ---


    cluster_center =  (20.0, 600.0, 200.0, 300, 400.0,800.0, 950.0)
    cluster_offsets = np.array([[-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
                                            [-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
                                            [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                            [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                            [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 30.0],
                                            [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                            [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0]])
    cluster_amps = np.array([[0.24, 0.24, 0.20, 0.30, 0.18, 0.26, 0.10],
                                            [0.14, 0.14, 0.10, 0.15, 0.10, 0.16, 0.10],
                                            [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                            [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                            [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                            [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                            [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1]])

    t0_chatter = 2.0
    ramp_sec = 1.0
    
    narrow_noise_sigma = 0.05
    white_noise_sigma = 0.02

    am_freqs_chatter=np.array([0.7, 1.5, 2.5])
    am_depths_chatter=np.array([3, 1.0, 0.5]) 

    am_freqs_chatter *= 7
    am_depths_chatter *= 2
    
    base_chatter_amp = 3
    grow_gain = 500
    grow_tau = 5

    f200_hz = 200.0
    f200_amp = 1
    low_freqs = np.array([8.0, 16.0, 20.0])
    low_amps = np.array([0.5, 0.9, 0.5])
    mid_freqs = np.array([55.0, 72.0, 95.0, 110.0, 135.0, 150.0])
    mid_amps = np.array([0.02, 0.03, 0.02, 0.01, 0.01, 0.01])

    alpha = 1000
    cluster_amps[0, :] *= 0.99*alpha   #20
    cluster_amps[1, :] *= 0.99*alpha  #600
    cluster_amps[2, :] *= 0.99*alpha   #200
    cluster_amps[3, :] *= 0.99*alpha   #300
    cluster_amps[4, :] *= 1*alpha   #400
    cluster_amps[5, :] *=1*alpha    #800
    cluster_amps[6, :] *= 1*alpha    #950

    low_amps *= 1000
    mid_amps *= 200


    sig, meta = make_chatter_like_signal(fs=fs, T=T, seed=seed, 
                                                            signal_chatter=True,
                                                            f_chatter=500.0,
                                                            t0_chatter=t0_chatter,
                                                            grow_gain=grow_gain,
                                                            grow_tau=grow_tau,
                                                            ramp_sec=ramp_sec,

                                                            low_amps= low_amps,
                                                            mid_amps= mid_amps,
                                                            low_freqs= low_freqs,
                                                            mid_freqs= mid_freqs,

                                                            f200_hz=f200_hz,
                                                            f200_amp=f200_amp,

                                                            cluster_center=cluster_center,
                                                            cluster_offsets=cluster_offsets,
                                                            cluster_amps=cluster_amps,

                                                            am_freqs_chatter=am_freqs_chatter,
                                                            am_depths_chatter=am_depths_chatter,
                                                            base_chatter_amp=base_chatter_amp ,

                                                            white_noise_sigma=white_noise_sigma,
                                                            narrow_noise_sigma=narrow_noise_sigma

                                                            )
    
    return meta['t'], sig


def sinus_6_C_SNR(
    fs : int, 
    T  : float, 
    chatter : bool = True, 
    exp: float = None, 
    Amp: float = 1,
    noise : bool = True, 
    SNR_dB : float = 10,
    stable_to_chatter: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera una señal sintética basada en la ecuación (9) del artículo de
    Altintas y Budak (1995).

    Modos:
      - chatter=True, stable_to_chatter=False  -> chatter presente en todo el registro
      - stable_to_chatter=True                -> estable al principio y chatter que crece
      - chatter=False y stable_to_chatter=False -> sin chatter (solo componentes 1–6 + ruido)
    """

    # ============================
    # Vector de tiempo
    # ============================
    t = np.arange(0, T, 1/fs)

    # ============================
    # Definición de A(t)
    # ============================
    if stable_to_chatter:
        # A(t) casi constante y pequeña hasta t0,
        # luego crecimiento suave tipo exponencial hacia Amp
        t0 = T / 2               # instante en el que empieza el chatter
        Amp_ini = 0.1 * Amp      # amplitud “estable” (pequeña)
        Amp_fin = Amp            # amplitud en régimen de chatter
        tau = 0.1 * T            # controla lo rápido que crece

        A_t = np.zeros_like(t)
        # Antes de t0: amplitud baja y casi constante
        A_t[t < t0] = Amp_ini

        # Después de t0: crecimiento exponencial suave hacia Amp_fin
        mask = t >= t0
        A_t[mask] = Amp_ini + (Amp_fin - Amp_ini) * (
            np.exp((t[mask] - t0) / tau)
        )

    else:
        # Comportamiento original
        if exp is not None:
            # Ojo: aquí es Amp * exp(exp * t), no Amp * t**exp como dice el comentario
            A_t = Amp * np.exp(exp * t)
        else:
            A_t = Amp * t

    # ============================
    # Componentes de la ecuación (9)
    # ============================
    c1 = 4   * np.sin(800  * np.pi * t)
    c2 = 1.5 * np.sin(1600 * np.pi * t)
    c3 = 4   * np.sin(2400 * np.pi * t)

    # chatter (modo de chatter amplitud modulada)
    if chatter or stable_to_chatter:
        c4 = A_t * (1 + 0.3 * np.sin(1995 * np.pi * t)) * np.cos(8991 * np.pi * t)
    else:
        c4 = np.zeros_like(t)

    c5 = np.sin(4800 * np.pi * t)
    c6 = 0.6 * np.sin(7200 * np.pi * t)

    # ============================
    # Ruido w(t) según SNR
    # ============================
    signal_no_noise = c1 + c2 + c3 + c5 + c6  # sin chatter

    if SNR_dB is not None and noise:
        Ps = np.mean(signal_no_noise**2)
        SNR_lin = 10**(SNR_dB / 10)
        Pn = Ps / SNR_lin
        sigma_n = np.sqrt(Pn)
        w = np.random.normal(0, sigma_n, size=t.shape)
    else:
        w = np.zeros_like(t)

    # ============================
    # Señales finales
    # ============================
    signal_chatter       = signal_no_noise + c4 + w      # con chatter (constante o creciente)
    signal_chatter_free  = signal_no_noise + w           # sin chatter

    if stable_to_chatter:
        # En este modo, ya hemos hecho el “estable -> chatter”
        # mediante A(t), así que devolvemos signal_chatter
        return t, signal_chatter
    elif chatter:
        return t, signal_chatter
    else:
        return t, signal_chatter_free

from matplotlib import pyplot as plt
if __name__ == "__main__":
    fs = 2000.0
    T = 6.0
    sig, t = signal_chatter_example(fs=fs, T=T, seed=123)

    plt.figure(figsize=(10, 4))
    plt.plot(t, sig)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Chatter-like Signal Example")
    plt.grid()
    plt.tight_layout()

    
    f, A = amplitude_spectrum(sig, fs=fs)
    
    plt.figure(figsize=(10, 4))
    plt.plot(f, A)
    plt.title("Espectro de amplitud de F(t)")
    plt.xlabel("Frecuencia (Hz)")
    plt.xlim(0, 1000)
    plt.ylabel("Amplitud (u.a.)")
    plt.show()
    
    