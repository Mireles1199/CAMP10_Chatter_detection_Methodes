

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


from C_emd_hht import make_chatter_like_signal, five_senos
from C_emd_hht import amplitude_spectrum
from C_emd_hht import detect_chatter_from_force
from C_emd_hht import plot_imfs, plot_imf_seleccionado, plot_tendencia, plot_HHS
from C_emd_hht import signal_chatter_example




def main() -> None:
    
    # --- 1) Señal de ejemplo (sintética). Sustituye por tu F real ---
    fs = 2_000.0
    T =6.0
    t = np.arange(int(T*fs)) / fs
    # ramp_sec = 1.0

    # cluster_center =  (20.0, 600.0, 200.0, 300, 400.0,800.0, 950.0)
    # cluster_offsets = np.array([[-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
    #                                         [-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
    #                                         [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
    #                                         [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
    #                                         [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
    #                                         [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
    #                                         [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0]])
    # cluster_amps = np.array([[0.24, 0.24, 0.20, 0.30, 0.18, 0.26, 0.10],
    #                                         [0.14, 0.14, 0.10, 0.15, 0.10, 0.16, 0.10],
    #                                         [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
    #                                         [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
    #                                         [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
    #                                         [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
    #                                         [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1]])



    # am_freqs_chatter=np.array([9.0, 17.0, 26.0])
    # am_depths_chatter=np.array([0.2, 0.25, 0.07]) 

    # am_freqs_chatter *= 1.0
    # am_depths_chatter *= 4.0

    # f200_hz = 200.0
    # f200_amp = 1
    # low_freqs = np.array([8.0, 16.0, 20.0])
    # low_amps = np.array([0.5, 0.9, 0.5])
    # mid_freqs = np.array([55.0, 72.0, 95.0, 110.0, 135.0, 150.0])
    # mid_amps = np.array([0.02, 0.03, 0.02, 0.01, 0.01, 0.01])


    # cluster_amps[0, :] *= 0.99   #20
    # cluster_amps[1, :] *= 0.99  #600
    # cluster_amps[2, :] *= 0.99   #200
    # cluster_amps[3, :] *= 0.99   #300
    # cluster_amps[4, :] *= 5   #400
    # cluster_amps[5, :] *=10    #800
    # cluster_amps[6, :] *= 7    #950

    # low_amps *= 30
    # mid_amps *= 100



    # F += (0.05*(1 + 4*np.clip((t-0.8)/0.6, 0, 1))) * np.sin(2*np.pi*520*t)  # modo ~520 Hz que crece
    # t, F = five_senos(fs, duracion=T, ruido_std=2, fase_aleatoria=True, seed=42)

    F,t = signal_chatter_example(fs=fs, T=T, seed=123)


    f, A = amplitude_spectrum(F, fs=fs)

    # Comentario: bloque de ejemplos/figuras (idéntico al original en lógica)
    plt.figure()
    plt.plot(t, F)
    plt.title("Señal de fuerza F(t) de ejemplo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Fuerza (u.a.)")
    plt.show()

    plt.figure()
    plt.plot(f, A)
    plt.title("Espectro de amplitud de F(t)")
    plt.xlabel("Frecuencia (Hz)")
    plt.xlim(0, 1000)
    plt.ylabel("Amplitud (u.a.)")

    #%%
    # --- 2) Ejecuta el pipeline (sin baseline/decisión) ---
    res = detect_chatter_from_force(
        F=F,
        fs=fs,
        bandlimit_pre=(-100, 2000),
        emd_method="emd",          # "emd" o "eemd" también valen
        ceemdan_noise_strength=0.2,
        ceemdan_ensemble_size=50,
        band_chatter=(450, 600),
        band_selection_margin=0.15,
        imf_selection="index",
        imf_index=0,
        # Hilbert
        phase_diff_mode = "first_diff",
        sg_window = 11,
        sg_polyorder= 2,
        f_inst_smooth_median = None,
        # HHS
        hhs_enable = True,
        hhs_fmax = 2000.0,
        hhs_fbin_hz = 5.0,
        count_win_samples = 100,
        energy_mode = "A2",
        thr_mode="none",
        thr_k_mad=0.0, # usar MAD sin multiplicador
        thr_percentile=50, # usar percentil 95
    )


    #%%
    # --- 3) Grafica TODOS los IMFs en UNA sola figura ---
    #    (puedes limitar con max_to_plot y distribuir en columnas con ncols)
    fig, axes = plot_imfs(
        imfs=res.imfs,
        plot_spectrum=True,
        fs=fs,
        max_to_plot=None,   # por ejemplo 8
        ncols=1,            # 2 o 3 si quieres cuadrícula
        show=True
    )

    #%%
    # --- 4) Grafica el IMF seleccionado y, opcionalmente, A(t) y f_inst(t) ---
    plot_imf_seleccionado(
        selected_imf=res.selected_imf,
        fs=fs,
        plot_spectrum=True,
        A=res.A,
        f_inst=res.f_inst,
        show=True
    )

    print("IMFs:", res.imfs.shape if res.imfs is not None else None)
    print("IMF seleccionado:", res.k_selected)
    print("Counts shape:", res.counts.shape)
    print("Umbral usado (energía):", res.meta.get("threshold"))

    #%%
    # --- 5) Grafica el HHS (Hilbert-Huang Spectrum) -
    plot_HHS(
        t =t,
        fgrid=res.fgrid,
        HHS=res.HHS,  
    )


    #%% ---------- Counts plot ----------
    t_counts = res.t_counts_samples / fs
    counts = res.counts

    # 1) Recta punteada
    plot_tendencia(t_counts, counts, tipo="pchip", ls='--', lw=2, color='k')

    # 2) Suavizada punteada (queda muy similar a la figura)
    plot_tendencia(t_counts, counts, tipo="polinomial", grado=4, ls='--', lw=2, color='k')

    # 3) Polinomial punteada (p. ej. grado 2)
    plot_tendencia(t_counts, counts, tipo="step_suave", grado=3, ls='--', lw=2, color='k')
    
    plt.show()


if __name__ == "__main__":
    main()