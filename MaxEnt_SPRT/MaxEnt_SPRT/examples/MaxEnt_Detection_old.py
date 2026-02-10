from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

# Importaciones del paquete reorganizado (nombres públicos en inglés; comentarios en español)
from MaxEnt_SPRT.lib.detector import (
    MaxEntSPRTConfig,
    MaxEntSPRTDetector,
)
from MaxEnt_SPRT.lib.entropy import (
    GaussianMaxEntEstimator,
    EmpiricalHistogramEntropyEstimator,
)
from MaxEnt_SPRT.utils.opr import sample_opr

# La función sinus_6_C_SNR proviene de tu generador externo de señales.
# Si no está instalado en tu entorno, instala/importa el módulo correspondiente.
try:
    from C_emd_hht import sinus_6_C_SNR  # type: ignore
except Exception as e:
    raise ImportError(
        "No se pudo importar 'sinus_6_C_SNR' desde 'C_emd_hht'. "
        "Asegúrate de tener ese módulo disponible en tu entorno."
    ) from e


# ==========================================================
# 9. EJEMPLO DE USO (MAIN) – DEMO INTEGRADA
# ==========================================================

if __name__ == "__main__":
    # --- Parámetros de simulación (ejemplo, puedes adaptarlos) ---
    rpm: float = 15_000.0        # revoluciones por minuto
    ratio_sampling: float = 250  # muestras por revolución
    fr: float = rpm / 60.0       # Hz, frecuencia de rotación
    fs: float = ratio_sampling * fr  # Hz, frecuencia de muestreo
    T: float = 1.0               # s, duración de la señal
    N_seg: int = 20              # nº de revoluciones por segmento
    seed: int = 42

    # ------------------- SEÑALES DE ENTRENAMIENTO -------------------
    t, y_free = sinus_6_C_SNR(
        fs=fs,
        T=T,
        chatter=False,
        exp=None,
        Amp=5,
        stable_to_chatter=False,
        noise=True,
        SNR_dB=10.0,
        seed=24,
    )

    t, y_chat = sinus_6_C_SNR(
        fs=fs,
        T=T,
        chatter=True,
        exp=None,
        Amp=5,
        stable_to_chatter=False,
        noise=True,
        SNR_dB=10.0,
        seed=seed,
    )

    print("Generated chatter-free and chatter-included signals.")
    print(f"Size of signal free: {y_free.size} samples.")
    print(f"Size of signal chatter: {y_chat.size} samples.")

    # Visualización rápida
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_free, label="Chatter-free")
    plt.plot(t, y_chat, label="Chatter-included", alpha=0.7)
    plt.legend()
    plt.title("Generated signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # ------------------- OPR DE ENTRENAMIENTO -------------------
    opr_free, t_opr_free = sample_opr(y_free, t, fs=fs, fr=fr)
    opr_chat, t_opr_chat = sample_opr(y_chat, t, fs=fs, fr=fr)

    print(f"Sampled OPR: {opr_free.size} samples free, {opr_chat.size} samples chatter.")

    plt.figure(figsize=(10, 4))
    plt.plot(t, y_free, label="Chatter-free", alpha=0.7)
    plt.scatter(t_opr_free, opr_free, label="OPR free", color="red", alpha=0.7, s=7)
    plt.legend()
    plt.title("Free OPR samples")
    plt.xlabel("Time (s)")
    plt.ylabel("OPR Value")

    plt.figure(figsize=(10, 4))
    plt.plot(t, y_chat, label="Chatter-included", alpha=0.7)
    plt.scatter(t_opr_chat, opr_chat, label="OPR chatter", color="red", alpha=0.7, s=7)
    plt.legend()
    plt.title("Chatter OPR samples")
    plt.xlabel("Time (s)")
    plt.ylabel("OPR Value")
    plt.show()

    # ------------------- DETECTOR END-TO-END GAUSSIANO -------------------
    detector_cfg = MaxEntSPRTConfig(alpha=0.01, beta=0.01, reset_on_H0=True)
    gaussian_estimator = GaussianMaxEntEstimator()
    detector = MaxEntSPRTDetector(config=detector_cfg, estimator=gaussian_estimator)

    # Entrenamiento offline a partir de OPR
    detector.fit_offline_from_opr(
        opr_free=opr_free,
        opr_t_free=t_opr_free,
        opr_chat=opr_chat,
        opr_t_chat=t_opr_chat,
        N_seg=N_seg,
    )

    models_trained = detector._check_models()
    print("OFFLINE MODEL (Gaussian MaxEnt):")
    print(f"  FREE:  mu0={models_trained.p0.mu:.5f}, sigma0={models_trained.p0.sigma:.5f}")
    print(f"  CHAT:  mu1={models_trained.p1.mu:.5f}, sigma1={models_trained.p1.sigma:.5f}")

    # Histograma de H_free y H_chat
    if detector.H_free is not None and detector.H_chat is not None:
        H_free = detector.H_free
        H_chat = detector.H_chat

        plt.figure(figsize=(10, 4))
        plt.hist(H_free, bins=15, alpha=0.5, density=True, label="H free")
        plt.hist(H_chat, bins=15, alpha=0.5, density=True, label="H chatter")
        plt.legend()
        plt.title("Histograms of MaxEnt indicators (Gaussian)")
        plt.xlabel("Entropy H")
        plt.ylabel("Density")
        plt.show()

        xs = np.linspace(
            min(H_free.min(), H_chat.min()) - 0.1,
            max(H_free.max(), H_chat.max()) + 0.1,
            200,
        )
        pdf0 = np.exp([models_trained.p0.logpdf(x) for x in xs])
        pdf1 = np.exp([models_trained.p1.logpdf(x) for x in xs])
        plt.plot(xs, pdf0, label="pdf p0(H) free")
        plt.plot(xs, pdf1, label="pdf p1(H) chatter")
        plt.legend()
        plt.title("MaxEnt indicator PDFs (Gaussian)")
        plt.xlabel("Entropy H")
        plt.ylabel("Probability Density Function (PDF)")
        plt.show()

    # ------------------- OPCIONAL: ESTIMADOR EMPÍRICO -------------------
    # hist_estimator = EmpiricalHistogramEntropyEstimator(bins=20)
    # detector_hist = MaxEntSPRTDetector(config=detector_cfg, estimator=hist_estimator)
    # detector_hist.fit_offline_from_opr(
    #     opr_free=opr_free,
    #     opr_t_free=t_opr_free,
    #     opr_chat=opr_chat,
    #     opr_t_chat=t_opr_chat,
    #     N_seg=N_seg,
    # )
    # Aquí detector_hist.models contendría modelos p0(H), p1(H) para el indicador empírico.

    # ------------------- FASE ONLINE: SEÑAL NUEVA -------------------
    t_on, y_on = sinus_6_C_SNR(
        fs=fs,
        T=T,
        chatter=False,
        exp=False,
        Amp=5,
        stable_to_chatter=False,
        noise=True,
        SNR_dB=10.0,
        seed=seed,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(t_on, y_on, label="Chatter test", alpha=0.7)
    plt.legend()
    plt.title("Test signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    sprt_result, H_seq_online, t_mid_segments = detector.detect_online_from_signal(
        y_online=y_on,
        t_online=t_on,
        rpm=rpm,
        ratio_sampling=ratio_sampling,
        N_seg=N_seg,
    )

    print(f"ONLINE FINAL STATE: {sprt_result.final_state}, decision at segment {sprt_result.decision_index}")

    # Visualización de H_seq y S_history
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(t_mid_segments, H_seq_online, marker="o")
    ax[0].set_ylabel("H (MaxEnt per segment)")
    ax[0].set_title("Evolution of MaxEnt indicator")

    ax[1].plot(t_mid_segments, sprt_result.S_history, marker="o")
    ax[1].axhline(sprt_result.a, linestyle="--", linewidth=0.8)
    ax[1].axhline(sprt_result.b, linestyle="--", linewidth=0.8)
    ax[1].set_ylabel("S_n (SPRT)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_title("Evolution of SPRT statistic")

    plt.tight_layout()
    plt.show()
