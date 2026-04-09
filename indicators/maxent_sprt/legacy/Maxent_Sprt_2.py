#%%
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, List
import numpy as np
import math
from C_emd_hht import signal_chatter_example, sinus_6_C_SNR

# ==========================================================
# 1. UTILIDADES BÁSICAS (tuyas + alguna extra)
# ==========================================================

def sample_opr(y: np.ndarray, t: np.ndarray, fs: float, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae datos once-per-revolution (1 muestra por revolución).
    ratio = fs/fr debe ser entero (por defecto 250). Tomamos el primer índice de cada revolución.
    """
    first = True
    last = False
    mid = False
    
    ratio = fs / fr
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("fs/fr debe ser entero para OPR exacto.")
    step = int(round(ratio))
    if first:
        y, t = y[::step], t[::step]
    elif last:
        y, t = y[step-1::step], t[step-1::step]
    elif mid:
        half_step = step // 2
        y, t = y[half_step::step], t[half_step::step]
    return y, t


def segment_opr(opr: np.ndarray, opr_t: np.ndarray, N_seg: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Divide la señal OPR en segmentos consecutivos de longitud N_seg.
    Se descarta el sobrante al final si no completa un segmento.
    """
    n_total = len(opr)
    n_segments = n_total // N_seg
    segments: List[np.ndarray] = []
    segments_t: List[np.ndarray] = []
    list_start_indices = []
    list_end_indices = []
    for k in range(n_segments):
        start = k * N_seg
        end = start + N_seg
        list_start_indices.append(start)
        list_end_indices.append(end)
        segments.append(opr[start:end])
        segments_t.append(opr_t[start:end])
    return segments, segments_t, list_start_indices, list_end_indices


@dataclass(frozen=True)
class GaussianPDF:
    mu: float
    sigma: float  # > 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mu):
            raise ValueError("mu no finito.")
        if not (np.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError("sigma debe ser finito y > 0.")

    def logpdf(self, x: float) -> float:
        """
        log N(x ; mu, sigma^2)
        """
        z = (x - self.mu) / self.sigma
        return -0.5 * (math.log(2.0 * math.pi) + 2.0 * math.log(self.sigma) + z * z)
    
    def pdf(self, x: float) -> float:
        """Devuelve el valor de la PDF en x"""
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - self.mu) / self.sigma) ** 2
        )
        
    def plot(self, num_points=300, ax=None,
            titre=None, xlabel=None, ylabel=None, **plot_kwargs):
        """Grafica la PDF normal con Matplotlib"""
        # Prepara eje si no te pasaron uno
        # if ax is None:
        #     fig, ax = plt.subplots()

        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        # Rango de X (3 sigmas a cada lado suele verse bonito)
        x = np.linspace(self.mu - 3*self.sigma, self.mu + 3*self.sigma, num_points)
        y = self.pdf(x)

        ax.plot(x, y, **plot_kwargs)
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        
  
        
        fig, ax = plt.subplots() 
        # Rango de X (3 sigmas a cada lado suele verse bonito)
        x = np.linspace(self.mu - 3*self.sigma, self.mu + 3*self.sigma, num_points)
        y = self.pdf(x)

        ax.plot(x, y, **plot_kwargs)
        ax.set_title(titre)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        

        

    def entropy_shannon(self) -> float:
        """
        Entropía de una normal N(mu, sigma^2):
        H = 0.5 * log(2*pi*e*sigma^2)
        """
        return 0.5 * math.log(2.0 * math.pi * math.e * (self.sigma ** 2))

    @staticmethod
    def from_samples(samples: Iterable[float], eps: float = 1e-12) -> "GaussianPDF":
        """
        Ajusta N(mu, sigma^2) a las muestras dadas.
        """
        x = np.asarray(list(samples), dtype=float)
        if x.size < 2:
            raise ValueError("Se requieren al menos 2 muestras para estimar sigma.")
        mu = float(np.mean(x))
        var = float(np.var(x, ddof=1))
        sigma = math.sqrt(max(var, eps))
        return GaussianPDF(mu=mu, sigma=sigma)


@dataclass
class MaxEntModels:
    """
    Modelos globales de la pdf del indicador MaxEnt (entropía) bajo:
    - H0: chatter-free  -> p0(H)
    - H1: early chatter -> p1(H)
    """
    p0: GaussianPDF
    p1: GaussianPDF


def fit_maxent_gaussians(
    samples_H0: Iterable[float],
    samples_H1: Iterable[float],
    min_sigma: float = 1e-12,
) -> MaxEntModels:
    """
    Ajusta dos gaussianas a:
    - samples_H0: valores del indicador MaxEnt en estado chatter-free
    - samples_H1: valores del indicador MaxEnt en estado chatter
    Estas pdf p0(H) y p1(H) se usan luego en el SPRT.
    """
    g0 = GaussianPDF.from_samples(samples_H0, eps=min_sigma)
    g1 = GaussianPDF.from_samples(samples_H1, eps=min_sigma)
    return MaxEntModels(p0=g0, p1=g1)


# ==========================================================
# 2. MAXENT LOCAL POR SEGMENTO  (nivel "segmento")
# ==========================================================

def entropy_from_segment(seg: np.ndarray) -> float:
    """
    Calcula el indicador MaxEnt de un segmento OPR:
    1) Asume que la pdf MaxEnt del segmento es normal (como en el paper).
       -> ajusta N(mu, sigma^2) a los datos OPR del segmento.
    2) Devuelve la entropía de esa normal: H = 0.5*log(2*pi*e*sigma^2).
    Este H es el indicador que se pasará al SPRT.
    """
    g = GaussianPDF.from_samples(seg)
    return g.entropy_shannon(), g


# ==========================================================
# 3. FASE OFFLINE: ENTRENAR MODELOS p0(H) Y p1(H)
# ==========================================================

def offline_train_maxent_sprt(
    opr_free: np.ndarray,
    opr_chat: np.ndarray,
    opr_t_free: np.ndarray,
    opr_t_chat: np.ndarray,
    N_seg: int,
    plot_diagnostics: bool = False,
) -> Tuple[MaxEntModels, np.ndarray, np.ndarray]:
    """
    FASE OFFLINE (prior knowledge), a partir de señales etiquetadas:
    - opr_free: OPR en condición chatter-free
    - opr_chat: OPR en condición early chatter
    - N_seg: nº de revoluciones (muestras OPR) por segmento

    Devuelve:
    - models: MaxEntModels con p0(H) y p1(H) (gaussianas sobre el indicador H)
    - H_free: vector de entropías segmentarias en estado libre (para diagnóstico)
    - H_chat: vector de entropías segmentarias en chatter (para diagnóstico)
    - t_mid_free: tiempos medios de los segmentos libres (para diagnóstico)
    - t_mid_chat: tiempos medios de los segmentos con chatter (para diagnóstico)
    """

    # 1) Segmentar OPR
    segments_free, segments_t_free, start_indices_free, end_indices_free = segment_opr(opr_free, opr_t_free, N_seg=N_seg)
    segments_chat, segments_t_chat, start_indices_chat, end_indices_chat = segment_opr(opr_chat, opr_t_chat, N_seg=N_seg)

    if plot_diagnostics:
        figure, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(opr_t_free, opr_free, label="OPR Stable", color="red", s=7, zorder=1)
        ymax= np.max(opr_free)+0.1*(np.max(opr_free))
        ymin = np.min(opr_free) * 1.1 if np.min(opr_free) < 0 else np.min(opr_free) * 0.9

        for start, end in zip(start_indices_free, end_indices_free):

            ax.vlines(opr_t_free[start], ymax=ymax , ymin=ymin ,
                      color='black', alpha=0.99, 
                      linewidth=1, zorder=2)
            
        ax.vlines(opr_t_free[end_indices_free[-1]], ymax=ymax , ymin=ymin ,
                      color='black', alpha=0.99,
                      linewidth=1, zorder=2)
        ax.set_title("Segmentación OPR Stable")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("OPR Displacement (x)")
        ax.legend()
        
        figure, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(opr_t_chat, opr_chat, label="OPR chatter", color="red", s=7, zorder=1)
        ymax= np.max(opr_chat)+0.1*(np.max(opr_chat)-np.min(opr_chat))
        ymin = np.min(opr_chat) * 1.1 if np.min(opr_chat) < 0 else np.min(opr_chat) * 0.9
        
        for start, end in zip(start_indices_chat, end_indices_chat):
            ax.vlines(opr_t_chat[start], ymax=ymax , ymin=ymin ,
                      color='black', alpha=0.99, 
                      linewidth=1, zorder=2)
        ax.vlines(opr_t_chat[end_indices_chat[-1]], ymax=ymax , ymin=ymin ,
                      color='black', alpha=0.99,
                      linewidth=1, zorder=2)
        ax.set_title("Segmentación OPR chatter")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("OPR Displacement (x)")
        ax.legend()

    if len(segments_free) == 0 or len(segments_chat) == 0:
        raise ValueError("No hay segmentos suficientes para entrenamiento. Revisa N_seg y longitud de opr_*.")  # noqa: E501

    # 2) Calcular entropía MaxEnt por segmento (indicador H)
    H_free, pdf_x_free = zip(*(entropy_from_segment(seg) for seg in segments_free))
    H_chat, pdf_x_chat = zip(*(entropy_from_segment(seg) for seg in segments_chat))
    
    if plot_diagnostics:
        pdf_limit = min(5, len(pdf_x_free), len(pdf_x_chat))
        figure, ax = plt.subplots(figsize=(10, 4))
        for i in range(pdf_limit):
            pdf_x_free[i].plot(ax=ax, titre=f"Segment {i+1} PDF Stable",
                                xlabel="Displacement (x)", ylabel="PDF - F(x)",
                               label=f"Segment {i+1} PDF Stable", alpha=0.99, linewidth=2)
         
        figure, ax = plt.subplots(figsize=(10, 4))   
        for i in range(pdf_limit):
            pdf_x_chat[i].plot(ax=ax, titre=f"Segment {i+1} PDF chatter",
                                xlabel="Displacement (x)", ylabel="PDF - F(x)",
                               label=f"Segment {i+1} PDF chatter", alpha=0.7, linewidth=2)
        

    
    t_mid_free = np.array([np.mean(seg_t) for seg_t in segments_t_free])
    t_mid_chat = np.array([np.mean(seg_t) for seg_t in segments_t_chat])

    # 3) Ajustar pdf del indicador MaxEnt en free y chatter (p0(H), p1(H))
    models = fit_maxent_gaussians(H_free, H_chat)
    
    H_free = np.array(H_free)
    H_chat = np.array(H_chat)
    t_mid_chat = np.array(t_mid_chat)
    t_mid_free = np.array(t_mid_free)

    return models, H_free, H_chat, t_mid_free, t_mid_chat


# ==========================================================
# 4. FASE ONLINE: SPRT SOBRE EL INDICADOR MaxEnt
# ==========================================================

def log_likelihood_ratio(H_obs: float, models: MaxEntModels) -> float:
    """
    LLR: log ( p1(H_obs) / p0(H_obs) )
    donde p0 y p1 son las pdf del indicador MaxEnt bajo H0/H1.
    """
    return models.p1.logpdf(H_obs) - models.p0.logpdf(H_obs)


def sprt_detect_sequence(
    H_seq: Iterable[float],
    models: MaxEntModels,
    alpha: float = 0.01,
    beta: float = 0.01,
    reset_on_H0: bool = True,
    plot_diagnostics: bool = False,
) -> Tuple[str, int, np.ndarray]:
    """
    Aplica el SPRT a una secuencia de observaciones H (entropías de segmentos)
    usando los modelos MaxEnt p0(H) y p1(H).

    Parámetros:
        H_seq: iterable de H_n (entropía por segmento) en el tiempo.
        models: MaxEntModels con p0(H) y p1(H).
        alpha: prob. máxima de falso positivo (rechazar H0 cuando es cierto).
        beta:  prob. máxima de falso negativo (aceptar H0 cuando es falso).
        reset_on_H0: si True, cuando S_n <= a se resetea S_n = 0 (como en el paper).
        plot_diagnostics: si True, muestra gráficos de diagnóstico.

    Devuelve:
        estado_final: "free", "chatter" o "indeterminado"
        idx_decision: índice del segmento donde se tomó la decisión (o -1)
        S_hist: historial de S_n (array)
        a: umbral inferior SPRT
        b: umbral superior SPRT
    """
    # Umbrales SPRT
    a = math.log(beta / (1.0 - alpha))
    b = math.log((1.0 - beta) / alpha)

    H_seq = list(H_seq)
    S_hist = np.zeros(len(H_seq), dtype=float)
    S = 0.0
    estado_final = "indeterminado"
    idx_decision = -1
    _llr_store = []
    
    if plot_diagnostics:
        _p_H0_store = []
        _p_H1_store = []

    for i, H_obs in enumerate(H_seq):
        if plot_diagnostics:
            _p_H0_store.append(models.p0.pdf(H_obs))
            _p_H1_store.append(models.p1.pdf(H_obs))
            
            
            
        llr = log_likelihood_ratio(H_obs, models)
        _llr_store.append(llr)
        S += llr
        S_hist[i] = S

        if S <= a:
            # Aceptamos H0 (estado chatter-free)
            estado_final = "free"
            idx_decision = i
            if reset_on_H0:
                S = 0.0  # reset como en el paper
            # IMPORTANTE: no rompemos el bucle si queremos seguir monitorizando.
            # Si quieres parar en cuanto confirmas free, podrías hacer un break.

        if S >= b:
            # Aceptamos H1 (early chatter)
            estado_final = "chatter"
            idx_decision = i
            # break
        
    if plot_diagnostics:
        
        figure, ax = plt.subplots(figsize=(10, 4))
        ax.plot(H_seq, marker="o", color='#ff7f0e')
        ax.set_title("Evolution of  MaxEnt (H_n)")
        ax.set_xlabel("Index of segment")
        ax.set_ylabel("H_n")
        
        figure, ax = plt.subplots(figsize=(10, 4))
        ax.plot(_p_H0_store, label="P0(H_n) PDF Stable", marker="o")
        ax.plot(_p_H1_store, label="P1(H_n) PDF chatter", marker="o")
        ax.legend()
        ax.set_title("PDFs  P0 y P1 for H_n")
        ax.set_xlabel("Index of segment")
        ax.set_ylabel(" Value PDF")
        
      
        figure, ax = plt.subplots(figsize=(10, 4))
        ax.plot(_llr_store, marker="o")
        ax.set_title("Evolution of LLR (log-likelihood ratio)")
        ax.set_xlabel("Index of segment")
        ax.set_ylabel("LLR")
        plt.show()

    return estado_final, idx_decision, S_hist, a, b


def online_maxent_sprt_from_signal(
    y_online: np.ndarray,
    t_online: np.ndarray,
    rpm: float,
    ratio_sampling: float,
    N_seg: int,
    models: MaxEntModels,
    alpha: float = 0.01,
    beta: float = 0.01,
    reset_on_H0: bool = True,
    plot_diagnostics: bool = False,
) -> Tuple[str, int, np.ndarray, np.ndarray]:
    """
    FASE ONLINE completa para una señal nueva:

    1) OPR: y_online -> opr_online
    2) Segmentación: opr_online -> segmentos
    3) MaxEnt local: segmentos -> H_seq (entropía por segmento)
    4) SPRT: H_seq + models (p0(H), p1(H)) -> decisión chatter/free

    Devuelve:
        estado_final: "free" o "chatter" o "indeterminado"
        idx_decision: índice del segmento donde se decidió
        H_seq: entropías por segmento
        S_hist: historial S_n del SPRT
    """
    # 1) OPR
    fr = rpm / 60.0      # Hz, frecuencia de rotación
    fs = ratio_sampling * fr  # Hz, frecuencia de muestreo
    opr_online, opr_t_online = sample_opr(y_online, t_online, fs=fs, fr=fr)

    # 2) Segmentación
    segments_online, segments_t_online, *_ = segment_opr(opr_online, opr_t_online, N_seg=N_seg)
    if len(segments_online) == 0:
        raise ValueError("No hay segmentos suficientes en la señal online.")

    # 3) Entropía MaxEnt de cada segmento
    H_seq, pdf_segment = zip(*(entropy_from_segment(seg) for seg in segments_online))
    H_seq = np.array(H_seq)
    t_mid_segments = np.array([np.mean(seg_t) for seg_t in segments_t_online])

    # 4) SPRT
    estado_final, idx_decision, S_hist, a, b = sprt_detect_sequence(
        H_seq, models, alpha=alpha, beta=beta, reset_on_H0=reset_on_H0,
        plot_diagnostics=plot_diagnostics
    )

    return estado_final, idx_decision, H_seq, S_hist, t_mid_segments, a, b


#%%
# ==========================================================
# 5. EJEMPLO DE USO (esqueleto, adaptado a tu código)
# ==========================================================

# if __name__ == "__main__":
import matplotlib.pyplot as plt

# --- Parámetros de simulación (ejemplo, adaptalo a los tuyos) ---
rpm = 20_000.0 # revoluciones por minuto
ratio_sampling = 250  # muestras por revolución
fr = rpm / 60.0      # Hz, frecuencia de rotación
fs = ratio_sampling * fr  # Hz, frecuencia de muestreo
T = 1.0        # s, duración de la señal
N_seg = 20    # nº de revoluciones por segmento (20–30 típico)
 # semilla para reproducibilidad

# Aquí asumimos que ya tienes definida sinus_6_C_SNR como en tu código
# Señal de entrenamiento: FREE
seed = 42
t, y_free = sinus_6_C_SNR(
    fs=fs, T=T,
    chatter=False,
    exp=None,
    Amp=5,
    stable_to_chatter=False,
    noise=True,
    SNR_dB=10.0,
    seed=seed
)

# Señal de entrenamiento: CHATTER
t, y_chat = sinus_6_C_SNR(
    fs=fs, T=T,
    chatter=True,
    exp=None,
    Amp=5,
    stable_to_chatter=False,
    noise=True,
    SNR_dB=10.0,
    seed=seed
)

print("Generated chatter-free and chatter-included signals.")
print(f"Size of signal free: {y_free.size} samples.")
print(f"Size of signal chatter: {y_chat.size} samples.")

plt.figure(figsize=(10, 4))
plt.plot(t, y_free, label="Chatter-free")
plt.plot(t, y_chat, label="Chatter-included", alpha=0.7)
plt.legend()
plt.title("Generated signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

#%%

# --- FASE OFFLINE: entrenar p0(H) y p1(H) ---
opr_free, t_opr_free = sample_opr(y_free, t, fs=fs, fr=fr)
opr_chat, t_opr_chat = sample_opr(y_chat, t, fs=fs, fr=fr)

print(f"Sampled OPR: {opr_free.size} samples free, {opr_chat.size} samples chatter.")

plt.figure(figsize=(10, 4))
plt.scatter(t_opr_free, opr_free, label="OPR  Displacement", color='red', alpha=1, s=7, zorder=1)
plt.plot(t, y_free, label="Displacement (x)", alpha=1, zorder=0)

plt.legend()
plt.title("OPR samples - Stable")
plt.xlabel("Time (s)")
plt.ylabel(" Displacement (x)")

plt.figure(figsize=(10, 4))
plt.plot(t, y_chat, label="Displacement (x)", alpha=0.99, zorder=0)
plt.scatter(t_opr_chat, opr_chat, label="OPR  Displacement", color='red', alpha=0.99, s=7, zorder=1)
plt.legend()
plt.title("OPR Displacement - Chatter")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (x)")
plt.show()

#%% --- Ajuste MaxEnt y diagnóstico ---

models, H_free, H_chat,*_ = offline_train_maxent_sprt(
    opr_free=opr_free,
    opr_chat=opr_chat,
    opr_t_free=t_opr_free,
    opr_t_chat=t_opr_chat,
    N_seg=N_seg,
    plot_diagnostics=True,
)

print("MODELO OFFLINE:")
print(f"  FREE:  mu0={models.p0.mu:.5f}, sigma0={models.p0.sigma:.5f}")
print(f"  CHAT:  mu1={models.p1.mu:.5f}, sigma1={models.p1.sigma:.5f}")

# (Opcional) ver histograma de H_free y H_chat + gaussianas
plt.figure(figsize=(10, 4))
plt.hist(H_free, bins=15, alpha=0.5, density=True, label='H Stable')
plt.hist(H_chat, bins=15, alpha=0.5, density=True, label='H chatter')
plt.legend()
plt.title("Histograms of MaxEnt indicators")
plt.xlabel("Entropy H")
plt.ylabel("Density")
plt.show()

#%% --- Visualización de PDFs ajustadas ---

xs = np.linspace(min(H_free.min(), H_chat.min()) - 0.1,
                    max(H_free.max(), H_chat.max()) + 0.1, 200)
pdf0 = np.exp([models.p0.logpdf(x) for x in xs])
pdf1 = np.exp([models.p1.logpdf(x) for x in xs])
plt.plot(xs, pdf0, label="PDF - P0(H) stable")
plt.plot(xs, pdf1, label="PDF - P1(H) chatter")
plt.legend()
plt.title("PDFs of MaxEnt OFFLINE models")
plt.xlabel("Entropy H")
plt.ylabel("Probability Density Function (PDF) - G(H)")
plt.show()

#%%

# --- FASE ONLINE: detección sobre una señal nueva ---
# Por ejemplo, una señal que pasa de free a chatter
t_on, y_on = sinus_6_C_SNR(
    fs=fs, T=1.0,
    chatter=True,
    exp=False,
    Amp=3,
    stable_to_chatter=False,  # por ejemplo: empieza estable y luego chatter
    noise=True,
    SNR_dB=10.0,
    seed=50
)

opr_on, t_opr_on = sample_opr(y_on, t_on, fs=fs, fr=fr)


print(f"Sampled OPR: {opr_on.size} samples online.")

plt.figure(figsize=(10, 4))
plt.scatter(t_opr_on, opr_on, label="OPR  Displacement", color='red', alpha=1, s=7, zorder=1)
plt.plot(t_on, y_on, label="Displacement (x)", alpha=1, zorder=0)
plt.legend()
plt.title("OPR samples - Case ONLINE")
plt.xlabel("Time (s)")
plt.ylabel(" Displacement (x)")

estado_final, idx_dec, H_seq, S_hist, t_mid_segments, a, b = online_maxent_sprt_from_signal(
    y_online=y_on,
    t_online=t_on,
    rpm=10_000.0,
    ratio_sampling=ratio_sampling,
    N_seg=20,
    models=models,
    alpha=0.01,
    beta=0.01,
    reset_on_H0=True,
    plot_diagnostics=True,
)

print(f"ESTADO FINAL ONLINE: {estado_final}, decisión en segmento {idx_dec}")

# Visualización rápida de H_seq y S_hist
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
# ax[0].plot(t_mid_segments, H_seq, marker="o")
ax[0].plot( H_seq, marker="o", color='#ff7f0e')
ax[0].set_ylabel("H (MaxEnt for segment)")
ax[0].set_title("Evolution of MaxEnt indicator")


# ax[1].plot(t_mid_segments, S_hist, marker="o")
ax[1].plot( S_hist, marker="o", color='#ff7f0e')
ax[1].axhline(a, color="g", linestyle="--", linewidth=0.8)
ax[1].axhline(b, color="r", linestyle="--", linewidth=0.8)
ax[1].set_ylabel("S_n (SPRT)")
# ax[1].set_xlabel("Times (s)")
ax[1].set_xlabel("Index of segment")
ax[1].set_title("Evolution of SPRT statistic")

plt.tight_layout()
plt.show()
