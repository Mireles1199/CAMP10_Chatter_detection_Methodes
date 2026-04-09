# %%
# coding: utf-8

"""
maxent_sprt_sim_validation.py

Simulation Validation I — Código reproducible:
- Señal sintética (ec. 9 y 10)
- Muestreo once-per-revolution (OPR)
- Suposición MaxEnt: pdf normal para chatter-free (H0) y early-chatter (H1)
- Validación de normalidad: Lilliefors (aprox.) y Jarque–Bera
- SPRT (Wald) con umbrales a partir de alpha, beta
- Demostración de detección y tiempo de cruce

Dependencias: numpy (solo estándar), math.
No requiere SciPy.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple, Dict
import math
import numpy as np
import matplotlib.pyplot as plt

from C_emd_hht import signal_chatter_example, sinus_6_C_SNR




def generate_timebase(T: float, fs: float) -> np.ndarray:
    """Crea vector temporal [0, T) con paso 1/fs."""
    T = float(T)
    fs = float(fs)
    n = int(np.floor(T * fs))
    return np.arange(n, dtype=float) / fs


def sample_opr(y: np.ndarray, t: np.ndarray, fs: float, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae datos once-per-revolution (1 muestra por revolución).
    ratio = fs/fr debe ser entero (por defecto 250). Tomamos el primer índice de cada revolución.
    """
    ratio = fs / fr
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("fs/fr debe ser entero para OPR exacto.")
    step = int(round(ratio))
    return y[::step], t[::step]


# ============================================================
# 2) MaxEnt (gaussiano) y entropía
# ============================================================

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
        z = (x - self.mu) / self.sigma
        return -0.5 * (math.log(2.0 * math.pi) + 2.0 * math.log(self.sigma) + z * z)

    def entropy_shannon(self) -> float:
        return 0.5 * math.log(2.0 * math.pi * math.e * (self.sigma ** 2))

    @staticmethod
    def from_samples(samples: Iterable[float], eps: float = 1e-12) -> "GaussianPDF":
        x = np.asarray(list(samples), dtype=float)
        if x.size < 2:
            raise ValueError("Se requieren al menos 2 muestras para estimar sigma.")
        mu = float(np.mean(x))
        var = float(np.var(x, ddof=1))
        sigma = math.sqrt(max(var, eps))
        return GaussianPDF(mu=mu, sigma=sigma)


# ============================================================
# 3) Tests de normalidad: Lilliefors (aprox.) y Jarque–Bera
# ============================================================

def lilliefors_normal_test(
    x: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Test de Lilliefors para normalidad (estimando mu y sigma).
    Devuelve (D, D_crit, es_normal). Aproximación a D_crit(α) ~ c(α)/sqrt(n).
    Constantes c(α): 0.89 (5%), 1.03 (1%). Usamos 0.89 para α=0.05.
    Referencia aproximada clásica; para n pequeño la aproximación es conservadora.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 5:
        raise ValueError("Se requieren >=5 datos para Lilliefors aproximado.")

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if sigma <= 0:
        return 0.0, 0.0, False

    xs = np.sort(x)
    z = (xs - mu) / sigma
    # CDF normal estándar por aproximación de error
    cdf = 0.5 * (1.0 + erf_approx(z / math.sqrt(2.0)))
    # ECDF
    i = np.arange(1, n + 1, dtype=float)
    ecdf = i / n
    D_plus = np.max(ecdf - cdf)
    D_minus = np.max(cdf - (i - 1) / n)
    D = float(max(D_plus, D_minus))

    c_alpha = 0.89 if 0.049 <= alpha <= 0.051 else 1.03  # fallback tosco
    D_crit = c_alpha / math.sqrt(n)
    return D, D_crit, (D <= D_crit)


def jarque_bera_test(
    x: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Test de Jarque–Bera para normalidad.
    JB = n/6 * (S^2 + (K-3)^2/4) ~ χ²_2 asintótico.
    Para χ² con k=2, CDF(x) = 1 - exp(-x/2) * (1 + x/2).
    Devuelve (JB, crit, es_normal) usando cuantíl asintótico.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 5:
        raise ValueError("Se requieren >=5 datos para JB.")
    x_centered = x - np.mean(x)
    s2 = float(np.mean(x_centered**2))
    if s2 <= 0:
        return 0.0, 0.0, False
    m3 = float(np.mean(x_centered**3))
    m4 = float(np.mean(x_centered**4))
    S = m3 / (s2 ** 1.5)
    K = m4 / (s2 ** 2)
    JB = n / 6.0 * (S * S + ((K - 3.0) ** 2) / 4.0)

    # Cuantíl crítico para χ²_2 al nivel 1-α.
    # Invertimos CDF sencilla: CDF(x)=1 - e^{-x/2}(1 + x/2).
    # No hay forma cerrada simple; usamos búsqueda binaria rápida.
    crit = chi2_isf_asymptotic(df=2, alpha=alpha)
    return float(JB), float(crit), (JB <= crit)


def erf_approx(z: np.ndarray) -> np.ndarray:
    """Aproximación numérica de erf(z) (Abramowitz-Stegun, elevada precisión)."""
    # Coeficientes de Abramowitz & Stegun 7.1.26
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(z)
    x = np.abs(z)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def chi2_isf_asymptotic(df: int, alpha: float) -> float:
    """
    Inversa de la supervivencia (ISF) para χ²(df) aproximada. Aquí df=2.
    Para df=2: CDF(x) = 1 - e^{-x/2} (1 + x/2). ISF = cuantíl 1-α.
    Resolvemos e^{-x/2}(1 + x/2) = α por bisección.
    """
    if df != 2:
        raise ValueError("Esta implementación asume df=2.")
    # Búsqueda en [0, Xmax]; la masa se concentra pronto.
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        val = math.exp(-mid / 2.0) * (1.0 + mid / 2.0)
        if val > alpha:  # necesitamos x más grande
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================
# 4) SPRT (Wald)
# ============================================================

@dataclass
class SPRT:
    p0: GaussianPDF
    p1: GaussianPDF
    alpha: float = 0.05
    beta: float = 0.05

    def __post_init__(self) -> None:
        if not (0 < self.alpha < 1 and 0 < self.beta < 1):
            raise ValueError("alpha y beta deben estar en (0,1).")
        self.a = math.log(self.beta / (1.0 - self.alpha))
        self.b = math.log((1.0 - self.beta) / self.alpha)
        if not (self.a < 0 < self.b):
            raise ValueError("Se espera a<0<b; revise alpha/beta.")

        self.S = 0.0
        self.n = 0
        self.decision: Optional[Literal["H0", "H1", "continue"]] = "continue"

    def llr(self, x: float) -> float:
        return self.p1.logpdf(x) - self.p0.logpdf(x)

    def update(self, x: float) -> Literal["H0", "H1", "continue"]:
        if self.decision != "continue":
            return self.decision
        self.S += self.llr(x)
        self.n += 1
        if self.S <= self.a:
            self.decision = "H0"
        elif self.S >= self.b:
            self.decision = "H1"
        else:
            self.decision = "continue"
        return self.decision

    def reset(self) -> None:
        self.S = 0.0
        self.n = 0
        self.decision = "continue"


# ============================================================
# 5) Pipeline de validación: ajuste, tests y detección
# ============================================================

@dataclass
class MaxEntModels:
    p0: GaussianPDF
    p1: GaussianPDF


def fit_maxent_gaussians(
    samples_H0: Iterable[float],
    samples_H1: Iterable[float],
    min_sigma: float = 1e-12,
) -> MaxEntModels:
    g0 = GaussianPDF.from_samples(samples_H0, eps=min_sigma)
    g1 = GaussianPDF.from_samples(samples_H1, eps=min_sigma)
    return MaxEntModels(p0=g0, p1=g1)


def build_sprt(models: MaxEntModels, alpha: float = 0.05, beta: float = 0.05) -> SPRT:
    return SPRT(p0=models.p0, p1=models.p1, alpha=alpha, beta=beta)


def segment_opr(opr: np.ndarray, N_seg: int) -> list[np.ndarray]:
    """
    Divide la señal OPR en segmentos consecutivos de longitud N_seg.
    Se descarta el sobrante al final si no completa un segmento.
    """
    n_total = len(opr)
    n_segments = n_total // N_seg
    segments = []
    for k in range(n_segments):
        start = k * N_seg
        end = start + N_seg
        segments.append(opr[start:end])
    return segments



# %%
# ============================================================
# 6) Demostración / main
# ============================================================

# def demo(
#     T: float = 1.0,
#     fs: float = 20000.0,
#     fr: float = 80.0,
#     ratio_required: int = 250,
#     seed: int = 1234,
#     alpha: float = 0.05,
#     beta: float = 0.05,
# ) -> Dict[str, object]:
#     """
#     Demostración reproducible:
#     - Genera señal chatter-free y con early chatter
#     - Extrae OPR
#     - Estima p0, p1 (gaussianas)
#     - Ejecuta Lilliefors y Jarque–Bera sobre OPR (y sobre MaxEnt si se desea)
#     - Corre SPRT sobre un stream que pasa a chatter y reporta el tiempo de detección

#     Params:
#       T: duración total (s)
#       fs: frecuencia de muestreo (Hz)
#       fr: frecuencia de rotación (Hz); fs/fr debe ser 250
#       ratio_required: razón requerida (entero, típicamente 250)
#       seed: RNG
#       alpha, beta: riesgos del SPRT

#     Returns: diccionario con resultados clave.
#     """

T= 1.0
# multiple = 3
rpm = 10_000.0
ratio_sampling = 250
fr = rpm / 60.0
Ts = 1/fr / ratio_sampling 
fs = 1/ Ts

seed= 5
alpha= 0.05
beta= 0.05

# if abs(fs / fr - ratio_required) > 1e-9:
#     raise ValueError("fs/fr debe ser exactamente = ratio_required (p. ej., 250).")

SNR_dB = 10



# Base temporal
t = generate_timebase(T=T, fs=fs)

print(f"Generated timebase with {t.size} samples.")

# Señales
t, y_free = sinus_6_C_SNR(fs=fs, T=T, 
                     chatter=False,
                     exp=None,
                     Amp=5,
                     stable_to_chatter=False,
                     noise=True,
                     SNR_dB=10.0)

t, y_chat = sinus_6_C_SNR(fs=fs, T=T, 
                     chatter=True,
                     exp=None,
                     Amp=5,
                     stable_to_chatter=False,
                     noise=True,
                     SNR_dB=10.0)

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

# OPR
opr_free,t_opr_free = sample_opr(y_free, t=t, fs=fs, fr=fr)
opr_chat,t_opr_chat = sample_opr(y_chat, t=t, fs=fs, fr=fr)
print(f"Sampled OPR: {opr_free.size} samples free, {opr_chat.size} samples chatter.")


plt.figure(figsize=(10, 4))
plt.plot(t, y_free, label="Chatter-free", alpha=0.7)
plt.scatter(t_opr_free, opr_free, label="OPR  Chatter-free", color='red', alpha=0.7, s=7)
plt.legend()
plt.title("Free OPR samples")
plt.xlabel("Sample index")
plt.ylabel(" OPR Value")

plt.figure(figsize=(10, 4))
plt.plot(t, y_chat, label="Chatter-included", alpha=0.7,  )
plt.scatter(t_opr_chat, opr_chat, label="OPR  Chatter-included", color='red', alpha=0.7, s=7)
plt.legend()
plt.title("Chatter OPR samples")
plt.xlabel("Sample index")
plt.ylabel("OPR Value")
plt.show()

#%%


# Ajuste MaxEnt gaussianas (ventanas de calibración)
# Tomamos 200 primeras revoluciones (si existen), sino todas las disponibles
N_seg = min(20, opr_free.size, opr_chat.size)
segments_free = segment_opr(opr_free, N_seg=N_seg)
segments_chat = segment_opr(opr_chat, N_seg=N_seg)

print(f"Segmentos free: {len(segments_free)}, tamaño cada uno: {segments_free[0].shape}")
print(f"Segmentos chat: {len(segments_chat)}, tamaño cada uno: {segments_chat[0].shape}")

#%%
models = fit_maxent_gaussians(opr_free[:N_seg], opr_chat[:N_seg])

# Tests de normalidad sobre OPR (H0 y H1)
D0, Dc0, ok_L0 = lilliefors_normal_test(opr_free, alpha=0.05)
D1, Dc1, ok_L1 = lilliefors_normal_test(opr_chat, alpha=0.05)
JB0, JBc0, ok_JB0 = jarque_bera_test(opr_free, alpha=0.05)
JB1, JBc1, ok_JB1 = jarque_bera_test(opr_chat, alpha=0.05)

print(f"Test Lilliefors H0: D={D0:.4f}, Dcrit={Dc0:.4f}, normal={ok_L0}")
print(f"Test Lilliefors H1: D={D1:.4f}, Dcrit={Dc1:.4f}, normal={ok_L1}")
print(f"Test Jarque–Bera H0: JB={JB0:.4f}, crit={JBc0:.4f}, normal={ok_JB0}")
print(f"Test Jarque–Bera H1: JB={JB1:.4f}, crit={JBc1:.4f}, normal={ok_JB1}")

#%%

# Construir SPRT
sprt = build_sprt(models, alpha=alpha, beta=beta)

# Stream: mitad estable, mitad chatter (concatenado en OPR)
half = opr_free.size // 2
stream = np.concatenate([opr_free[:half], opr_chat[:half]])

# Ejecutar SPRT y detectar cruce de b
decision_time_idx = None
for i, x in enumerate(stream, start=1):
    decision = sprt.update(float(x))
    if decision == "H1":
        decision_time_idx = i
        break

# Convertir índice OPR a tiempo en segundos
time_per_rev = 1.0 / fr
detection_time_s = None if decision_time_idx is None else (decision_time_idx * time_per_rev)

results = {
    "p0": models.p0,
    "p1": models.p1,
    "lilliefors": {
        "H0": {"D": D0, "D_crit": Dc0, "normal": ok_L0},
        "H1": {"D": D1, "D_crit": Dc1, "normal": ok_L1},
    },
    "jarque_bera": {
        "H0": {"JB": JB0, "crit": JBc0, "normal": ok_JB0},
        "H1": {"JB": JB1, "crit": JBc1, "normal": ok_JB1},
    },
    "SPRT": {
        "a": sprt.a,
        "b": sprt.b,
        "S_n": sprt.S,
        "n": sprt.n,
        "decision": sprt.decision,
        "detection_time_s": detection_time_s,
    },
    "meta": {
        "T": T,
        "fs": fs,
        "fr": fr,
        "ratio": fs / fr,
        "n_opr_free": int(opr_free.size),
        "n_opr_chatter": int(opr_chat.size),
    },
}
# return results


# if __name__ == "__main__":
# out = demo()
out = results

p0, p1 = out["p0"], out["p1"]

print("=== MaxEnt (Gaussian) Models ===")
print(f"H0: mu={p0.mu:.4f}, sigma={p0.sigma:.4f}, H={p0.entropy_shannon():.4f} nats")
print(f"H1: mu={p1.mu:.4f}, sigma={p1.sigma:.4f}, H={p1.entropy_shannon():.4f} nats")
print("\n=== Normality tests (α=0.05) ===")


L0 = out["lilliefors"]["H0"]; L1 = out["lilliefors"]["H1"]
J0 = out["jarque_bera"]["H0"]; J1 = out["jarque_bera"]["H1"]
print(f"Lilliefors H0: D={L0['D']:.4f}, Dcrit={L0['D_crit']:.4f}, normal={L0['normal']}")
print(f"Lilliefors H1: D={L1['D']:.4f}, Dcrit={L1['D_crit']:.4f}, normal={L1['normal']}")
print(f"Jarque–Bera H0: JB={J0['JB']:.4f}, crit={J0['crit']:.4f}, normal={J0['normal']}")
print(f"Jarque–Bera H1: JB={J1['JB']:.4f}, crit={J1['crit']:.4f}, normal={J1['normal']}")
print("\n=== SPRT ===")


S = out["SPRT"]
print(f"a={S['a']:.3f}, b={S['b']:.3f}, S_n={S['S_n']:.3f}, n={S['n']}, decision={S['decision']}")
print(f"Detection time (s): {S['detection_time_s']}")
