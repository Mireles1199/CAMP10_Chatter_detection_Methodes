# frf_fase_visual.py
"""
FRF y FASE — Visual y paso a paso (SDOF masa–resorte–amortiguador)

Qué muestra (cada sección tiene prints + figuras):
  1) Definición del SDOF y FRF H(ω) = 1 / (k - m ω^2 + j c ω)
  2) Bode magnitud (lineal y dB) + marcador de ω_n
  3) Bode fase desenrollada + interpretación (antes / en / después de resonancia)
  4) Marcadores ω_n (natural) y ω_d (amortiguada) sobre la magnitud
  5) Diagrama de Argand en una frecuencia ω* (vector a + j b y su ángulo φ)
  6) Serie temporal a ω*: F(t) y x(t) mostrando el desfase (φ)
  7) Retardo equivalente τ = -φ/ω* y verificación x(t) ≈ x_amp cos(ω*(t-τ))
  8) Mini “prueba Fourier”: ∫ x(t) g(t) e^{-j ω t} dt grande en ω*, pequeño en 2ω*

Requisitos:
  - Python 3.x
  - numpy, matplotlib

Reglas gráficas:
  - Matplotlib (sin seaborn)
  - Una figura por gráfico
  - No se fijan colores manualmente

Uso:
  python frf_fase_visual.py
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# ============================ Utilidades ============================

def gauss_window(t: np.ndarray, sigma: float) -> np.ndarray:
    """Ventana gaussiana centrada en el promedio de t."""
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0")
    t0 = t.mean()
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)

def unwrap_phase(z: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(z))


# ============================ Parámetros del sistema ============================

m = 1.0          # masa [kg]
k = 1000.0       # rigidez [N/m]
zeta = 0.05      # razón de amortiguamiento (adimensional)

wn = np.sqrt(k / m)                      # frecuencia natural no amortiguada [rad/s]
c = 2.0 * zeta * m * wn                  # amortiguamiento viscoso [N·s/m]
wd = wn * np.sqrt(1 - zeta**2)           # frecuencia natural amortiguada [rad/s]

w = np.linspace(0.1, 4.0 * wn, 4000)     # malla de frecuencia [rad/s]

print("=== Parámetros SDOF ===")
print(f"m = {m} kg, k = {k} N/m, ζ = {zeta:.3f}")
print(f"ω_n = {wn:.3f} rad/s  (f_n = {wn/(2*np.pi):.3f} Hz)")
print(f"c = {c:.3f} N·s/m,   ω_d = {wd:.3f} rad/s")
print()

# ============================ 1) FRF: definición y cálculo ============================

# FRF desplazamiento/fuerza: H(ω) = 1 / (k - m ω^2 + j c ω)
H = 1.0 / (k - m * w**2 + 1j * c * w)
mag = np.abs(H)
phase = unwrap_phase(np.angle(H))     # fase desenrollada en radianes
mag_db = 20 * np.log10(mag + 1e-30)   # dB opcional (evita log(0))

# Pequeñas comprobaciones
assert H.shape == w.shape, "Dimensiones inesperadas en H(ω)"
assert np.isfinite(mag).all(), "Magnitud no finita detectada"
assert np.isfinite(phase).all(), "Fase no finita detectada"

# ============================ 2) Bode: Magnitud (lineal) ============================

plt.figure()
plt.title("FRF SDOF — Magnitud |H(ω)| (desplazamiento / fuerza)")
plt.plot(w, mag, label="|H(ω)|")
plt.axvline(wn, linestyle="--", label="ω_n")
plt.xlabel("ω [rad/s]")
plt.ylabel("|H(ω)| [m/N]")
plt.grid(True)
plt.legend()


# ============================ 2b) Bode: Magnitud (dB) ============================

plt.figure()
plt.title("FRF SDOF — Magnitud 20·log10 |H(ω)|")
plt.plot(w, mag_db, label="20·log10 |H(ω)|")
plt.axvline(wn, linestyle="--", label="ω_n")
plt.xlabel("ω [rad/s]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.legend()


# ============================ 3) Bode: Fase desenrollada ============================

plt.figure()
plt.title("FRF SDOF — Fase desenrollada ∠H(ω)")
plt.plot(w, phase * 180 / np.pi, label="fase [°]")
plt.axvline(wn, linestyle="--", label="ω_n")
plt.xlabel("ω [rad/s]")
plt.ylabel("fase [°]")
plt.grid(True)
plt.legend()


print("Lectura de fase en FRF:")
print(" - Antes de la resonancia: fase ≈ 0° (respuesta casi en fase con la fuerza).")
print(" - Cerca de la resonancia: transición rápida de fase.")
print(" - Muy por encima: fase ≈ -180° (respuesta casi opuesta).")
print()

# ============================ 4) Marcadores ω_n y ω_d sobre magnitud ============================

plt.figure()
plt.title("Marcadores de resonancia en la magnitud")
plt.plot(w, mag, label="|H(ω)|")
plt.axvline(wn, linestyle="--", label="ω_n (natural)")
plt.axvline(wd, linestyle=":", label="ω_d (amortiguada)")
plt.xlabel("ω [rad/s]")
plt.ylabel("|H(ω)| [m/N]")
plt.grid(True)
plt.legend()


# ============================ 5) Diagrama de Argand en ω* ============================

w_star = 0.9 * wn
H_star = 1.0 / (k - m * w_star**2 + 1j * c * w_star)
a, b = H_star.real, H_star.imag
phi_star = np.angle(H_star)  # rad

print("=== Punto de análisis (ω*) ===")
print(f"ω* = {w_star:.3f} rad/s (~0.9 ω_n)")
print(f"H(ω*) = {a:.4e} + j{b:.4e}")
print(f"|H(ω*)| = {np.abs(H_star):.4e} [m/N]")
print(f"∠H(ω*) = {phi_star * 180 / np.pi:.2f}°")
print()

plt.figure()
plt.title("Diagrama de Argand de H(ω*)")
plt.axhline(0); plt.axvline(0)
plt.plot([0, a], [0, b], label="H(ω*)")
plt.scatter([a], [b])
plt.xlabel("Re{H}"); plt.ylabel("Im{H}")
plt.grid(True); plt.legend()


# ============================ 6) Señales temporales a ω*: F(t) y x(t) ============================

F0 = 1.0  # N
T_star = 2 * np.pi / w_star
t_end = 10 * T_star
t = np.linspace(0, t_end, 4000)

F_t = F0 * np.cos(w_star * t)                     # fuerza
x_amp = np.abs(H_star) * F0                       # amplitud de salida
x_phi = np.angle(H_star)                          # fase de salida
x_t = x_amp * np.cos(w_star * t + x_phi)          # respuesta

plt.figure()
plt.title("Entrada y salida a una frecuencia ω*")
plt.plot(t, F_t, label="F(t) [N]")
plt.plot(t, x_t, linestyle="--", label="x(t) [m] (desfasada)")
plt.xlabel("t [s]"); plt.ylabel("amplitud")
plt.grid(True); plt.legend()


# ============================ 7) Retardo equivalente τ = -φ/ω* ============================

tau = -x_phi / w_star
print("=== Retardo temporal equivalente ===")
print(f"φ(ω*) = {x_phi * 180/np.pi:.2f}°  →  τ = -φ/ω* = {tau:.6f} s")
print("Interpretación: x(t) ≈ x_amp · cos(ω* · (t - τ))")
print()

# Verificación
x_t_delay = x_amp * np.cos(w_star * (t - tau))
max_diff = np.max(np.abs(x_t - x_t_delay))
assert max_diff < 1e-12, "La equivalencia fase↔retardo debería ser exacta para senoidales puras"

plt.figure()
plt.title("Verificación del retardo equivalente")
plt.plot(t, x_t, label="x(t) = x_amp cos(ω* t + φ)")
plt.plot(t, x_t_delay, linestyle="--", label="x_amp cos(ω*(t - τ))")
plt.xlabel("t [s]"); plt.ylabel("amplitud")
plt.grid(True); plt.legend()


print(f"Máxima diferencia numérica entre ambas formas: {max_diff:.3e}")

# ============================ 8) Mini “prueba Fourier” ============================

# La integral de x(t)·g(t)·e^{-j ω t} es grande cuando ω coincide con el contenido principal.
sigma_win = 0.2 * t_end
g = gauss_window(t, sigma_win)
X_wstar = np.trapz(x_t * g * np.exp(-1j * w_star * t), t)
X_2wstar = np.trapz(x_t * g * np.exp(-1j * (2 * w_star) * t), t)

print("\n=== Demostración estilo Fourier ===")
print(f"|∫ x(t) g(t) e^(-j ω* t) dt| = {np.abs(X_wstar):.4e}   (grande)")
print(f"|∫ x(t) g(t) e^(-j 2ω* t) dt| = {np.abs(X_2wstar):.4e} (pequeño)")
print("Interpretación: x(t) contiene principalmente la frecuencia ω*, no 2ω*.\n")

print("Listo. Cierra ventanas para terminar.")

plt.show()
