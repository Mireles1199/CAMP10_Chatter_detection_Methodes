# ft_intuitiva_visual.py
"""
Transformada de Fourier (FT) — Guía visual e intuitiva con código

Qué muestra (alineado con tu resumen):
  1) Señal ejemplo x(t) = cos(ω0 t) y su descomposición exponencial
  2) Idea de "correlación" con ondas de prueba: ∫ x(t) g(t) e^{-i ω t} dt
  3) Espectro continuo (numérico) |X(ω)|: picos en ±ω0
  4) Por qué aparece e^{-i ω t} y qué mide la fase: Argand de X(ω*)
  5) Efecto de la ventana g(t): anchura temporal ↔ resolución en frecuencia
  6) Diferencia entre usar ω (rad/s) y f (Hz): verificación numérica

Reglas de gráficos:
  - Matplotlib (sin seaborn)
  - Una figura por gráfico (sin subplots múltiples)
  - No se fijan colores manualmente

Uso:
  python ft_intuitiva_visual.py
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# ============================ Utilidades ============================

def gaussian(t: np.ndarray, sigma: float) -> np.ndarray:
    """Ventana gaussiana centrada en 0: g(t) = exp(-t^2 / (2 σ^2))."""
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0")
    return np.exp(-(t**2) / (2.0 * sigma**2))

def numeric_FT(x_t: np.ndarray, t: np.ndarray, omega: np.ndarray, window: np.ndarray | None = None) -> np.ndarray:
    """
    Aproxima X(ω) = ∫ x(t) g(t) e^{-i ω t} dt mediante regla del trapecio.
    - x_t: señal en el tiempo
    - t: malla temporal UNIFORME
    - omega: array de frecuencias angulares (rad/s)
    - window: ventana g(t) (si None, g(t)=1)
    Retorna: X(ω) para cada ω (complejo)
    """
    if t.size < 3 or not np.allclose(np.diff(np.diff(t)), 0, atol=1e-12):
        raise ValueError("t debe ser malla uniforme y con >= 3 puntos")
    dt = t[1] - t[0]
    if window is None:
        w_t = np.ones_like(t)
    else:
        w_t = window
        if w_t.shape != t.shape:
            raise ValueError("window y t deben tener la misma forma")

    # pesos de trapecio
    trap = np.ones_like(t)
    trap[0] = 0.5
    trap[-1] = 0.5

    # Matriz E_{k,n} = e^{-i ω_k t_n}
    E = np.exp(-1j * np.outer(omega, t))
    Xw = (E @ (x_t * w_t * trap)) * dt
    return Xw

def unwrap_phase(z: np.ndarray) -> np.ndarray:
    """Fase desenrollada de un array complejo."""
    return np.unwrap(np.angle(z))


# ============================ Parámetros base ============================

# Señal ejemplo: x(t) = cos(ω0 t)
w0 = 10.0       # frecuencia angular objetivo (rad/s)
A  = 1.0        # amplitud
t = np.linspace(-2.0, 2.0, 8001)  # malla temporal densa y simétrica
dt = t[1] - t[0]

# Ventana para evitar efectos de borde (y ejemplificar resolución)
sigma = 0.25
g = gaussian(t, sigma)

# Rejilla de frecuencias angulares para el "espectro"
omega = np.linspace(-40.0, 40.0, 4001)

print("=== Parámetros ===")
print(f"ω0 = {w0:.3f} rad/s")
print(f"malla temporal: N={t.size}, dt={dt:.4e} s, duración={t[-1]-t[0]:.3f} s")
print(f"ventana gaussiana: σ={sigma:.3f} s")
print(f"rejilla de ω: K={omega.size}, Δω={omega[1]-omega[0]:.3f} rad/s")
print()


# ============================ 1) Señal y descomposición exponencial ============================

x = A * np.cos(w0 * t)

# Verificación numérica de cos(ωt) = Re{(e^{iωt} + e^{-iωt})/2}
recomp = 0.5 * (np.exp(1j*w0*t) + np.exp(-1j*w0*t)).real
err_cos = np.max(np.abs(x - recomp))
print("[1] Descomposición exponencial del coseno:")
print(f"    max|cos - Re[(e^{{iωt}}+e^{{-iωt}})/2]| = {err_cos:.3e}  (≈ precisión de máquina)")
print()

# Figura: señal x(t)
plt.figure()
plt.title("Señal temporal: x(t) = cos(ω0 t)")
plt.plot(t, x, label="x(t)")
plt.xlabel("t [s]"); plt.ylabel("amplitud")
plt.grid(True); plt.legend()



# ============================ 2) Correlación con ondas de prueba ============================

# Elegimos dos ω de prueba: ω0 (debería dar grande) y 2ω0 (pequeño)
omega_star = w0
omega_2star = 2*w0

X_omega_star  = np.trapz(x * g * np.exp(-1j*omega_star*t), t)
X_2omega_star = np.trapz(x * g * np.exp(-1j*omega_2star*t), t)

print("[2] Correlación con ondas de prueba (con ventana g):")
print(f"    |∫ x(t) g(t) e^(-i ω0 t) dt|   = {np.abs(X_omega_star):.4e}  (GRANDE)")
print(f"    |∫ x(t) g(t) e^(-i 2ω0 t) dt| = {np.abs(X_2omega_star):.4e}  (PEQUEÑO)")
print("    → La integral 'se enciende' cuando ω coincide con el contenido de x(t).")
print()


# ============================ 3) Espectro continuo (numérico) ============================

X = numeric_FT(x, t, omega, window=g)     # X(ω) = ∫ x(t) g(t) e^{-i ω t} dt
magX = np.abs(X)

# Figura: |X(ω)|
plt.figure()
plt.title("Espectro numérico |X(ω)| con ventana gaussiana")
plt.plot(omega, magX, label="|X(ω)|")
plt.axvline(+w0, linestyle="--", label="+ω0")
plt.axvline(-w0, linestyle="--", label="-ω0")
plt.xlabel("ω [rad/s]"); plt.ylabel("|X(ω)|")
plt.grid(True); plt.legend()


# Comprobación de picos
idx_pos = np.argmax(magX[omega >= 0])
omega_pos = omega[omega >= 0][idx_pos]
idx_neg = np.argmax(magX[omega <= 0])
omega_neg = omega[omega <= 0][idx_neg]
print("[3] Picos espectrales detectados:")
print(f"    pico positivo en ω ≈ {omega_pos:.3f} rad/s (esperado ≈ +{w0})")
print(f"    pico negativo en ω ≈ {omega_neg:.3f} rad/s (esperado ≈ -{w0})")
print()


# ============================ 4) Fase: qué mide y cómo leerla ============================

# Elegimos ω* = ω0 y miramos X(ω*) en el plano complejo (Argand)
X_star = np.trapz(x * g * np.exp(-1j*omega_star*t), t)
a, b = X_star.real, X_star.imag
phi_star = np.angle(X_star)    # fase en rad

print("[4] Fase en X(ω*):")
print(f"    X(ω*) = {a:.4e} + j{b:.4e}")
print(f"    |X(ω*)| = {np.abs(X_star):.4e}")
print(f"    ∠X(ω*) = {phi_star*180/np.pi:.2f}°")
print("    → La fase indica adelanto/atraso relativo de x(t) frente a la onda de referencia e^{-i ω* t}.")
print()

# Figura: Argand de X(ω*)
plt.figure()
plt.title("Diagrama de Argand de X(ω*)")
plt.axhline(0); plt.axvline(0)
plt.plot([0, a], [0, b], label="X(ω*)")
plt.scatter([a], [b])
plt.xlabel("Re{X}"); plt.ylabel("Im{X}")
plt.grid(True); plt.legend()



# ============================ 5) Efecto de la ventana: resolución frecuencia-tiempo ============================

# Comparamos dos sigmas: angosta (mejor frecuencia, peor tiempo) vs ancha (mejor tiempo, peor frecuencia)
sigma_narrow = 0.5      # ventana más ancha en tiempo → más estrecha en frecuencia (OJO: gauss: área ↑, ancho tiempo ↑)
sigma_wide   = 0.1      # ventana más angosta en tiempo → más ancha en frecuencia

g_narrow = gaussian(t, sigma_narrow)
g_wide   = gaussian(t, sigma_wide)

X_narrow = numeric_FT(x, t, omega, window=g_narrow)
X_wide   = numeric_FT(x, t, omega, window=g_wide)

plt.figure()
plt.title("Efecto de la ventana: |X(ω)| para distintas σ")
plt.plot(omega, np.abs(X_narrow), label=f"|X| con σ={sigma_narrow}")
plt.plot(omega, np.abs(X_wide),   linestyle="--", label=f"|X| con σ={sigma_wide}")
plt.xlabel("ω [rad/s]"); plt.ylabel("|X(ω)|")
plt.grid(True); plt.legend()


print("[5] Ventana y compromiso:")
print("    - σ grande ⇒ g(t) más ancha en tiempo ⇒ lóbulos espectrales más finos (mejor resolución en ω).")
print("    - σ pequeña ⇒ g(t) más angosta ⇒ espectro más ancho (peor resolución en ω).")
print()


# ============================ 6) ω (rad/s) vs f (Hz) ============================

# Verificación numérica simple de ω = 2π f: para f0 = ω0 / (2π) la onda de prueba coincide
f0 = w0 / (2*np.pi)
omega_from_f = 2*np.pi * f0
X_from_f = np.trapz(x * g * np.exp(-1j*omega_from_f*t), t)

print("[6] Relación ω (rad/s) y f (Hz): ω = 2π f")
print(f"    f0 = ω0 / (2π) = {f0:.4f} Hz  →  2π f0 = {omega_from_f:.4f} rad/s")
print(f"    |X(2π f0)| = {np.abs(X_from_f):.4e} (debe coincidir con |X(ω0)| dentro de error numérico)")
print()

print("Listo. Cierra las ventanas de las figuras para terminar.")
plt.show()
