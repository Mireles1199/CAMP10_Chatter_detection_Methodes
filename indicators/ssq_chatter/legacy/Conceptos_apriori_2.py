"""
Fundamentos para entender S_x(μ, ξ) sin gráficas
- Demostraciones numéricas con logs y asserts
- Solo NumPy (sin Matplotlib)

Qué verifica:
  [1] coseno = (e^{iωt}+e^{-iωt})/2 (parte real)
  [2] Ventana gaussiana y su FT analítica vs FT numérica (trapecio)
  [3] Modulación en tiempo => desplazamiento en frecuencia
  [4] Desplazamiento temporal => fase ≈ -ω μ (magnitud invariante)
  [5] Simetría de espectro para función real/par
  [6] Proto-STFT: S_x(μ, ξ) cerrado vs. integral numérica

Autor: tú+yo
"""

from __future__ import annotations
import numpy as np

# ----------------------------- Utilidades generales -----------------------------

def gauss(t: np.ndarray, sigma: float) -> np.ndarray:
    """g(t) = exp(-t^2/(2 σ^2))"""
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0")
    return np.exp(-(t**2) / (2.0 * sigma**2))

def gauss_hat_analytic(omega: np.ndarray, sigma: float) -> np.ndarray:
    """FT continua bajo convención: ĝ(ω) = ∫ g(t) e^{-i ω t} dt (frecuencia angular)"""
    return sigma * np.sqrt(2*np.pi) * np.exp(-0.5 * (sigma**2) * (omega**2))

def numeric_continuous_FT(x_t: np.ndarray, t: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Aproxima ĝ(ω) = ∫ x(t) e^{-i ω t} dt por trapecio.
    Retorna array de tamaño len(omega).
    """
    if len(t) < 3 or not np.allclose(np.diff(np.diff(t)), 0, atol=1e-12):
        raise ValueError("t debe ser malla uniforme y con >=3 puntos")
    dt = t[1] - t[0]
    w = np.ones_like(t); w[0] = 0.5; w[-1] = 0.5  # pesos trapecio
    # Matriz E_{k,n} = e^{-i ω_k t_n}
    E = np.exp(-1j * np.outer(omega, t))
    return (E @ (x_t * w)) * dt

def unwrap_phase(z: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(z))

def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Error relativo máximo en magnitud entre arrays complejos/reales."""
    num = np.abs(np.abs(a) - np.abs(b))
    den = np.abs(b) + 1e-12
    return float(np.max(num / den))

# ----------------------------- Parámetros base -----------------------------

A = 1.0
w0 = 10.0               # rad/s
mu = 0.30               # s
sigma = 0.10            # s

t = np.linspace(-1.0, 1.0, 4001)      # malla temporal (uniforme)
omega = np.linspace(-60, 60, 2401)    # rad/s para FTs

print("\n=== Parámetros ===")
print(f"A={A}, ω0={w0} rad/s, μ={mu} s, σ={sigma} s")
print(f"malla temporal: N={t.size}, dt={t[1]-t[0]:.4e} s")
print(f"malla de frecuencias: K={omega.size}, Δω={omega[1]-omega[0]:.4e} rad/s")

# ----------------------------- [1] Coseno vs exponenciales -----------------------------
print("\n[1] Coseno = parte real de (e^{iωt}+e^{-iωt})/2")
cos_ref = np.cos(w0 * t)
cos_from_exp = 0.5 * (np.exp(1j*w0*t) + np.exp(-1j*w0*t)).real
imag_part = 0.5 * (np.exp(1j*w0*t) + np.exp(-1j*w0*t)).imag

err1 = float(np.max(np.abs(cos_ref - cos_from_exp)))
err1_im = float(np.max(np.abs(imag_part)))

print(f"  max|diff real| = {err1:.3e} (esperado ~1e-15)")
print(f"  max|parte imag| = {err1_im:.3e} (debe ser ~0)")
assert err1 < 1e-12 and err1_im < 1e-12, "Fallo en identidad del coseno"

# ----------------------------- [2] FT gaussiana: analítica vs numérica -----------------
print("\n[2] FT de g(t): analítica vs. numérica (trapecio)")
g_t = gauss(t, sigma)
g_hat_num = numeric_continuous_FT(g_t, t, omega)
g_hat_ana = gauss_hat_analytic(omega, sigma)

err2 = rel_err(g_hat_num, g_hat_ana)
print(f"  error relativo máximo (|ĝ_num|-|ĝ_ana|)/|ĝ_ana| = {err2:.3e}")
assert err2 < 1e-2, "FT numérica de la gaussiana no coincide (revisa N, rango o dt)"

# ----------------------------- [3] Modulación -> desplazamiento espectral --------------
print("\n[3] Modulación: FT{g(t) e^{iω0 t}} ≈ ĝ(ω - ω0)")
g_mod = g_t * np.exp(1j * w0 * t)
gmod_hat_num = numeric_continuous_FT(g_mod, t, omega)
gmod_hat_ref = gauss_hat_analytic(omega - w0, sigma)

err3 = rel_err(gmod_hat_num, gmod_hat_ref)
print(f"  error relativo máximo = {err3:.3e}")
assert err3 < 5e-2, "Fallo en propiedad de modulación"

# ----------------------------- [4] Desplazamiento temporal -----------------------------
print("\n[4] Desplazamiento temporal: FT{g(t-μ)} = e^{-i ω μ} ĝ(ω)")
g_shift = gauss(t - mu, sigma)
gshift_hat_num = numeric_continuous_FT(g_shift, t, omega)

# Fase teórica relativa: -ω μ. Comprobamos pendiente por ajuste lineal.
phase_diff = unwrap_phase(gshift_hat_num / (g_hat_ana + 1e-18))
coef = np.polyfit(omega, phase_diff, 1)   # fase ≈ a*ω + b
slope_est = float(coef[0])                 # ~ -μ
print(f"  pendiente fase ≈ {slope_est:.5f} rad/(rad/s); esperada = {-mu:.5f}")
assert np.isclose(slope_est, -mu, atol=2e-3), "Pendiente de fase no coincide con -μ"

# Magnitudes iguales
err4_mag = rel_err(np.abs(gshift_hat_num), np.abs(g_hat_ana))
print(f"  magnitud invariante: error relativo = {err4_mag:.3e}")
assert err4_mag < 1e-3, "La magnitud debería ser igual tras desplazar en tiempo"

# ----------------------------- [5] Simetría para función real/par ----------------------
print("\n[5] Simetría (g real y par) -> |ĝ(ω)| = |ĝ(-ω)|")
symm = float(np.max(np.abs(np.abs(g_hat_ana) - np.abs(g_hat_ana[::-1]))))
print(f"  max |ĝ(ω)| - |ĝ(-ω)| = {symm:.3e}")
assert symm < 1e-10, "No se cumple paridad numéricamente (rango/malla insuficientes)"

# ----------------------------- [6] Proto-STFT S_x(μ, ξ) --------------------------------
print("\n[6] Proto-STFT: S_x(μ, ξ) para x(t)=A cos(ω0 t) — cerrado vs integral")

def Sx_closed(mu: float, xi: np.ndarray, A: float, w0: float, sigma: float) -> np.ndarray:
    term1 = np.exp(1j * w0 * mu) * gauss_hat_analytic(xi - w0, sigma)
    term2 = np.exp(-1j * w0 * mu) * gauss_hat_analytic(xi + w0, sigma)
    return 0.5 * A * (term1 + term2)

def Sx_integral(mu: float, xi: float, A: float, w0: float, sigma: float,
                t: np.ndarray) -> complex:
    g_loc = gauss(t - mu, sigma)
    integrand = 0.5 * A * (np.exp(1j*w0*t) + np.exp(-1j*w0*t)) * g_loc * np.exp(-1j*xi*(t - mu))
    return np.trapz(integrand, t)

xi_grid = np.linspace(-40, 40, 1601)
S_closed = Sx_closed(mu, xi_grid, A, w0, sigma)
S_num = np.array([Sx_integral(mu, xi, A, w0, sigma, t) for xi in xi_grid])

err6_abs = float(np.max(np.abs(S_closed - S_num)))
idx_peak = int(np.argmax(np.abs(S_closed)))
xi_peak = float(xi_grid[idx_peak])
mag_peak = float(np.abs(S_closed[idx_peak]))

print(f"  max |S_closed - S_num| = {err6_abs:.3e}")
print(f"  pico |S_x| aprox en ξ ≈ {xi_peak:.2f} rad/s; |S|_max ≈ {mag_peak:.3e}")
print("  esperado: dos lóbulos alrededor de ±ω0 (solo reportamos el positivo numéricamente).")
assert err6_abs < 5e-3, "S_x cerrado no coincide con la integral (revisa N o rango temporal)"

# ----------------------------- Resumen final ------------------------------------------
print("\n=== RESUMEN DE CHECKS ===")
print(" [1] Identidad del coseno.................. OK")
print(" [2] ĝ analítica ≈ ĝ numérica.............. OK")
print(" [3] Modulación desplaza espectro.......... OK")
print(" [4] Desplazamiento temporal (fase -ωμ).... OK")
print(" [5] Simetría |ĝ(ω)| = |ĝ(-ω)|............. OK")
print(" [6] S_x(μ, ξ) cerrado ≈ numérico.......... OK")

print("\nSugerencias para explorar:")
print(" - Cambia σ para ver el compromiso tiempo↔frecuencia (σ grande -> espectro estrecho).")
print(" - Cambia μ: debería cambiar la fase de S_x, no su magnitud (señal estacionaria).")
print(" - Cambia ω0 y mira dónde cae el pico en ξ.")
print(" - Reduce la longitud de t o aumenta dt y observa cómo rompes los asserts (diagnóstico).")
