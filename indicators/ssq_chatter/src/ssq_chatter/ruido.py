import numpy as np
import matplotlib.pyplot as plt

# Semilla para reproducibilidad
rng = np.random.default_rng(0)

# Eje temporal
t = np.linspace(0, 1, 500)  # 1 segundo, 500 muestras

# Señal limpia (por ejemplo, una sinusoide de 5 Hz)
f0 = 5  # Hz
x_clean = np.sin(2 * np.pi * f0 * t)

# Parámetro del ruido
ruido_std = 0.3  # desviación estándar del ruido

# Ruido gaussiano: n(t) ~ N(0, ruido_std^2)
noise = rng.normal(0.0, ruido_std, size=t.shape)

# Señal con ruido: x_noisy(t) = x_clean(t) + n(t)
x_noisy = x_clean + noise

# --- Gráficas ---
fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# 1) Señal limpia
axs[0].plot(t, x_clean)
axs[0].set_title("Señal limpia $x_{\\text{clean}}(t)$")
axs[0].set_ylabel("Amplitud")

# 2) Ruido
axs[1].plot(t, noise)
axs[1].set_title(
    "Ruido gaussiano $n(t) \\sim \\mathcal{N}(0, \\sigma^2)$ con "
    f"$\\sigma = {ruido_std}$"
)
axs[1].set_ylabel("Amplitud")

# 3) Señal limpia + ruido
axs[2].plot(t, x_noisy, label="Señal con ruido $x_{\\text{noisy}}(t)$")
axs[2].plot(t, x_clean, linestyle="--", alpha=0.7, label="Señal limpia")
axs[2].set_title("$x_{\\text{noisy}}(t) = x_{\\text{clean}}(t) + n(t)$")
axs[2].set_xlabel("Tiempo [s]")
axs[2].set_ylabel("Amplitud")
axs[2].legend()

plt.tight_layout()
plt.show()
