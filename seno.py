import numpy as np
import matplotlib.pyplot as plt

# ====== PARÁMETROS AJUSTABLES ======
amplitud = 1          # Altura de la onda
periodo = 2 * np.pi  # Periodo (distancia entre ciclos)
fase = 0             # Desplazamiento horizontal
desplazamiento = 0   # Desplazamiento vertical

# Configuración del rango de x
x_inicio = 0
x_fin = 60
puntos = 10000

# ====== GENERAR DATOS ======
x = np.linspace(x_inicio, x_fin, puntos)

# Fórmula del seno ajustable
y = amplitud * np.sin((2 * np.pi / periodo) * x + fase) + desplazamiento

# ====== CONFIGURACIÓN DEL PLOT ======
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Función seno", color="blue", linewidth=4)

plt.title("Gráfica de función seno")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")

# plt.axhline(0)  # eje horizontal
# plt.axvline(0)  # eje vertical

# plt.grid()
# plt.legend()

# ====== MOSTRAR ======
plt.show()