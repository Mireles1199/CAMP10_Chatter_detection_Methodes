import numpy as np
import matplotlib.pyplot as plt


def shoelace_oriented(x, y):
    """
    Área orientada de una curva cerrada usando shoelace.
    Si la curva no está cerrada, esta función igualmente conecta
    el último punto con el primero debido a np.roll.
    """
    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def edge_area(x1, y1, x2, y2):
    """
    Contribución al área orientada del segmento que va de
    (x1, y1) a (x2, y2).
    """
    return 0.5 * (x1 * y2 - y1 * x2)


def open_path_area(x, y):
    """
    Área orientada acumulada SOLO por los segmentos existentes
    de una trayectoria abierta.

    No incluye el cierre desde el último punto al primero.
    """
    return 0.5 * float(
        np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    )


# ============================================================
# Parámetros
# ============================================================

N_loops = 4
d = 2          # d < 2r => puede haber solape entre círculos
n = 1000

# Ángulo: lazo casi cerrado, pero no completamente cerrado
theta = np.linspace(np.pi / 2 * 0.5, 2 * np.pi, n)

# Prueba 1: radios variables
r = np.linspace(1, 1.5, N_loops)

# Prueba 2: radios iguales
# r = np.linspace(1, 1, N_loops)


# ============================================================
# Construcción de lazos
# ============================================================

loops = []          # lazos abiertos
loops_closed = []   # lazos cerrados individualmente

areas_arcos = []
areas_cierres_locales = []
areas_lazos_cerrados = []

for k in range(N_loops):
    xc = k * d
    yc = 0.0

    x = xc + r[k] * np.cos(theta)
    y = yc + r[k] * np.sin(theta)

    loops.append((x, y))

    # Área del arco abierto, SIN cerrar el lazo
    A_arco = open_path_area(x, y)

    # Área del cierre local: último punto -> primer punto
    A_cierre_local = edge_area(x[-1], y[-1], x[0], y[0])

    # Área del lazo cerrado individualmente
    A_lazo_cerrado = A_arco + A_cierre_local

    areas_arcos.append(A_arco)
    areas_cierres_locales.append(A_cierre_local)
    areas_lazos_cerrados.append(A_lazo_cerrado)

    # Para graficar el lazo cerrado individualmente
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    loops_closed.append((x_closed, y_closed))


# ============================================================
# Caso A: suma de áreas cerrando cada lazo individualmente
# ============================================================

A_arcos_total = sum(areas_arcos)
A_cierres_locales_total = sum(areas_cierres_locales)
A_lazos_cerrados_total_orientada = sum(areas_lazos_cerrados)

# Esta es la que tú estabas usando:
A_acumulada_abs = sum(abs(A) for A in areas_lazos_cerrados)


# ============================================================
# Caso B: trayectoria completa cerrada una sola vez
# ============================================================

# Concatenar todos los lazos abiertos
x_all_open = np.concatenate([x for x, y in loops])
y_all_open = np.concatenate([y for x, y in loops])

# Área de todo el trayecto abierto, incluyendo las conexiones
# automáticas entre final de un lazo e inicio del siguiente
A_trayecto_abierto_con_conexiones = open_path_area(x_all_open, y_all_open)

# Cierre global: último punto del último lazo -> primer punto del primer lazo
A_cierre_global = edge_area(
    x_all_open[-1], y_all_open[-1],
    x_all_open[0], y_all_open[0]
)

A_trayecto_completo_orientada = (
    A_trayecto_abierto_con_conexiones
    + A_cierre_global
)

# Equivalente usando shoelace sobre el trayecto cerrado
x_all_closed = np.append(x_all_open, x_all_open[0])
y_all_closed = np.append(y_all_open, y_all_open[0])

A_trayecto_shoelace = shoelace_oriented(x_all_closed, y_all_closed)


# ============================================================
# Separación explícita de conexiones globales
# ============================================================

areas_conexiones_entre_lazos = []

for k in range(N_loops - 1):
    x1 = loops[k][0][-1]
    y1 = loops[k][1][-1]

    x2 = loops[k + 1][0][0]
    y2 = loops[k + 1][1][0]

    A_conexion = edge_area(x1, y1, x2, y2)
    areas_conexiones_entre_lazos.append(A_conexion)

A_conexiones_entre_lazos_total = sum(areas_conexiones_entre_lazos)

# Las conexiones globales son:
# - conexiones entre lazos
# - cierre global del último lazo al primer lazo
A_conexiones_globales_total = (
    A_conexiones_entre_lazos_total
    + A_cierre_global
)

# El trayecto completo puede escribirse como:
A_trayecto_por_partes = (
    A_arcos_total
    + A_conexiones_globales_total
)

# La diferencia entre ambos métodos viene de:
diferencia_cierres = (
    A_conexiones_globales_total
    - A_cierres_locales_total
)


# ============================================================
# Impresión de resultados
# ============================================================

print("\n==============================")
print("ÁREAS POR LAZO")
print("==============================")

for i in range(N_loops):
    print(f"\nLazo {i + 1}")
    print(f"  Radio                         : {r[i]:.6f}")
    print(f"  Área del arco abierto          : {areas_arcos[i]: .6f}")
    print(f"  Área del cierre local          : {areas_cierres_locales[i]: .6f}")
    print(f"  Área lazo cerrado              : {areas_lazos_cerrados[i]: .6f}")
    print(f"  |Área lazo cerrado|            : {abs(areas_lazos_cerrados[i]): .6f}")


print("\n==============================")
print("CASO A: CERRAR CADA LAZO")
print("==============================")
print(f"Suma áreas de arcos                         : {A_arcos_total: .6f}")
print(f"Suma áreas de cierres locales               : {A_cierres_locales_total: .6f}")
print(f"Suma orientada de lazos cerrados            : {A_lazos_cerrados_total_orientada: .6f}")
print(f"Suma de valores absolutos de lazos cerrados : {A_acumulada_abs: .6f}")


print("\n==============================")
print("CASO B: CERRAR TODO UNA SOLA VEZ")
print("==============================")
print(f"Área trayecto abierto con conexiones        : {A_trayecto_abierto_con_conexiones: .6f}")
print(f"Área cierre global                          : {A_cierre_global: .6f}")
print(f"Área trayecto completo orientada            : {A_trayecto_completo_orientada: .6f}")
print(f"Área trayecto completo con shoelace          : {A_trayecto_shoelace: .6f}")
print(f"|Área trayecto completo|                    : {abs(A_trayecto_completo_orientada): .6f}")


print("\n==============================")
print("CONEXIONES GLOBALES")
print("==============================")

for i, A_con in enumerate(areas_conexiones_entre_lazos, 1):
    print(f"Conexión lazo {i} -> lazo {i + 1}: {A_con: .6f}")

print(f"Cierre global último -> primero             : {A_cierre_global: .6f}")
print(f"Suma conexiones entre lazos                 : {A_conexiones_entre_lazos_total: .6f}")
print(f"Conexiones globales totales                 : {A_conexiones_globales_total: .6f}")


print("\n==============================")
print("COMPARACIÓN CLAVE")
print("==============================")
print(f"Cierres locales totales                     : {A_cierres_locales_total: .6f}")
print(f"Conexiones globales totales                 : {A_conexiones_globales_total: .6f}")
print(f"Diferencia conexiones - cierres locales     : {diferencia_cierres: .6f}")

print("\nVerificación:")
print(f"Trayecto por partes                         : {A_trayecto_por_partes: .6f}")
print(f"Trayecto completo orientado                 : {A_trayecto_completo_orientada: .6f}")
print(f"Diferencia numérica                         : {A_trayecto_por_partes - A_trayecto_completo_orientada: .12f}")

print("\n==============================")
print("ERROR COMO LO ESTABAS MIDIENDO")
print("==============================")

error_pct = (
    (abs(A_trayecto_completo_orientada) - A_acumulada_abs)
    / A_acumulada_abs
    * 100
)

print(f"Área acumulada por lazos, con abs           : {A_acumulada_abs: .6f}")
print(f"|Área trayecto completo|                    : {abs(A_trayecto_completo_orientada): .6f}")
print(f"Diferencia acumulada - trayecto             : {A_acumulada_abs - abs(A_trayecto_completo_orientada): .6f}")
print(f"Relación acumulada / trayecto               : {A_acumulada_abs / abs(A_trayecto_completo_orientada): .6f}")
print(f"Error (%)                                   : {error_pct: .6f}")


# ============================================================
# Figura
# ============================================================

plt.figure(figsize=(9, 4.5))

# Lazos cerrados individualmente
for k, (x, y) in enumerate(loops_closed):
    plt.plot(x, y, label=f"Lazo {k + 1}")

# Trayecto completo cerrado una sola vez
plt.plot(
    x_all_closed,
    y_all_closed,
    "k--",
    alpha=0.35,
    linewidth=4,
    label="Trayecto completo cerrado una vez"
)

# Dibujar conexiones entre lazos en negro
for k in range(N_loops - 1):
    x1 = loops[k][0][-1]
    y1 = loops[k][1][-1]
    x2 = loops[k + 1][0][0]
    y2 = loops[k + 1][1][0]

    plt.plot([x1, x2], [y1, y2], "k-", linewidth=2, alpha=0.7)

# Dibujar cierre global último -> primero
plt.plot(
    [x_all_open[-1], x_all_open[0]],
    [y_all_open[-1], y_all_open[0]],
    "k-",
    linewidth=2,
    alpha=0.7,
    label="Conexiones/cierre global"
)

# Dibujar cierres locales en gris punteado
for k, (x, y) in enumerate(loops):
    plt.plot(
        [x[-1], x[0]],
        [y[-1], y[0]],
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label="Cierres locales" if k == 0 else None
    )

plt.axhline(0, linewidth=0.8)
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Área por lazo vs área del trayecto completo")
plt.legend()
plt.tight_layout()
plt.show()