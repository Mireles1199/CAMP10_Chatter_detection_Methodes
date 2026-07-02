import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Fonctions d'aire
# ============================================================

def shoelace_oriented(x, y):
    """
    Aire orientée fermée avec fermeture automatique.
    Peut être positive, négative ou proche de zéro.
    """
    if len(x) < 3:
        return np.nan

    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def shoelace_abs(x, y):
    """
    Magnitude de l'aire orientée fermée.
    """
    return abs(shoelace_oriented(x, y))


def winding_number_point(px, py, x, y):
    """
    Nombre d'enroulement autour du point (px, py).

    Retourne environ :
        +1 si la courbe entoure le point en sens positif
        -1 si la courbe entoure le point en sens négatif
         0 si le point est dehors
    """
    dx = x - px
    dy = y - py

    dx_next = np.roll(dx, -1)
    dy_next = np.roll(dy, -1)

    cross = dx * dy_next - dy * dx_next
    dot = dx * dx_next + dy * dy_next

    angles = np.arctan2(cross, dot)

    return np.sum(angles) / (2 * np.pi)


# ============================================================
# 2. Créer une trajectoire en forme de 8
# ============================================================

N = 2000
t = np.linspace(0, 2 * np.pi, N + 1)

# Figure 8 classique
x = np.sin(t)
v = np.sin(2 * t)

# Cette courbe se croise au centre.
# Les deux lobes sont :
#   lobe droit :  t de 0 à pi
#   lobe gauche : t de pi à 2pi

idx_mid = np.argmin(np.abs(t - np.pi))

x_lobe_1 = x[:idx_mid + 1]
v_lobe_1 = v[:idx_mid + 1]

x_lobe_2 = x[idx_mid:]
v_lobe_2 = v[idx_mid:]


# ============================================================
# 3. Aires orientées
# ============================================================

A_total_oriented = shoelace_oriented(x, v)
A_total_abs_direct = abs(A_total_oriented)

A_lobe_1 = shoelace_oriented(x_lobe_1, v_lobe_1)
A_lobe_2 = shoelace_oriented(x_lobe_2, v_lobe_2)

A_lobes_oriented_sum = A_lobe_1 + A_lobe_2
A_lobes_abs_sum = abs(A_lobe_1) + abs(A_lobe_2)


# ============================================================
# 4. Figure 1 : trajectoire complète en forme de 8
# ============================================================

plt.figure(figsize=(6, 6))

plt.plot(x, v, label="trajectoire complète")
plt.scatter(x[0], v[0], marker="x", s=80, label="début")
plt.scatter(x[idx_mid], v[idx_mid], s=60, label="intersection / coupure")
plt.scatter(x[-1], v[-1], marker="x", s=80, label="fin")

# Quelques flèches pour montrer le sens de parcours
for k in np.linspace(100, N - 100, 10, dtype=int):
    plt.arrow(
        x[k],
        v[k],
        x[k + 8] - x[k],
        v[k + 8] - v[k],
        head_width=0.04,
        length_includes_head=True
    )

# Puntos clave en t = 0, π/4, π/2, 3π/4, 5π/4, 3π/2, 7π/4
t_key = [0, np.pi/4, np.pi/2, 3*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4]
t_labels = ["0", "π/4", "π/2", "3π/4", "5π/4", "3π/2", "7π/4"]
offsets = [(0.04, 0.06), (0.05, 0.06), (0.05, 0.06), (0.05, -0.10),
           (-0.12, -0.10), (0.04, -0.10), (0.05, 0.06)]
for tk, tlbl, (dx_off, dy_off) in zip(t_key, t_labels, offsets):
    idx_k = np.argmin(np.abs(t - tk))
    xk, vk = x[idx_k], v[idx_k]
    plt.scatter(xk, vk, s=55, zorder=5)
    plt.annotate(
        f"t={tlbl}\n({xk:.2f}, {vk:.2f})",
        xy=(xk, vk),
        xytext=(xk + dx_off, vk + dy_off),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
        ha="center"
    )

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "1. Trajectoire en forme de 8\n"
    f"Aire orientée totale = {A_total_oriented:.6f}"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 5. Figure 2 : séparation des deux lobes
# ============================================================

plt.figure(figsize=(6, 6))

ligne1, = plt.plot(x_lobe_1, v_lobe_1, label=f"lobe 1 : A = {A_lobe_1:.6f}")
couleur1 = ligne1.get_color()
plt.fill(x_lobe_1, v_lobe_1, alpha=0.20, color=couleur1)

ligne2, = plt.plot(x_lobe_2, v_lobe_2, label=f"lobe 2 : A = {A_lobe_2:.6f}")
couleur2 = ligne2.get_color()
plt.fill(x_lobe_2, v_lobe_2, alpha=0.20, color=couleur2)

plt.scatter(0, 0, s=70, label="intersection")

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "2. Les deux régions du 8\n"
    "Une région peut compter positif, l'autre négatif"
)
plt.axis("equal")
plt.legend(fontsize=13)
plt.tight_layout()


# ============================================================
# 6. Figure 3 : comparaison des aires
# ============================================================

labels = [
    "A totale\norientée",
    "|A totale|",
    "A lobe 1",
    "A lobe 2",
    "A lobe 1\n+ A lobe 2",
    "|A lobe 1|\n+ |A lobe 2|"
]

values = [
    A_total_oriented,
    A_total_abs_direct,
    A_lobe_1,
    A_lobe_2,
    A_lobes_oriented_sum,
    A_lobes_abs_sum
]

plt.figure(figsize=(10, 4))
bars = plt.bar(labels, values)
plt.axhline(0, linewidth=0.8)
plt.ylabel("aire")
plt.title("3. Aire orientée vs somme géométrique des lobes")

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f"{h:.3f}",
        ha="center",
        va="bottom" if h >= 0 else "top",
        fontsize=9
    )

plt.tight_layout()


# ============================================================
# 7. Figure 4 : carte de multiplicité / winding number
# ============================================================

# Grille de points pour visualiser la multiplicité.
nx = 120
ny = 120

xg = np.linspace(-1.2, 1.2, nx)
yg = np.linspace(-1.2, 1.2, ny)

W = np.zeros((ny, nx))

for iy, yy in enumerate(yg):
    for ix, xx in enumerate(xg):
        W[iy, ix] = winding_number_point(xx, yy, x, v)

# Arrondir pour voir les régions entières : -1, 0, +1
W_round = np.round(W)

levels = np.arange(W_round.min() - 0.5, W_round.max() + 1.5, 1)

plt.figure(figsize=(6, 6))

contour = plt.contourf(
    xg,
    yg,
    W_round,
    levels=[-1.5, -0.5, 0.5, 1.5],
    alpha=0.35
)

cbar = plt.colorbar(contour)
cbar.set_label("multiplicité / winding number", fontsize=13)
cbar.set_ticks(np.arange(W_round.min(), W_round.max() + 1, 1))
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
cbar.ax.tick_params(labelsize=13)

plt.contour(
    xg,
    yg,
    W_round,
    levels=levels,
    colors="black",
    linewidths=0.5,
    alpha=0.5
)

plt.plot(x, v, color="black", linewidth=1.5, label="Trajectoire")
plt.scatter(0, 0, s=70, label="intersection", zorder=10, marker="x",)

# Punto de inicio: cuadrado rojo
plt.scatter(x[0], v[0], s=120, marker="s", color="red", zorder=2, label="Debut")
# Puntos clave
for tk, tlbl in zip(t_key, t_labels):
    idx_k = np.argmin(np.abs(t - tk))
    plt.scatter(x[idx_k], v[idx_k], s=60, color="orange", zorder=8)
# Flechas de sentido de recorrido
for k in np.linspace(50, N - 50, 12, dtype=int):
    plt.arrow(
        x[k], v[k],
        x[k + 10] - x[k], v[k + 10] - v[k],
        head_width=0.08, head_length=0.12,
        fc="black", ec="black",
        length_includes_head=False,
        zorder=7
    )

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "4. Multiplicité des régions\n"
    "n = +1 se somme, n = -1 se soustrait, n = 0 dehors"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()

plt.show()


# ============================================================
# 8. Résultats numériques
# ============================================================

print("==================================================")
print("RÉSULTATS POUR LA TRAJECTOIRE EN 8")
print("==================================================")
print(f"Aire orientée totale du 8 :        {A_total_oriented:.10f}")
print(f"Magnitude directe |A totale| :     {A_total_abs_direct:.10f}")
print("------------------------------------")
print(f"Aire orientée lobe 1 :             {A_lobe_1:.10f}")
print(f"Aire orientée lobe 2 :             {A_lobe_2:.10f}")
print(f"Somme orientée des lobes :         {A_lobes_oriented_sum:.10f}")
print(f"Somme des magnitudes des lobes :   {A_lobes_abs_sum:.10f}")
print("==================================================")
print("INTERPRÉTATION")
print("==================================================")
print(
    "La formule de Green/shoelace calcule une aire orientée.\n"
    "Dans une trajectoire auto-intersectée, les régions sont comptées\n"
    "avec une multiplicité orientée.\n\n"
    "Ici, un lobe est parcouru avec un signe et l'autre avec le signe opposé.\n"
    "Donc l'aire totale orientée peut être proche de zéro même si les deux\n"
    "lobes ont une aire géométrique non nulle.\n\n"
    "C'est pourquoi :\n"
    "    |A totale orientée| != |A lobe 1| + |A lobe 2|\n\n"
    "La somme |A lobe 1| + |A lobe 2| représente mieux la magnitude\n"
    "géométrique totale des régions du 8."
)
print("==================================================")