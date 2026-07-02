import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Fonctions d'aire et de multiplicité
# ============================================================

def shoelace_oriented(x, y):
    """
    Aire orientée fermée avec fermeture automatique.
    """
    if len(x) < 3:
        return np.nan

    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def shoelace_open_contribution(x, y):
    """
    Contribution orientée ouverte, sans fermeture.
    """
    if len(x) < 2:
        return np.nan

    return 0.5 * float(
        np.sum(x[:-1] * y[1:] - y[:-1] * x[1:])
    )


def closure_contribution(x_start, y_start, x_end, y_end):
    """
    Contribution orientée du segment de fermeture :
    point final -> point initial.
    """
    return 0.5 * float(
        x_end * y_start - y_end * x_start
    )


def winding_number_point(px, py, x, y):
    """
    Nombre d'enroulement de la courbe fermée autour du point (px, py).
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
# 2. Paramètres de la spirale
# ============================================================

N_cycles = 4
points_par_cycle = 800

r0 = 0.4
croissance = 0.09

centre_x = 0.0
centre_v = 0.0

theta_debut = 0.0
theta_fin = 2 * np.pi * N_cycles

N = N_cycles * points_par_cycle

theta = np.linspace(theta_debut, theta_fin, N + 1)

r = r0 + croissance * theta

x = centre_x + r * np.cos(theta)
v = centre_v + r * np.sin(theta)


# ============================================================
# 3. Fermeture globale de la spirale
# ============================================================

# Shoelace ferme automatiquement :
# dernier point -> premier point

A_spirale_oriented = shoelace_oriented(x, v)
A_spirale_abs = abs(A_spirale_oriented)

C_spirale_open = shoelace_open_contribution(x, v)

K_spirale = closure_contribution(
    x_start=x[0],
    y_start=v[0],
    x_end=x[-1],
    y_end=v[-1]
)

# Vérification : A = C + K
verification = A_spirale_oriented - (C_spirale_open + K_spirale)


# ============================================================
# 4. Découpage par tours pour comparaison
# ============================================================

tours = []

for i in range(N_cycles):
    debut = i * points_par_cycle
    fin = (i + 1) * points_par_cycle + 1

    x_i = x[debut:fin]
    v_i = v[debut:fin]

    A_i = shoelace_oriented(x_i, v_i)
    C_i = shoelace_open_contribution(x_i, v_i)
    K_i = closure_contribution(
        x_start=x_i[0],
        y_start=v_i[0],
        x_end=x_i[-1],
        y_end=v_i[-1]
    )

    tours.append((x_i, v_i, A_i, C_i, K_i))


A_tours = np.array([item[2] for item in tours])
C_tours = np.array([item[3] for item in tours])
K_tours = np.array([item[4] for item in tours])

A_tours_sum = np.sum(A_tours)
A_tours_abs_sum = np.sum(np.abs(A_tours))

C_tours_sum = np.sum(C_tours)
K_tours_sum = np.sum(K_tours)


# ============================================================
# 5. Figure 1 : signaux x(theta), v(theta), r(theta)
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(theta, x, label="x(θ)")
plt.plot(theta, v, label="v(θ)")
plt.plot(theta, r, "--", label="r(θ)")

plt.xlabel("phase θ [rad]")
plt.ylabel("amplitude")
plt.title("1. Signaux de la spirale à amplitude croissante")
plt.legend()
plt.tight_layout()


# ============================================================
# 6. Figure 2 : spirale ouverte + fermeture globale
# ============================================================

plt.figure(figsize=(6, 6))

plt.plot(x, v, label="spirale")
plt.plot(
    [x[-1], x[0]],
    [v[-1], v[0]],
    "--",
    linewidth=2,
    label="fermeture globale"
)

plt.scatter(x[0], v[0], marker="x", s=80, label="début")
plt.scatter(x[-1], v[-1], marker="x", s=80, label="fin")

# Quelques flèches pour le sens de parcours
for k in np.linspace(50, len(x) - 60, 12, dtype=int):
    plt.arrow(
        x[k],
        v[k],
        x[k + 10] - x[k],
        v[k + 10] - v[k],
        head_width=0.04,
        length_includes_head=True
    )

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "2. Spirale fermée globalement\n"
    f"A orientée = {A_spirale_oriented:.6f}"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 7. Figure 3 : tours fermés localement
# ============================================================

plt.figure(figsize=(7, 7))

for i, (x_i, v_i, A_i, C_i, K_i) in enumerate(tours, start=1):
    line, = plt.plot(x_i, v_i, label=f"tour {i}: A = {A_i:.3f}")
    color = line.get_color()

    plt.plot(
        [x_i[-1], x_i[0]],
        [v_i[-1], v_i[0]],
        "--",
        color=color,
        linewidth=1.5
    )

    plt.fill(x_i, v_i, alpha=0.10, color=color)

    plt.scatter(x_i[0], v_i[0], s=25, color=color)
    plt.scatter(x_i[-1], v_i[-1], s=25, color=color)

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "3. Tours de la spirale fermés localement\n"
    f"ΣA tours = {A_tours_sum:.6f}"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 8. Figure 4 : comparaison des aires et fermetures
# ============================================================

labels = [
    "A spirale\nfermée globale",
    "ΣA tours\nfermés locaux",
    "Σ|A tours|",
    "C spirale\nouverte",
    "ΣC tours\nouvertes",
    "K spirale\nfermeture",
    "ΣK tours\nfermetures"
]

values = [
    A_spirale_oriented,
    A_tours_sum,
    A_tours_abs_sum,
    C_spirale_open,
    C_tours_sum,
    K_spirale,
    K_tours_sum
]

plt.figure(figsize=(11, 4))
bars = plt.bar(labels, values)
plt.axhline(0, linewidth=0.8)
plt.ylabel("valeur orientée")
plt.title("4. Comparaison : spirale globale vs tours locaux")

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f"{h:.3f}",
        ha="center",
        va="bottom" if h >= 0 else "top",
        fontsize=8,
        rotation=0
    )

plt.tight_layout()


# ============================================================
# 9. Figure 5 : carte de multiplicité / winding number
# ============================================================

# On analyse la courbe fermée : spirale + fermeture globale.
# Comme shoelace avec np.roll ferme automatiquement,
# le winding number utilise aussi la fermeture dernier -> premier.

marge = 0.3
xmin, xmax = x.min() - marge, x.max() + marge
ymin, ymax = v.min() - marge, v.max() + marge

nx = 140
ny = 140

xg = np.linspace(xmin, xmax, nx)
yg = np.linspace(ymin, ymax, ny)

W = np.zeros((ny, nx))

for iy, yy in enumerate(yg):
    for ix, xx in enumerate(xg):
        W[iy, ix] = winding_number_point(xx, yy, x, v)

W_round = np.round(W)

plt.figure(figsize=(7, 7))

levels = np.arange(W_round.min() - 0.5, W_round.max() + 1.5, 1)

contour =plt.contourf(
    xg,
    yg,
    W_round,
    levels=levels,
    alpha=0.35
)

cbar = plt.colorbar(contour)
cbar.set_label("multiplicité / winding number")
cbar.set_ticks(np.arange(W_round.min(), W_round.max() + 1, 1))
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%d"))

plt.contour(
    xg,
    yg,
    W_round,
    levels=levels,
    colors="black",
    linewidths=0.5,
    alpha=0.5
)

plt.plot(x, v, color="black", linewidth=1.4, label="spirale")
plt.plot(
    [x[-1], x[0]],
    [v[-1], v[0]],
    "--",
    color="black",
    linewidth=2,
    label="fermeture globale"
)

plt.scatter(x[0], v[0], marker="x", s=80, label="début")
plt.scatter(x[-1], v[-1], marker="x", s=80, label="fin")

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "5. Multiplicité de la spirale fermée\n"
    "Les régions internes peuvent avoir n = 1, 2, 3, ..."
)
plt.axis("equal")
plt.legend()
plt.tight_layout()

plt.show()


# ============================================================
# 10. Résultats numériques
# ============================================================

print("==================================================")
print("RÉSULTATS POUR LA SPIRALE À AMPLITUDE CROISSANTE")
print("==================================================")
print(f"Nombre de cycles :                       {N_cycles}")
print(f"Rayon initial r0 :                       {r0}")
print(f"Croissance radiale :                     {croissance}")
print("--------------------------------------------------")
print("SPIRALE GLOBALE")
print("--------------------------------------------------")
print(f"C spirale ouverte :                      {C_spirale_open:.10f}")
print(f"K fermeture globale :                    {K_spirale:.10f}")
print(f"A spirale orientée fermée globale :      {A_spirale_oriented:.10f}")
print(f"Vérification A - (C + K) :               {verification:.3e}")
print("--------------------------------------------------")
print("TOURS LOCAUX")
print("--------------------------------------------------")
print(f"ΣC tours ouvertes :                      {C_tours_sum:.10f}")
print(f"ΣK tours fermetures :                    {K_tours_sum:.10f}")
print(f"ΣA tours fermés localement :             {A_tours_sum:.10f}")
print(f"Σ|A tours| :                             {A_tours_abs_sum:.10f}")
print("--------------------------------------------------")
print("DIFFÉRENCES")
print("--------------------------------------------------")
print(f"A globale - ΣA tours :                   {A_spirale_oriented - A_tours_sum:.10f}")
print(f"C globale - ΣC tours :                   {C_spirale_open - C_tours_sum:.10e}")
print(f"K globale - ΣK tours :                   {K_spirale - K_tours_sum:.10f}")
print("==================================================")
print("INTERPRÉTATION")
print("==================================================")
print(
    "La spirale n'est pas une simple courbe fermée : on la ferme avec un segment global.\n"
    "Le winding number montre combien de fois chaque région est entourée.\n"
    "Les régions proches du centre peuvent avoir une multiplicité plus grande,\n"
    "parce que la spirale les entoure plusieurs fois.\n\n"
    "Donc l'aire orientée globale n'est pas seulement une aire géométrique simple :\n"
    "elle correspond à une somme de régions pondérées par leur multiplicité.\n\n"
    "En découpant par tours, chaque tour a sa propre fermeture locale.\n"
    "Cela donne une autre décomposition de l'aire, utile si l'on veut une mesure\n"
    "cycle par cycle."
)
print("==================================================")