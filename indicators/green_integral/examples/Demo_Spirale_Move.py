import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


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
# 2. Paramètres de la spirale avec centre mobile en x
# ============================================================

N_cycles = 4
points_par_cycle = 800

r0 = 0.4
croissance = 0.09

centre_x0 = 0.0
centre_v0 = 0.0

# Déplacement du centre uniquement en x.
# Plus cette valeur est grande, plus le centre avance horizontalement.
vitesse_centre_x = 0.15
vitesse_centre_v = 0.00

theta_debut = 0.0
theta_fin = 2 * np.pi * N_cycles

N = N_cycles * points_par_cycle

theta = np.linspace(theta_debut, theta_fin, N + 1)

r = r0 + croissance * theta

centre_x = centre_x0 + vitesse_centre_x * theta
centre_v = centre_v0 + vitesse_centre_v * theta

x = centre_x + r * np.cos(theta)
v = centre_v + r * np.sin(theta)


# ============================================================
# 3. Fermeture globale de la trajectoire
# ============================================================

A_global_oriented = shoelace_oriented(x, v)
A_global_abs = abs(A_global_oriented)

C_global_open = shoelace_open_contribution(x, v)

K_global = closure_contribution(
    x_start=x[0],
    y_start=v[0],
    x_end=x[-1],
    y_end=v[-1]
)

verification = A_global_oriented - (C_global_open + K_global)


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
# 5. Figure 1 : signaux x(theta), v(theta), r(theta), centre_x(theta)
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(theta, x, label="x(θ)")
plt.plot(theta, v, label="v(θ)")
plt.plot(theta, r, "--", label="r(θ)")
plt.plot(theta, centre_x, ":", label="centre_x(θ)")
plt.plot(theta, centre_v, ":", label="centre_v(θ)")

plt.xlabel("phase θ [rad]")
plt.ylabel("amplitude")
plt.title("1. Signaux : spirale avec centre mobile uniquement en x")
plt.legend()
plt.tight_layout()


# ============================================================
# 6. Figure 2 : trajectoire + fermeture globale
# ============================================================

plt.figure(figsize=(7, 6))

plt.plot(x, v, label="trajectoire")
plt.plot(centre_x, centre_v, "--", label="trajectoire du centre")

plt.plot(
    [x[-1], x[0]],
    [v[-1], v[0]],
    "--",
    linewidth=2,
    label="fermeture globale"
)

plt.scatter(x[0], v[0], marker="x", s=80, label="début")
plt.scatter(x[-1], v[-1], marker="x", s=80, label="fin")

plt.scatter(centre_x[0], centre_v[0], marker="o", s=70, label="centre initial")
plt.scatter(centre_x[-1], centre_v[-1], marker="o", s=70, label="centre final")

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
    "2. Spirale avec centre mobile en x\n"
    f"A orientée globale = {A_global_oriented:.6f}"
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

plt.plot(centre_x, centre_v, "--", label="trajectoire du centre")

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "3. Tours fermés localement\n"
    f"ΣA tours = {A_tours_sum:.6f}"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 8. Figure 4 : comparaison des aires et fermetures
# ============================================================

labels = [
    "A globale\nfermée",
    "ΣA tours\nlocaux",
    "Σ|A tours|",
    "C globale\nouverte",
    "ΣC tours\nouvertes",
    "K globale\nfermeture",
    "ΣK tours\nfermetures"
]

values = [
    A_global_oriented,
    A_tours_sum,
    A_tours_abs_sum,
    C_global_open,
    C_tours_sum,
    K_global,
    K_tours_sum
]

plt.figure(figsize=(11, 4))
bars = plt.bar(labels, values)
plt.axhline(0, linewidth=0.8)
plt.ylabel("valeur orientée")
plt.title("4. Comparaison : trajectoire globale vs tours locaux")

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f"{h:.3f}",
        ha="center",
        va="bottom" if h >= 0 else "top",
        fontsize=8,
        rotation=90
    )

plt.tight_layout()


# ============================================================
# 9. Figure 5 : carte de multiplicité / winding number
# ============================================================

marge = 0.4
xmin, xmax = x.min() - marge, x.max() + marge
ymin, ymax = v.min() - marge, v.max() + marge

nx = 500
ny = 500

xg = np.linspace(xmin, xmax, nx)
yg = np.linspace(ymin, ymax, ny)

W = np.zeros((ny, nx))

for iy, yy in enumerate(yg):
    for ix, xx in enumerate(xg):
        W[iy, ix] = winding_number_point(xx, yy, x, v)

W_round = np.round(W)

plt.figure(figsize=(8, 6))

levels = np.arange(W_round.min() - 0.5, W_round.max() + 1.5, 1)

contour = plt.contourf(
    xg,
    yg,
    W_round,
    levels=levels,
    alpha=0.75,
    cmap="inferno"
)

cbar = plt.colorbar(contour)
cbar.set_label("multiplicité / winding number")
cbar.set_ticks(np.arange(W_round.min(), W_round.max() + 1, 1))
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))



plt.plot(x, v, color="black", linewidth=1.4, label="trajectoire")
plt.plot(centre_x, centre_v, "--", color="black", linewidth=1.2, label="centre")

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
    "5. Multiplicité de la trajectoire fermée\n"
    "n indique combien de fois chaque région est entourée"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()

plt.show()


# ============================================================
# 10. Résultats numériques
# ============================================================

print("==================================================")
print("RÉSULTATS : SPIRALE AVEC CENTRE MOBILE EN x")
print("==================================================")
print(f"Nombre de cycles :                       {N_cycles}")
print(f"Rayon initial r0 :                       {r0}")
print(f"Croissance radiale :                     {croissance}")
print(f"Vitesse centre x :                       {vitesse_centre_x}")
print(f"Vitesse centre v :                       {vitesse_centre_v}")
print("--------------------------------------------------")
print("TRAJECTOIRE GLOBALE")
print("--------------------------------------------------")
print(f"C globale ouverte :                      {C_global_open:.10f}")
print(f"K fermeture globale :                    {K_global:.10f}")
print(f"A globale orientée fermée :              {A_global_oriented:.10f}")
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
print(f"A globale - ΣA tours :                   {A_global_oriented - A_tours_sum:.10f}")
print(f"C globale - ΣC tours :                   {C_global_open - C_tours_sum:.10e}")
print(f"K globale - ΣK tours :                   {K_global - K_tours_sum:.10f}")
print("==================================================")
print("INTERPRÉTATION")
print("==================================================")
print(
    "Cette trajectoire est une spirale dont l'amplitude augmente,\n"
    "mais dont le centre se déplace uniquement en x.\n\n"
    "La trajectoire globale est fermée par un segment artificiel.\n"
    "Si ce segment ou la trajectoire elle-même crée des zones auto-intersectées,\n"
    "l'aire orientée globale compte les régions avec multiplicité.\n\n"
    "Les tours locaux donnent une autre lecture : chaque tour est fermé\n"
    "par son propre segment local. Cela permet de comparer l'aire globale\n"
    "avec une mesure cycle par cycle."
)
print("==================================================")