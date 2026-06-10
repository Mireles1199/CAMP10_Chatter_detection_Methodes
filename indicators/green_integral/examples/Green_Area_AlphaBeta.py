import numpy as np
import matplotlib.pyplot as plt


def shoelace_oriented(x, y):
    """
    Aire orientée d'une courbe fermée avec la formule du lacet.

    Si la courbe n'est pas fermée, cette fonction relie quand même
    le dernier point au premier grâce à np.roll.
    """
    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


# ============================================================
# 1. Paramètres contrôlables
# ============================================================

N_cycles = 5
points_par_cycle = 10000

croissance = 0.00
# croissance = 0.00  -> rayons constants
# croissance = 0.05  -> spirale visible

centre_mobile = True
# centre_mobile = False -> centre fixe
# centre_mobile = True  -> centre qui se déplace

centre_x0 = 0.0
centre_v0 = 0.0

vitesse_centre_x = 0.1
vitesse_centre_v = 0.00

frequence = 0.5  # cycles par unité de temps
omega = 2 * np.pi * frequence

# Période réelle d'un cycle
T_cycle = 1 / frequence


# ============================================================
# 2. Contrôle des morceaux de bêta en degrés de phase
# ============================================================

tramo_beta_inicio_grados = 0
tramo_beta_final_grados = 0

# Exemples :
# 0 degrés   -> pas de morceau ajouté
# 90 degrés  -> un quart de cycle
# 180 degrés -> un demi-cycle
# 360 degrés -> un cycle complet

tramo_beta_inicio_cycles = tramo_beta_inicio_grados / 360.0
tramo_beta_final_cycles = tramo_beta_final_grados / 360.0

tramo_beta_inicio = tramo_beta_inicio_cycles * T_cycle
tramo_beta_final = tramo_beta_final_cycles * T_cycle


# ============================================================
# 3. Contrôle des morceaux de chaque alpha en degrés de phase
# ============================================================

tramo_alpha_inicio_grados = +10
tramo_alpha_final_grados =500

# Exemples :
# tramo_alpha_inicio_grados = 0.0 ; tramo_alpha_final_grados = 0.0
#   -> chaque alpha commence et finit exactement aux coupures naturelles.
#
# tramo_alpha_inicio_grados = 30.0 ; tramo_alpha_final_grados = 0.0
#   -> chaque alpha commence 30 degrés avant sa coupure naturelle.
#
# tramo_alpha_inicio_grados = 0.0 ; tramo_alpha_final_grados = 45.0
#   -> chaque alpha finit 45 degrés après sa coupure naturelle.
#
# tramo_alpha_inicio_grados = 30.0 ; tramo_alpha_final_grados = 45.0
#   -> chaque alpha contient un morceau avant et un morceau après.

tramo_alpha_inicio_cycles = tramo_alpha_inicio_grados / 360.0
tramo_alpha_final_cycles = tramo_alpha_final_grados / 360.0

tramo_alpha_inicio = tramo_alpha_inicio_cycles * T_cycle
tramo_alpha_final = tramo_alpha_final_cycles * T_cycle


# ============================================================
# 4. Création de la fenêtre temporelle bêta
# ============================================================

t_debut_beta = -tramo_beta_inicio
t_fin_beta = N_cycles * T_cycle + tramo_beta_final

nombre_points_total = int(
    (
        N_cycles
        + tramo_beta_inicio_cycles
        + tramo_beta_final_cycles
    )
    * points_par_cycle
) + 1

t = np.linspace(
    t_debut_beta,
    t_fin_beta,
    nombre_points_total
)

# Temps naturels où commencent/finissent les cycles alpha
temps_coupures = np.array([i * T_cycle for i in range(N_cycles + 1)])
indices_coupures = [np.argmin(np.abs(t - tc)) for tc in temps_coupures]


# ============================================================
# 5. Centre fixe ou mobile
# ============================================================

if centre_mobile:
    centre_x = centre_x0 + vitesse_centre_x * t
    centre_v = centre_v0 + vitesse_centre_v * t
else:
    centre_x = np.full_like(t, centre_x0)
    centre_v = np.full_like(t, centre_v0)


# ============================================================
# 6. Signaux temporels x(t), v(t)
# ============================================================

A = 1 + croissance * t

x = centre_x + A * np.cos(omega * t)
v = centre_v + A * np.sin(omega * t)

# Important :
# Aux coupures naturelles entre cycles alpha :
#
#     sin(omega*t) = 0
#
# donc :
#
#     v = centre_v
#
# Si centre_v0 = 0 et vitesse_centre_v = 0,
# alors les coupures naturelles alpha se font exactement à v = 0.
#
# Si vitesse_centre_v != 0,
# alors les coupures naturelles alpha se font à v = centre_v(t),
# pas nécessairement à v = 0.


# ============================================================
# 7. Vérification des coupures naturelles
# ============================================================

print("====================================")
print("VÉRIFICATION DES COUPURES NATURELLES")
print("====================================")

for i, idx in enumerate(indices_coupures):
    print(
        f"coupure naturelle {i}: "
        f"t = {t[idx]:.10f}, "
        f"x = {x[idx]:.10f}, "
        f"v = {v[idx]:.10e}, "
        f"centre_v = {centre_v[idx]:.10e}, "
        f"v - centre_v = {v[idx] - centre_v[idx]:.10e}"
    )

print("====================================")


# ============================================================
# 8. Graphique des signaux temporels
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(t, x, label="x(t)")
plt.plot(t, v, label="v(t)")

# Points de coupure naturelle alpha
for idx in indices_coupures:
    plt.scatter(t[idx], v[idx], s=25)

# Début et fin de bêta
plt.scatter(t[0], v[0], s=45, marker="x", label="début bêta")
plt.scatter(t[-1], v[-1], s=45, marker="x", label="fin bêta")

plt.xlabel("temps t")
plt.ylabel("amplitude")
plt.title("Signaux temporels avec morceaux bêta et alpha")
plt.legend()
plt.tight_layout()


# ============================================================
# 9. Diagramme de phase complet
# ============================================================

plt.figure(figsize=(6, 6))
plt.plot(x, v, label="trajectoire bêta complète")
plt.plot(centre_x, centre_v, "--", label="trajectoire du centre")

plt.scatter(centre_x[0], centre_v[0], s=80, marker="x", label="centre initial")
plt.scatter(centre_x[-1], centre_v[-1], s=80, marker="x", label="centre final")

# Points de coupure naturelle alpha
for idx in indices_coupures:
    plt.scatter(x[idx], v[idx], s=25)

plt.scatter(x[0], v[0], s=45, marker="x", label="début bêta")
plt.scatter(x[-1], v[-1], s=45, marker="x", label="fin bêta")

plt.xlabel("x")
plt.ylabel("v")
plt.title("Diagramme de phase x-v")
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 10. Trajectoire bêta et aire bêta
# ============================================================

aire_beta = shoelace_oriented(x, v)

plt.figure(figsize=(6, 6))
plt.plot(x, v, label="trajectoire bêta")
plt.fill(x, v, alpha=0.25, label="aire bêta")

plt.plot(centre_x, centre_v, "--", label="trajectoire du centre")

plt.scatter(x[0], v[0], s=45, marker="x", label="début bêta", zorder=3)
plt.scatter(x[-1], v[-1], s=45, marker="x", label="fin bêta", zorder=3)

# Fermeture artificielle globale de bêta
plt.plot(
    [x[-1], x[0]],
    [v[-1], v[0]],
    "--",
    label="fermeture artificielle bêta"
)

plt.xlabel("x")
plt.ylabel("v")
plt.title(f"Trajectoire bêta\nAire bêta = {aire_beta:.6f}")
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 11. Sous-trajectoires alpha et aires alpha
# ============================================================

aires_alpha = []

plt.figure(figsize=(7, 7))

for i in range(N_cycles):
    # Temps naturel de l'alpha i
    t_alpha_debut_naturel = i * T_cycle
    t_alpha_fin_naturelle = (i + 1) * T_cycle

    # Temps étendu de l'alpha i
    t_alpha_debut = t_alpha_debut_naturel - tramo_alpha_inicio
    t_alpha_fin = t_alpha_fin_naturelle + tramo_alpha_final

    # Sécurité : on ne sort pas de la fenêtre bêta disponible
    t_alpha_debut = max(t_alpha_debut, t[0])
    t_alpha_fin = min(t_alpha_fin, t[-1])

    # Indices les plus proches
    debut = np.argmin(np.abs(t - t_alpha_debut))
    fin = np.argmin(np.abs(t - t_alpha_fin)) + 1

    x_i = x[debut:fin]
    v_i = v[debut:fin]

    aire_i = shoelace_oriented(x_i, v_i)
    aires_alpha.append(aire_i)

    linea_alpha, = plt.plot(
        x_i,
        v_i,
        label=f"alpha {i + 1}: A = {aire_i:.6f}"
    )

    color_alpha = linea_alpha.get_color()

    # Fermeture artificielle de chaque alpha étendu
    # avec la même couleur que la trajectoire alpha.
    plt.plot(
        [x_i[-1], x_i[0]],
        [v_i[-1], v_i[0]],
        "--",
        linewidth=1.5,
        color=color_alpha
    )

    # Points de début et fin du alpha étendu
    plt.scatter(x_i[0], v_i[0], s=20, color=color_alpha)
    plt.scatter(x_i[-1], v_i[-1], s=20, color=color_alpha)

plt.plot(centre_x, centre_v, "--", label="trajectoire du centre")

# Début et fin de bêta
plt.scatter(x[0], v[0], s=55, marker="x", label="début bêta")
plt.scatter(x[-1], v[-1], s=55, marker="x", label="fin bêta")

plt.xlabel("x")
plt.ylabel("v")
plt.title("Sous-trajectoires alpha avec morceaux avant/après")
plt.axis("equal")
plt.legend()
plt.tight_layout()

plt.show()


# ============================================================
# 12. Comparaison numérique
# ============================================================

somme_alpha = np.sum(aires_alpha)
difference = aire_beta - somme_alpha

print("====================================")
print("RÉSULTATS")
print("====================================")
print(f"Nombre de cycles alpha :             {N_cycles}")
print(f"Points par cycle :                   {points_par_cycle}")
print(f"Fréquence :                          {frequence}")
print(f"Période d'un cycle :                 {T_cycle}")
print(f"Croissance :                         {croissance}")
print(f"Centre mobile :                      {centre_mobile}")
print(f"Centre initial x :                   {centre_x0}")
print(f"Centre initial v :                   {centre_v0}")
print(f"Vitesse centre x :                   {vitesse_centre_x}")
print(f"Vitesse centre v :                   {vitesse_centre_v}")
print("------------------------------------")
print(f"Tramo début bêta en degrés :         {tramo_beta_inicio_grados}")
print(f"Tramo fin bêta en degrés :           {tramo_beta_final_grados}")
print(f"Tramo début alpha en degrés :        {tramo_alpha_inicio_grados}")
print(f"Tramo fin alpha en degrés :          {tramo_alpha_final_grados}")
print("------------------------------------")
print(f"Aire bêta :                          {aire_beta:.10f}")
print(f"Somme des aires alpha :              {somme_alpha:.10f}")
print(f"Bêta - somme alpha :                 {difference:.10f}")
print("------------------------------------")

for i, aire_i in enumerate(aires_alpha, start=1):
    print(f"Aire alpha {i} :                      {aire_i:.10f}")

print("====================================")