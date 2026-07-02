import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Fonctions d'aire
# ============================================================

def shoelace_open_contribution(x, y):
    """
    Contribution orientée d'une trajectoire ouverte.
    Ne ferme PAS la courbe.
    """
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


def shoelace_closed_area(x, y):
    """
    Aire orientée fermée avec la formule du lacet.
    Ferme automatiquement la courbe :
    dernier point -> premier point.
    """
    return shoelace_open_contribution(x, y) + closure_contribution(
        x_start=x[0],
        y_start=y[0],
        x_end=x[-1],
        y_end=y[-1]
    )


# ============================================================
# 2. Paramètres contrôlables
# ============================================================

N_cycles_alpha = 5
points_par_cycle = 1000

scenario = "cercles_compatibles"
# Options :
# scenario = "cercles_compatibles"
# scenario = "spirale"
# scenario = "centre_mobile"
# scenario = "spirale_centre_mobile"

rayon = 1.0
croissance = 0.08

deplacement_centre_x = 1
deplacement_centre_y = 0.00

# Fenêtre bêta arbitraire en phase
fase_inicio_beta_grados = -45
fase_final_beta_grados = 135

# Phase de coupure des alpha.
#
# Cas particuliers :
#   0°   -> coupure à v = 0, côté droit
#   180° -> coupure à v = 0, côté gauche
#   90°  -> coupure près du maximum de v
#   270° -> coupure près du minimum de v
#
# Pour retrouver le cas précédent :
#   fase_corte_alpha_grados = 0.0
fase_corte_alpha_grados = 0

# Mode de coupure alpha:
#   "aligned_v0"      -> tous les cycles coupés à la même phase (référence v=0)
#   "per_cycle_offset" -> chaque cycle a sa propre phase de coupure
mode_coupure_alpha = "per_cycle_offset"

# Si mode_coupure_alpha == "per_cycle_offset", cette liste donne
# la phase de coupure de chaque cycle (en degrés).
# Elle est répétée / tronquée si besoin.
phases_coupure_alpha_grados = [0, 0, 0, 0, 0]


# ============================================================
# 3. Fenêtre bêta en phase
# ============================================================

fase_inicio_beta = np.deg2rad(fase_inicio_beta_grados)
fase_final_beta = np.deg2rad(fase_final_beta_grados)
fase_corte_alpha = np.deg2rad(fase_corte_alpha_grados)

theta_debut_beta = fase_inicio_beta
theta_fin_beta = 2 * np.pi * N_cycles_alpha + fase_final_beta

theta_total = theta_fin_beta - theta_debut_beta

nombre_points_total = int(
    theta_total / (2 * np.pi) * points_par_cycle
) + 1

theta = np.linspace(
    theta_debut_beta,
    theta_fin_beta,
    nombre_points_total
)


# ============================================================
# 4. Génération de la trajectoire bêta
# ============================================================

if scenario == "cercles_compatibles":
    # Rayon constant, centre fixe.
    r = rayon * np.ones_like(theta)
    cx = np.zeros_like(theta)
    cy = np.zeros_like(theta)

elif scenario == "spirale":
    # Rayon croissant.
    r = rayon + croissance * theta
    cx = np.zeros_like(theta)
    cy = np.zeros_like(theta)

elif scenario == "centre_mobile":
    # Rayon constant, centre mobile.
    r = rayon * np.ones_like(theta)
    cx = deplacement_centre_x * theta / (2 * np.pi)
    cy = deplacement_centre_y * theta / (2 * np.pi)

elif scenario == "spirale_centre_mobile":
    # Rayon croissant, centre mobile.
    r = rayon + croissance * theta
    cx = deplacement_centre_x * theta / (2 * np.pi)
    cy = deplacement_centre_y * theta / (2 * np.pi)
else:
    raise ValueError("Scenario inconnu.")


x_beta = cx + r * np.cos(theta)
y_beta = cy + r * np.sin(theta)


# ============================================================
# 5. Définition des coupures alpha
# ============================================================

theta_coupures = []

if mode_coupure_alpha == "aligned_v0":
    # Tous les cycles partagent la même phase de coupe.
    k_min = int(np.ceil((theta_debut_beta - fase_corte_alpha) / (2 * np.pi)))
    k_max = int(np.floor((theta_fin_beta - fase_corte_alpha) / (2 * np.pi)))

    for k in range(k_min, k_max + 1):
        theta_k = fase_corte_alpha + 2 * np.pi * k
        if theta_debut_beta <= theta_k <= theta_fin_beta:
            theta_coupures.append(theta_k)

elif mode_coupure_alpha == "per_cycle_offset":
    # Chaque cycle a sa propre phase de coupure.
    n_cycles = int(np.floor((theta_fin_beta - theta_debut_beta) / (2 * np.pi)))
    if not phases_coupure_alpha_grados:
        raise ValueError("phases_coupure_alpha_grados ne peut pas être vide.")

    for j in range(n_cycles + 1):
        phase_deg = phases_coupure_alpha_grados[j % len(phases_coupure_alpha_grados)]
        theta_k = np.deg2rad(phase_deg) + 2 * np.pi * j
        if theta_debut_beta <= theta_k <= theta_fin_beta:
            theta_coupures.append(theta_k)

else:
    raise ValueError(f"Mode de coupure inconnu: {mode_coupure_alpha}")

theta_coupures = np.array(theta_coupures)

indices_coupures = [
    int(np.argmin(np.abs(theta - th)))
    for th in theta_coupures
]

print("==================================================")
print("DIAGNOSTIC CYCLE CUTS")
print("==================================================")
print(f"Mode de coupure alpha : {mode_coupure_alpha}")
print(f"Phase de référence (aligned_v0) : {fase_corte_alpha_grados:.3e}°")
print(f"Phases par cycle (per_cycle_offset) : {[f'{p:.3e}' for p in phases_coupure_alpha_grados]}")
print(f"Nombre de coups détectées : {len(theta_coupures):.3e}")
print("theta_coupures (rad) :")
for i, th in enumerate(theta_coupures, start=1):
    print(f"  {i:>2d}: {th:.6e}")
print("indices_coupures :")
print(" ", indices_coupures)
print("theta_coupures (deg) :")
for i, th in enumerate(theta_coupures, start=1):
    print(f"  {i:>2d}: {np.rad2deg(th):.6e}°")
print("==================================================")


# ============================================================
# 6. Figure 0 : signaux x et v
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(theta, x_beta, label="x(θ)")
plt.plot(theta, y_beta, label="v(θ)")

plt.scatter(theta[0], x_beta[0], marker="x", s=70, label="début bêta x")
plt.scatter(theta[-1], x_beta[-1], marker="x", s=70, label="fin bêta x")

plt.scatter(theta[0], y_beta[0], marker="o", s=45, label="début bêta v")
plt.scatter(theta[-1], y_beta[-1], marker="o", s=45, label="fin bêta v")

# Coupures alpha
for idx in indices_coupures:
    plt.scatter(theta[idx], y_beta[idx], s=35)

plt.xlabel("phase θ [rad]")
plt.ylabel("amplitude")
plt.title(
    "0. Signaux x(θ) et v(θ)\n"
    f"coupure alpha = {fase_corte_alpha_grados}°"
)
plt.legend()
plt.tight_layout()


# ============================================================
# 7. Construction des alpha
# ============================================================

alphas = []

for j in range(len(indices_coupures) - 1):
    debut = indices_coupures[j]
    fin = indices_coupures[j + 1] + 1

    x_i = x_beta[debut:fin]
    y_i = y_beta[debut:fin]
    theta_i = theta[debut:fin]

    alphas.append((x_i, y_i, theta_i, debut, fin))


# ============================================================
# 8. Résidus : morceaux de bêta hors cycles alpha complets
# ============================================================

residus = []

if len(indices_coupures) > 0:
    # Résidu initial : début bêta -> première coupure alpha
    if indices_coupures[0] > 0:
        debut = 0
        fin = indices_coupures[0] + 1
        residus.append(("résidu initial", x_beta[debut:fin], y_beta[debut:fin]))

    # Résidu final : dernière coupure alpha -> fin bêta
    if indices_coupures[-1] < len(theta) - 1:
        debut = indices_coupures[-1]
        fin = len(theta)
        residus.append(("résidu final", x_beta[debut:fin], y_beta[debut:fin]))
else:
    residus.append(("bêta entière sans cycle complet", x_beta, y_beta))


# ============================================================
# 9. Vérification trajectoire :
#    résidus + alpha reconstruisent bêta
# ============================================================

segments_complets = []

if len(indices_coupures) > 0 and indices_coupures[0] > 0:
    segments_complets.append(
        (
            x_beta[0:indices_coupures[0] + 1],
            y_beta[0:indices_coupures[0] + 1]
        )
    )

for x_i, y_i, theta_i, debut, fin in alphas:
    segments_complets.append((x_i, y_i))

if len(indices_coupures) > 0 and indices_coupures[-1] < len(theta) - 1:
    segments_complets.append(
        (
            x_beta[indices_coupures[-1]:],
            y_beta[indices_coupures[-1]:]
        )
    )

if len(segments_complets) == 0:
    segments_complets.append((x_beta, y_beta))

x_recon_parts = [segments_complets[0][0]]
y_recon_parts = [segments_complets[0][1]]

for x_seg, y_seg in segments_complets[1:]:
    x_recon_parts.append(x_seg[1:])
    y_recon_parts.append(y_seg[1:])

x_recon = np.concatenate(x_recon_parts)
y_recon = np.concatenate(y_recon_parts)

erreur_reconstruction_x = np.max(np.abs(x_recon - x_beta))
erreur_reconstruction_y = np.max(np.abs(y_recon - y_beta))


# ============================================================
# 10. Contributions ouvertes
# ============================================================

C_beta = shoelace_open_contribution(x_beta, y_beta)

C_alpha = []
for x_i, y_i, theta_i, debut, fin in alphas:
    C_alpha.append(shoelace_open_contribution(x_i, y_i))

C_alpha_sum = np.sum(C_alpha) if len(C_alpha) > 0 else 0.0

C_residus = []
for nom, x_r, y_r in residus:
    C_residus.append(shoelace_open_contribution(x_r, y_r))

C_residus_sum = np.sum(C_residus) if len(C_residus) > 0 else 0.0

C_partition_sum = C_alpha_sum + C_residus_sum

difference_ouverte_partition = C_beta - C_partition_sum


# ============================================================
# 11. Aires fermées
# ============================================================

A_beta = shoelace_closed_area(x_beta, y_beta)

A_alpha = []
for x_i, y_i, theta_i, debut, fin in alphas:
    A_alpha.append(shoelace_closed_area(x_i, y_i))

A_alpha_sum = np.sum(A_alpha) if len(A_alpha) > 0 else 0.0

A_residus = []
for nom, x_r, y_r in residus:
    A_residus.append(shoelace_closed_area(x_r, y_r))

A_residus_sum = np.sum(A_residus) if len(A_residus) > 0 else 0.0

A_partition_closed_sum = A_alpha_sum + A_residus_sum

difference_fermee_alpha_only = A_beta - A_alpha_sum
difference_fermee_partition = A_beta - A_partition_closed_sum


# ============================================================
# 12. Contributions de fermeture
# ============================================================

K_beta = closure_contribution(
    x_start=x_beta[0],
    y_start=y_beta[0],
    x_end=x_beta[-1],
    y_end=y_beta[-1]
)

K_alpha = []
for x_i, y_i, theta_i, debut, fin in alphas:
    K_alpha.append(
        closure_contribution(
            x_start=x_i[0],
            y_start=y_i[0],
            x_end=x_i[-1],
            y_end=y_i[-1]
        )
    )

K_alpha_sum = np.sum(K_alpha) if len(K_alpha) > 0 else 0.0

K_residus = []
for nom, x_r, y_r in residus:
    K_residus.append(
        closure_contribution(
            x_start=x_r[0],
            y_start=y_r[0],
            x_end=x_r[-1],
            y_end=y_r[-1]
        )
    )

K_residus_sum = np.sum(K_residus) if len(K_residus) > 0 else 0.0

K_partition_sum = K_alpha_sum + K_residus_sum

difference_fermetures_partition = K_beta - K_partition_sum


# ============================================================
# 13. Figure 1 : bêta arbitraire
# ============================================================

plt.figure(figsize=(6, 6))
plt.plot(x_beta, y_beta, label="bêta ouverte")

plt.scatter(x_beta[0], y_beta[0], marker="x", s=80, label="début bêta")
plt.scatter(x_beta[-1], y_beta[-1], marker="x", s=80, label="fin bêta")

for idx in indices_coupures:
    plt.scatter(x_beta[idx], y_beta[idx], s=35)

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "1. Fenêtre bêta arbitraire\n"
    f"début = {fase_inicio_beta_grados}°, fin = {fase_final_beta_grados}°"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 14. Figure 2 : bêta = résidus + alpha
# ============================================================

plt.figure(figsize=(7, 7))

for nom, x_r, y_r in residus:
    plt.plot(x_r, y_r, linewidth=3, label=nom)

for i, (x_i, y_i, theta_i, debut, fin) in enumerate(alphas, start=1):
    ligne, = plt.plot(x_i, y_i, label=f"alpha {i}")
    couleur = ligne.get_color()
    plt.scatter(x_i[0], y_i[0], s=25, color=couleur)
    plt.scatter(x_i[-1], y_i[-1], s=25, color=couleur)

plt.scatter(x_beta[0], y_beta[0], marker="x", s=80, label="début bêta")
plt.scatter(x_beta[-1], y_beta[-1], marker="x", s=80, label="fin bêta")

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    "2. Bêta découpée : résidus + cycles alpha complets\n"
    f"coupure alpha = {fase_corte_alpha_grados}°"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 15. Figure 3 : fermeture globale bêta
# ============================================================

plt.figure(figsize=(6, 6))

plt.plot(x_beta, y_beta, label="bêta")
plt.fill(x_beta, y_beta, alpha=0.20, label="aire bêta fermée")

plt.plot(
    [x_beta[-1], x_beta[0]],
    [y_beta[-1], y_beta[0]],
    "--",
    linewidth=2,
    label="fermeture globale bêta"
)

plt.scatter(x_beta[0], y_beta[0], marker="x", s=80, label="début bêta")
plt.scatter(x_beta[-1], y_beta[-1], marker="x", s=80, label="fin bêta")

plt.xlabel("x")
plt.ylabel("v")
plt.title(f"3. Aire bêta fermée globalement\nAβ = {A_beta:.6f}")
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 16. Figure 4 : fermetures alpha locales
# ============================================================

plt.figure(figsize=(7, 7))

for i, (x_i, y_i, theta_i, debut, fin) in enumerate(alphas, start=1):
    ligne, = plt.plot(x_i, y_i, label=f"alpha {i}")
    couleur = ligne.get_color()

    plt.plot(
        [x_i[-1], x_i[0]],
        [y_i[-1], y_i[0]],
        "--",
        linewidth=1.8,
        color=couleur
    )

    plt.fill(x_i, y_i, alpha=0.12, color=couleur)

    plt.scatter(x_i[0], y_i[0], s=25, color=couleur)
    plt.scatter(x_i[-1], y_i[-1], s=25, color=couleur)

plt.scatter(x_beta[0], y_beta[0], marker="x", s=80, label="début bêta")
plt.scatter(x_beta[-1], y_beta[-1], marker="x", s=80, label="fin bêta")

plt.xlabel("x")
plt.ylabel("v")
plt.title(
    f"4. Aires alpha fermées localement\n"
    f"coupure alpha = {fase_corte_alpha_grados}°, ΣAα = {A_alpha_sum:.6f}"
)
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 17. Figure 5 : fermetures de la partition complète
# ============================================================

plt.figure(figsize=(7, 7))

for nom, x_r, y_r in residus:
    ligne, = plt.plot(x_r, y_r, linewidth=3, label=nom)
    couleur = ligne.get_color()

    plt.plot(
        [x_r[-1], x_r[0]],
        [y_r[-1], y_r[0]],
        "--",
        linewidth=1.8,
        color=couleur
    )

for i, (x_i, y_i, theta_i, debut, fin) in enumerate(alphas, start=1):
    ligne, = plt.plot(x_i, y_i, label=f"alpha {i}")
    couleur = ligne.get_color()

    plt.plot(
        [x_i[-1], x_i[0]],
        [y_i[-1], y_i[0]],
        "--",
        linewidth=1.4,
        color=couleur
    )

plt.xlabel("x")
plt.ylabel("v")
plt.title("5. Fermetures locales de chaque morceau de la partition")
plt.axis("equal")
plt.legend()
plt.tight_layout()


# ============================================================
# 18. Figure 6 : résumé numérique en barres
# ============================================================

def ajouter_valeurs_barres(ax, bars, fmt="{:.3f}"):
    """
    Ajoute les valeurs numériques au-dessus ou au-dessous des barres.
    """
    for bar in bars:
        hauteur = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2

        if hauteur >= 0:
            va = "bottom"
            y_pos = hauteur
        else:
            va = "top"
            y_pos = hauteur

        ax.text(
            x_pos,
            y_pos,
            fmt.format(hauteur),
            ha="center",
            va=va,
            fontsize=8,
            rotation=0
        )


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


# ------------------------------------------------------------
# 18.1 Contributions ouvertes
# ------------------------------------------------------------

labels_ouvertes = [
    "Cβ\nouverte",
    "ΣCα\nouvertes",
    "ΣC résidus\nouvertes",
    "ΣC partition\ncomplète",
    "Cβ -\npartition"
]

valeurs_ouvertes = [
    C_beta,
    C_alpha_sum,
    C_residus_sum,
    C_partition_sum,
    difference_ouverte_partition
]

bars = axes[0].bar(labels_ouvertes, valeurs_ouvertes)

axes[0].set_title("Contributions ouvertes")
axes[0].set_ylabel("valeur orientée")
axes[0].axhline(0, linewidth=0.8)
ajouter_valeurs_barres(axes[0], bars)

axes[0].text(
    0.5,
    -0.28,
    "Ici, bêta ouverte = alpha + résidus.\n"
    "La différence doit être ≈ 0.",
    transform=axes[0].transAxes,
    ha="center",
    va="top",
    fontsize=9
)


# ------------------------------------------------------------
# 18.2 Aires fermées
# ------------------------------------------------------------

labels_fermees = [
    "Aβ\nfermée",
    "ΣAα\nfermées",
    "ΣA résidus\nfermés",
    "ΣA partition\nlocale",
    "Aβ -\nΣAα",
    "Aβ -\npartition"
]

valeurs_fermees = [
    A_beta,
    A_alpha_sum,
    A_residus_sum,
    A_partition_closed_sum,
    difference_fermee_alpha_only,
    difference_fermee_partition
]

bars = axes[1].bar(labels_fermees, valeurs_fermees)

axes[1].set_title("Aires fermées")
axes[1].axhline(0, linewidth=0.8)
ajouter_valeurs_barres(axes[1], bars)

axes[1].text(
    0.5,
    -0.28,
    "Ici, chaque morceau est fermé localement.\n"
    "Donc la somme peut différer de Aβ.",
    transform=axes[1].transAxes,
    ha="center",
    va="top",
    fontsize=9
)


# ------------------------------------------------------------
# 18.3 Contributions des fermetures
# ------------------------------------------------------------

labels_fermetures = [
    "Kβ\nfermeture",
    "ΣKα\nfermetures",
    "ΣK résidus\nfermetures",
    "ΣK partition\nfermetures",
    "Kβ -\npartition"
]

valeurs_fermetures = [
    K_beta,
    K_alpha_sum,
    K_residus_sum,
    K_partition_sum,
    difference_fermetures_partition
]

bars = axes[2].bar(labels_fermetures, valeurs_fermetures)

axes[2].set_title("Contributions des fermetures")
axes[2].axhline(0, linewidth=0.8)
ajouter_valeurs_barres(axes[2], bars)

axes[2].text(
    0.5,
    -0.28,
    "La différence des aires fermées vient\n"
    "exactement de la différence des fermetures.",
    transform=axes[2].transAxes,
    ha="center",
    va="top",
    fontsize=9
)


# ------------------------------------------------------------
# Titre global
# ------------------------------------------------------------

fig.suptitle(
    "Résumé : contributions ouvertes, aires fermées et fermetures",
    fontsize=14
)

plt.tight_layout()
plt.show()


# ============================================================
# 19. Résultats numériques
# ============================================================

print("==================================================")
print("RÉSULTATS")
print("==================================================")
print(f"Scénario : {scenario}")
print(f"Cycles alpha complets demandés : {N_cycles_alpha:.3e}")
print(f"Points par cycle : {points_par_cycle:.3e}")
print(f"Début bêta en degrés : {fase_inicio_beta_grados:.3e}")
print(f"Fin bêta en degrés : {fase_final_beta_grados:.3e}")
print(f"Phase de coupure alpha en degrés : {fase_corte_alpha_grados:.3e}")
print(f"Nombre de coupures trouvées : {len(indices_coupures):.3e}")
print(f"Nombre d'alphas complets : {len(alphas):.3e}")
print(f"Nombre de résidus : {len(residus):.3e}")
print("--------------------------------------------------")
print("RECONSTRUCTION DE LA TRAJECTOIRE")
print("--------------------------------------------------")
print(f"Erreur max reconstruction x : {erreur_reconstruction_x:.3e}")
print(f"Erreur max reconstruction y : {erreur_reconstruction_y:.3e}")
print("Interprétation : résidus + alpha reconstruisent bêta comme trajectoire.")
print("--------------------------------------------------")
print("CONTRIBUTIONS OUVERTES")
print("--------------------------------------------------")
print(f"C_beta ouverte :                  {C_beta:.6e}")
print(f"Somme C_alpha ouvertes :          {C_alpha_sum:.6e}")
print(f"Somme C_résidus ouvertes :        {C_residus_sum:.6e}")
print(f"Somme C_partition complète :      {C_partition_sum:.6e}")
print(f"C_beta - C_partition :            {difference_ouverte_partition:.6e}")
print("--------------------------------------------------")
print("AIRES FERMÉES")
print("--------------------------------------------------")
print(f"A_beta fermée globale :           {A_beta:.6e}")
print(f"Somme A_alpha fermées :           {A_alpha_sum:.6e}")
print(f"Somme A_résidus fermés :          {A_residus_sum:.6e}")
print(f"Somme A_partition fermée locale : {A_partition_closed_sum:.6e}")
print(f"A_beta - Somme A_alpha :          {difference_fermee_alpha_only:.6e}")
print(f"A_beta - Somme A_partition :      {difference_fermee_partition:.6e}")
print("--------------------------------------------------")
print("CONTRIBUTIONS DES FERMETURES")
print("--------------------------------------------------")
print(f"K_beta fermeture globale :        {K_beta:.6e}")
print(f"Somme K_alpha fermetures :        {K_alpha_sum:.6e}")
print(f"Somme K_résidus fermetures :      {K_residus_sum:.6e}")
print(f"Somme K_partition fermetures :    {K_partition_sum:.6e}")
print(f"K_beta - K_partition :            {difference_fermetures_partition:.6e}")
print("--------------------------------------------------")
print("VÉRIFICATION CLÉ")
print("--------------------------------------------------")
print(
    "Différence fermée partition - différence fermetures : "
    f"{difference_fermee_partition - difference_fermetures_partition:.6e}"
)
print("==================================================")
print("INTERPRÉTATION")
print("==================================================")
print(
    "1) Les cycles alpha seuls ne couvrent pas forcément toute bêta, "
    "car bêta commence et finit à une phase arbitraire.\n"
    "2) En ajoutant les résidus initial/final, la partition reconstruira bêta.\n"
    "3) Les contributions ouvertes se somment correctement.\n"
    "4) Les aires fermées diffèrent parce que bêta a une fermeture globale, "
    "tandis que chaque alpha/résidu a sa fermeture locale.\n"
    "5) La phase de coupure alpha contrôle la position du segment de fermeture "
    "local de chaque cycle. Le cas 0° correspond au cas particulier v = 0."
)
print("==================================================")