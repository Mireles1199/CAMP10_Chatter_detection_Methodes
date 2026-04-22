from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch


# ============================================================
# Style général des figures
# ============================================================

def configure_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.5,
    })


configure_plot_style()


# ============================================================
# Noyau SSQ-STFT
# ============================================================

def gaussian_window(win_len: int = 2000,
                    n_fft: int = 8192,
                    sigma: float = 333.0) -> np.ndarray:
    """
    Construit une fenêtre gaussienne de longueur win_len et l'intègre
    dans un vecteur de longueur n_fft, centrée au milieu du tableau.
    """
    g_small = np.zeros(win_len, dtype=float)
    n = np.arange(win_len)
    n0 = (win_len - 1) / 2.0
    g_small = np.exp(-0.5 * ((n - n0) / sigma) ** 2)

    g = np.zeros(n_fft, dtype=float)
    start = (n_fft - win_len) // 2
    g[start:start + win_len] = g_small
    return g


def spectral_derivative_window(window: np.ndarray, fs: float) -> np.ndarray:
    """
    Dérivée exacte par différenciation spectrale.
    Retourne g'(t) en unités par seconde.
    """
    N = len(window)
    G = np.fft.fft(window)

    # axe angulaire en rad/échantillon, ordre DFT "wrap"
    xi = np.concatenate([
        np.arange(0, N // 2 + 1),
        np.arange(-N // 2 + 1, 0)
    ]) * (2.0 * np.pi / N)

    # annuler Nyquist pour préserver la sortie réelle
    xi[N // 2] = 0.0

    gprime = np.fft.ifft(1j * xi * G).real
    gprime *= fs  # dérivée par échantillon -> dérivée par seconde
    return gprime


def frame_signal_centered(x: np.ndarray,
                          n_fft: int = 8192,
                          hop_len: int = 750) -> np.ndarray:
    """
    Extrait des trames complètes et applique ifftshift à l'intérieur de chaque trame.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < n_fft:
        raise ValueError("Le signal doit avoir au moins n_fft échantillons.")

    n_frames = 1 + (len(x) - n_fft) // hop_len
    frames = np.zeros((n_fft, n_frames), dtype=float)

    for j in range(n_frames):
        start = j * hop_len
        seg = x[start:start + n_fft]
        frames[:, j] = np.fft.ifftshift(seg)

    return frames


def stft_with_derivative(x: np.ndarray,
                         fs: float = 50_000,
                         win_len: int = 2000,
                         sigma: float = 333.0,
                         n_fft: int = 8192,
                         hop_len: int = 750):
    """
    Calcule Sx et dSx avec une fenêtre gaussienne et sa dérivée.
    """
    g_centered = gaussian_window(win_len=win_len, n_fft=n_fft, sigma=sigma)
    gp_centered = spectral_derivative_window(g_centered, fs=fs)

    # Alignement avec les trames déjà décalées par ifftshift
    g = np.fft.ifftshift(g_centered)
    gp = np.fft.ifftshift(gp_centered)

    frames = frame_signal_centered(x, n_fft=n_fft, hop_len=hop_len)

    Sx = np.fft.rfft(frames * g[:, None], axis=0)
    dSx = np.fft.rfft(frames * gp[:, None], axis=0)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    times = (np.arange(frames.shape[1]) * hop_len + n_fft / 2.0) / fs

    return Sx, dSx, freqs, times, g_centered, gp_centered


def instantaneous_frequency(Sx: np.ndarray,
                            dSx: np.ndarray,
                            freqs: np.ndarray,
                            gamma: float = 1e-6) -> np.ndarray:
    """
    Estime la fréquence instantanée :
        w_hat = |xi - Im(dSx / Sx)/(2*pi)|
    """
    mag = np.abs(Sx)
    power = Sx.real**2 + Sx.imag**2
    mask = mag >= gamma

    # Im(dSx / Sx) sans division complexe explicite
    num = dSx.imag * Sx.real - dSx.real * Sx.imag

    ratio_im = np.zeros_like(power, dtype=float)
    ratio_im[mask] = num[mask] / power[mask]

    w_hat = np.full_like(power, np.inf, dtype=float)
    omega = np.abs(freqs[:, None] - ratio_im / (2.0 * np.pi))
    w_hat[mask] = omega[mask]
    return w_hat


def synchrosqueeze(Sx: np.ndarray,
                   w_hat: np.ndarray,
                   freqs: np.ndarray) -> np.ndarray:
    """
    Réassignation linéaire SSQ-STFT.
    """
    df = freqs[1] - freqs[0]
    Tx = np.zeros_like(Sx)

    for j in range(Sx.shape[1]):
        valid = np.isfinite(w_hat[:, j])
        if not np.any(valid):
            continue

        k = np.rint(w_hat[valid, j] / df).astype(int)
        k = np.clip(k, 0, len(freqs) - 1)
        np.add.at(Tx[:, j], k, Sx[valid, j] * df)

    return Tx


def ssq_stft(x: np.ndarray,
             fs: float = 50_000,
             win_len: int = 2000,
             sigma: float = 333.0,
             n_fft: int = 8192,
             hop_len: int = 750,
             gamma: float = 1e-6):
    """
    Pipeline complet SSQ-STFT.
    """
    Sx, dSx, freqs, times, g, gp = stft_with_derivative(
        x=x,
        fs=fs,
        win_len=win_len,
        sigma=sigma,
        n_fft=n_fft,
        hop_len=hop_len,
    )
    w_hat = instantaneous_frequency(Sx, dSx, freqs, gamma=gamma)
    Tx = synchrosqueeze(Sx, w_hat, freqs)
    return Tx, Sx, w_hat, freqs, times, g, gp


# ============================================================
# Utilitaires de visualisation
# ============================================================

def _save(fig, path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, pad_inches=0.03)


def _to_db(X, floor_db=-80.0):
    mag = np.abs(X)
    ref = np.max(mag)
    if ref <= 0:
        return np.full_like(mag, floor_db, dtype=float)
    mag = np.maximum(mag / ref, 10 ** (floor_db / 20.0))
    return 20.0 * np.log10(mag)


def _extent(times, freqs):
    return [times[0], times[-1], freqs[0], freqs[-1]]


def _add_vertical_colorbar(fig, im, label, left=0.90, bottom=0.14, width=0.018, height=0.70):
    cax = fig.add_axes([left, bottom, width, height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label)
    return cbar


# ============================================================
# Figure 1 — Signal temporel
# ============================================================

def plot_input_signal(x, fs, savepath=None):
    t = np.arange(len(x)) / fs

    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88)

    ax.plot(t, x)
    ax.set_title("Figure 1 — Signal d'entrée dans le domaine temporel", pad=10)
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Amplitude")
    ax.margins(x=0)

    _save(fig, savepath)
    return fig


# ============================================================
# Figure 2 — Fenêtre et dérivée
# ============================================================

def plot_windows(g, gp, fs, savepath=None):
    n = np.arange(len(g))
    t_ms = 1e3 * (n - len(g) // 2) / fs

    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 5.2), sharex=True,
        gridspec_kw={"hspace": 0.18}
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)

    axes[0].plot(t_ms, g)
    axes[0].axvline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel(r"$g[n]$")
    axes[0].set_title("Figure 2 — Fenêtre gaussienne et fenêtre dérivée", pad=8)

    axes[1].plot(t_ms, gp)
    axes[1].axvline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Temps local [ms]")
    axes[1].set_ylabel(r"$g'[n]$")

    _save(fig, savepath)
    return fig


# ============================================================
# Figure 3 — Diagramme trame + ifftshift
# ============================================================

def plot_frame_ifftshift_diagram(n_fft=8192, savepath=None):
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.94)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x0, w = 0.08, 0.84
    y_top, y_bot = 0.66, 0.24
    h = 0.12

    c1 = "#d9dde3"
    c2 = "#9ea7b3"
    edge = "#2f3640"

    # Barres
    ax.add_patch(Rectangle((x0, y_top), w/2, h, facecolor=c1, edgecolor=edge, linewidth=1.2))
    ax.add_patch(Rectangle((x0 + w/2, y_top), w/2, h, facecolor=c2, edgecolor=edge, linewidth=1.2))

    ax.add_patch(Rectangle((x0, y_bot), w/2, h, facecolor=c2, edgecolor=edge, linewidth=1.2))
    ax.add_patch(Rectangle((x0 + w/2, y_bot), w/2, h, facecolor=c1, edgecolor=edge, linewidth=1.2))

    # Lignes de référence
    ax.plot([x0 + w/2, x0 + w/2], [y_top - 0.03, y_top + h + 0.03],
            linestyle="--", linewidth=1.0, color=edge)
    ax.plot([x0, x0], [y_bot - 0.03, y_bot + h + 0.03],
            linestyle="--", linewidth=1.0, color=edge)

    # Titre
    ax.text(0.5, 0.94, "Figure 3 — Réordonnancement de la trame par ifftshift",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Étiquettes
    ax.text(x0, y_top + h + 0.09, "Trame originale",
            ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax.text(x0, y_bot + h + 0.09, "Trame après ifftshift",
            ha="left", va="bottom", fontsize=11, fontweight="bold")

    ax.text(x0 + w/4, y_top + h/2, "Première moitié de la trame",
            ha="center", va="center", fontsize=10)
    ax.text(x0 + 3*w/4, y_top + h/2, "Deuxième moitié de la trame",
            ha="center", va="center", fontsize=10)

    ax.text(x0 + w/4, y_bot + h/2, "Deuxième moitié",
            ha="center", va="center", fontsize=10)
    ax.text(x0 + 3*w/4, y_bot + h/2, "Première moitié",
            ha="center", va="center", fontsize=10)

    # Texte central
    ax.text(x0 + w/2, 0.54, "Centre de la trame à l'indice N/2",
            ha="center", va="center", fontsize=10)

    ax.text(x0, y_bot - 0.10, "Centre de la trame décalé à l'indice 0",
            ha="left", va="top", fontsize=10)

    # Flèches plus nettes
    ax.add_patch(FancyArrowPatch(
        (x0 + 0.23*w, y_top - 0.02),
        (x0 + 0.73*w, y_bot + h + 0.02),
        arrowstyle="->", mutation_scale=16, linewidth=1.3, color=edge
    ))
    ax.add_patch(FancyArrowPatch(
        (x0 + 0.73*w, y_top - 0.02),
        (x0 + 0.23*w, y_bot + h + 0.02),
        arrowstyle="->", mutation_scale=16, linewidth=1.3, color=edge
    ))

    _save(fig, savepath)
    return fig


# ============================================================
# Figure 4 — STFT
# ============================================================

def plot_stft(Sx, times, freqs, fmax=None, floor_db=-80.0, savepath=None):
    db = _to_db(Sx, floor_db=floor_db)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.13, top=0.88)

    im = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=_extent(times, freqs),
        vmin=floor_db,
        vmax=0.0,
        interpolation="none",
        cmap="magma"
    )

    if fmax is not None:
        ax.set_ylim(0, fmax)

    ax.set_title(r"Figure 4 — Magnitude de la STFT, $|S_x(\xi,t)|$", pad=10)
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Fréquence [Hz]")

    _add_vertical_colorbar(fig, im, "Magnitude relative [dB]", left=0.90, bottom=0.14, height=0.70)

    _save(fig, savepath)
    return fig


# ============================================================
# Figure 5 — Fréquence instantanée estimée
# ============================================================

def plot_instantaneous_frequency_map(w_hat, Sx, times, freqs,
                                     gamma=1e-6, fmax=None,
                                     floor_db=-60.0, savepath=None):
    valid = np.isfinite(w_hat) & (np.abs(Sx) >= gamma)
    W = np.ma.array(w_hat, mask=~valid)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.13, top=0.88)

    vmax = fmax if fmax is not None else np.nanpercentile(w_hat[np.isfinite(w_hat)], 99)

    im = ax.imshow(
        W,
        origin="lower",
        aspect="auto",
        extent=_extent(times, freqs),
        interpolation="none",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax
    )

    # Contours doux de |Sx| pour le contexte
    stft_db = _to_db(Sx, floor_db=floor_db)
    TT, FF = np.meshgrid(times, freqs)
    ax.contour(
        TT, FF, stft_db,
        levels=np.linspace(-30, -8, 5),
        linewidths=0.5,
        colors="white",
        alpha=0.35
    )

    if fmax is not None:
        ax.set_ylim(0, fmax)

    ax.set_title(r"Figure 5 — Carte de fréquence instantanée estimée, $\hat{\omega}(\xi,t)$", pad=10)
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Fréquence [Hz]")

    _add_vertical_colorbar(
        fig, im,
        "Fréquence instantanée estimée [Hz]",
        left=0.90, bottom=0.14, height=0.70
    )

    _save(fig, savepath)
    return fig


# ============================================================
# Figure 6 — Schéma conceptuel SSQ
# ============================================================

def plot_ssq_concept_diagram(savepath=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.82, wspace=0.22)

    bins = np.arange(15)
    center = 8

    spread = np.exp(-0.5 * ((bins - center) / 1.6) ** 2)
    spread /= spread.max()

    squeezed = np.zeros_like(spread)
    squeezed[center] = spread.sum() / 2.7

    ax1.vlines(bins, 0, spread, linewidth=2.0)
    ax1.plot(bins, spread, "o", markersize=4)

    ax2.vlines(bins, 0, squeezed, linewidth=2.0)
    ax2.plot(bins, squeezed, "o", markersize=4)

    for ax, title in zip(
        (ax1, ax2),
        ("Coefficients STFT voisins", "Coefficient SSQ réassigné")
    ):
        ax.set_xlim(-0.7, 14.7)
        ax.set_ylim(0, 1.18)
        ax.set_xlabel("Bin de fréquence")
        ax.set_ylabel("Magnitude")
        ax.set_title(title, pad=8)
        ax.grid(True, alpha=0.15)

    for b in [5, 6, 7, 8, 9, 10, 11]:
        con = ConnectionPatch(
            xyA=(b, min(spread[b], 0.95)),
            coordsA=ax1.transData,
            xyB=(center, min(squeezed[center], 0.95)),
            coordsB=ax2.transData,
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=12,
            linewidth=1.0,
            alpha=0.80
        )
        fig.add_artist(con)

    ax2.text(center + 0.35, 1.05, r"$\hat{\omega}\approx f_0$", fontsize=10)

    fig.suptitle("Figure 6 — Principe de réassignation dans le synchrosqueezing",
                 y=0.96, fontsize=13)

    _save(fig, savepath)
    return fig


# ============================================================
# Figure 7 — Comparaison STFT vs SSQ
# ============================================================

def plot_stft_vs_ssq(Sx, Tx, times, freqs,
                     fmax=None, floor_db=-80.0, savepath=None):
    db_stft = _to_db(Sx, floor_db=floor_db)
    db_ssq = _to_db(Tx, floor_db=floor_db)

    fig = plt.figure(figsize=(13.2, 5.4))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1, 1, 0.035],
        left=0.07, right=0.96, bottom=0.12, top=0.84, wspace=0.10
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    cax = fig.add_subplot(gs[0, 2])

    im1 = ax1.imshow(
        db_stft,
        origin="lower",
        aspect="auto",
        extent=_extent(times, freqs),
        vmin=floor_db,
        vmax=0.0,
        interpolation="none",
        cmap="magma"
    )

    im2 = ax2.imshow(
        db_ssq,
        origin="lower",
        aspect="auto",
        extent=_extent(times, freqs),
        vmin=floor_db,
        vmax=0.0,
        interpolation="none",
        cmap="magma"
    )

    ax1.set_title(r"(a) STFT, $|S_x|$", pad=8)
    ax2.set_title(r"(b) SSQ-STFT, $|T_x|$", pad=8)

    ax1.set_xlabel("Temps [s]")
    ax2.set_xlabel("Temps [s]")
    ax1.set_ylabel("Fréquence [Hz]")

    if fmax is not None:
        ax1.set_ylim(0, fmax)
        ax2.set_ylim(0, fmax)

    plt.setp(ax2.get_yticklabels(), visible=False)

    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Magnitude relative [dB]")

    fig.suptitle(
        "Figure 7 — Comparaison entre STFT et SSQ-STFT",
        y=0.94,
        fontsize=13
    )

    _save(fig, savepath)
    return fig

# ============================================================
# Figure 8 — Validation synthétique
# ============================================================

def synthetic_amfm_signal(fs=50_000, duration=2.0):
    t = np.arange(int(fs * duration)) / fs
    f_inst = 4800.0 + 650.0 * np.sin(2.0 * np.pi * 0.55 * t)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    amp = 1.0 + 0.18 * np.cos(2.0 * np.pi * 1.3 * t)
    x = amp * np.cos(phase)
    return x, t, f_inst


def plot_synthetic_validation(fs=50_000,
                              win_len=2000,
                              sigma=333.0,
                              n_fft=8192,
                              hop_len=750,
                              gamma=1e-6,
                              fmax=8000.0,
                              floor_db=-80.0,
                              savepath=None):
    x_syn, t_syn, f_true = synthetic_amfm_signal(fs=fs, duration=2.0)

    Tx, Sx, _, freqs, times, _, _ = ssq_stft(
        x_syn,
        fs=fs,
        win_len=win_len,
        sigma=sigma,
        n_fft=n_fft,
        hop_len=hop_len,
        gamma=gamma
    )

    db_stft = _to_db(Sx, floor_db=floor_db)
    db_ssq = _to_db(Tx, floor_db=floor_db)
    f_true_frames = np.interp(times, t_syn, f_true)

    fig = plt.figure(figsize=(13.2, 5.4))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1, 1, 0.035],
        left=0.07, right=0.96, bottom=0.12, top=0.84, wspace=0.10
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    cax = fig.add_subplot(gs[0, 2])

    ax1.imshow(
        db_stft,
        origin="lower",
        aspect="auto",
        extent=_extent(times, freqs),
        vmin=floor_db,
        vmax=0.0,
        interpolation="none",
        cmap="magma"
    )
    ax1.plot(times, f_true_frames, linestyle="--", linewidth=1.6, color="cyan")
    ax1.set_title("STFT sur signal AM-FM synthétique", pad=8)
    ax1.set_xlabel("Temps [s]")
    ax1.set_ylabel("Fréquence [Hz]")
    ax1.set_ylim(0, fmax)

    im = ax2.imshow(
        db_ssq,
        origin="lower",
        aspect="auto",
        extent=_extent(times, freqs),
        vmin=floor_db,
        vmax=0.0,
        interpolation="none",
        cmap="magma"
    )
    ax2.plot(times, f_true_frames, linestyle="--", linewidth=1.6, color="cyan")
    ax2.set_title("SSQ-STFT sur signal AM-FM synthétique", pad=8)
    ax2.set_xlabel("Temps [s]")
    ax2.set_ylim(0, fmax)

    plt.setp(ax2.get_yticklabels(), visible=False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Magnitude relative [dB]")

    fig.suptitle(
        "Figure 8 — Validation visuelle sur une composante AM-FM synthétique",
        y=0.94,
        fontsize=13
    )

    _save(fig, savepath)
    return fig

# ============================================================
# Générateur complet des Figures 1–8
# ============================================================

def generate_all_figures(x: np.ndarray,
                         fs: float = 50_000,
                         win_len: int = 2000,
                         sigma: float = 333.0,
                         n_fft: int = 8192,
                         hop_len: int = 750,
                         gamma: float = 1e-6,
                         fmax_display: float = 12_000.0,
                         out_dir: Optional[str] = "figuras_ssq") -> Dict[str, Any]:
    """
    Génère et sauvegarde les Figures 1–8.
    """
    out = Path(out_dir) if out_dir is not None else None

    Tx, Sx, w_hat, freqs, times, g, gp = ssq_stft(
        x=x,
        fs=fs,
        win_len=win_len,
        sigma=sigma,
        n_fft=n_fft,
        hop_len=hop_len,
        gamma=gamma
    )

    figs: Dict[str, plt.Figure] = {}

    figs["fig1_signal"] = plot_input_signal(
        x, fs,
        savepath=None if out is None else out / "fig01_signal_temporal.pdf"
    )

    figs["fig2_windows"] = plot_windows(
        g, gp, fs,
        savepath=None if out is None else out / "fig02_ventana_y_derivada.pdf"
    )

    figs["fig3_ifftshift"] = plot_frame_ifftshift_diagram(
        n_fft=n_fft,
        savepath=None if out is None else out / "fig03_diagrama_ifftshift.pdf"
    )

    figs["fig4_stft"] = plot_stft(
        Sx, times, freqs,
        fmax=fmax_display,
        savepath=None if out is None else out / "fig04_stft.pdf"
    )

    figs["fig5_if_map"] = plot_instantaneous_frequency_map(
        w_hat, Sx, times, freqs,
        gamma=gamma,
        fmax=fmax_display,
        savepath=None if out is None else out / "fig05_frecuencia_instantanea.pdf"
    )

    figs["fig6_concept"] = plot_ssq_concept_diagram(
        savepath=None if out is None else out / "fig06_concepto_ssq.pdf"
    )

    figs["fig7_compare"] = plot_stft_vs_ssq(
        Sx, Tx, times, freqs,
        fmax=fmax_display,
        savepath=None if out is None else out / "fig07_stft_vs_ssq.pdf"
    )

    figs["fig8_synthetic"] = plot_synthetic_validation(
        fs=fs,
        win_len=win_len,
        sigma=sigma,
        n_fft=n_fft,
        hop_len=hop_len,
        gamma=gamma,
        fmax=min(fmax_display, 8000.0),
        savepath=None if out is None else out / "fig08_validacion_sintetica.pdf"
    )

    return {
        "Tx": Tx,
        "Sx": Sx,
        "w_hat": w_hat,
        "freqs": freqs,
        "times": times,
        "g": g,
        "gprime": gp,
        "figures": figs,
    }


# ============================================================
# Exemple d'utilisation
# ============================================================

if __name__ == "__main__":
    fs = 50_000

    # Remplace ceci par ton signal réel de fraisage :
    # x = ton_signal_reel.astype(float)

    # Exemple temporel de test :
    N = 220_000
    t = np.arange(N) / fs
    f0 = 1400 + 180*np.cos(2*np.pi*0.40*t) + 90*np.sin(2*np.pi*0.85*t)

    phi = 2*np.pi*np.cumsum(f0) / fs

    x_armonicos  = (
        1.00*(1 + 0.20*np.cos(2*np.pi*5*t)) * np.cos(1*phi) +
        0.75*(1 + 0.12*np.cos(2*np.pi*4*t + 0.5)) * np.cos(2*phi + 0.3) +
        0.55*(1 + 0.10*np.cos(2*np.pi*6*t + 1.1)) * np.cos(3*phi + 0.8) +
        0.38*(1 + 0.08*np.cos(2*np.pi*3*t + 0.2)) * np.cos(4*phi + 1.4) +
        0.25*np.cos(5*phi + 2.0)
    )

    fs = 50_000
    N = 260_000
    t = np.arange(N) / fs

    def smooth_gate(t, centers, width, weights=None):
        y = np.zeros_like(t)
        if weights is None:
            weights = np.ones(len(centers))
        for c, w in zip(centers, weights):
            y += w * np.exp(-0.5 * ((t - c) / width) ** 2)
        y /= np.max(y) + 1e-12
        return y

    # Branches principales : plusieurs courbes AM-FM qui se croisent
    f1 = 1400 + 220*np.sin(2*np.pi*0.35*t + 0.1) + 120*np.cos(2*np.pi*0.90*t)
    f2 = 2200 + 320*np.sin(2*np.pi*0.55*t + 1.0) - 140*np.cos(2*np.pi*0.60*t + 0.4)
    f3 = 3200 + 420*np.sin(2*np.pi*0.42*t + 2.1) + 180*np.cos(2*np.pi*0.75*t + 1.4)
    f4 = 4600 + 520*np.sin(2*np.pi*0.30*t + 0.7) - 200*np.cos(2*np.pi*1.05*t + 2.2)
    f5 = 6200 + 650*np.sin(2*np.pi*0.48*t + 1.8) + 220*np.cos(2*np.pi*0.52*t + 0.6)
    f6 = 8200 + 750*np.sin(2*np.pi*0.37*t + 2.7) - 260*np.cos(2*np.pi*0.82*t + 1.1)

    # Deux chirps qui traversent tout et aident à former une "étoile"
    f7 = 1800 + 1800*t
    f8 = 9500 - 1600*t

    # Phases par intégration
    phi1 = 2*np.pi*np.cumsum(f1) / fs
    phi2 = 2*np.pi*np.cumsum(f2) / fs
    phi3 = 2*np.pi*np.cumsum(f3) / fs
    phi4 = 2*np.pi*np.cumsum(f4) / fs
    phi5 = 2*np.pi*np.cumsum(f5) / fs
    phi6 = 2*np.pi*np.cumsum(f6) / fs
    phi7 = 2*np.pi*np.cumsum(f7) / fs
    phi8 = 2*np.pi*np.cumsum(f8) / fs

    # Enveloppes pour faire apparaître des "cellules" dans la STFT
    g1 = 0.9 + 0.1*np.cos(2*np.pi*3.0*t)
    g2 = smooth_gate(t, [0.6, 1.8, 3.0, 4.1], width=0.22)
    g3 = smooth_gate(t, [0.9, 2.1, 3.5, 4.6], width=0.18)
    g4 = 0.7 + 0.3*np.cos(2*np.pi*2.3*t + 1.0)
    g5 = smooth_gate(t, [0.5, 1.3, 2.7, 3.9, 4.7], width=0.16)
    g6 = 0.75 + 0.25*np.cos(2*np.pi*1.7*t + 0.5)
    g7 = smooth_gate(t, [1.0, 2.5, 4.0], width=0.28)
    g8 = smooth_gate(t, [0.8, 2.0, 3.2, 4.4], width=0.24)

    # Signal final
    x_red = (
        1.00 * g1 * np.cos(phi1) +
        0.95 * g2 * np.cos(phi2) +
        0.85 * g3 * np.cos(phi3) +
        0.75 * g4 * np.cos(phi4) +
        0.65 * g5 * np.cos(phi5) +
        0.55 * g6 * np.cos(phi6) +
        0.50 * g7 * np.cos(phi7) +
        0.45 * g8 * np.cos(phi8)
    )

    x_use = x_red
    pow = 1
    results = generate_all_figures(
        x=x_use,
        fs=fs,
        win_len=700,
        sigma=100.0,
        n_fft=2**(10+pow),
        hop_len=160,
        gamma=1e-6,
        fmax_display=10_000,
        out_dir="figuras_ssq"
    )

    plt.show()
    