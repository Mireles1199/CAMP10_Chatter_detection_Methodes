import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from C_emd_hht import make_chatter_like_signal, five_senos
from C_emd_hht import amplitude_spectrum
from C_emd_hht import detect_chatter_from_force
from C_emd_hht import plot_imfs, plot_imf_seleccionado, plot_tendencia, plot_HHS
from C_emd_hht import signal_chatter_example, sinus_6_C_SNR

# --- 1) Synthetic example signal. Replace with your real F ---
fs = 10_000.0
T = 5.0
t = np.arange(int(T * fs)) / fs

t, F = five_senos(fs, duracion=T, ruido_std=0.2, fase_aleatoria=True, seed=42)
# t, F = signal_chatter_example(fs=fs, T=T, seed=123)
t, F = sinus_6_C_SNR(fs=fs, T=T, 
                     chatter=True,
                     exp=None,
                     Amp=5,
                     stable_to_chatter=False,
                     noise=True,
                     SNR_dB=10.0)

fmax = 5000.0


f, A = amplitude_spectrum(F, fs=fs,)

# Example/plot block (same logic as original)
plt.figure()
plt.plot(t, F)
plt.title("Example force signal F(t)")
plt.xlabel("Time (s)")
plt.ylabel("Force (a.u.)")
plt.show()

plt.figure()
plt.plot(f, A)
plt.title("Amplitude spectrum of F(t)")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, fmax)
plt.ylabel("Amplitude (a.u.)")

#%%
# --- 2) Run the pipeline (no baseline/decision) ---
res = detect_chatter_from_force(
    F=F,
    fs=fs,
    bandlimit_pre=(-100, fmax),
    emd_method="emd",          # "emd" or "eemd" also valid
    ceemdan_noise_strength=0.2,
    ceemdan_ensemble_size=50,
    band_chatter=(450, 600),
    band_selection_margin=0.15,
    imf_selection="index",
    imf_index=0,
    # Hilbert
    phase_diff_mode="first_diff",
    sg_window=11,
    sg_polyorder=2,
    f_inst_smooth_median=None,
    # HHS
    hhs_enable=True,
    hhs_fmax=5000.0,
    hhs_fbin_hz=5.0,
    count_win_samples=100,
    energy_mode="A2",
    thr_mode="none",
    thr_k_mad=0.0,
    thr_percentile=50,
)

#%%
# --- 3) Plot ALL IMFs in a single figure ---
#    (you can limit with max_to_plot and arrange columns with ncols)
fig, axes = plot_imfs(
    imfs=res.imfs,
    plot_spectrum=True,
    fs=fs,
    f_max=fmax,
    max_to_plot=None,
    ncols=1,
    show=True
)

#%%
# --- 4) Plot selected IMF and optionally A(t) and f_inst(t) ---
plot_imf_seleccionado(
    selected_imf=res.selected_imf,
    fs=fs,
    f_max = fmax,
    plot_spectrum=True,
    A=res.A,
    f_inst=res.f_inst,
    show=True
)

print("IMFs:", res.imfs.shape if res.imfs is not None else None)
print("Selected IMF:", res.k_selected)
print("Counts shape:", res.counts.shape)
print("Threshold used (energy):", res.meta.get("threshold"))

#%%
# --- 5) Plot the HHS (Hilbert-Huang Spectrum) ---
plot_HHS(
    t=t,
    fgrid=res.fgrid,
    HHS=res.HHS,
    fmax=fmax,
)

#%% ---------- Counts plot ----------
t_counts = res.t_counts_samples / fs
counts = res.counts

# 1) Dashed line
# plot_tendencia(t_counts, counts, tipo="pchip", ls='--', lw=2, color='k')

# # 2) Smoothed dashed line (very similar to the original figure)
# plot_tendencia(t_counts, counts, tipo="polinomial", grado=4, ls='--', lw=2, color='k')

# 3) Smoothed step-like polynomial (e.g., degree 2)
plot_tendencia(t_counts, counts, tipo="step_suave", grado=3, ls='--', lw=2, color='k')

plt.show()
