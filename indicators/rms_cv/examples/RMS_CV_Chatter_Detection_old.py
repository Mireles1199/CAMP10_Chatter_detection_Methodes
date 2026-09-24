# Comentario: ejemplo mínimo de uso de la librería
from __future__ import annotations
import os
import sys
import numpy as np
from collections import defaultdict

# Force the local src/ over the fixed-path editable install (see
# indicators/COMMON_TEMPLATE.md §8) so this script always runs against the
# code in this worktree, not whatever checkout `pip install -e` points at.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rms_cv import five_senos, signal_1, rms_sequence, CVOnlineConfig, CVOnlineMonitor
from rms_cv import plot_signal, plot_rms, plot_cv
import matplotlib.pyplot as plt

from C_emd_hht import signal_chatter_example, sinus_6_C_SNR

def main() -> None:

    #%%
    fs: float = 5000.0
    dur: float = 5.0

    t, sig = five_senos(fs, duracion=dur, ruido_std=0.2, fase_aleatoria=True, seed=42)
    t, sig = signal_chatter_example(fs=fs, T=dur, seed=123)
    t, sig = sinus_6_C_SNR(fs=fs, T=dur, 
                         chatter=True,
                         exp=None,
                         Amp=5,
                         stable_to_chatter=False,
                         noise=True,
                         SNR_dB=10.0)


    plot_signal(t, sig, title="Synthetic signal (stable -> chatter)")

    samples_per_window: int = 1000
    window_sec: float = samples_per_window / fs
    overlap_pct: float = 0.5
    dt_rms: float = window_sec * (1.0 - overlap_pct)

    out = rms_sequence(sig, fs, window_sec=dt_rms, overlap_pct=overlap_pct, detrend=False, pad_mode="none")
    rms_vals = out["rms"]
    times = out["times"]
    plot_rms(times, rms_vals, title="RMS sequence")

    cfg = CVOnlineConfig(
        n_max=20, use_unbiased_std=True, eps=1e-12,
        cv_threshold=0.15, rms_threshold=0.9,
        n_min_cv=2, warmup_ignore_alerts=False,
        dt_rms=dt_rms, start_time=0.0
    )
    mon = CVOnlineMonitor(cfg)

    results = defaultdict(list)
    for r in rms_vals:
        res = mon.update(float(r))
        for k, v in res.items():
            results[k].append(v)

    plot_cv(results["time"], results["cv"], cfg.cv_threshold,  title="CV over time")
    plt.show()



if __name__ == "__main__":
    main()
