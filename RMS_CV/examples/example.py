# Comentario: ejemplo mínimo de uso de la librería
from __future__ import annotations
import numpy as np
from collections import defaultdict
from rms_cv import five_senos, signal_1, rms_sequence, CVOnlineConfig, CVOnlineMonitor
from rms_cv import plot_signal, plot_rms, plot_cv
import matplotlib.pyplot as plt

def main() -> None:
    fs: float = 5000.0
    dur: float = 10.0

    t1, sig_s = five_senos(fs, duracion=dur, ruido_std=0.2, fase_aleatoria=True, seed=42)
    t2, sig_c = signal_1(fs=fs, T=dur, tpf=200.0, chatter_freqs=[150.0, 300.0],
                         t_chatter_start=5.0, noise_std=0.2)
    sig = np.concatenate([sig_s[:len(t1)//2], sig_c[len(t2)//2:]])
    t = np.linspace(0.0, dur, sig.size, endpoint=False)

    plot_signal(t, sig, title="Synthetic signal (stable -> chatter)")

    window_sec: float = 0.05
    overlap_pct: float = 0.0
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

    plot_cv(results["time"], results["cv"], title="CV over time")
    plt.show()

if __name__ == "__main__":
    main()
