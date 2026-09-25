"""
test_reference_signal.py
=========================
Self-check for the ``reference_signal`` extension point (see
``indicators/COMMON_TEMPLATE.md``): an optional externally-labelled
``SignalData`` that, when given, overrides the internally-derived
stable/training region instead of ``training_intervals``/legacy cut. Also
covers ``reference_signal_chatter``, MaxEnt's own mirror of it for the
chatter/P1 side (not part of the shared contract).

Synthetic signal, no real dataset needed. Run directly: asserts only.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MaxEnt_SPRT import SignalData, run_maxent_sprt


def _make_signal(fs: float, t_total: float, seed: int, chatter_from: float | None = None) -> SignalData:
    rng = np.random.default_rng(seed)
    n = int(t_total * fs)
    t = np.arange(n) / fs
    x = rng.normal(0.0, 1.0, size=n)
    if chatter_from is not None:
        x[t >= chatter_from] += rng.normal(0.0, 4.0, size=(t >= chatter_from).sum())
    return SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=fs)


def _base_config(
    reference_signal: SignalData | None = None,
    reference_signal_chatter: SignalData | None = None,
) -> dict:
    cfg = {
        "id": "MaxEnt_SPRT",
        "func": "Default",
        "param_mode": "native",
        "params": {
            "rpm": 6000.0,
            "N_seg": 200,
            "t_stable_total": 2.0,
            "alpha": 0.01,
            "beta": 0.01,
            "reset_on_H0": True,
            "cut_start_time": 0.0,
            "cut_end_time": 4.0,
            "segmentation": "raw",
            "N_samples_per_seg": 200,
        },
    }
    if reference_signal is not None:
        cfg["reference_signal"] = reference_signal
    if reference_signal_chatter is not None:
        cfg["reference_signal_chatter"] = reference_signal_chatter
    return cfg


def _seam_test() -> None:
    """Two reference pieces at very different noise levels (sigma=1 vs sigma=10),
    lengths not multiples of N_samples_per_seg. If windowing ever concatenated
    the pieces before segmenting (the seam-contamination bug), a window
    straddling the seam would mix both noise levels and land strictly between
    the two pieces' theoretical entropies -- this asserts that never happens.
    """
    fs = 5000.0
    N_samples_per_seg = 200

    def _noise_piece(sigma: float, n_samples: int, seed: int) -> SignalData:
        rng = np.random.default_rng(seed)
        t = np.arange(n_samples) / fs
        x = rng.normal(0.0, sigma, size=n_samples)
        return SignalData(t_analysis=t, signal_analysis=x, path=f"synthetic_sigma{sigma}",
                           fs=fs, meta={"signal_id": f"sigma{sigma}_{n_samples}"})

    piece_low  = _noise_piece(sigma=1.0,  n_samples=950, seed=10)
    piece_high = _noise_piece(sigma=10.0, n_samples=730, seed=11)
    signal = _make_signal(fs=fs, t_total=4.0, seed=99, chatter_from=2.0)

    result = run_maxent_sprt(signal, _base_config(reference_signal=[piece_low, piece_high]))

    assert result.meta["reference_n_pieces"] == 2, result.meta["reference_n_pieces"]
    n_windows = result.meta["n_windows_per_piece_free"]
    assert n_windows == [950 // N_samples_per_seg, 730 // N_samples_per_seg], n_windows

    detector = result.meta["detector"]
    assert detector.H_free.size == sum(n_windows)

    H_low_theory  = 0.5 * np.log(2 * np.pi * np.e * 1.0 ** 2)
    H_high_theory = 0.5 * np.log(2 * np.pi * np.e * 10.0 ** 2)
    for h in detector.H_free:
        near_low  = abs(h - H_low_theory)  < 0.5
        near_high = abs(h - H_high_theory) < 0.5
        assert near_low or near_high, (
            f"H={h:.4f} falls between the two pieces' entropy bands "
            f"({H_low_theory:.4f} / {H_high_theory:.4f}) -> seam contamination"
        )

    print("test_reference_signal (seam pool): OK")


def main() -> None:
    signal = _make_signal(fs=5000.0, t_total=4.0, seed=0, chatter_from=2.0)

    # ── internal split (no reference_signal): existing behaviour, unaffected ──
    result_internal = run_maxent_sprt(signal, _base_config(reference_signal=None))
    assert result_internal.meta["training_source"] == "internal"
    internal_stable_size = result_internal.meta["Size_signal_free"]
    expected_stable_size = int(((signal.t_analysis >= 0.0) & (signal.t_analysis <= 2.0)).sum())
    assert internal_stable_size == expected_stable_size

    # ── external reference overrides the stable region ────────────────────────
    reference = _make_signal(fs=5000.0, t_total=1.5, seed=1)  # deliberately different size
    result_ext = run_maxent_sprt(signal, _base_config(reference_signal=reference))
    assert result_ext.meta["training_source"] == "external_reference"
    assert result_ext.meta["chatter_source"] == "internal"
    assert result_ext.meta["Size_signal_free"] == reference.signal_analysis.size
    assert result_ext.meta["Size_signal_free"] != internal_stable_size
    # chatter counterpart still comes from the legacy cut_end_time fallback
    assert result_ext.meta["Size_signal_chatter"] > 0

    # ── reference_signal_chatter overrides the chatter/P1 side too ────────────
    reference_chatter = _make_signal(fs=5000.0, t_total=0.8, seed=2)  # deliberately different size
    result_both = run_maxent_sprt(
        signal,
        _base_config(reference_signal=reference, reference_signal_chatter=reference_chatter),
    )
    assert result_both.meta["training_source"] == "external_reference"
    assert result_both.meta["chatter_source"] == "external_reference"
    assert result_both.meta["Size_signal_free"] == reference.signal_analysis.size
    assert result_both.meta["Size_signal_chatter"] == reference_chatter.signal_analysis.size

    print("test_reference_signal: OK")

    _seam_test()


if __name__ == "__main__":
    main()
