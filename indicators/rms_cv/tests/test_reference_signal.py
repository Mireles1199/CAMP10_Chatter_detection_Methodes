"""Assert-based self-check for the reference_signal extension point (Fase 3).

Proves -- not just visually, with real assertions -- that when
INDICATOR_CONFIG["reference_signal"] is given, rms_cv_pipeline actually reads
its mu/sigma/threshold from that external signal's own CV series instead of
from the analyzed signal, and that omitting it reproduces the original
internal stable_region behaviour unchanged.

Run directly: python test_reference_signal.py
"""
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from rms_cv import SignalData, run_rms_cv, rms_sequence, CVOnlineConfig, CVOnlineMonitor
from rms_cv.lib.cv_monitor import CVStableRegionDetector


def _make_bursty_signal(n_samples: int, fs: float, seed: int, block: int = 500) -> SignalData:
    """White noise whose amplitude alternates between quiet/loud every `block`
    samples -- gives a genuinely non-trivial CV population (RMS jumps between
    consecutive n_max-frame windows)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    n_blocks = n_samples // block + 1
    amp = np.repeat(np.where(np.arange(n_blocks) % 2 == 0, 0.02, 0.2), block)[:n_samples]
    x = amp * rng.standard_normal(n_samples)
    return SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=fs)


def _make_constant_signal(n_samples: int, fs: float) -> SignalData:
    """Pure sine: RMS is essentially identical across every window -> CV ~ 0.
    CV is scale-invariant, so it's the *shape* (constant vs. bursty), not the
    amplitude, that must differ between the two signals for this check to mean
    anything."""
    t = np.arange(n_samples) / fs
    x = 0.1 * np.sin(2 * np.pi * 5 * t)
    return SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=fs)


def _cv_series(signal: SignalData, n_max: int, samples_per_window: int) -> np.ndarray:
    """Independently reproduce the CV series rms_cv_pipeline computes for a signal --
    used to check meta["cv_training_values"] against a ground truth the pipeline
    itself did not hand us."""
    window_sec = samples_per_window / signal.fs
    out = rms_sequence(signal.signal_analysis, signal.fs, window_sec=window_sec, overlap_pct=0.0)
    mon = CVOnlineMonitor(CVOnlineConfig(n_max=n_max))
    return np.array([mon.update(float(r))["cv"] for r in out["rms"]])


def test_reference_signal_is_actually_used():
    fs = 1000.0
    n_max, samples_per_window = 10, 100

    # Analyzed signal: bursty (quiet/loud blocks) -> its own CV has real variability.
    analyzed = _make_bursty_signal(20_000, fs, seed=1)
    # Reference signal: constant-amplitude sine -> near-zero CV population,
    # unambiguously different from the analyzed signal's own.
    reference = _make_constant_signal(20_000, fs)

    common_params = {
        "n_max": n_max, "samples_per_window": samples_per_window,
        "cv_threshold": None, "frac_stable": 0.5, "z": 3.0,
    }

    cfg_internal = {"func": "Default", "params": dict(common_params)}
    cfg_external = {"func": "Default", "params": dict(common_params),
                     "reference_signal": reference}

    res_internal = run_rms_cv(analyzed, cfg_internal)
    res_external = run_rms_cv(analyzed, cfg_external)

    # 1) training_source correctly tags each run.
    assert res_internal.meta["training_source"] == "internal"
    assert res_external.meta["training_source"] == "external_reference"
    assert res_external.meta["cv_threshold_method"] == "external_reference"

    # 2) the external run's mu/sigma really come from the external CV population,
    #    not a silent no-op that keeps using the analyzed signal's own stats.
    #    Ground truth goes through CVStableRegionDetector directly (same as the
    #    pipeline) rather than a plain mean, since it may fall back to MAD/median
    #    when the Lilliefors normality test rejects the population.
    ref_cv = _cv_series(reference, n_max, samples_per_window)
    expected_ext = CVStableRegionDetector(frac_stable=1.0, z=3.0).detect(
        ref_cv, idx_stable=np.arange(ref_cv.size)
    )
    assert np.isclose(res_external.meta["mu_stable"], expected_ext["mu"], rtol=1e-9), (
        "external mu_stable does not match the reference signal's own CV population -- "
        "reference_signal is not actually being consumed."
    )

    # 3) meta["cv_training_values"] literally is that population (not a subset,
    #    not the analyzed signal's), per the "use ALL of it, no cropping" contract.
    training_vals = res_external.meta["cv_training_values"]
    assert training_vals.size == ref_cv.size
    assert np.allclose(np.sort(training_vals), np.sort(ref_cv))

    # 4) the two runs must disagree, given how different the two populations are --
    #    proves the branch has a real, visible effect on the indicator's output,
    #    not just on some unused metadata field.
    assert abs(res_internal.meta["mu_stable"] - res_external.meta["mu_stable"]) > 0.1
    assert res_internal.meta["cv_threshold_used"] != res_external.meta["cv_threshold_used"]

    # 5) omitting reference_signal reproduces the pre-existing internal behaviour:
    #    mu_stable matches the first frac_stable fraction of the analyzed signal's
    #    OWN CV series (ground truth computed independently here, same detector).
    analyzed_cv = _cv_series(analyzed, n_max, samples_per_window)
    expected_int = CVStableRegionDetector(frac_stable=0.5, z=3.0).detect(analyzed_cv)
    assert np.isclose(res_internal.meta["mu_stable"], expected_int["mu"], rtol=1e-9)

    print("test_reference_signal_is_actually_used: OK")


if __name__ == "__main__":
    test_reference_signal_is_actually_used()
