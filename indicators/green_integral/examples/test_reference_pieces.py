"""Self-check for `reference_signal` as a LIST of pieces (per-piece windowing).

Bug this fixes: with a single stitched-together reference signal (the old
`reference_combined.h5`), the windowing pipeline ran end-to-end across the
whole thing, so some windows straddled the seam between two unrelated
pieces (e.g. the tail of case_003 and the head of case_004) — those mixed
windows' areas fed straight into the mu +- z*sigma population.

Fix: `reference_signal` accepts `List[SignalData]` (a bare SignalData is
still accepted, treated as one piece). Each piece is windowed SEPARATELY
with its own call to the pipeline; only the resulting per-window areas/times
are concatenated into the training pool — never the raw signals.

Verifies with two synthetic pieces of very different amplitude (so a seam
window would produce an area value in neither piece's own range, an easy
tell if concatenation ever creeps back in):

1. `global_data["reference_n_pieces"] == 2`.
2. `sum(reference_piece_window_counts) == training_areas.size` (the pool is
   exactly the per-piece window counts added up, nothing dropped or
   duplicated).
3. `training_areas` splits cleanly into two non-overlapping ranges, one per
   piece's own amplitude scale — proof no window mixed the two pieces.
4. A single bare SignalData (not a list) still works exactly as before
   (backward compatible with the Phase 3 behavior already shipped).
5. plot_training_distribution()'s "Training Curve" breaks the line (a NaN
   row) between two pooled pieces instead of drawing a straight segment
   connecting the end of one piece to the start of the next — which would
   represent nothing real, since the two are physically unrelated.
"""

from __future__ import annotations
import sys
import pathlib
import numpy as np

_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral.logging_setup import configure_logging, LOGGING_LEVELS
configure_logging(level=LOGGING_LEVELS["warning"])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from green_integral import StdSignalData, run_green_std
from green_integral.viz.plots import plot_training_distribution

fs = 5000.0
f_modal = 150.0


def _make_signal(t: np.ndarray, amp: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return amp * np.sin(2.0 * np.pi * f_modal * t) + 1e-4 * rng.standard_normal(len(t))


BASE_PARAMS = dict(f_modal=f_modal, num_T=4, dt=0.01, data_filtrated=True,
                    while_loop_extend=True, use_area_threshold=True, z_sigma=3.0)

# Analyzed signal: same shape as test_reference_signal.py, unrelated content.
t_an = np.arange(0.0, 1.0, 1.0 / fs)
x_an = _make_signal(t_an, amp=1.0)
an_std = StdSignalData(t_analysis=t_an, signal_analysis=x_an, path="analysis", fs=fs)

# Two reference pieces, very different amplitude scale.
t_piece = np.arange(0.0, 1.0, 1.0 / fs)
piece_low = StdSignalData(t_analysis=t_piece, signal_analysis=_make_signal(t_piece, amp=1.0),
                           path="piece_low", fs=fs, meta={"name": "piece_low"})
piece_high = StdSignalData(t_analysis=t_piece, signal_analysis=_make_signal(t_piece, amp=10.0),
                            path="piece_high", fs=fs, meta={"name": "piece_high"})


def _run(reference_signal):
    cfg = {"func": "Default", "param_mode": "native", "params": dict(BASE_PARAMS)}
    if reference_signal is not None:
        cfg["reference_signal"] = reference_signal
    return run_green_std(an_std, cfg)


# 1-3. List of 2 pieces: n_pieces, pool size, no cross-piece mixing.
res = _run([piece_low, piece_high])
gd = res.meta["raw_result"].global_data

assert gd["reference_n_pieces"] == 2, gd["reference_n_pieces"]
counts = gd["reference_piece_window_counts"]
assert len(counts) == 2 and all(c > 0 for c in counts), counts

train_areas = np.asarray(gd["training_areas"], dtype=float)
assert train_areas.size == sum(counts), (train_areas.size, counts)

# Two well-separated windows, one per piece's own amplitude scale (shoelace
# area grows ~amplitude^2, so amp=10 windows sit roughly 2 orders of
# magnitude above amp=1 windows) -- a seam-mixed window would land between
# them, which never happens here since pieces are windowed separately.
sorted_areas = np.sort(train_areas)
gap_idx = np.argmax(np.diff(sorted_areas))
low_cluster, high_cluster = sorted_areas[:gap_idx + 1], sorted_areas[gap_idx + 1:]
assert len(low_cluster) == counts[0] and len(high_cluster) == counts[1], (
    len(low_cluster), len(high_cluster), counts
)
assert low_cluster.max() < high_cluster.min() / 10, (
    "expected a clear amplitude-scale gap between the two pieces' areas"
)

# 4. A single bare SignalData (not a list) still works (Phase 3 compat).
res_single = _run(piece_low)
gd_single = res_single.meta["raw_result"].global_data
assert gd_single["reference_n_pieces"] == 1, gd_single["reference_n_pieces"]
assert np.asarray(gd_single["training_areas"]).size == gd_single["reference_piece_window_counts"][0]

# 5. "Training Curve" line breaks between the two pieces (no seam-connecting
# straight line) -- the two pieces' own windows are each internally regular
# in time, so the only place a NaN can legitimately land is the boundary.
plt.close("all")
# log_transform=False: this test's result comes from func="Default", whose
# area_mu_3sigma is in linear space (Lyapunov's is log10 -- see plots_lyapunov
# vs plots_green_integral's own log_transform choice).
figs = plot_training_distribution(gd, name="pieces", log_transform=False)
assert len(figs) == 2, f"expected [histogram, curve], got {len(figs)} figure(s)"
curve_ax = figs[1].axes[0]
curve_line = next(l for l in curve_ax.get_lines() if l.get_label().startswith("Training pop."))
y_curve = curve_line.get_ydata()
n_nan = int(np.sum(np.isnan(y_curve)))
assert n_nan >= 1, "expected at least one NaN break between the two pieces"
# exactly at the piece boundary (right after the first piece's own windows)
nan_positions = np.where(np.isnan(y_curve))[0]
assert counts[0] in nan_positions, (
    f"NaN break at {nan_positions}, expected at index {counts[0]} (end of piece 1)"
)

# The mu/mu+-z*sigma axhlines must still be the real threshold values, not
# clobbered by the gap-shading loop's own (array-index) local variables --
# regression check for exactly that bug (loop reused the names `lo`/`hi`,
# stomping the outer threshold values with small integers like 149/100 and
# blowing up the whole plot's Y scale).
thr = gd["area_mu_3sigma"]
hline_labels = {l.get_label(): l for l in curve_ax.get_lines() if l.get_label().startswith("$\\mu")}
assert len(hline_labels) == 3, hline_labels.keys()
for line in hline_labels.values():
    y = float(np.asarray(line.get_ydata())[0])
    assert y in (thr["mu"], thr["upper"], thr["lower"]), (
        f"axhline at y={y} doesn't match any real threshold value "
        f"(mu={thr['mu']}, upper={thr['upper']}, lower={thr['lower']}) -- "
        f"looks like a leftover index value instead"
    )
plt.close("all")

print("OK — reference_signal per-piece self-checks passed.")
