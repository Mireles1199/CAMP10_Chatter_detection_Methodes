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

from green_integral import StdSignalData, run_green_std

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

print("OK — reference_signal per-piece self-checks passed.")
