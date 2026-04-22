"""Quick test: segment_signal_raw produces correct blocks, offline_train accepts segmentation='raw'."""
import importlib.util, pathlib, sys, numpy as np

BASE = pathlib.Path(
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria"
    r"\CAMP10_Chatter_detection_Methodes\indicators\maxent_sprt\src"
)
sys.path.insert(0, str(BASE))

from MaxEnt_SPRT.utils.opr import segment_signal_raw

# ---------- segment_signal_raw basics -----------------------------------------
y = np.random.randn(10_000)
t = np.linspace(0, 1, 10_000)
N = 500

segs, segs_t = segment_signal_raw(y, t, N_samples_per_seg=N)
assert len(segs) == 20 and all(len(s) == N for s in segs), "basic block count FAIL"

segs_ov, _ = segment_signal_raw(y, t, N_samples_per_seg=N, step=250)
# floor((10000 - 500) / 250) + 1 = 39
assert len(segs_ov) == 39, f"overlap block count FAIL: got {len(segs_ov)}"

# ---------- offline_train with segmentation='raw' ----------------------------
from MaxEnt_SPRT.lib.offline import offline_train_maxent_sprt

rng = np.random.default_rng(0)
sig_free = rng.normal(0.0, 1.0, 5000)
sig_chat = rng.normal(0.5, 1.5, 5000)
t_arr    = np.linspace(0, 1, 5000)

models, H_free, H_chat, *_ = offline_train_maxent_sprt(
    opr_free=sig_free, opr_chat=sig_chat,
    opr_t_free=t_arr,  opr_t_chat=t_arr,
    N_seg=10,            # only used by 'opr' path (ignored here)
    segmentation="raw",
    N_samples_per_seg=500,
)
assert H_free.shape[0] == 10 and H_chat.shape[0] == 10, "entropy seq len FAIL"
assert models.p0.mu != models.p1.mu, "models not distinguishable FAIL"

print("All assertions passed")
print(f"  segment_signal_raw:  20 blocks (no overlap), 38 blocks (50% overlap)")
print(f"  offline_train (raw): H_free={H_free.shape}, mu0={models.p0.mu:.4f}, mu1={models.p1.mu:.4f}")
