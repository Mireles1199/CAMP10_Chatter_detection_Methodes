"""Quick backward-compat + overlap correctness test for segment_opr."""
import importlib.util, pathlib, numpy as np

spec = importlib.util.spec_from_file_location(
    "opr",
    pathlib.Path(__file__).parent / "src/MaxEnt_SPRT/utils/opr.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
segment_opr = m.segment_opr

opr = np.arange(50, dtype=float)
t   = np.linspace(0, 1, 50)

# backward compat: step=None == step=N_seg
s1, _ = segment_opr(opr, t, N_seg=5)
s2, _ = segment_opr(opr, t, N_seg=5, step=5)
assert len(s1) == len(s2) and all(np.array_equal(a, b) for a, b in zip(s1, s2)), \
    "backward compat FAIL"

# overlap: N_seg=5, step=2 -> floor((50-5)/2)+1 = 23 segments
s3, _ = segment_opr(opr, t, N_seg=5, step=2)
assert len(s3) == 23, f"overlap count FAIL: expected 23, got {len(s3)}"

# first segment content check
assert np.array_equal(s3[0], opr[0:5])
assert np.array_equal(s3[1], opr[2:7])

print("All assertions passed")
print(f"  no-overlap  : {len(s1)} segs")
print(f"  overlap 60% : {len(s3)} segs  (step=2, N_seg=5)")
