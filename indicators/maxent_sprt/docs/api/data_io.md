# Data & I/O

Input types, signal preprocessing, OPR sampling, segmentation, and HDF5 file reading utilities.
Start here when integrating the detector with a new data source.

[← Models & Entropy](models_entropy.md){ .md-button } [Visualization →](visualization.md){ .md-button }

---

## `SignalData` — Pipeline input type { #signaldata }

The container passed to every detection function. Wraps the raw signal array and its time vector.

```python
from MaxEnt_SPRT.utils.types import SignalData

signal = SignalData(
    y: np.ndarray,   # vibration samples
    t: np.ndarray,   # time vector (same length as y)
)
```

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/utils/types/index.html){ .md-button .md-button--primary }

---

## `IndicatorResult` — Detection output { #indicatorresult }

Returned by every top-level detection call. Bundles the SPRT result with the full
entropy sequence and segment timing for plotting.

```python
result.sprt_result   # SPRTResult — final decision + cumulative statistic
result.H_seq         # np.ndarray — entropy value per segment
result.t_mid         # np.ndarray — time midpoint per segment
```

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/utils/types/index.html){ .md-button }

---

## OPR Sampling & Segmentation

Once-Per-Revolution (OPR) resampling and fixed-length segmentation — the preprocessing
steps that convert a raw vibration signal into analysis-ready windows.

```python
from MaxEnt_SPRT.utils.opr import sample_opr, segment_opr

# Downsample to ratio_sampling samples per revolution
opr, opr_t = sample_opr(y, t, fs=fs, fr=fr)

# Split into N_seg equal segments
segments, seg_times = segment_opr(opr, opr_t, N_seg=N_seg)
```

| Function | Returns |
|---|---|
| `sample_opr(y, t, fs, fr)` | `(opr, opr_t)` downsampled arrays |
| `segment_opr(opr, opr_t, N_seg)` | `(List[ndarray], List[ndarray])` segment lists |

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/utils/opr/index.html){ .md-button }

---

## `HDF5Reader` — Experimental data loader

Reads HDF5 files from measurement hardware and provides dictionary-like access to
nested datasets.

```python
from MaxEnt_SPRT.utils.hdf5_utils import HDF5Reader

reader = HDF5Reader("experiment.h5")
reader.list_paths()            # all dataset paths in the file
reader.get_element("accel", "x") # access nested key hierarchy
reader.find_first("rpm")       # search by partial key name
```

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/utils/hdf5_utils/index.html){ .md-button }
