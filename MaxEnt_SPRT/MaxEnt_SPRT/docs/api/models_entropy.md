# Models & Entropy

Statistical modeling components used to represent the stable and chatter regimes.
These are the probabilistic foundations that the LLR model and SPRT engine consume.

[← Detector & SPRT](detection.md){ .md-button } [Data & I/O →](data_io.md){ .md-button }

---

## `MaxEntModels` & `fit_maxent_gaussians` — Model pair

A `MaxEntModels` object holds two fitted Gaussian PDFs — `p0` for the stable regime
and `p1` for the chatter regime. Use `fit_maxent_gaussians` to build it from entropy samples.

```python
from MaxEnt_SPRT.models.maxent import fit_maxent_gaussians

models = fit_maxent_gaussians(
    samples_H0=H_free,   # entropy samples from stable condition
    samples_H1=H_chat,   # entropy samples from chatter condition
)
# models.p0 → GaussianPDF (stable)
# models.p1 → GaussianPDF (chatter)
```

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/models/maxent/index.html){ .md-button .md-button--primary }

---

## `GaussianPDF` — Core probability model

Frozen Gaussian defined by `mu` and `sigma`. Provides the three operations the
detector pipeline needs: log-density, Shannon entropy, and sample-based fitting.

```python
GaussianPDF.logpdf(x: float) -> float          # log p(x)
GaussianPDF.entropy_shannon()  -> float          # H = 0.5 * log(2πe σ²)
GaussianPDF.from_samples(samples) -> GaussianPDF # static constructor
```

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/models/prob/index.html){ .md-button }

---

## `EntropyEstimator` & `entropy_from_segments` — Entropy extraction

Converts a list of raw signal segments into a scalar entropy sequence fed into the SPRT.

```python
from MaxEnt_SPRT.lib.entropy import entropy_from_segments, GaussianMaxEntEstimator

H = entropy_from_segments(segments, estimator=GaussianMaxEntEstimator())
```

| Estimator | Method |
|---|---|
| `GaussianMaxEntEstimator` | Fits a Gaussian per segment (default) |
| `EmpiricalHistogramEntropyEstimator` | Uses histogram bin counts |

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/lib/entropy/index.html){ .md-button }
