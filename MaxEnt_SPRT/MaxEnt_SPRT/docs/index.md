# MaxEnt-SPRT: Chatter Detection Indicator

**MaxEnt-SPRT** is a Python library that detects machining chatter *online* using two well-established statistical tools: **Maximum Entropy (MaxEnt)** modelling and **Wald's Sequential Probability Ratio Test (SPRT)**.

[Open User Guide](guide/quickstart.md){ .md-button .md-button--primary }
[Open Technical Docs](technical/index.html){ .md-button }

---

[Developer Hub](developer_hub.md){ .md-button }

---

## What is Chatter?

Chatter is a self-excited vibration that appears during machining (milling, turning, boring) when the cutting force couples with the structural dynamics of the machine–“tool–“workpiece system. It manifests as a sudden, large-amplitude oscillation that:

- Damages the tool and the workpiece surface.
- Produces a characteristic loud noise.
- Limits depth-of-cut and spindle speed.

Detecting chatter *early* — before it fully develops — allows a controller to adjust the process parameters and avoid damage.

---

## Why MaxEnt + SPRT?

| Challenge | Solution |
|---|---|
| The vibration signal changes character when chatter starts | Entropy of signal segments increases at chatter onset |
| We need to decide *online*, segment by segment | SPRT makes binary decisions sequentially with controlled error rates |
| We want to avoid false alarms | SPRT thresholds $\alpha$ (false positive) and $\beta$ (false negative) directly control error rates |
| We need a probability model for entropy values | Maximum Entropy gives the *least-biased* Gaussian PDF from observed mean/variance |

---

## Quick Installation

```bash
# From the project folder
pip install .
```

See [Installation](guide/installation.md) for details.

---
## Pipeline

The following diagram summarizes the complete detection workflow.

```mermaid
flowchart TD

    A["Raw vibration signal"]

    subgraph P1["Preprocessing"]
        B["OPR Downsampling<br/>One sample per revolution<br/>f_r = rpm / 60"]
        C["Segmentation<br/>N_seg revolutions per segment"]
        D["Entropy Extraction<br/>Compute one entropy value H<br/>for each segment"]
    end

    subgraph P2["Statistical Modeling"]
        E["Offline Training<br/>Estimate P₀(H) from stable data<br/>Estimate P₁(H) from chatter data  "]
    end

    subgraph P3["Sequential Detection (SPRT)"]
        F["Online Monitoring<br/>Process the entropy sequence"]
        G["Statistic Update<br/>Accumulate decision statistic S"]
        H{"Decision rule"}
        I["Accept H₁<br/>CHATTER detected"]
        J["Accept H₀<br/>FREE state"]
        K["Continue sampling<br/>Collect more evidence"]
    end

    subgraph P4["Output"]
        L["IndicatorResult<br/>Return timestamps where S ≥ b"]
    end

    A --> B --> C --> D --> E --> F --> G --> H
    H -- "S ≥ b" --> I --> L
    H -- "S ≤ a" --> J --> F
    H -- "a < S < b" --> K --> F

    classDef input fill:#f8f9fa,stroke:#5f6368,stroke-width:1.5px,color:#111,rx:8,ry:8;
    classDef prep fill:#e8f0fe,stroke:#1a73e8,stroke-width:1.5px,color:#111,rx:8,ry:8;
    classDef model fill:#e6f4ea,stroke:#188038,stroke-width:1.5px,color:#111,rx:8,ry:8;
    classDef detect fill:#fef7e0,stroke:#f9ab00,stroke-width:1.5px,color:#111,rx:8,ry:8;
    classDef decision fill:#fde7e9,stroke:#d93025,stroke-width:2px,color:#111,rx:10,ry:10;
    classDef positive fill:#fce8e6,stroke:#d93025,stroke-width:2px,color:#111,rx:8,ry:8;
    classDef negative fill:#e6f4ea,stroke:#188038,stroke-width:2px,color:#111,rx:8,ry:8;
    classDef neutral fill:#f1f3f4,stroke:#5f6368,stroke-width:1.5px,color:#111,rx:8,ry:8;
    classDef output fill:#ede7f6,stroke:#7e57c2,stroke-width:1.5px,color:#111,rx:8,ry:8;

    class A input;
    class B,C,D prep;
    class E model;
    class F,G detect;
    class H decision;
    class I positive;
    class J negative;
    class K neutral;
    class L output;
```

---
## Quick Example

```python
import numpy as np
from MaxEnt_SPRT import SignalData, run_maxent_sprt, plots_maxent_sprt

# --- Synthetic signal: stable sine (0-5 s) + high-frequency chirp (5-10 s) ---
fs = 20_000.0
t = np.arange(0, 10, 1/fs)
rng = np.random.default_rng(42)
stable  = np.sin(2 * np.pi * 200 * t[t < 5])  + rng.normal(0, 0.05, (t < 5).sum())
chatter = np.sin(2 * np.pi * (200 + 60 * t[t >= 5]) * t[t >= 5]) + rng.normal(0, 0.05, (t >= 5).sum())
signal  = np.concatenate([stable, chatter])

# --- Package the signal ---
sig = SignalData(t_analysis=t, signal_analysis=signal, fs=fs, path="synthetic")

# --- Configure the indicator ---
config = {
    "func": "Default",
    "params": {
        "rpm": 12_000.0,
        "ratio_sampling": 50.0,
        "N_seg": 2,
        "t_stable_total": 5.0,
        "alpha": 0.05,
        "beta": 0.05,
        "reset_on_H0": True,
        "cut_start_time": 0.0,
        "cut_end_time": 10.0,
    },
}

# --- Run + plot ---
result = run_maxent_sprt(sig, config)
plots_maxent_sprt(signal=sig, result=result, show_signal=True)
```

For a complete step-by-step walkthrough, see [Quick Start](guide/quickstart.md).

---

## Documentation Structure

| Section | Contents |
|---|---|
| **Theory** | MaxEnt principle, SPRT test, OPR sampling — what the math means |
| **User Guide** | Installation, Quick Start, API walkthrough, real data loading |
| **API Reference** | All classes and functions with parameters and return types |
| **Technical Documentation** | Detailed Sphinx pages for internal modules, full APIs, and developer-oriented implementation detail |

[Go To Technical Documentation](technical/index.html){ .md-button .md-button--primary }

---

## Recommended Reading Paths

### Guided Path

Short route to understand the method, run it, and find the main callable surfaces.

1. [OPR Sampling](theory/opr.md)
2. [Quick Start](guide/quickstart.md)
3. [Run & Plot](guide/run_and_plot.md)
4. [API Overview](api/index.md)

### Deep Dive Path

Longer route for implementation details, internal structure, and full generated APIs.

1. [Technical Overview](technical/overview.html)
2. [Internal Architecture](technical/architecture.html)
3. [Developer Notes](technical/developer_notes.html)
4. [Full AutoAPI Reference](technical/autoapi/index.html)

