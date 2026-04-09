# Detector & SPRT

Classes responsible for sequential hypothesis testing and chatter decision logic.
This layer consumes the MaxEnt probability models and produces a binary chatter/stable decision over time.

[← Workflow](workflow.md){ .md-button } [Models & Entropy →](models_entropy.md){ .md-button }

---

## `MaxEntSPRTDetector` — End-to-end detector

The central object when you need more control than `run_maxent_sprt` provides.
Exposes separate offline fitting and online detection steps.

```python
detector = MaxEntSPRTDetector(config=MaxEntSPRTConfig(alpha=0.01, beta=0.01))

# --- Offline: fit on labelled reference data ---
detector.fit_offline_from_signals(
    y_free, t_free, y_chat, t_chat,
    rpm=rpm, ratio_sampling=ratio_sampling, N_seg=N_seg,
)

# --- Online: score a new signal ---
sprt_result, H_seq, t_mid = detector.detect_online_from_signal(
    y_online, t_online,
    rpm=rpm, ratio_sampling=ratio_sampling, N_seg=N_seg,
)
```

| Method | Purpose |
|---|---|
| `fit_offline_from_signals(...)` | Train from raw vibration arrays + RPM |
| `fit_offline_from_opr(...)` | Train from pre-resampled OPR data |
| `detect_online_from_signal(...)` | Score a raw signal → `(SPRTResult, H_seq, t_mid)` |
| `detect_from_H_seq(H_seq)` | Score a precomputed entropy sequence → `SPRTResult` |

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/lib/detector/index.html){ .md-button .md-button--primary }

---

## `MaxEntSPRTConfig` — Detector configuration

High-level configuration shared between offline and online stages.

```python
MaxEntSPRTConfig(
    alpha: float = 0.01,      # false-alarm probability
    beta: float  = 0.01,      # missed-detection probability
    reset_on_H0: bool = True, # reset cumulative statistic on stable decision
)
```

---

## `SPRTResult` — Decision output

Dataclass returned by every detection call.

```python
SPRTResult(
    final_state: str,          # "chatter" | "free" | "indeterminado"
    decision_index: int,       # sample index at which a decision was reached (-1 if none)
    S_history: np.ndarray,     # cumulative LLR at each segment
    a: float,                  # lower threshold (H0)
    b: float,                  # upper threshold (H1 / chatter)
)
```

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/lib/sprt/index.html){ .md-button }

---

## `GaussianIndicatorLLR` — Log-likelihood ratio

Evaluates `llr(h_obs) = log(p1(h_obs) / p0(h_obs))` for a single entropy observation.
Used internally by the SPRT engine; useful for custom scoring loops.

```python
llr_model = GaussianIndicatorLLR(models=detector.models)
value = llr_model.llr(h_obs=3.2)
```

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/lib/llr/index.html){ .md-button }
