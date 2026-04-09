# Workflow

End-to-end pipeline that orchestrates all stages — from raw signal to chatter decision output.
For most use cases, this is the only module you need to import directly.

[← API Overview](index.md){ .md-button } [Detector & SPRT →](detection.md){ .md-button }

---

## `run_maxent_sprt` — Main entry point

```python
from MaxEnt_SPRT import run_maxent_sprt

result = run_maxent_sprt(signal, INDICATOR_CONFIG)
```

Receives a [`SignalData`](data_io.md#signaldata) object and a configuration dictionary, runs
the full offline training + online scoring pipeline internally, and returns an
[`IndicatorResult`](data_io.md#indicatorresult) ready for plotting.

**Signature**

```python
run_maxent_sprt(
    signal: SignalData,
    INDICATOR_CONFIG: dict,
) -> IndicatorResult
```

All tuning parameters (segment count, error rates `alpha`/`beta`, RPM, etc.) are
passed via the `INDICATOR_CONFIG` dictionary — see the [Quick Start](../guide/quickstart.md)
and [INDICATOR_CONFIG](../guide/indicator_config.md) guides for the full key list.

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/lib/runner/index.html){ .md-button .md-button--primary }

---

## Package root exports

The top-level `MaxEnt_SPRT` package re-exports the main function and the most
used types so you can import from one place:

```python
from MaxEnt_SPRT import run_maxent_sprt, SignalData, IndicatorResult
```

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/index.html){ .md-button }
