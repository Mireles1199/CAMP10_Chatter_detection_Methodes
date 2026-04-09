# API Reference

Reference documentation organized by functional domain. Each section groups the modules that work together within the MaxEnt-SPRT pipeline.

[Quick Start](../guide/quickstart.md){ .md-button } [Developer Hub](../developer_hub.md){ .md-button }

---

## Domains

<div class="grid cards" markdown>

- :material-play-circle-outline: **Workflow**

    ---

    End-to-end execution pipeline. Main entry point for running the detector.

    [Open](workflow.md)

- :material-radar: **Detector & SPRT**

    ---

    Sequential decision logic, detector configuration, and LLR computation.

    [Open](detection.md)

- :material-function-variant: **Models & Entropy**

    ---

    Gaussian probabilistic models and per-segment entropy estimation.

    [Open](models_entropy.md)

- :material-database-outline: **Data & I/O**

    ---

    Common types, OPR sampling, segmentation, and HDF5 file reading.

    [Open](data_io.md)

- :material-chart-line: **Visualization**

    ---

    Diagnostic and result plotting functions for detection output.

    [Open](visualization.md)

</div>

---

## Reading Order

If you are exploring the API for the first time, follow this order:

1. [Data & I/O](data_io.md) — understand `SignalData` and `OPR` first.
2. [Workflow](workflow.md) — `run_maxent_sprt` is the top-level entry point.
3. [Detector & SPRT](detection.md) — internals of `MaxEntSPRTDetector` and `SPRT`.
4. [Models & Entropy](models_entropy.md) — the statistical foundations.
5. [Visualization](visualization.md) — diagnose and plot results.

