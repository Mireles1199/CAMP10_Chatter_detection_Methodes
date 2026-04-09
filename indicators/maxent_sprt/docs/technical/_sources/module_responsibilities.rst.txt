Module Responsibilities
=======================

This page clarifies ownership boundaries so contributors can quickly identify where
new features or fixes should be implemented.

Responsibility map
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Module
     - Responsibility
     - Typical changes
   * - ``MaxEnt_SPRT.lib.runner``
     - Top-level orchestration entrypoint and configuration-driven execution.
     - Add new runtime knobs, adjust orchestration sequence, update integration behavior.
   * - ``MaxEnt_SPRT.lib.detector``
     - End-to-end detector object combining offline fit and online scoring interfaces.
     - Add new detector methods, expose diagnostics, evolve public detector API.
   * - ``MaxEnt_SPRT.lib.sprt``
     - Sequential hypothesis engine, thresholds, and result container.
     - Change decision logic, threshold policy, or cumulative-state semantics.
   * - ``MaxEnt_SPRT.lib.llr``
     - LLR models connecting probabilistic models to SPRT scoring.
     - Implement alternative LLR forms or custom model adapters.
   * - ``MaxEnt_SPRT.lib.entropy``
     - Segment-to-indicator conversion via entropy estimators.
     - Add entropy estimators, tune preprocessing assumptions, validate segment behavior.
   * - ``MaxEnt_SPRT.models.maxent``
     - Construction of stable/chatter model pair from labeled samples.
     - Add fit safeguards, expose model diagnostics, support new model families.
   * - ``MaxEnt_SPRT.models.prob``
     - Core Gaussian PDF object and entropy/log-density operations.
     - Extend probabilistic primitives, improve numerical stability.
   * - ``MaxEnt_SPRT.utils.opr``
     - OPR sampling and fixed-window segmentation helpers.
     - Add sampling modes, improve edge-case handling, optimize segmentation.
   * - ``MaxEnt_SPRT.utils.hdf5_utils``
     - HDF5 reading and dataset navigation utilities.
     - Support new HDF5 schemas or metadata conventions.
   * - ``MaxEnt_SPRT.utils.types``
     - Shared dataclasses exchanged across package layers.
     - Add fields to result containers, refine typing contracts.
   * - ``MaxEnt_SPRT.viz.maxent_sprt_plots``
     - Diagnostic plotting and publication-oriented figure output.
     - Add plot panels, styling modes, or annotation utilities.

Extension rules of thumb
------------------------

- If a change modifies *what gets computed* end-to-end, start in ``lib.runner`` or ``lib.detector``.
- If a change modifies *how evidence accumulates*, start in ``lib.llr`` or ``lib.sprt``.
- If a change modifies *feature extraction*, start in ``lib.entropy`` and ``utils.opr``.
- If a change modifies *model assumptions*, start in ``models.maxent`` and ``models.prob``.
- If a change modifies *output format or visualization*, start in ``utils.types`` and ``viz.maxent_sprt_plots``.
