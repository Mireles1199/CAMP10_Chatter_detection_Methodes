Execution Flow
==============

This page documents the concrete runtime path from raw signal data to final chatter decision.

Pipeline sequence
-----------------

1. **Input normalization**

   ``run_maxent_sprt`` validates the input payload and resolves detector settings from
   ``INDICATOR_CONFIG``.

   - API: `MaxEnt_SPRT.lib.runner <autoapi/MaxEnt_SPRT/lib/runner/index.html>`_

2. **OPR sampling and segmentation**

   Raw vibration signals are converted to once-per-revolution samples and split into
   fixed-size windows used by downstream entropy extraction.

   - API: `MaxEnt_SPRT.utils.opr <autoapi/MaxEnt_SPRT/utils/opr/index.html>`_

3. **Entropy extraction per segment**

   Each segment is converted into a scalar entropy indicator sequence ``H_seq``.

   - API: `MaxEnt_SPRT.lib.entropy <autoapi/MaxEnt_SPRT/lib/entropy/index.html>`_

4. **Offline model fitting (stable/chatter)**

   The detector fits Gaussian MaxEnt models ``p0`` and ``p1`` from labeled stable/chatter
   samples.

   - API: `MaxEnt_SPRT.models.maxent <autoapi/MaxEnt_SPRT/models/maxent/index.html>`_
   - API: `MaxEnt_SPRT.models.prob <autoapi/MaxEnt_SPRT/models/prob/index.html>`_

5. **LLR scoring**

   The per-segment log-likelihood ratio is evaluated as ``log(p1(H) / p0(H))``.

   - API: `MaxEnt_SPRT.lib.llr <autoapi/MaxEnt_SPRT/lib/llr/index.html>`_

6. **Sequential decision (SPRT)**

   The cumulative statistic crosses threshold ``a`` (stable) or ``b`` (chatter),
   producing a final state and decision index.

   - API: `MaxEnt_SPRT.lib.sprt <autoapi/MaxEnt_SPRT/lib/sprt/index.html>`_

7. **Result packaging and plotting**

   The final result is stored in ``IndicatorResult`` and can be visualized with the
   plotting utilities.

   - API: `MaxEnt_SPRT.utils.types <autoapi/MaxEnt_SPRT/utils/types/index.html>`_
   - API: `MaxEnt_SPRT.viz.maxent_sprt_plots <autoapi/MaxEnt_SPRT/viz/maxent_sprt_plots/index.html>`_

Developer checkpoints
---------------------

When debugging or extending the pipeline, these checkpoints are the fastest sanity checks:

- OPR output lengths are consistent with expected revolutions
- ``H_seq`` has no NaN/Inf values
- ``p0`` and ``p1`` are fitted from correctly labeled windows
- SPRT thresholds from ``alpha``/``beta`` are within expected ranges
- Decision index maps to a valid ``t_mid`` timestamp
