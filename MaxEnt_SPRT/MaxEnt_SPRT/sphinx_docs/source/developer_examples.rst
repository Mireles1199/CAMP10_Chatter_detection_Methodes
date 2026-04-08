Developer Examples
==================

These snippets focus on implementation-level usage patterns used while extending or
validating the package internals.

Minimal end-to-end call
-----------------------

.. code-block:: python

   from MaxEnt_SPRT import run_maxent_sprt, SignalData

   signal = SignalData(y=y_arr, t=t_arr)
   result = run_maxent_sprt(signal, INDICATOR_CONFIG)
   print(result.sprt_result.final_state, result.sprt_result.decision_index)

Manual detector workflow
------------------------

.. code-block:: python

   from MaxEnt_SPRT.lib.detector import MaxEntSPRTDetector, MaxEntSPRTConfig

   detector = MaxEntSPRTDetector(
       config=MaxEntSPRTConfig(alpha=0.01, beta=0.01, reset_on_H0=True)
   )

   detector.fit_offline_from_signals(
       y_free=y_free,
       t_free=t_free,
       y_chat=y_chat,
       t_chat=t_chat,
       rpm=rpm,
       ratio_sampling=ratio_sampling,
       N_seg=N_seg,
   )

   sprt_result, H_seq, t_mid = detector.detect_online_from_signal(
       y_online=y_online,
       t_online=t_online,
       rpm=rpm,
       ratio_sampling=ratio_sampling,
       N_seg=N_seg,
   )

Working with precomputed entropy
--------------------------------

.. code-block:: python

   # Useful when entropy is computed in another tool
   sprt_result = detector.detect_from_H_seq(H_seq)

   if sprt_result.final_state == "chatter":
       print("Chatter detected at segment", sprt_result.decision_index)

Diagnostic plotting for validation
----------------------------------

.. code-block:: python

   from MaxEnt_SPRT.viz.maxent_sprt_plots import plots_maxent_sprt

   fig = plots_maxent_sprt(
       signal=signal,
       result=result,
       show_signal=True,
       zoom_x=(0.0, 4.0),
       vlines=[2.1],
   )
   fig.savefig("diagnostic_detection.png", dpi=200)

Common extension pattern
------------------------

When adding a new indicator or estimator variant:

1. Implement and validate it in the corresponding ``lib`` or ``models`` module.
2. Plug it into ``MaxEntSPRTDetector`` while preserving existing return types.
3. Rebuild docs and verify AutoAPI output includes the new public surface.
4. Add one short usage example to keep technical docs actionable.

Writing math in Sphinx pages
----------------------------

Use the native Sphinx directive in ``.rst`` files:

.. code-block:: rst

    .. math::

        S_n = S_{n-1} + \Lambda_n

Rendered equation:

.. math::

    S_n = S_{n-1} + \Lambda_n
