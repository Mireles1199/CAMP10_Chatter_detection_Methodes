"""
effective_window
================
Phase 1 — Common Observation Window Framework for chatter detection.

Theory
------
Each indicator is given the same desired window length T_des [s]:

  Modal basis  :  T_des = n_m / f_modal
  Revolution   :  T_des = n_r * T_rev  = n_r * 60 / rpm

From T_des each indicator's internal parameter set is algebraically resolved:
  RMS-CV   — solve one of {N, ρ, n_max}  from T_w = N/fs * [1+(n_max-1)(1-ρ)]
  MaxEnt   — solve N_seg from T_w = N_seg * T_rev
  SST-SVD  — solve one of {n_A, w, h_ratio} from T_w = w + (n_A-1)*h  [ms]

Public API
----------
  SignalData           — signal container (all channels + meta)
  WindowBasis          — MODAL / REVOLUTION
  RoundingPolicy       — FLOOR / CEIL / ROUND / NONE
  WindowSpec           — T_des specification (basis, n_cycles, f_modal, rpm)
  ParameterResolutionConfig  — which variable to solve and rounding policy
  IndicatorWindowConfig      — one indicator's full configuration
  RunnerConfig               — top-level runner configuration
  WindowRunner               — main orchestrator; call .run(signal, config)
  WindowResult               — result container for all indicators
  IndicatorReport            — per-indicator result

Quick start
-----------
>>> from effective_window import (
...     SignalData, WindowSpec, WindowBasis, RoundingPolicy,
...     ParameterResolutionConfig, IndicatorWindowConfig, RunnerConfig,
...     WindowRunner,
... )
>>> spec = WindowSpec(basis=WindowBasis.REVOLUTION, n_cycles=5, rpm=12_000)
>>> rms_res = ParameterResolutionConfig(
...     solved_var="N",
...     fixed_vars={"rho": 0.50, "n_max": 8},
...     rounding=RoundingPolicy.FLOOR,
... )
>>> cfg = RunnerConfig(
...     window_spec=spec,
...     indicators=[
...         IndicatorWindowConfig("rms_cv", base_params={}, resolution=rms_res),
...     ],
...     show_plots=False,
...     debug_level=1,
... )
>>> signal = SignalData(t_analysis=t, signal_analysis=v, fs=fs, path="demo")
>>> result = WindowRunner().run(signal, cfg)
>>> print(result.summary())
"""

from .signal_data import SignalData

from .config import (
    WindowBasis,
    RoundingPolicy,
    WindowSpec,
    ParameterResolutionConfig,
    IndicatorWindowConfig,
    RunnerConfig,
)

from .runner import WindowRunner, WindowResult, IndicatorReport

from .debug import DebugManager

from . import resolvers, constraints, adapters

__all__ = [
    # Signal
    "SignalData",
    # Config
    "WindowBasis",
    "RoundingPolicy",
    "WindowSpec",
    "ParameterResolutionConfig",
    "IndicatorWindowConfig",
    "RunnerConfig",
    # Runner + results
    "WindowRunner",
    "WindowResult",
    "IndicatorReport",
    # Utilities
    "DebugManager",
    # Sub-modules
    "resolvers",
    "constraints",
    "adapters",
]
