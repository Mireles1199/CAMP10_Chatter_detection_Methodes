"""
plotting/__init__.py
====================
Public API of the effective_window.plotting sub-package.

runner.py imports these names directly:

  Production (show_plots=True):
    plot_indicator_overview   — per-indicator 2-panel signal+I(t) overview
    plot_window_geometry      — per-indicator window decomposition bars
    plot_all_indicators       — stacked I(t) for all successful indicators
    plot_parameter_table      — matplotlib table: params + T_w + ΔT_w

  Debug (debug_level >= 3):
    plot_resolution_steps     — raw→rounded + T_w vs T_des (per indicator)
    plot_constraint_report    — semáforo per constraint level (per indicator)

  Optional / manual:
    plot_delta_Tw_comparison  — |ΔT_w| bar chart across indicators
    plot_feasibility_summary  — heatmap: constraint pass/fail per indicator

  Style:
    configure_global_style    — sets global matplotlib rcParams (must be called once)
"""

from .style import configure_global_style  # noqa: F401

from .plot_combined import (  # noqa: F401
    plot_indicator_overview,
    plot_window_geometry,
    plot_all_indicators,
    plot_parameter_table,
    plot_resolution_steps,
    plot_constraint_report,
    plot_delta_Tw_comparison,
    plot_feasibility_summary,
)

# Per-indicator modules exposed for direct use or testing
from . import plot_rms_cv   # noqa: F401
from . import plot_maxent   # noqa: F401
from . import plot_sst_svd  # noqa: F401

__all__ = [
    "configure_global_style",
    # combined / dispatch
    "plot_indicator_overview",
    "plot_window_geometry",
    "plot_all_indicators",
    "plot_parameter_table",
    "plot_resolution_steps",
    "plot_constraint_report",
    "plot_delta_Tw_comparison",
    "plot_feasibility_summary",
    # per-indicator sub-modules
    "plot_rms_cv",
    "plot_maxent",
    "plot_sst_svd",
]
