"""
runner.py
=========
Main orchestrator for the effective-window framework.

``WindowRunner.run(signal, config)`` applies the full Phase 1 pipeline:

    For each indicator in config.indicators:
        1. Compute T_des from WindowSpec
        2. Resolve internal parameters algebraically (WindowResolver)
        3. Apply rounding policy
        4. Compute T_w_actual and ΔT_w
        5. Check admissibility constraints (ConstraintChecker)
        6. Skip or proceed based on strict_constraints flag
        7. Build INDICATOR_CONFIG via adapter
        8. Run the indicator via adapter
        9. Collect IndicatorReport

Returns a WindowResult containing all IndicatorReports and T_des.

Extension to Phase 2
---------------------
Phase 2 can wrap ``WindowRunner`` (or subclass it) to:
  - iterate over multiple T_des values (or n_cycles grids)
  - collect WindowResult objects and apply multi-objective optimisation
  - access the raw ResolverResult.raw_value floats via IndicatorReport
    for continuous-space optimisation
"""

from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adapters import get_adapter
from .config import IndicatorWindowConfig, RunnerConfig
from .constraints import ConstraintReport, get_checker
from .debug import DebugManager
from .resolvers import ResolverResult, get_resolver
from .signal_data import SignalData


# ──────────────────────────────────────────────────────────────────────────────
# Output containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IndicatorReport:
    """
    Result of running one indicator through the effective-window pipeline.

    Attributes
    ----------
    indicator_id : str
        Identifier of the indicator (e.g. ``"RMS_CV"``).
    T_des : float
        Target effective window [s] that was imposed.
    resolved_params : dict
        Parameters after resolution and rounding (fixed + solved).
    T_w_actual : float or None
        Actual effective window [s] computed with resolved parameters.
        None if resolution failed before a valid result was produced.
    delta_T_w : float or None
        T_w_actual − T_des [s].  None if T_w_actual is None.
    constraint_report : ConstraintReport or None
        Full constraint check outcome.  None if constraints were not run.
    indicator_config : dict or None
        The final INDICATOR_CONFIG dict passed to the runner.
    result : IndicatorResult or None
        Library result object.  None if the run was skipped or failed.
    skipped : bool
        True if the indicator was not run (constraint failure or exception).
    skip_reason : str
        Human-readable reason for skipping (empty when skipped=False).
    resolver_result : ResolverResult or None
        Full resolver output (raw value, rounded value, etc.) for traceability.
    """

    indicator_id: str
    T_des: float
    resolved_params: Dict[str, Any] = field(default_factory=dict)
    T_w_actual: Optional[float] = None
    delta_T_w: Optional[float] = None
    constraint_report: Optional[ConstraintReport] = None
    indicator_config: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    skipped: bool = False
    skip_reason: str = ""
    resolver_result: Optional[ResolverResult] = None


@dataclass
class WindowResult:
    """
    Aggregated output of one :class:`WindowRunner` invocation.

    Attributes
    ----------
    T_des : float
        Common effective decision window [s] imposed on all indicators.
    window_spec : WindowSpec
        The WindowSpec that produced T_des.
    reports : list of IndicatorReport
        One report per indicator registered in the RunnerConfig.
    """

    T_des: float
    window_spec: Any          # WindowSpec (avoid circular import)
    reports: List[IndicatorReport] = field(default_factory=list)

    # ── convenience accessors ─────────────────────────────────────────────────

    def get(self, indicator_id: str) -> Optional[IndicatorReport]:
        """Return the report for *indicator_id*, or None if not found."""
        for r in self.reports:
            if r.indicator_id == indicator_id:
                return r
        return None

    def successful(self) -> List[IndicatorReport]:
        """Return only the reports where the indicator ran successfully."""
        return [r for r in self.reports if not r.skipped and r.result is not None]

    def skipped(self) -> List[IndicatorReport]:
        """Return only the skipped reports."""
        return [r for r in self.reports if r.skipped]

    def summary(self) -> str:
        """One-line summary: T_des + per-indicator status."""
        lines = [f"T_des = {self.T_des*1000:.2f} ms"]
        for r in self.reports:
            status = "SKIP" if r.skipped else "OK  "
            dT = f"ΔT_w = {r.delta_T_w*1000:+.2f} ms" if r.delta_T_w is not None else "ΔT_w = n/a"
            lines.append(f"  [{status}] {r.indicator_id:15s}  {dT}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

class WindowRunner:
    """
    Orchestrates the effective-window Phase 1 pipeline across all indicators.

    Usage
    -----
    >>> runner = WindowRunner()
    >>> result = runner.run(signal, config)
    >>> print(result.summary())

    Phase 2 integration
    --------------------
    Phase 2 optimisation objects can call :meth:`run` repeatedly with
    different ``RunnerConfig`` objects (varying ``n_cycles`` or fixed
    parameter values) and collect ``WindowResult`` objects for
    multi-objective evaluation.  The ``resolver_result.raw_value`` fields
    in each ``IndicatorReport`` expose the continuous float values needed
    by continuous-space optimisers.
    """

    def run(
        self,
        signal: SignalData,
        config: RunnerConfig,
    ) -> WindowResult:
        """
        Execute the full effective-window pipeline.

        Parameters
        ----------
        signal : SignalData
            Input signal (framework-level container).
        config : RunnerConfig
            Top-level configuration: WindowSpec + per-indicator configs.

        Returns
        -------
        WindowResult
        """
        dbg = DebugManager(level=config.debug_level)

        # ── 0. Compute T_des ─────────────────────────────────────────────────
        T_des = config.window_spec.compute_T_des()
        dbg.log_T_des(T_des, config.window_spec)

        reports: List[IndicatorReport] = []

        for ind_cfg in config.indicators:
            report = self._run_one(signal, ind_cfg, T_des, config, dbg)
            reports.append(report)

        result = WindowResult(
            T_des=T_des,
            window_spec=config.window_spec,
            reports=reports,
        )

        # ── optional plots ───────────────────────────────────────────────────
        if config.show_plots:
            self._generate_plots(signal, result, dbg)

        dbg.log(f"\n{result.summary()}", level=1)
        return result

    # ── per-indicator pipeline ────────────────────────────────────────────────

    def _run_one(
        self,
        signal: SignalData,
        ind_cfg: IndicatorWindowConfig,
        T_des: float,
        runner_cfg: RunnerConfig,
        dbg: DebugManager,
    ) -> IndicatorReport:
        indicator_id = ind_cfg.indicator_id
        report = IndicatorReport(indicator_id=indicator_id, T_des=T_des)

        # ── 1. Resolve parameters ─────────────────────────────────────────────
        try:
            resolver = get_resolver(indicator_id)
            res_result = resolver.resolve(
                T_des=T_des,
                fs=signal.fs,
                config=ind_cfg.resolution,
            )
        except Exception as exc:
            reason = f"Resolution failed: {exc}"
            dbg.log_warning(f"[{indicator_id}] {reason}")
            report.skipped = True
            report.skip_reason = reason
            dbg.log_run(indicator_id, skipped=True, reason=reason)
            return report

        report.resolver_result = res_result
        report.resolved_params = res_result.resolved_params
        report.T_w_actual = res_result.T_w_actual
        report.delta_T_w  = res_result.delta_T_w

        dbg.log_resolution(
            indicator_id,
            res_result.solved_var,
            res_result.raw_value,
            res_result.rounded_value,
            res_result.T_w_actual,
            T_des,
        )

        # ── 2. Constraint checks ──────────────────────────────────────────────
        try:
            checker = get_checker(indicator_id)
            cr = checker.check(res_result.resolved_params, ind_cfg.resolution)
        except Exception as exc:
            dbg.log_warning(
                f"[{indicator_id}] ConstraintChecker raised: {exc}. Skipping."
            )
            report.skipped = True
            report.skip_reason = f"Constraint checker error: {exc}"
            return report

        report.constraint_report = cr
        dbg.log_constraint(indicator_id, cr)

        # Advisory for SST h_ratio outside library preference
        if indicator_id == "SST_SVD" and cr.details.get("_runner_hop_advisory"):
            h_ratio = res_result.resolved_params.get("h_ratio", "?")
            dbg.log_warning(
                f"[SST_SVD] h_ratio = {h_ratio:.4f} is outside the library's "
                f"preferred range [0.25, 0.50]. Pipeline called directly; "
                "theory constraint 0 < h ≤ w is satisfied."
            )

        if not cr.passed and ind_cfg.strict_constraints:
            reason = f"Constraint failure (level={cr.level_failed}): {cr.message}"
            report.skipped = True
            report.skip_reason = reason
            dbg.log_run(indicator_id, skipped=True, reason=reason)
            return report

        # ── 3. Build INDICATOR_CONFIG ─────────────────────────────────────────
        try:
            adapter = get_adapter(indicator_id)
            ind_config = adapter.build_config(
                base_params=copy.deepcopy(ind_cfg.base_params),
                resolved_params=res_result.resolved_params,
            )
            report.indicator_config = ind_config
        except Exception as exc:
            reason = f"Config build failed: {exc}"
            dbg.log_warning(f"[{indicator_id}] {reason}")
            report.skipped = True
            report.skip_reason = reason
            return report

        # ── 4. Run indicator ──────────────────────────────────────────────────
        try:
            ind_result = adapter.run(signal, ind_config)
            report.result = ind_result
        except Exception as exc:
            reason = f"Indicator run failed: {exc}"
            dbg.log_warning(
                f"[{indicator_id}] {reason}\n{traceback.format_exc()}"
            )
            report.skipped = True
            report.skip_reason = reason
            dbg.log_run(indicator_id, skipped=True, reason=reason)
            return report

        dbg.log_run(indicator_id, skipped=False)

        # ── 5. Debug plots ────────────────────────────────────────────────────
        if runner_cfg.debug_level >= 3:
            self._generate_debug_plots(signal, report)

        return report

    # ── plot dispatch ─────────────────────────────────────────────────────────

    def _generate_plots(
        self,
        signal: SignalData,
        result: WindowResult,
        dbg: DebugManager,
    ) -> None:
        """Generate production plots for all indicators."""
        try:
            from .plotting import (
                plot_indicator_overview,
                plot_window_geometry,
                plot_all_indicators,
                plot_parameter_table,
            )
            for report in result.successful():
                plot_indicator_overview(signal, report)
                plot_window_geometry(report)
            if len(result.successful()) > 1:
                plot_all_indicators(signal, result)
            plot_parameter_table(result)
        except Exception as exc:
            dbg.log_warning(f"Plotting failed: {exc}")

    def _generate_debug_plots(
        self,
        signal: SignalData,
        report: IndicatorReport,
    ) -> None:
        """Generate debug plots for one indicator."""
        try:
            from .plotting import (
                plot_resolution_steps,
                plot_constraint_report,
            )
            if report.resolver_result is not None:
                plot_resolution_steps(report)
            if report.constraint_report is not None:
                plot_constraint_report(report)
        except Exception as exc:
            import logging
            logging.getLogger("effective_window").warning(
                f"Debug plot failed for {report.indicator_id}: {exc}"
            )
