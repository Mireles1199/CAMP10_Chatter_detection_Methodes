"""
debug.py
========
Centralised debug / logging manager for the effective-window framework.

Quick-start
-----------
    # ── Debug levels ─────────────────────────────────────────────────────────
    #  0  OFF      — no output at all (production runs)
    #  1  INFO     — key events: T_des computed, parameter resolved, skip/ok
    #  2  VERBOSE  — per-indicator detail: raw value, rounded value, ΔT_w,
    #                  constraint check results
    #  3  DEBUG    — everything above + triggers all debug plots automatically
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Usage:
    #   dbg = DebugManager(level=0)   ← silent production
    #   dbg = DebugManager(level=1)   ← light monitoring
    #   dbg = DebugManager(level=2)   ← development detail
    #   dbg = DebugManager(level=3)   ← full debug + plots
    # ─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

# ── Custom level between DEBUG(10) and INFO(20) ───────────────────────────────
_VERBOSE_LEVEL = 15
logging.addLevelName(_VERBOSE_LEVEL, "VERBOSE")


def _verbose(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(_VERBOSE_LEVEL):
        self._log(_VERBOSE_LEVEL, message, args, **kwargs)


if not hasattr(logging.Logger, "verbose"):
    logging.Logger.verbose = _verbose  # type: ignore[attr-defined]

if not hasattr(logging.Logger, "info_plus"):
    logging.Logger.info_plus = _verbose  # type: ignore[attr-defined]


class DebugManager:
    """
    Centralised debug and logging manager for the effective-window framework.

    Replaces scattered ``print`` statements and ``if debug:`` guards with a
    single, levelled logger that also controls whether debug plots are emitted.

    Parameters
    ----------
    level : int
        Verbosity level (0–3).  See module docstring for the level table.
    name : str
        Logger name (visible in log output).  Defaults to ``"effective_window"``.
    """

    # ── level → Python logging level mapping ─────────────────────────────────
    _LEVEL_MAP = {
        0: logging.WARNING,    # OFF  — only warnings and errors pass through
        1: logging.INFO,       # INFO
        2: _VERBOSE_LEVEL,     # VERBOSE
        3: logging.DEBUG,      # DEBUG
    }

    def __init__(self, level: int = 0, name: str = "effective_window") -> None:
        self.level = max(0, min(level, 3))
        self._logger = logging.getLogger(name)
        self._configure_logger()

    # ── internal setup ────────────────────────────────────────────────────────

    def _configure_logger(self) -> None:
        log_level = self._LEVEL_MAP.get(self.level, logging.WARNING)
        self._logger.setLevel(log_level)

        # Avoid adding duplicate handlers when called multiple times
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
            self._logger.propagate = False

    # ── public logging API ────────────────────────────────────────────────────

    def log(self, message: str, level: int = 1) -> None:
        """
        Emit *message* if the manager's level >= *level*.

        Parameters
        ----------
        message : str
            Text to log.
        level : int
            Minimum debug level required to show this message (1–3).
        """
        if self.level < level:
            return
        py_level = self._LEVEL_MAP.get(level, logging.INFO)
        self._logger.log(py_level, message)

    def log_T_des(self, T_des: float, window_spec: Any) -> None:
        """Log the computed T_des value (level 1)."""
        self.log(
            f"T_des = {T_des*1000:.3f} ms  "
            f"[basis={window_spec.basis.value}, n_cycles={window_spec.n_cycles}]",
            level=1,
        )

    def log_resolution(
        self,
        indicator_id: str,
        solved_var: str,
        raw_value: float,
        rounded_value: float,
        T_w_actual: float,
        T_des: float,
    ) -> None:
        """Log resolution result for one indicator (level 2)."""
        delta_ms = (T_w_actual - T_des) * 1000.0
        self.log(
            f"[{indicator_id}] resolve '{solved_var}': "
            f"raw={raw_value:.6g} → rounded={rounded_value:.6g} | "
            f"T_w={T_w_actual*1000:.3f} ms  T_des={T_des*1000:.3f} ms  "
            f"ΔT_w={delta_ms:+.3f} ms",
            level=2,
        )

    def log_constraint(self, indicator_id: str, report: Any) -> None:
        """Log constraint check outcome (level 2)."""
        status = "PASS" if report.passed else f"FAIL [level={report.level_failed}]"
        msg = f"[{indicator_id}] constraints: {status}"
        if not report.passed:
            msg += f" — {report.message}"
        self.log(msg, level=2)

    def log_run(self, indicator_id: str, skipped: bool, reason: str = "") -> None:
        """Log whether an indicator was run or skipped (level 1)."""
        if skipped:
            self.log(f"[{indicator_id}] SKIPPED — {reason}", level=1)
        else:
            self.log(f"[{indicator_id}] OK — run completed", level=1)

    def log_warning(self, message: str) -> None:
        """Always emit a warning regardless of debug level."""
        self._logger.warning(message)

    def log_error(self, message: str) -> None:
        """Always emit an error regardless of debug level."""
        self._logger.error(message)

    # ── plot control ──────────────────────────────────────────────────────────

    @property
    def show_debug_plots(self) -> bool:
        """True when debug level is 3 — triggers automatic debug plot generation."""
        return self.level >= 3
