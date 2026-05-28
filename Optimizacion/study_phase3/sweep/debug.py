"""
sweep/debug.py
==============
Centralised debug / logging manager for the sweep study framework.

Copied from effective_window/debug.py and adapted for sweep study usage.

Quick-start
-----------
    # ── Debug levels ─────────────────────────────────────────────────────────
    #  0  OFF      — no output at all (production runs)
    #  1  INFO     — key events: combo started, run ok/fail, K_total grid step
    #  2  VERBOSE  — per-combo detail: config dict, metric values
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
    Centralised debug and logging manager for the sweep study framework.

    Parameters
    ----------
    level : int
        Verbosity level (0–3).  See module docstring for the level table.
    name : str
        Logger name (visible in log output).  Defaults to ``"sweep"``.
    """

    # ── level → Python logging level mapping ─────────────────────────────────
    _LEVEL_MAP = {
        0: logging.WARNING,    # OFF  — only warnings and errors pass through
        1: logging.INFO,       # INFO
        2: _VERBOSE_LEVEL,     # VERBOSE
        3: logging.DEBUG,      # DEBUG
    }

    def __init__(self, level: int = 0, name: str = "sweep") -> None:
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

    def log_combo(
        self,
        run_id: str,
        indicator: str,
        K_total: int,
        N_win: Optional[int],
        step: int,
        n_accum: Optional[int],
    ) -> None:
        """Log the start of a combo run (level 2)."""
        nw = str(N_win)    if N_win    is not None else "   -"
        na = str(n_accum) if n_accum is not None else "   -"
        self.log(
            f">> [{indicator:<10s}]  K={K_total:>3d}  win={nw:>4s}  step={step:>3d}  acc={na:>5s}   id={run_id}",
            level=2,
        )

    def log_run_ok(self, run_id: str, t_d_first: float, t_d_first_true: float, delta_t_d: float, N_fa: int) -> None:
        """Log a successful run result (level 2)."""
        import math
        td1 = f"{t_d_first* 1e3:>10.1f} ms" if not math.isnan(t_d_first)      else "n/a"
        td2 = f"{t_d_first_true* 1e3:>10.1f} ms" if not math.isnan(t_d_first_true) else "n/a"
        self.log(
            f"   OK  t_d={td1}  t_d1={td2}  Dt_d={delta_t_d * 1e3:>+8.1f} ms  N_fa={N_fa}",
            level=2,
        )

    def log_run_fail(self, run_id: str, error: str) -> None:
        """Always emit a warning on failed run."""
        self._logger.warning("FAIL run_id=%s  error=%s", run_id, error)

    def log_k_step(self, K_total: int, n_combos: int) -> None:
        """Log entry into a new K_total grid point (level 2)."""
        self.log(f"\n", level=2)
        self.log(f"── K_total={K_total}  ({n_combos} combos) ──", level=2)

    def log_warning(self, message: str) -> None:
        """Always emit a warning regardless of debug level."""
        self._logger.warning(message)

    def log_error(self, message: str) -> None:
        """Always emit an error regardless of debug level."""
        self._logger.error(message)

    def log_detail(self, message: str) -> None:
        """Log a verbose detail message (level 2)."""
        self.log(message, level=2)

    # ── plot control ──────────────────────────────────────────────────────────

    @property
    def show_debug_plots(self) -> bool:
        """True when debug level is 3 — triggers automatic debug plot generation."""
        return self.level >= 3
