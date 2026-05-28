"""Centralized debug manager for the green_integral indicator."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


class DebugManager:
    """Centralized debug controller.

    debug_level
    -----------
    0 = off   — no debug output
    1 = minimal — key events (window counts, phase transitions)
    2 = full  — per-window details, all signal plots
    """

    def __init__(
        self,
        debug_level: int = 0,
        window_range: Tuple[int, Optional[int]] = (0, None),
        save_figures: bool = False,
    ) -> None:
        self.debug_level = debug_level
        self.window_min: int = window_range[0]
        self.window_max: Optional[int] = window_range[1]
        self.save_figures: bool = save_figures
        self._logger = logging.getLogger("green_integral.debug")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def log(self, msg: str, level: int = 1) -> None:
        """Emit *msg* when ``debug_level >= level``."""
        if self.debug_level >= level:
            if level <= 1:
                self._logger.info(msg)
            else:
                self._logger.debug(msg)

    def log_window_progress(self, num_window: int, total: Optional[int] = None) -> None:
        """Log window progress every 100 windows (requires debug_level >= 1)."""
        if self.debug_level >= 1 and (num_window + 1) % 100 == 0:
            suffix = f"/{total}" if total is not None else ""
            self._logger.info("Window %d%s processed", num_window + 1, suffix)

    # ------------------------------------------------------------------
    # Window range check
    # ------------------------------------------------------------------
    def is_window_in_debug_range(self, num_window: int) -> bool:
        """Return ``True`` only when ``debug_level >= 2`` and window index is within range."""
        if self.debug_level < 2:
            return False
        if self.window_max is None:
            return num_window >= self.window_min
        return self.window_min <= num_window <= self.window_max

    # ------------------------------------------------------------------
    # Quick plots (only when debug_level >= 2)
    # ------------------------------------------------------------------
    def plot_signal(
        self,
        t: np.ndarray,
        y: np.ndarray,
        title: str = "Signal",
        xlabel: str = "Time (s)",
        ylabel: str = "Amplitude",
    ) -> None:
        """Quick single-axis plot (debug_level >= 2 only)."""
        if self.debug_level < 2:
            return
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, y, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.4)
        fig.tight_layout()
        if self.save_figures:
            fig.savefig(f"debug_{title.replace(' ', '_')}.png", bbox_inches="tight")
        plt.show()

    def plot_windows_context(
        self,
        i: int,
        data_window: dict,
        all_windows: list,
        ax=None,
        title: Optional[str] = None,
    ) -> None:
        """Overlay current, past, and future windows for visual context."""
        if self.debug_level < 2:
            return

        def _plot_one(w: dict, color: str, name: str) -> None:
            t = np.asarray(w["exp_fit_times"])
            v = np.asarray(w["exp_fit_values"])
            if t.size == 0:
                return
            ax.axvline(t[0], color=color, linestyle="-", linewidth=2, label=f"{name} Start")
            ax.axvline(t[-1], color=color, linestyle="--", linewidth=2, label=f"{name} End")
            ax.scatter(t, v, color=color, s=75, marker="o", label=f"{name} Points")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        _plot_one(data_window, "red", "Actual")
        if i - 1 >= 0:
            _plot_one(all_windows[i - 1], "green", "Past")
        if i + 1 < len(all_windows):
            _plot_one(all_windows[i + 1], "blue", "Future")
        if i + 2 < len(all_windows):
            _plot_one(all_windows[i + 2], "orange", "Far Future")

        ax.set_title(title or f"Debug i={i}")
        ax.grid(True, lw=0.4)

        handles, labels = ax.get_legend_handles_labels()
        seen: dict = {}
        for h, lbl in zip(handles, labels):
            if lbl not in seen:
                seen[lbl] = h
        ax.legend(seen.values(), seen.keys(), loc="best")
