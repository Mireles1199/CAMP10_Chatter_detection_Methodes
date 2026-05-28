"""Crossing-index grouper for the green_integral indicator."""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


class CrossingGrouper:
    """Clusters crossing indices by proximity and selects one representative per group.

    Parameters
    ----------
    max_distance : int, optional
        Maximum index-distance between consecutive crossings to be placed in the
        same group.  Estimated automatically from the 25th-percentile of inter-
        crossing distances when ``None``.
    selection_strategy : {"first", "mean", "median", "center"}
        Rule for picking the representative index from each group.
    """

    def __init__(
        self,
        max_distance: Optional[int] = None,
        selection_strategy: Literal["first", "mean", "median", "center"] = "first",
    ) -> None:
        self.max_distance = max_distance
        self.selection_strategy = selection_strategy
        self._groups: List[List[int]] = []
        self.representatives: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cluster(
        self,
        crossing_indices: List[int],
        crossing_data: Optional[List[float]] = None,
    ) -> List[int]:
        """Group *crossing_indices* and return representative indices."""
        if len(crossing_indices) == 0:
            return []

        if self.max_distance is None:
            diffs = np.diff(crossing_indices)
            if len(diffs) == 0:
                self.max_distance = 1
            else:
                self.max_distance = max(1, int(np.percentile(diffs, 25)))

        self._groups = [[crossing_indices[0]]]
        for idx in crossing_indices[1:]:
            if idx - self._groups[-1][-1] <= self.max_distance:
                self._groups[-1].append(idx)
            else:
                self._groups.append([idx])

        self.representatives = []
        return self._select_representatives(self._groups)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _select_representatives(self, groups: List[List[int]]) -> List[int]:
        for group in groups:
            if self.selection_strategy == "first":
                self.representatives.append(group[0])
            elif self.selection_strategy == "mean":
                self.representatives.append(int(np.mean(group)))
            elif self.selection_strategy == "median":
                self.representatives.append(int(np.median(group)))
            elif self.selection_strategy == "center":
                center = np.mean(group)
                closest = min(group, key=lambda x: abs(x - center))
                self.representatives.append(closest)
            else:
                raise ValueError(f"Unknown selection_strategy: {self.selection_strategy!r}")
        return self.representatives

    # ------------------------------------------------------------------
    # Debug plot
    # ------------------------------------------------------------------
    def plot(
        self,
        original_data_t: Optional[np.ndarray] = None,
        original_data_v: Optional[np.ndarray] = None,
        original_data_crossing: Optional[np.ndarray] = None,
        velocity_used_crossing: Optional[np.ndarray] = None,
        num_window: Optional[int] = None,
    ) -> None:
        """Visualize grouped crossings and selected representatives."""
        if not self._groups:
            raise RuntimeError("Call cluster() before plot().")

        colors = colormaps.get_cmap("tab10").resampled(len(self._groups))

        if original_data_t is not None and original_data_crossing is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
            ax1.set_title(f"Grouped Crossings — window {num_window}")
            ax1.set_xlabel("Index")
            ax2.set_xlabel("Time (s)")
            ax3.set_xlabel("Time (s)")
            ax2.plot(original_data_t, original_data_v, color="blue", lw=1)
            ax2.axhline(0, color="gray", ls="--", lw=1.5, alpha=0.75)
            ax3.axhline(0, color="gray", ls="--", lw=1.5, alpha=0.75)
            if velocity_used_crossing is not None:
                ax3.plot(original_data_t, velocity_used_crossing, color="blue", lw=1)

            reps_set = set(self.representatives)
            for i, group in enumerate(self._groups):
                mask = np.isin(original_data_crossing[:, -1], group)
                cross_g = original_data_crossing[mask]
                mask_r = np.isin(original_data_crossing[:, -1], list(reps_set))
                cross_r = original_data_crossing[mask_r]
                for j in range(len(group) - 1):
                    ax1.axvline(group[j], color=colors(i), ls="--", lw=1, alpha=0.75)
                    if j < len(cross_g):
                        ax2.axvline(cross_g[j][0], color=colors(i), ls="--", lw=1, alpha=0.75)
                        ax3.axvline(cross_g[j][0], color=colors(i), ls="--", lw=1, alpha=0.75)
                if i < len(cross_r):
                    ax2.axvline(cross_r[i][0], color=colors(i), ls="-", lw=2, label=f"G{i+1}")
                    ax3.axvline(cross_r[i][0], color=colors(i), ls="-", lw=2, label=f"G{i+1}")
                ax1.axvline(self.representatives[i], color=colors(i), ls="-", lw=2, label=f"G{i+1}")
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            for i, group in enumerate(self._groups):
                for idx in group:
                    ax.axvline(idx, color=colors(i), ls="--", lw=1, alpha=0.75)
                ax.axvline(
                    self.representatives[i], color=colors(i), ls="-", lw=2, label=f"G{i+1}"
                )
            ax.set_title("Grouped Crossings and Representatives")
            ax.set_xlabel("Index")
            fig.tight_layout()
        plt.show()
