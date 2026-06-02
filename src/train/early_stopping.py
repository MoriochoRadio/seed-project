"""Framework-agnostic early-stopping helper.

Pure-NumPy module — does not depend on torch — so it can be unit-tested in
environments without a deep-learning framework installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EarlyStopping:
    """Track a monitored metric and decide when to stop training.

    Parameters
    ----------
    patience : int
        # of epochs with no improvement after which training stops.
    min_delta : float
        Minimum change in monitored metric counted as improvement.
    mode : "max" | "min"
        Whether the monitored metric is to be maximised (e.g. F1, accuracy)
        or minimised (e.g. validation loss, ECE).
    """

    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "max"

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self._best: float = -math.inf if self.mode == "max" else math.inf
        self._counter: int = 0
        self._best_epoch: int = -1
        self.should_stop: bool = False

    @property
    def best(self) -> float:
        return self._best

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def num_bad_epochs(self) -> int:
        return self._counter

    def step(self, metric: float, epoch: int) -> bool:
        """Update with the latest metric. Returns True if it was an improvement."""
        if not np.isfinite(metric):
            self._counter += 1
            self.should_stop = self._counter >= self.patience
            return False

        if self.mode == "max":
            improved = metric > self._best + self.min_delta
        else:
            improved = metric < self._best - self.min_delta

        if improved:
            self._best = float(metric)
            self._best_epoch = epoch
            self._counter = 0
        else:
            self._counter += 1
        self.should_stop = self._counter >= self.patience
        return improved
