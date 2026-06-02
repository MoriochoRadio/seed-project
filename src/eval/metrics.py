"""Classification metrics — accuracy, F1, AUROC, Cohen's κ, ECE.

All metrics gracefully degrade when only a single class is present (early-stage
training, tiny test buckets) — instead of emitting sklearn warnings and NaNs,
we return sensible defaults (κ=0, AUROC=NaN) and pass explicit ``labels=`` so
the confusion-matrix shape is always correct.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1_macro: float
    auroc: float
    cohen_kappa: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "auroc": self.auroc,
            "cohen_kappa": self.cohen_kappa,
        }


def _all_labels(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> List[int]:
    """Best-effort enumeration of all class indices that could appear."""
    seen = set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())
    if y_prob is not None and y_prob.ndim == 2:
        seen.update(range(y_prob.shape[1]))
    return sorted(int(x) for x in seen)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> ClassificationMetrics:
    if len(y_true) == 0:
        return ClassificationMetrics(float("nan"), float("nan"), float("nan"), float("nan"))

    labels = _all_labels(y_true, y_pred, y_prob)
    n_unique_true = len(np.unique(y_true))

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))

    # Cohen's κ is undefined when only one class is present in y_true ∪ y_pred.
    if n_unique_true < 2 or len(np.unique(y_pred)) < 2:
        kap = 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                kap = float(cohen_kappa_score(y_true, y_pred, labels=labels))
                if not np.isfinite(kap):
                    kap = 0.0
            except ValueError:
                kap = 0.0

    # AUROC needs ≥2 classes in y_true and per-class probabilities.
    if y_prob is not None and n_unique_true > 1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auroc = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        labels=list(range(y_prob.shape[1])),
                    )
                )
        except (ValueError, IndexError):
            auroc = float("nan")
    else:
        auroc = float("nan")
    return ClassificationMetrics(acc, f1, auroc, kap)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE — averaged absolute confidence gap across equal-width probability bins."""
    if len(y_true) == 0:
        return float("nan")
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if in_bin.any():
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += abs(avg_conf - avg_acc) * in_bin.mean()
    return float(ece)
