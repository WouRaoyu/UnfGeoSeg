"""Evaluation metrics shared by all experiment scripts.

Covers the manuscript's reported quantities:

* classification quality: Accuracy, Precision, Recall, F1 (class-wise + macro),
  hit rate, false-alarm rate;
* localization: nearest-chainage error (held-out TFR), boundary/contact error
  along borehole trajectories (mean/median surface distance);
* reliability: mean softmax probability, entropy, variance (Table 6).

All functions operate on plain numpy arrays so they are independent of nnU-Net.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> Dict[str, int]:
    yt = y_true == cls
    yp = y_pred == cls
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    tn = int(np.sum(~yt & ~yp))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf1(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    hit_rate = recall
    false_alarm = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_rate": hit_rate,
        "false_alarm": false_alarm,
    }


def classwise_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Per-class metrics for foreground classes ``1..len(class_names)``."""
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names, start=1):
        out[name] = prf1(confusion_counts(y_true, y_pred, i))
    return out


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------
def nearest_chainage_error(
    pred_positions: np.ndarray, true_positions: np.ndarray
) -> float:
    """Mean nearest distance (along chainage) between predicted and true class
    occurrences. ``*_positions`` are 1-D chainage values; empty -> nan."""
    if len(pred_positions) == 0 or len(true_positions) == 0:
        return float("nan")
    d = np.abs(np.asarray(pred_positions)[:, None] - np.asarray(true_positions)[None, :])
    return float(d.min(axis=1).mean())


def boundary_error_1d(
    pred_labels: np.ndarray, true_labels: np.ndarray, spacing: float = 1.0
) -> Dict[str, float]:
    """Boundary/contact error along a 1-D borehole trajectory.

    Compares the class-transition positions of prediction vs. log and returns
    mean/median absolute boundary displacement (in physical units via
    ``spacing``)."""
    pred_b = np.where(np.diff(pred_labels) != 0)[0]
    true_b = np.where(np.diff(true_labels) != 0)[0]
    if len(pred_b) == 0 or len(true_b) == 0:
        return {"mean_boundary_error": float("nan"), "median_boundary_error": float("nan")}
    d = np.abs(pred_b[:, None] - true_b[None, :]).min(axis=1) * spacing
    return {
        "mean_boundary_error": float(np.mean(d)),
        "median_boundary_error": float(np.median(d)),
    }


# ---------------------------------------------------------------------------
# Reliability / uncertainty metrics (Table 6)
# ---------------------------------------------------------------------------
def reliability_metrics(
    proba: np.ndarray, mask: np.ndarray | None = None
) -> Dict[str, float]:
    """Mean softmax probability, entropy and variance over voxels.

    ``proba`` is (C, ...) per-voxel class probabilities. ``mask`` restricts the
    voxels considered (e.g. a class region or a borehole trajectory)."""
    C = proba.shape[0]
    flat = proba.reshape(C, -1)
    if mask is not None:
        sel = mask.reshape(-1).astype(bool)
        flat = flat[:, sel]
    if flat.shape[1] == 0:
        return {"mean_softmax": float("nan"), "entropy": float("nan"), "variance": float("nan")}
    max_p = flat.max(axis=0)
    entropy = -np.sum(flat * np.log(np.clip(flat, 1e-8, 1.0)), axis=0)
    variance = flat.var(axis=0)
    return {
        "mean_softmax": float(np.mean(max_p)),
        "entropy": float(np.mean(entropy)),
        "variance": float(np.mean(variance)),
    }


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over confidence bins. ``correct`` is a boolean per-sample array."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct).astype(float)
    if confidences.size == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = confidences.size
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if not np.any(m):
            continue
        acc = correct[m].mean()
        conf = confidences[m].mean()
        ece += (np.sum(m) / n) * abs(acc - conf)
    return float(ece)
