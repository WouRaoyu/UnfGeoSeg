"""Experiment 1 - held-out coarse-classifier performance (Table 2).

Evaluates the Random-Forest coarse classifier under a leakage-controlled
protocol (leave-one-tunnel-out by default; each group = one tunnel project, or
one volume when a single project is present). Reports class-wise Accuracy,
Precision, Recall, F1 aggregated over held-out folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..coarse.rf_classifier import CoarseClassifier
from ..config import Config, load_config
from .eval_metrics import classwise_metrics, prf1, confusion_counts
from .records import build_records
from .report import write_table


def run(
    dataset_dir: str | Path,
    config: Config | None = None,
    out_dir: str | Path = "results/e1_coarse",
    model: str = "random_forest",
    n_per_class: int = 2000,
) -> List[Dict]:
    cfg = config or load_config()
    dataset_dir = Path(dataset_dir)
    classes = cfg.classes

    rec = build_records(
        dataset_dir,
        cfg.channels,
        cfg.half_window,
        cfg.statistics,
        cfg.mode_decimals,
        n_per_class=n_per_class,
        seed=42,
        class_name=classes[0],
        strict_per_class=len(cfg.geology_classes) > 1,
    )
    X, y, groups = rec["X"], rec["y"], rec["groups"]
    # Leave-one-tunnel-out; fall back to leave-one-volume-out for a single-project
    # (smoke) dataset so held-out folds are still meaningful.
    if len(set(groups.tolist())) < 2:
        groups = rec["cases"]
    unique_groups = sorted(set(groups.tolist()))

    # Leave-one-group-out predictions pooled across folds.
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    rf_params = cfg.get("coarse", "random_forest", default={})
    for held in unique_groups:
        test = groups == held
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        clf = CoarseClassifier(
            model=model, num_classes=len(classes), params=rf_params
        )
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        y_true_all.extend(y[test].tolist())
        y_pred_all.extend(y_pred.tolist())

    y_true = np.asarray(y_true_all)
    y_pred = np.asarray(y_pred_all)

    rows: List[Dict] = []
    per_class = classwise_metrics(y_true, y_pred, classes)
    for name in classes:
        m = per_class[name]
        rows.append(
            {
                "Category": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 score": m["f1"],
            }
        )
    # macro row
    if rows:
        rows.append(
            {
                "Category": "macro_avg",
                "Accuracy": float(np.mean([r["Accuracy"] for r in rows])),
                "Precision": float(np.mean([r["Precision"] for r in rows])),
                "Recall": float(np.mean([r["Recall"] for r in rows])),
                "F1 score": float(np.mean([r["F1 score"] for r in rows])),
            }
        )

    out = Path(out_dir) / f"table2_coarse_{model}"
    write_table(
        rows,
        out,
        columns=["Category", "Accuracy", "Precision", "Recall", "F1 score"],
    )
    return rows
