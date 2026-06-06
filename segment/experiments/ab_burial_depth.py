"""Ablation - contribution of burial depth (Table 4).

Compares the coarse classifier trained with all three channels (vp, vs, depth)
against vp+vs only, under the same leakage-controlled split and identical
sampling, reporting class-wise Precision/Recall/F1/Accuracy for both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..coarse.rf_classifier import CoarseClassifier
from ..config import Config, load_config
from ..process_contract import feature_columns
from .eval_metrics import classwise_metrics
from .records import build_records
from .report import write_table


def _loto_metrics(X, y, groups, classes, rf_params) -> Dict[str, Dict[str, float]]:
    y_true, y_pred = [], []
    for held in sorted(set(groups.tolist())):
        test = groups == held
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        clf = CoarseClassifier(model="random_forest", num_classes=len(classes), params=rf_params)
        clf.fit(X[train], y[train])
        y_pred.extend(clf.predict(X[test]).tolist())
        y_true.extend(y[test].tolist())
    return classwise_metrics(np.asarray(y_true), np.asarray(y_pred), classes)


def run(dataset_dir, config: Config | None = None, out_dir="results/ab_burial_depth",
        n_per_class: int = 2000) -> List[Dict]:
    cfg = config or load_config()
    classes = cfg.classes
    channels = [c for c in cfg.channels if c not in {"confidence", "probfg"}]
    n_stats = len(cfg.statistics)
    process_mode = cfg.process_feature_mode
    rf_params = cfg.get("coarse", "random_forest", default={})

    rec = build_records(dataset_dir, channels, cfg.half_window, cfg.statistics,
                        cfg.mode_decimals, n_per_class=n_per_class, seed=42,
                        class_name=classes[0],
                        strict_per_class=len(cfg.geology_classes) > 1,
                        process_feature_mode=process_mode,
                        process_depth=cfg.process_depth,
                        process_size=cfg.process_size)
    X, y, groups = rec["X"], rec["y"], rec["groups"]
    if len(set(groups.tolist())) < 2:
        groups = rec["cases"]

    if process_mode:
        cols = feature_columns(process_mode)
        keep_no_depth = [i for i, name in enumerate(cols) if not name.startswith("depth")]
    else:
        # depth is the last channel -> its features are the final n_stats columns
        depth_cols = slice((len(channels) - 1) * n_stats, len(channels) * n_stats)
        keep_no_depth = [
            i for i in range(X.shape[1]) if not (depth_cols.start <= i < depth_cols.stop)
        ]

    with_db = _loto_metrics(X, y, groups, classes, rf_params)
    without_db = _loto_metrics(X[:, keep_no_depth], y, groups, classes, rf_params)

    rows: List[Dict] = []
    for metric in ("precision", "recall", "f1", "accuracy"):
        for name in classes:
            rows.append({
                "Metric": metric.capitalize(),
                "Class": name,
                "With Db": with_db[name][metric],
                "Without Db": without_db[name][metric],
            })
    write_table(rows, Path(out_dir) / "table4_burial_depth",
                columns=["Metric", "Class", "With Db", "Without Db"])
    return rows
