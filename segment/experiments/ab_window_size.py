"""Ablation - optimum local sampling window size (Fig. 12).

Sweeps the half-window over a set of configurations and reports class-wise F1 of
the coarse classifier under the leakage-controlled split, identifying the peak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..coarse.rf_classifier import CoarseClassifier
from ..config import Config, load_config
from .eval_metrics import classwise_metrics
from .records import build_records
from .report import write_table

DEFAULT_HALF_WINDOWS = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]


def _loto_f1(X, y, groups, classes, rf_params) -> Dict[str, float]:
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
    m = classwise_metrics(np.asarray(y_true), np.asarray(y_pred), classes)
    return {name: m[name]["f1"] for name in classes}


def run(dataset_dir, config: Config | None = None, out_dir="results/ab_window_size",
        half_windows: Sequence[Sequence[int]] = None, n_per_class: int = 1500) -> List[Dict]:
    cfg = config or load_config()
    classes = cfg.classes
    rf_params = cfg.get("coarse", "random_forest", default={})
    half_windows = half_windows or DEFAULT_HALF_WINDOWS

    rows: List[Dict] = []
    for idx, hw in enumerate(half_windows):
        rec = build_records(dataset_dir, cfg.channels, hw, cfg.statistics,
                            cfg.mode_decimals, n_per_class=n_per_class, seed=42,
                            class_name=classes[0],
                            strict_per_class=len(cfg.geology_classes) > 1)
        X, y, groups = rec["X"], rec["y"], rec["groups"]
        if len(set(groups.tolist())) < 2:
            groups = rec["cases"]
        f1 = _loto_f1(X, y, groups, classes, rf_params)
        row = {"Index": idx, "half_window": "x".join(map(str, hw)),
               "size_voxels": int(np.prod([2 * h + 1 for h in hw]))}
        row.update({name: f1[name] for name in classes})
        row["macro_F1"] = float(np.mean([f1[name] for name in classes]))
        rows.append(row)

    best = max(rows, key=lambda r: r["macro_F1"])
    for r in rows:
        r["best"] = "*" if r is best else ""
    write_table(rows, Path(out_dir) / "fig12_window_size",
                columns=["Index", "half_window", "size_voxels", *classes, "macro_F1", "best"])
    return rows
