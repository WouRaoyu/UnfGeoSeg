"""Ablation - sensitivity to TSP inversion uncertainty.

The coarse classifier is evaluated under multiplicative Vp/Vs perturbations
(+/-3/5/10%). Because all sampling statistics (mean/median/mode/max/min) are
homogeneous of degree one, a multiplicative velocity perturbation scales the
corresponding feature columns exactly, so the perturbed evaluation is computed
without recomputing the windows. Reports class-wise F1 change.

The full-volume fine-grained perturbation (re-running 3D-TransUNet inference on
perturbed volumes, incl. the spatial-smoothing/boundary-attenuation test) is
driven by ``scripts/run_all.ps1``; this module provides the compact coarse
sensitivity used to set the representative perturbation level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..coarse.rf_classifier import CoarseClassifier
from ..config import Config, load_config
from ..process_contract import feature_columns
from .eval_metrics import classwise_metrics
from .records import build_records
from .report import write_table


def _velocity_cols(n_channels: int, n_stats: int) -> List[int]:
    # channels are [vp, vs, depth]; perturb vp (ch0) and vs (ch1)
    cols = []
    for ch in (0, 1):
        cols.extend(range(ch * n_stats, (ch + 1) * n_stats))
    return cols


def _loto_f1(X, y, groups, classes, rf_params, clfs) -> Dict[str, float]:
    """Evaluate pre-trained per-fold classifiers ``clfs`` on (possibly perturbed)
    features ``X``."""
    y_true, y_pred = [], []
    for held, clf in clfs.items():
        test = groups == held
        if test.sum() == 0:
            continue
        y_pred.extend(clf.predict(X[test]).tolist())
        y_true.extend(y[test].tolist())
    m = classwise_metrics(np.asarray(y_true), np.asarray(y_pred), classes)
    return {name: m[name]["f1"] for name in classes}


def run(dataset_dir, config: Config | None = None, out_dir="results/ab_tsp_perturbation",
        levels: Sequence[float] = None, n_per_class: int = 1500) -> List[Dict]:
    cfg = config or load_config()
    classes = cfg.classes
    channels = [c for c in cfg.channels if c not in {"confidence", "probfg"}]
    n_stats = len(cfg.statistics)
    process_mode = cfg.process_feature_mode
    rf_params = cfg.get("coarse", "random_forest", default={})
    levels = levels or cfg.get("perturbation", "velocity_levels", default=[0.03, 0.05, 0.10])

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

    # train one classifier per held-out fold on UNPERTURBED features
    clfs = {}
    for held in sorted(set(groups.tolist())):
        train = groups != held
        if train.sum() == 0 or (groups == held).sum() == 0:
            continue
        clf = CoarseClassifier(model="random_forest", num_classes=len(classes), params=rf_params)
        clf.fit(X[train], y[train])
        clfs[held] = clf

    if process_mode:
        cols = feature_columns(process_mode)
        vcols = [
            i for i, name in enumerate(cols)
            if name.startswith("vp") or name.startswith("vs")
        ]
    else:
        vcols = _velocity_cols(len(channels), n_stats)
    rows: List[Dict] = []

    def add_row(label, Xp, interp):
        f1 = _loto_f1(Xp, y, groups, classes, rf_params, clfs)
        row = {"Perturbation": label}
        row.update({f"{name} F1": f1[name] for name in classes})
        row["macro F1"] = float(np.mean([f1[name] for name in classes]))
        row["Interpretation"] = interp
        rows.append(row)

    add_row("Original Vp/Vs", X, "Baseline")
    for lvl in levels:
        for sign, tag in ((1 + lvl, "+"), (1 - lvl, "-")):
            Xp = X.copy()
            Xp[:, vcols] *= sign
            add_row(f"Vp/Vs {tag}{int(lvl*100)}%", Xp,
                    "mild" if lvl <= 0.03 else "moderate" if lvl <= 0.05 else "strong")

    cols = ["Perturbation", *[f"{n} F1" for n in classes], "macro F1", "Interpretation"]
    write_table(rows, Path(out_dir) / "tsp_perturbation_coarse", columns=cols)
    return rows
