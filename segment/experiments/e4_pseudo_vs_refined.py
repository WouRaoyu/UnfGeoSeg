"""Experiment 4 - raw pseudo-labels vs refined 3D-TransUNet output.

Both the coarse hard pseudo-labels and the refined segmentation are scored
against the same reference labels (held-out GT / TFR), per class, so the table
shows whether the fine model improves the weak supervision rather than merely
reproducing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..config import Config, load_config
from .eval_metrics import classwise_metrics
from .predictions import list_predicted_cases, load_reference, load_segmentation
from .report import write_table


def _pool_metrics(
    case_ids, get_pred, ref_dir, classes, file_ending, class_name, strict_per_class
):
    yt, yp = [], []
    for case in case_ids:
        ref = load_reference(ref_dir, case, file_ending, class_name, strict_per_class).ravel()
        pred = get_pred(case).ravel()
        n = min(ref.size, pred.size)
        yt.append(ref[:n])
        yp.append(pred[:n])
    yt = np.concatenate(yt) if yt else np.array([])
    yp = np.concatenate(yp) if yp else np.array([])
    return classwise_metrics(yt, yp, classes)


def run(
    refined_pred_dir,
    pseudolabel_dir,
    reference_labels_dir,
    config: Config | None = None,
    out_dir="results/e4_pseudo_vs_refined",
    file_ending: str = ".nii.gz",
) -> List[Dict]:
    cfg = config or load_config()
    classes = cfg.classes
    class_name = classes[0]
    strict_per_class = len(cfg.geology_classes) > 1
    cases = list_predicted_cases(refined_pred_dir, file_ending)

    raw = _pool_metrics(cases, lambda c: load_segmentation(pseudolabel_dir, c, file_ending),
                        reference_labels_dir, classes, file_ending, class_name, strict_per_class)
    refined = _pool_metrics(cases, lambda c: load_segmentation(refined_pred_dir, c, file_ending),
                            reference_labels_dir, classes, file_ending, class_name, strict_per_class)

    rows: List[Dict] = []
    for name in classes:
        rows.append({
            "Class": name,
            "Pseudo F1": raw[name]["f1"],
            "Refined F1": refined[name]["f1"],
            "F1 gain": refined[name]["f1"] - raw[name]["f1"],
            "Pseudo hit": raw[name]["hit_rate"],
            "Refined hit": refined[name]["hit_rate"],
        })
    write_table(rows, Path(out_dir) / "e4_pseudo_vs_refined",
                columns=["Class", "Pseudo F1", "Refined F1", "F1 gain", "Pseudo hit", "Refined hit"])
    return rows
