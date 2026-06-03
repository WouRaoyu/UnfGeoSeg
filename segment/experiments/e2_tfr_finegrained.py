"""Experiment 2 - held-out TFR (excavation-face) validation of the fine model.

The refined segmentation is sampled at held-out excavation-face sections (within
a +/- tolerance band) and compared with the face records. Reports class-wise
Accuracy/Precision/Recall/F1, hit rate and nearest-chainage error. These support
face/path-level consistency, not dense 3-D accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import Config, load_config
from .eval_metrics import classwise_metrics, nearest_chainage_error
from .predictions import list_predicted_cases, load_reference, load_segmentation
from .report import write_table
from .validation_records import faces_from_label, predicted_face_class


def run(
    refined_pred_dir,
    reference_labels_dir,
    config: Config | None = None,
    out_dir="results/e2_tfr",
    axial_axis: int | None = None,
    n_sections: int = 16,
    file_ending: str = ".nii.gz",
) -> List[Dict]:
    cfg = config or load_config()
    classes = cfg.classes
    class_name = classes[0]
    strict_per_class = len(cfg.geology_classes) > 1
    axial_axis = cfg.axial_axis if axial_axis is None else axial_axis
    tol = int(cfg.get("validation", "tolerance_sections", default=1))
    rng = np.random.default_rng(42)

    cases = list_predicted_cases(refined_pred_dir, file_ending)
    y_true, y_pred = [], []
    chainage_true: Dict[int, List[int]] = {i: [] for i in range(1, len(classes) + 1)}
    chainage_pred: Dict[int, List[int]] = {i: [] for i in range(1, len(classes) + 1)}

    for case in cases:
        ref = load_reference(
            reference_labels_dir, case, file_ending, class_name, strict_per_class
        )
        pred = load_segmentation(refined_pred_dir, case, file_ending)
        for rec in faces_from_label(ref, axial_axis, n_sections, rng):
            pc = predicted_face_class(pred, axial_axis, rec["section"], tol)
            y_true.append(rec["class"])
            y_pred.append(pc)
            if rec["class"] > 0:
                chainage_true[rec["class"]].append(rec["section"])
            if pc > 0:
                chainage_pred[pc].append(rec["section"])

    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    per_class = classwise_metrics(yt, yp, classes)
    rows: List[Dict] = []
    for i, name in enumerate(classes, start=1):
        m = per_class[name]
        rows.append({
            "Class": name,
            "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
            "F1-score": m["f1"], "Hit rate": m["hit_rate"],
            "Mean chainage error": nearest_chainage_error(chainage_pred[i], chainage_true[i]),
        })
    write_table(rows, Path(out_dir) / "e2_tfr_finegrained",
                columns=["Class", "Accuracy", "Precision", "Recall", "F1-score",
                         "Hit rate", "Mean chainage error"])
    return rows
