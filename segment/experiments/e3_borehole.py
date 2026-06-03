"""Experiment 3 - independent borehole/probe-hole validation.

Each logged interval is mapped onto the voxel grid along the borehole trajectory
and the predicted class/probability is compared with the geological log. Reports
class-wise Precision/Recall/F1, hit rate, and boundary/contact error (mean/median
surface distance along the 1-D trajectory). Boreholes are reserved independent
evidence -- never used in training/pseudo-labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import Config, load_config
from .eval_metrics import boundary_error_1d, classwise_metrics
from .predictions import list_predicted_cases, load_reference, load_segmentation
from .report import write_table
from .validation_records import boreholes_from_label, predicted_trajectory


def run(
    refined_pred_dir,
    reference_labels_dir,
    config: Config | None = None,
    out_dir="results/e3_borehole",
    axial_axis: int | None = None,
    n_boreholes_per_case: int = 8,
    spacing: float = 1.0,
    file_ending: str = ".nii.gz",
) -> List[Dict]:
    cfg = config or load_config()
    classes = cfg.classes
    class_name = classes[0]
    strict_per_class = len(cfg.geology_classes) > 1
    axial_axis = cfg.axial_axis if axial_axis is None else axial_axis
    rng = np.random.default_rng(7)

    cases = list_predicted_cases(refined_pred_dir, file_ending)
    yt_all, yp_all = [], []
    boundary_errors: List[float] = []
    interval_counts = {i: 0 for i in range(1, len(classes) + 1)}

    for case in cases:
        ref = load_reference(
            reference_labels_dir, case, file_ending, class_name, strict_per_class
        )
        pred = load_segmentation(refined_pred_dir, case, file_ending)
        for bh in boreholes_from_label(ref, axial_axis, n_boreholes_per_case, rng):
            true_traj = bh["labels"]
            pred_traj = predicted_trajectory(pred, axial_axis, bh["col"])
            n = min(true_traj.size, pred_traj.size)
            yt_all.append(true_traj[:n])
            yp_all.append(pred_traj[:n])
            be = boundary_error_1d(pred_traj[:n], true_traj[:n], spacing)
            if not np.isnan(be["mean_boundary_error"]):
                boundary_errors.append(be["mean_boundary_error"])
            for c in range(1, len(classes) + 1):
                interval_counts[c] += int(np.sum(true_traj[:n] == c))

    yt = np.concatenate(yt_all) if yt_all else np.array([])
    yp = np.concatenate(yp_all) if yp_all else np.array([])
    per_class = classwise_metrics(yt, yp, classes)
    rows: List[Dict] = []
    for i, name in enumerate(classes, start=1):
        m = per_class[name]
        rows.append({
            "Class": name,
            "No. of borehole intervals": interval_counts[i],
            "Precision": m["precision"], "Recall": m["recall"], "F1-score": m["f1"],
            "Hit rate": m["hit_rate"],
            "Mean boundary error (m)": float(np.mean(boundary_errors)) if boundary_errors else float("nan"),
            "Median boundary error (m)": float(np.median(boundary_errors)) if boundary_errors else float("nan"),
        })
    write_table(rows, Path(out_dir) / "e3_borehole",
                columns=["Class", "No. of borehole intervals", "Precision", "Recall",
                         "F1-score", "Hit rate", "Mean boundary error (m)",
                         "Median boundary error (m)"])
    return rows
