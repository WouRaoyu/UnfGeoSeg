"""Unified fine-stage comparison report.

The report is intentionally not a pseudo-label copying score. It combines
held-out/reference metrics with diagnostics that show where refined predictions
depart from the coarse pseudo-labels: low-confidence probfg voxels, pseudo
boundary-like regions, and foreground-occupancy buckets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import Config, load_config
from ..data.splits import foreground_ratio_bin
from ..io import read_volume
from .eval_metrics import expected_calibration_error
from .predictions import list_predicted_cases, load_probabilities, load_reference, load_segmentation
from .report import write_table


def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = float(np.sum(yt & yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    dice = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) else 1.0
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0
    return {"dice": dice, "iou": iou, "f1": dice}


def _foreground_probability(pred_dir: str | Path, case: str, hard: np.ndarray) -> np.ndarray:
    proba = load_probabilities(pred_dir, case)
    if proba is None:
        return (hard > 0).astype(np.float32)
    if proba.shape[0] == 1:
        return np.clip(proba[0].astype(np.float32), 0.0, 1.0)
    return np.clip(proba[1].astype(np.float32), 0.0, 1.0)


def _probfg_path(pseudolabel_dir: Path, case: str, file_ending: str) -> Path | None:
    for name in (f"probfg_{case}{file_ending}", f"prob_{case}{file_ending}"):
        path = pseudolabel_dir / name
        if path.exists():
            return path
    if pseudolabel_dir.name.startswith("labels"):
        split = pseudolabel_dir.name.removeprefix("labels")
        exported = pseudolabel_dir.parent / f"probs{split}" / f"{case}{file_ending}"
        if exported.exists():
            return exported
    return None


def _load_pseudo_and_probfg(
    pseudolabel_dir: str | Path | None,
    case: str,
    file_ending: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if pseudolabel_dir is None:
        return None, None
    pdir = Path(pseudolabel_dir)
    hard_path = pdir / f"{case}{file_ending}"
    hard = None
    if hard_path.exists():
        hard, _ = read_volume(hard_path)
        hard = (hard > 0).astype(np.uint8)
    probfg = None
    prob_path = _probfg_path(pdir, case, file_ending)
    if prob_path is not None:
        probfg, _ = read_volume(prob_path)
        probfg = np.clip(probfg.astype(np.float32), 0.0, 1.0)
    return hard, probfg


def _mixed_like_from_pseudo(pseudo: np.ndarray) -> np.ndarray:
    pseudo = pseudo.astype(bool)
    mixed = np.zeros_like(pseudo, dtype=bool)
    for axis in range(3):
        diff = np.diff(pseudo, axis=axis) != 0
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(1, None)
        sl_b[axis] = slice(None, -1)
        mixed[tuple(sl_a)] |= diff
        mixed[tuple(sl_b)] |= diff
    return mixed


def _safe_mean(values: List[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def run(
    method_pred_dirs: Dict[str, str],
    reference_labels_dir: str,
    config: Config | None = None,
    pseudolabel_dir: str | None = None,
    out_dir: str = "results/fine_report",
    file_ending: str = ".nii.gz",
    data_source: str = "heldout_reference",
) -> Dict[str, List[Dict]]:
    cfg = config or load_config()
    classes = cfg.classes
    class_name = classes[0]
    strict_per_class = len(cfg.geology_classes) > 1
    first_dir = next(iter(method_pred_dirs.values()))
    cases = list_predicted_cases(first_dir, file_ending)

    summary_rows: List[Dict] = []
    audit_acc: Dict[tuple[str, str, str], Dict[str, float]] = {}

    for method, pred_dir in method_pred_dirs.items():
        dice_vals: List[float] = []
        iou_vals: List[float] = []
        f1_vals: List[float] = []
        brier_vals: List[float] = []
        ece_vals: List[float] = []
        entropy_vals: List[float] = []
        copy_num = copy_den = 0.0
        disagree_num = disagree_den = 0.0
        low_num = low_den = 0.0
        high_num = high_den = 0.0
        mixed_num = mixed_den = 0.0

        for case in cases:
            ref = load_reference(
                reference_labels_dir,
                case,
                file_ending,
                class_name,
                strict_per_class,
            )
            pred = (load_segmentation(pred_dir, case, file_ending) > 0).astype(np.uint8)
            n = min(ref.size, pred.size)
            ref_flat = (ref.ravel()[:n] > 0).astype(np.uint8)
            pred_flat = pred.ravel()[:n].astype(np.uint8)
            scores = _binary_scores(ref_flat, pred_flat)
            dice_vals.append(scores["dice"])
            iou_vals.append(scores["iou"])
            f1_vals.append(scores["f1"])

            pfg = _foreground_probability(pred_dir, case, pred).ravel()[:n]
            brier_vals.append(float(np.mean((pfg - ref_flat) ** 2)))
            conf = np.maximum(pfg, 1.0 - pfg)
            correct = ((pfg >= 0.5).astype(np.uint8) == ref_flat)
            ece_vals.append(expected_calibration_error(conf, correct))
            entropy_vals.append(
                float(np.mean(-(pfg * np.log(np.clip(pfg, 1e-8, 1.0))
                                + (1.0 - pfg) * np.log(np.clip(1.0 - pfg, 1e-8, 1.0)))))
            )

            pseudo, probfg = _load_pseudo_and_probfg(pseudolabel_dir, case, file_ending)
            if pseudo is None:
                continue
            pseudo_flat = pseudo.ravel()[:n].astype(np.uint8)
            disagree = pred_flat != pseudo_flat
            copy_num += float(np.sum(~disagree))
            copy_den += float(n)
            disagree_num += float(np.sum(disagree))
            disagree_den += float(n)

            if probfg is not None:
                prob_flat = probfg.ravel()[:n]
                certainty = 2.0 * np.abs(prob_flat - 0.5)
                low = certainty < 0.5
                high = certainty >= 0.8
                low_num += float(np.sum(disagree & low))
                low_den += float(np.sum(low))
                high_num += float(np.sum(disagree & high))
                high_den += float(np.sum(high))
                for bucket_name, mask in (
                    ("probfg_low_confidence", low),
                    ("probfg_high_confidence", high),
                ):
                    key = (method, "probfg_confidence", bucket_name)
                    acc = audit_acc.setdefault(key, {"voxels": 0.0, "disagree": 0.0})
                    acc["voxels"] += float(np.sum(mask))
                    acc["disagree"] += float(np.sum(disagree & mask))

            mixed_like = _mixed_like_from_pseudo(pseudo).ravel()[:n]
            mixed_num += float(np.sum(disagree & mixed_like))
            mixed_den += float(np.sum(mixed_like))
            key = (method, "region", "pseudo_mixed_like")
            acc = audit_acc.setdefault(key, {"voxels": 0.0, "disagree": 0.0})
            acc["voxels"] += float(np.sum(mixed_like))
            acc["disagree"] += float(np.sum(disagree & mixed_like))

            fg_bucket = foreground_ratio_bin(float(np.count_nonzero(pseudo_flat) / max(n, 1)))
            key = (method, "case_pseudo_foreground", fg_bucket)
            acc = audit_acc.setdefault(key, {"voxels": 0.0, "disagree": 0.0})
            acc["voxels"] += float(n)
            acc["disagree"] += float(np.sum(disagree))

        summary_rows.append({
            "Method": method,
            "Data source": data_source,
            "Cases": len(cases),
            "Case macro Dice": _safe_mean(dice_vals),
            "Case macro IoU": _safe_mean(iou_vals),
            "Case macro F1": _safe_mean(f1_vals),
            "Brier": _safe_mean(brier_vals),
            "ECE": _safe_mean(ece_vals),
            "Entropy": _safe_mean(entropy_vals),
            "Pseudo-copy ratio": _ratio(copy_num, copy_den),
            "Pseudo disagreement": _ratio(disagree_num, disagree_den),
            "Low-conf disagreement": _ratio(low_num, low_den),
            "High-conf disagreement": _ratio(high_num, high_den),
            "Mixed-like disagreement": _ratio(mixed_num, mixed_den),
        })

    audit_rows = []
    for (method, group, bucket), acc in sorted(audit_acc.items()):
        audit_rows.append({
            "Method": method,
            "Group": group,
            "Bucket": bucket,
            "Voxels": int(acc["voxels"]),
            "Pseudo disagreement": _ratio(acc["disagree"], acc["voxels"]),
        })

    out = Path(out_dir)
    write_table(
        summary_rows,
        out / "fine_report_summary",
        columns=[
            "Method", "Data source", "Cases", "Case macro Dice",
            "Case macro IoU", "Case macro F1", "Brier", "ECE", "Entropy",
            "Pseudo-copy ratio", "Pseudo disagreement", "Low-conf disagreement",
            "High-conf disagreement", "Mixed-like disagreement",
        ],
    )
    write_table(
        audit_rows,
        out / "fine_report_disagreement",
        columns=["Method", "Group", "Bucket", "Voxels", "Pseudo disagreement"],
    )
    return {"summary": summary_rows, "audit": audit_rows}
