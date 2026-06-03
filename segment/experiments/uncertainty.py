"""Reliability / uncertainty assessment (Table 6).

For each method (raw pseudo-labels, nnU-Net, 3D-TransUNet, proposed) the mean
SoftMax probability, entropy and variance are reported per class, computed over
the voxels predicted as that class. These reliability indicators are subordinate
to the direct sparse validation (held-out TFR / borehole) and characterize where
predictions are confident vs. uncertain.

Inputs are per-method directories of ``<case>.npz`` probability volumes (from
``nnUNetv2_predict --save_probabilities``). The pseudo-label column is built
from the coarse binary foreground probability ``prob_<case>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import Config, load_config
from ..io import read_volume
from .eval_metrics import reliability_metrics
from .predictions import list_predicted_cases, load_probabilities
from .report import write_table


def _pseudolabel_proba(pseudolabel_dir: Path, case: str, n_classes: int, file_ending: str):
    hard, _ = read_volume(pseudolabel_dir / f"{case}{file_ending}")
    prob_path = pseudolabel_dir / f"prob_{case}{file_ending}"
    if prob_path.exists() and n_classes == 2:
        fg_prob, _ = read_volume(prob_path)
        fg_prob = np.clip(fg_prob.astype(np.float32), 0.0, 1.0)
        return np.stack([1.0 - fg_prob, fg_prob], axis=0)

    conf_path = pseudolabel_dir / f"conf_{case}{file_ending}"
    if conf_path.exists():
        conf, _ = read_volume(conf_path)
    else:
        # Backward compatibility: older pseudo-label folders stored assigned
        # hard-label confidence in prob_<case>.
        conf, _ = read_volume(prob_path)
    proba = np.full((n_classes, *hard.shape), (1.0) / max(n_classes - 1, 1), dtype=np.float32)
    other = (1.0 - conf) / max(n_classes - 1, 1)
    for c in range(n_classes):
        m = hard == c
        proba[c] = np.where(m, conf, other)
    proba /= proba.sum(axis=0, keepdims=True)
    return proba


def _method_metrics(case_ids, get_proba, classes) -> Dict[str, Dict[str, float]]:
    n_cls = len(classes) + 1
    acc = {name: {"mean_softmax": [], "entropy": [], "variance": []} for name in classes}
    for case in case_ids:
        proba = get_proba(case)
        if proba is None:
            continue
        argmax = proba.argmax(axis=0)
        for i, name in enumerate(classes, start=1):
            mask = argmax == i
            if mask.sum() == 0:
                continue
            r = reliability_metrics(proba, mask)
            for k in ("mean_softmax", "entropy", "variance"):
                acc[name][k].append(r[k])
    return {name: {k: float(np.nanmean(v)) if v else float("nan") for k, v in d.items()}
            for name, d in acc.items()}


def run(
    method_prob_dirs: Dict[str, str],
    classes_config: Config | None = None,
    pseudolabel_dir: str | None = None,
    out_dir="results/uncertainty",
    file_ending: str = ".nii.gz",
) -> List[Dict]:
    cfg = classes_config or load_config()
    classes = cfg.classes
    n_cls = len(classes) + 1

    # establish the case list from the first available method dir
    ref_dir = next(iter(method_prob_dirs.values()))
    cases = list_predicted_cases(ref_dir, file_ending)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    if pseudolabel_dir is not None:
        results["Pseudo-label"] = _method_metrics(
            cases, lambda c: _pseudolabel_proba(Path(pseudolabel_dir), c, n_cls, file_ending), classes
        )
    for method, pdir in method_prob_dirs.items():
        results[method] = _method_metrics(
            cases, lambda c, d=pdir: load_probabilities(d, c), classes
        )

    rows: List[Dict] = []
    for metric in ("mean_softmax", "entropy", "variance"):
        for name in classes:
            row = {"Metric": metric, "Class": name}
            for method in results:
                row[method] = results[method][name][metric]
            rows.append(row)
    write_table(rows, Path(out_dir) / "table6_uncertainty",
                columns=["Metric", "Class", *results.keys()])
    return rows
