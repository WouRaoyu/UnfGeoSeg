"""Experiment 5 - sensitivity to the probability-loss coefficient lambda.

Aggregates held-out TFR F1, borehole F1, boundary/chainage error and mean
entropy across models trained with different lambda (incl. lambda=0, the
no-probability-constraint baseline). Training itself is driven by
``scripts/run_all.ps1`` (which sets ``UNFAVORSEG_LAMBDA`` and trains one model
per value); this module collects the per-lambda prediction directories and
assembles the table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import Config, load_config
from . import e2_tfr_finegrained, e3_borehole, uncertainty
from .predictions import list_predicted_cases, load_probabilities
from .report import write_table


def _mean_entropy(pred_dir, cases) -> float:
    from .eval_metrics import reliability_metrics

    vals = []
    for c in cases:
        p = load_probabilities(pred_dir, c)
        if p is not None:
            vals.append(reliability_metrics(p)["entropy"])
    return float(np.mean(vals)) if vals else float("nan")


def run(
    lambda_to_pred_dir: Dict[float, str],
    reference_labels_dir: str,
    config: Config | None = None,
    out_dir="results/e5_lambda",
    file_ending: str = ".nii.gz",
) -> List[Dict]:
    cfg = config or load_config()
    rows: List[Dict] = []
    for lam in sorted(lambda_to_pred_dir):
        pred_dir = lambda_to_pred_dir[lam]
        tfr = e2_tfr_finegrained.run(pred_dir, reference_labels_dir, cfg,
                                     out_dir=f"{out_dir}/lam_{lam}_tfr", file_ending=file_ending)
        bh = e3_borehole.run(pred_dir, reference_labels_dir, cfg,
                             out_dir=f"{out_dir}/lam_{lam}_bh", file_ending=file_ending)
        cases = list_predicted_cases(pred_dir, file_ending)
        tfr_f1 = float(np.nanmean([r["F1-score"] for r in tfr]))
        bh_f1 = float(np.nanmean([r["F1-score"] for r in bh]))
        bnd = float(np.nanmean([r["Mean boundary error (m)"] for r in bh]))
        rows.append({
            "lambda": lam,
            "Held-out TFR F1": tfr_f1,
            "Borehole F1": bh_f1,
            "Mean boundary error": bnd,
            "Mean entropy": _mean_entropy(pred_dir, cases),
            "Decision": "no constraint" if lam == 0 else "",
        })
    # mark best by TFR F1
    if rows:
        best = max(rows, key=lambda r: r["Held-out TFR F1"])
        best["Decision"] = (best["Decision"] + " selected").strip()
    write_table(rows, Path(out_dir) / "lambda_sensitivity",
                columns=["lambda", "Held-out TFR F1", "Borehole F1",
                         "Mean boundary error", "Mean entropy", "Decision"])
    return rows
