"""Load nnU-Net inference outputs for the fine-stage experiments.

``nnUNetv2_predict --save_probabilities`` writes, per case, ``<case>.nii.gz``
(hard segmentation) and ``<case>.npz`` (key ``probabilities``, shape
``(C, z, y, x)``). These helpers expose them as numpy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..io import read_volume, resolve_label_path


def list_predicted_cases(pred_dir: str | Path, file_ending: str = ".nii.gz") -> List[str]:
    pred_dir = Path(pred_dir)
    return sorted(p.name[: -len(file_ending)] for p in pred_dir.glob(f"*{file_ending}")
                  if not p.name.startswith("prob_") and not p.name.startswith("conf_"))


def load_segmentation(pred_dir: str | Path, case: str, file_ending: str = ".nii.gz") -> np.ndarray:
    arr, _ = read_volume(Path(pred_dir) / f"{case}{file_ending}")
    return arr


def load_probabilities(pred_dir: str | Path, case: str) -> Optional[np.ndarray]:
    npz = Path(pred_dir) / f"{case}.npz"
    if not npz.exists():
        return None
    with np.load(npz) as d:
        key = "probabilities" if "probabilities" in d else list(d.keys())[0]
        return d[key]


def load_reference(
    labels_dir: str | Path,
    case: str,
    file_ending: str = ".nii.gz",
    class_name: str | None = None,
    strict_per_class: bool = False,
) -> np.ndarray:
    path = resolve_label_path(
        labels_dir, case, class_name, file_ending, strict_per_class=strict_per_class
    )
    arr, _ = read_volume(path)
    if class_name is not None:
        arr = (arr > 0).astype(np.uint8)
    return arr
