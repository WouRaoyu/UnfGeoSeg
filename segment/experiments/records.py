"""Build coarse-stage training records from an nnU-Net dataset.

The manuscript trains the coarse classifier on sparse tunnel-face records. On a
densely-labelled dataset (e.g. the binary smoke dataset) we emulate that by
stratified sampling of voxels from each labelled volume and attaching the local
statistical feature vector to each sampled voxel. Returns features, labels and a
per-record group id (tunnel/case) for leakage-controlled splitting.

Each unfavorable-geology type is an INDEPENDENT 0/1 target: pass ``class_name``
to read that type's per-class binary mask (``labelsTr/<case>_<type>.nii.gz``)
and sample a binary {0,1} problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..coarse.features import (
    compute_feature_volumes,
    compute_process_feature_volumes,
    gather_features,
)
from ..io import list_cases, read_case, read_volume, resolve_label_path


def sample_case_records(
    channel_volumes: Sequence[np.ndarray],
    label_volume: np.ndarray,
    half_window: Sequence[int],
    statistics: Sequence[str],
    mode_decimals: int,
    n_per_class: int,
    rng: np.random.Generator,
    process_feature_mode: str | None = None,
    process_depth: int = 4,
    process_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified voxel sampling for one case -> (features, labels)."""
    if process_feature_mode:
        feats = compute_process_feature_volumes(
            channel_volumes, process_depth, process_size, process_feature_mode
        )
    else:
        feats = compute_feature_volumes(
            channel_volumes, half_window, statistics, mode_decimals
        )
    coords: List[np.ndarray] = []
    labels: List[int] = []
    for cls in np.unique(label_volume):
        idx = np.argwhere(label_volume == cls)
        if idx.shape[0] == 0:
            continue
        take = min(n_per_class, idx.shape[0])
        sel = rng.choice(idx.shape[0], size=take, replace=False)
        coords.append(idx[sel])
        labels.extend([int(cls)] * take)
    if not coords:
        return np.empty((0, feats.shape[0]), np.float32), np.empty((0,), np.int64)
    coords_arr = np.concatenate(coords, axis=0)
    X = gather_features(feats, coords_arr)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def build_records(
    dataset_dir: str | Path,
    channels: Sequence[str],
    half_window: Sequence[int],
    statistics: Sequence[str],
    mode_decimals: int,
    n_per_class: int = 2000,
    case_to_tunnel: Dict[str, str] | None = None,
    file_ending: str = ".nii.gz",
    seed: int = 42,
    drop_prob_cases: bool = True,
    class_name: str | None = None,
    strict_per_class: bool = False,
    process_feature_mode: str | None = None,
    process_depth: int = 4,
    process_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Assemble records across all labelled cases.

    ``class_name`` selects the independent geology type whose per-class binary
    mask is read and binarized (``> 0``); labels are then a {0,1} problem. Use
    ``strict_per_class`` for multi-type datasets to avoid accidentally reading
    an old mutually-exclusive ``<case>.nii.gz`` label map as "any foreground".

    Returns ``{"X": (N,F), "y": (N,), "groups": (N,) str}``.
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    n_phys = len(channels)
    rng = np.random.default_rng(seed)

    all_case_ids = list_cases(images_dir, file_ending)
    Xs, ys, groups, cases = [], [], [], []
    for case in all_case_ids:
        if drop_prob_cases and case.startswith("prob_"):
            continue
        label_path = resolve_label_path(
            labels_dir, case, class_name, file_ending, strict_per_class=strict_per_class
        )
        if not label_path.exists():
            continue
        vol, _ = read_case(images_dir, case, n_phys, file_ending)
        label, _ = read_volume(label_path)
        # independent binary target: any positive voxel is foreground (1)
        label = (label > 0).astype(np.uint8)
        X, y = sample_case_records(
            list(vol),
            label,
            half_window,
            statistics,
            mode_decimals,
            n_per_class,
            rng,
            process_feature_mode=process_feature_mode,
            process_depth=process_depth,
            process_size=process_size,
        )
        if X.shape[0] == 0:
            continue
        tunnel = (case_to_tunnel or {}).get(case) or case.rsplit("_", 1)[0]
        Xs.append(X)
        ys.append(y)
        groups.extend([tunnel] * X.shape[0])
        cases.extend([case] * X.shape[0])
    if not Xs:
        raise RuntimeError(f"No labelled records found in {dataset_dir}")
    return {
        "X": np.concatenate(Xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "groups": np.asarray(groups),
        "cases": np.asarray(cases),
    }
