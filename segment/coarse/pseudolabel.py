"""Voxel-level pseudo-label and confidence generation (Manuscript: Voxel-level
pseudo-label and probability generation).

A neighborhood sliding window is evaluated for every voxel of a training volume:
the coarse classifier predicts a category and probability distribution from the
window statistics, producing a *soft pseudo-label volume* that carries both the
predicted category (hard label) and its confidence probability (soft label).
Voxels outside the valid region are cropped to background.

Each geology type is an independent binary run, so the coarse classifier is
binary (``0=background``, ``1=<type>``) and the outputs match the on-disk
convention of ``Dataset005_Hardness``:

* ``labelsTr/<case>.nii.gz`` -- hard pseudo-label (uint8, ``0/1`` for this type)
* ``labelsTr/prob_<case>.nii.gz`` -- foreground probability ``P(class=1)``
* ``labelsTr/conf_<case>.nii.gz`` -- confidence of the assigned hard class

The full categorical distribution needed by the KL term is reconstructed in the
fine stage from ``(hard label, confidence)`` (see ``segment.fine.loss``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from ..io import Geometry, write_volume
from .features import compute_feature_volumes
from .rf_classifier import CoarseClassifier


@dataclass
class PseudoLabelVolume:
    hard: np.ndarray          # (z, y, x) uint8 category
    confidence: np.ndarray    # (z, y, x) float32 in [0, 1]
    foreground_probability: np.ndarray  # (z, y, x) float32 P(class=1)
    proba: Optional[np.ndarray] = None  # (K+1, z, y, x) full distribution


def generate_pseudolabels(
    channel_volumes: Sequence[np.ndarray],
    classifier: CoarseClassifier,
    half_window: Sequence[int],
    statistics: Sequence[str] = ("mean", "median", "mode", "max", "min"),
    mode_decimals: int = 2,
    valid_mask: Optional[np.ndarray] = None,
    chunk: int = 200_000,
    return_proba: bool = False,
) -> PseudoLabelVolume:
    """Dense sliding-box inference over a whole volume."""
    feats = compute_feature_volumes(
        channel_volumes, half_window, statistics, mode_decimals
    )  # (F, z, y, x)
    shape = channel_volumes[0].shape
    n_cls = classifier.num_classes + 1

    if valid_mask is None:
        valid_mask = np.ones(shape, dtype=bool)
    coords = np.argwhere(valid_mask)  # (M, 3)

    hard = np.zeros(shape, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)
    foreground_probability = np.zeros(shape, dtype=np.float32)
    proba_vol = np.zeros((n_cls, *shape), dtype=np.float32) if return_proba else None

    F = feats.shape[0]
    feats_flat = feats.reshape(F, -1)  # (F, Z*Y*X)
    lin = np.ravel_multi_index((coords[:, 0], coords[:, 1], coords[:, 2]), shape)

    for start in range(0, coords.shape[0], chunk):
        sel = lin[start : start + chunk]
        X = feats_flat[:, sel].T  # (chunk, F)
        proba = classifier.predict_proba(X)  # (chunk, K+1)
        cls = np.argmax(proba, axis=1)
        conf = proba[np.arange(proba.shape[0]), cls]
        fg_prob = proba[:, 1] if proba.shape[1] > 1 else np.zeros(proba.shape[0], dtype=np.float32)
        c = coords[start : start + chunk]
        hard[c[:, 0], c[:, 1], c[:, 2]] = cls.astype(np.uint8)
        confidence[c[:, 0], c[:, 1], c[:, 2]] = conf.astype(np.float32)
        foreground_probability[c[:, 0], c[:, 1], c[:, 2]] = fg_prob.astype(np.float32)
        if return_proba:
            for k in range(n_cls):
                proba_vol[k, c[:, 0], c[:, 1], c[:, 2]] = proba[:, k]

    return PseudoLabelVolume(
        hard=hard,
        confidence=confidence,
        foreground_probability=foreground_probability,
        proba=proba_vol,
    )


def write_pseudolabels(
    pl: PseudoLabelVolume,
    geometry: Geometry,
    labels_dir: str | Path,
    case_id: str,
    file_ending: str = ".nii.gz",
) -> Tuple[Path, Path, Path]:
    """Write hard label, foreground probability and hard-label confidence."""
    labels_dir = Path(labels_dir)
    hard_path = labels_dir / f"{case_id}{file_ending}"
    prob_path = labels_dir / f"prob_{case_id}{file_ending}"
    conf_path = labels_dir / f"conf_{case_id}{file_ending}"
    write_volume(pl.hard, geometry, hard_path, dtype=np.uint8)
    write_volume(pl.foreground_probability, geometry, prob_path, dtype=np.float32)
    write_volume(pl.confidence, geometry, conf_path, dtype=np.float32)
    return hard_path, prob_path, conf_path
