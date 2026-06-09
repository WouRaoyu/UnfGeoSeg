"""Leakage-controlled validation splits (Manuscript: Datasets and
leakage-controlled validation design).

Adjacent tunnel faces are spatially autocorrelated and their local sampling
windows overlap, so random record-level splitting inflates accuracy. Two
protocols are provided:

* ``leave_one_tunnel_out`` -- case/volume-level folds for the nnU-Net (fine)
  stage; each fold holds out all volumes of one tunnel project.
* ``kfold_cases`` -- case/volume-level folds for single-project datasets where
  leave-one-tunnel-out would otherwise degenerate to one validation volume.
* ``blocked_chainage_split`` -- record-level split along the axial (chainage)
  axis into train/test blocks separated by a buffer gap no smaller than the
  axial half-window, used by the coarse Random-Forest stage.

Borehole / probe-hole records are never placed in a training fold (they are
reserved as independent validation evidence) -- callers pass them separately.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np


def leave_one_tunnel_out(case_to_tunnel: Dict[str, str]) -> List[Dict[str, List[str]]]:
    """Produce nnU-Net ``splits_final.json`` folds (one per tunnel project).

    ``case_to_tunnel`` maps an nnU-Net case identifier to its tunnel/project id.
    Returns ``[{"train": [...], "val": [...]}, ...]``.
    """
    tunnels = sorted(set(case_to_tunnel.values()))
    folds: List[Dict[str, List[str]]] = []
    for held in tunnels:
        val = sorted(c for c, t in case_to_tunnel.items() if t == held)
        train = sorted(c for c, t in case_to_tunnel.items() if t != held)
        folds.append({"train": train, "val": val})
    return folds


def kfold_cases(case_ids: Sequence[str], n_splits: int = 5) -> List[Dict[str, List[str]]]:
    """Produce deterministic case-level nnU-Net folds for one project.

    Cases are sorted and split into contiguous, size-balanced validation blocks.
    This keeps neighboring case ids together when they encode acquisition order
    while avoiding the unstable one-volume validation folds.
    """
    cases = sorted(case_ids)
    if len(cases) < 2:
        raise ValueError("Need at least two cases to create train/val folds")
    n_splits = min(int(n_splits), len(cases))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    folds: List[Dict[str, List[str]]] = []
    for val_arr in np.array_split(np.asarray(cases, dtype=object), n_splits):
        val = [str(c) for c in val_arr.tolist()]
        val_set = set(val)
        train = [c for c in cases if c not in val_set]
        folds.append({"train": train, "val": val})
    return folds


def foreground_ratio_bin(
    ratio: float,
    sparse_threshold: float = 0.05,
    dense_threshold: float = 0.95,
    eps: float = 1e-8,
) -> str:
    """Bucket a case by foreground occupancy for fold stratification."""
    ratio = float(ratio)
    if ratio <= eps:
        return "all_background"
    if ratio < sparse_threshold:
        return "sparse_foreground"
    if ratio >= 1.0 - eps:
        return "all_foreground"
    if ratio > dense_threshold:
        return "dense_foreground"
    return "mixed"


def stratified_kfold_cases(
    case_ids: Sequence[str],
    foreground_ratios: Mapping[str, float],
    n_splits: int = 5,
) -> List[Dict[str, List[str]]]:
    """Create deterministic folds balanced by foreground occupancy buckets.

    This is useful for small tunnel-volume datasets where all-background or
    all-foreground cases are valid but can dominate one fold if cases are split
    only by sorted id.
    """
    cases = sorted(case_ids)
    if len(cases) < 2:
        raise ValueError("Need at least two cases to create train/val folds")
    n_splits = min(int(n_splits), len(cases))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    missing = [c for c in cases if c not in foreground_ratios]
    if missing:
        raise ValueError(f"Missing foreground ratios for {len(missing)} cases")

    val_by_fold: List[List[str]] = [[] for _ in range(n_splits)]
    buckets: Dict[str, List[str]] = {}
    for case in cases:
        bucket = foreground_ratio_bin(foreground_ratios[case])
        buckets.setdefault(bucket, []).append(case)

    # Split each bucket across folds so extremes are spread instead of clustered.
    for bucket_cases in buckets.values():
        for fold_idx, arr in enumerate(np.array_split(np.asarray(bucket_cases, dtype=object), n_splits)):
            val_by_fold[fold_idx].extend(str(c) for c in arr.tolist())

    folds: List[Dict[str, List[str]]] = []
    all_cases = set(cases)
    for val in val_by_fold:
        val = sorted(val)
        train = sorted(all_cases.difference(val))
        folds.append({"train": train, "val": val})
    return folds


def blocked_chainage_split(
    chainage: Sequence[float],
    block_length: float,
    buffer: float,
    test_block_stride: int = 2,
) -> Dict[str, np.ndarray]:
    """Split record indices along the axial axis into train/test by blocks.

    Records are bucketed into contiguous blocks of ``block_length`` (in the same
    units as ``chainage``). Every ``test_block_stride``-th block is assigned to
    test; records within ``buffer`` of a test block on either side are dropped
    so train and test windows cannot overlap.

    Returns ``{"train": idx, "test": idx, "buffer": idx}`` (integer arrays).
    """
    chainage = np.asarray(chainage, dtype=np.float64)
    if chainage.size == 0:
        empty = np.array([], dtype=int)
        return {"train": empty, "test": empty.copy(), "buffer": empty.copy()}

    c0 = chainage.min()
    block_id = np.floor((chainage - c0) / block_length).astype(int)
    test_blocks = set(np.unique(block_id)[::test_block_stride].tolist())

    is_test = np.array([b in test_blocks for b in block_id])

    # buffer: training records whose chainage is within `buffer` of any test record
    test_chainage = chainage[is_test]
    in_buffer = np.zeros_like(is_test)
    if test_chainage.size:
        dist = np.abs(chainage[:, None] - test_chainage[None, :]).min(axis=1)
        in_buffer = (~is_test) & (dist <= buffer)

    idx = np.arange(chainage.size)
    return {
        "train": idx[~is_test & ~in_buffer],
        "test": idx[is_test],
        "buffer": idx[in_buffer],
    }


def default_case_to_tunnel(case_ids: Sequence[str]) -> Dict[str, str]:
    """Heuristic tunnel grouping from nnU-Net case ids of the form
    ``<tunnel>_<number>`` (e.g. ``hardness_002`` -> tunnel ``hardness``).

    When every case shares one prefix (single-project smoke dataset) the caller
    should supply an explicit mapping instead so leave-one-tunnel-out is
    meaningful.
    """
    mapping: Dict[str, str] = {}
    for cid in case_ids:
        tunnel = cid.rsplit("_", 1)[0] if "_" in cid else cid
        mapping[cid] = tunnel
    return mapping
