"""Vectorized local statistical feature volumes (fast path for Eq. 2-4).

The exact per-point reference implementation lives in
``segment.data.sampling``. For Random-Forest training over thousands of
tunnel-face records and for dense sliding-box pseudo-label inference over whole
volumes, computing the window statistics one voxel at a time is far too slow, so
here each statistic is computed for *every* voxel at once with separable
``scipy.ndimage`` filters. The RF therefore trains on exactly the same features
that are later used at inference time (no train/inference mismatch).

Window: a box of half-extents ``(dz, dy, dx)`` -> full size ``2*half + 1`` per
axis, border-clipped via ``mode="nearest"``.

``mode`` (the statistic) is, for continuous velocity/depth fields, defined as the
windowed median of the values rounded to ``mode_decimals`` -- a deterministic,
separable surrogate for the most-frequent rounded value (documented).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage

STAT_ORDER = ("mean", "median", "mode", "max", "min")


def _box_size(half_window: Sequence[int]) -> List[int]:
    return [2 * int(h) + 1 for h in half_window]


def compute_stat_volume(
    vol: np.ndarray,
    stat: str,
    half_window: Sequence[int],
    mode_decimals: int = 2,
) -> np.ndarray:
    size = _box_size(half_window)
    vol = vol.astype(np.float32, copy=False)
    if stat == "mean":
        return ndimage.uniform_filter(vol, size=size, mode="nearest")
    if stat == "max":
        return ndimage.maximum_filter(vol, size=size, mode="nearest")
    if stat == "min":
        return ndimage.minimum_filter(vol, size=size, mode="nearest")
    if stat == "median":
        return ndimage.median_filter(vol, size=size, mode="nearest")
    if stat == "mode":
        rounded = np.round(vol, mode_decimals)
        return ndimage.median_filter(rounded, size=size, mode="nearest")
    raise ValueError(f"Unknown statistic: {stat!r}")


def compute_feature_volumes(
    channel_volumes: Sequence[np.ndarray],
    half_window: Sequence[int],
    statistics: Sequence[str] = STAT_ORDER,
    mode_decimals: int = 2,
) -> np.ndarray:
    """Stack all (channel, statistic) feature volumes.

    Returns an array of shape ``(F, z, y, x)`` where ``F = C * len(statistics)``
    ordered ``[ch0_stat0, ch0_stat1, ..., ch1_stat0, ...]`` (matching
    :func:`segment.data.sampling.feature_names`).

    ``median`` is the costly filter; when both ``median`` and ``mode`` are
    requested the (rounded) median is reused for ``mode`` so the expensive
    ``median_filter`` runs only once per channel.
    """
    feats: List[np.ndarray] = []
    for vol in channel_volumes:
        median_cache = None
        if "median" in statistics or "mode" in statistics:
            median_cache = compute_stat_volume(vol, "median", half_window, mode_decimals)
        for stat in statistics:
            if stat == "median":
                feats.append(median_cache)
            elif stat == "mode":
                feats.append(np.round(median_cache, mode_decimals))
            else:
                feats.append(compute_stat_volume(vol, stat, half_window, mode_decimals))
    return np.stack(feats, axis=0)


def gather_features(feature_volumes: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Gather (M, F) feature rows at integer ``coords`` (M, 3) [z, y, x]."""
    coords = np.asarray(coords, dtype=np.int64)
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    return feature_volumes[:, z, y, x].T  # (M, F)
