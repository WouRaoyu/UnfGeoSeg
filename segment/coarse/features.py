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

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .. import process_contract

STAT_ORDER = ("mean", "median", "mode", "max", "min")
PROCESS_STAT_ORDER = ("mean", "q25", "median", "q75")


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
    if stat == "q25":
        return ndimage.percentile_filter(vol, percentile=25, size=size, mode="nearest")
    if stat == "q75":
        return ndimage.percentile_filter(vol, percentile=75, size=size, mode="nearest")
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


def _window_mean(vol: np.ndarray, size: Sequence[int]) -> np.ndarray:
    return ndimage.uniform_filter(vol.astype(np.float32, copy=False), size=size, mode="nearest")


def _window_stats(vol: np.ndarray, size: Sequence[int]) -> Dict[str, np.ndarray]:
    vol = vol.astype(np.float32, copy=False)
    mean = _window_mean(vol, size)
    mean2 = _window_mean(vol * vol, size)
    var = np.maximum(mean2 - mean * mean, 0.0)
    std = np.sqrt(var).astype(np.float32)
    q25 = ndimage.percentile_filter(vol, percentile=25, size=size, mode="nearest")
    median = ndimage.median_filter(vol, size=size, mode="nearest")
    q75 = ndimage.percentile_filter(vol, percentile=75, size=size, mode="nearest")
    iqr = (q75 - q25).astype(np.float32)
    mean3 = _window_mean(vol * vol * vol, size)
    m3 = mean3 - 3.0 * mean * mean2 + 2.0 * mean * mean * mean
    skew = np.divide(
        m3,
        np.maximum(std * std * std, 1e-6),
        out=np.zeros_like(mean, dtype=np.float32),
        where=std > 1e-6,
    )
    cv = np.divide(
        std,
        mean,
        out=np.zeros_like(mean, dtype=np.float32),
        where=mean != 0,
    )
    return {
        "mean": mean.astype(np.float32),
        "q25": q25.astype(np.float32),
        "median": median.astype(np.float32),
        "q75": q75.astype(np.float32),
        "std": std,
        "cv": cv.astype(np.float32),
        "skew": skew.astype(np.float32),
        "iqr": iqr,
    }


def _sample_nearest(vol: np.ndarray, offset: Tuple[int, int, int]) -> np.ndarray:
    z = np.clip(np.arange(vol.shape[0]) + offset[0], 0, vol.shape[0] - 1)
    y = np.clip(np.arange(vol.shape[1]) + offset[1], 0, vol.shape[1] - 1)
    x = np.clip(np.arange(vol.shape[2]) + offset[2], 0, vol.shape[2] - 1)
    return vol[np.ix_(z, y, x)].astype(np.float32)


def _fixed_sample_offsets(base: Sequence[int]) -> List[Tuple[int, int, int]]:
    dz, dy, dx = (process_contract.min_box(v) for v in base)
    oz = max(1, dz // 4)
    oy = max(1, dy // 4)
    ox = max(1, dx // 4)
    return [
        (0, 0, 0),
        (-oz, 0, 0),
        (oz, 0, 0),
        (0, -oy, 0),
        (0, oy, 0),
        (0, 0, -ox),
        (0, 0, ox),
    ]


def _directional_features(vol: np.ndarray, base_iqr: np.ndarray) -> List[np.ndarray]:
    gz, gy, gx = np.gradient(vol.astype(np.float32, copy=False))
    grad_mag = np.sqrt(gz * gz + gy * gy + gx * gx).astype(np.float32)
    return [
        grad_mag,
        gz.astype(np.float32),
        gy.astype(np.float32),
        gx.astype(np.float32),
        base_iqr.astype(np.float32),
    ]


def compute_process_feature_volumes(
    channel_volumes: Sequence[np.ndarray],
    depth: int,
    size: int,
    feature_mode: str = "baseline",
) -> np.ndarray:
    """Stack feature volumes in the exact column order of ``process/contract.h``.

    This is intended for Python-side experiments that need to compare against
    the standalone VDB workflow. It mirrors the contract column order and the
    common baseline/multiscale/directional/hybrid_spatial statistics.
    """
    if process_contract.is_spatial_v1_feature_mode(feature_mode):
        raise NotImplementedError(
            "Python experiments do not implement process spatial_v1 features yet"
        )
    if len(channel_volumes) != len(process_contract.FEATURE_GRIDS):
        raise ValueError(
            "process feature contract requires channels [vp, vs, depth]"
        )

    base = (int(depth), int(size), int(size))
    windows = process_contract.feature_windows(base, feature_mode)
    stats_by_window = []
    feats: List[np.ndarray] = []

    for win in windows:
        per_channel = [_window_stats(vol, win.size) for vol in channel_volumes]
        stats_by_window.append(per_channel)
        for stat in PROCESS_STAT_ORDER:
            for stats in per_channel:
                feats.append(stats[stat])

    base_stats = stats_by_window[0]

    if process_contract.has_directional_features(feature_mode):
        for vol, stats in zip(channel_volumes, base_stats):
            feats.extend(_directional_features(vol, stats["iqr"]))

    if process_contract.is_hybrid_feature_mode(feature_mode):
        for stat in ("std", "cv", "skew", "iqr"):
            for stats in base_stats:
                feats.append(stats[stat])
        for offset in _fixed_sample_offsets(base):
            for vol in channel_volumes:
                feats.append(_sample_nearest(vol, offset))
        for _vol in channel_volumes:
            feats.append(np.ones_like(channel_volumes[0], dtype=np.float32))

    return np.stack(feats, axis=0).astype(np.float32, copy=False)


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
