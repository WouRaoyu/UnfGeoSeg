"""Local statistical feature sampling (Manuscript Eq. 2-4).

For a center voxel ``(zi, yi, xi)`` a window ``W`` with half-extents
``(dz, dy, dx)`` is taken from each physical-property volume (vp, vs, depth) and
a statistical feature vector ``fstat = [mean, median, mode, max, min]`` is
computed. The per-channel feature vectors are concatenated to form the coarse
classifier input ``f_input = [fstat(vp), fstat(vs), fstat(depth)]``.

The same routine is reused at pseudo-label time, where it is evaluated on a
dense sliding grid over the volume.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

_STAT_FUNCS = ("mean", "median", "mode", "max", "min")


def _mode_continuous(values: np.ndarray, decimals: int) -> float:
    """Mode of continuous values, defined as the most frequent value after
    rounding to ``decimals`` (velocity/depth fields are continuous, so the raw
    mode is degenerate). Falls back to the median for an empty window."""
    if values.size == 0:
        return float("nan")
    rounded = np.round(values, decimals)
    vals, counts = np.unique(rounded, return_counts=True)
    return float(vals[int(np.argmax(counts))])


def window_statistics(
    window: np.ndarray,
    statistics: Sequence[str] = _STAT_FUNCS,
    mode_decimals: int = 2,
) -> np.ndarray:
    """Compute the requested statistics for one window (any shape). NaNs are
    ignored so masked / out-of-volume voxels do not corrupt the aggregate."""
    flat = window.reshape(-1)
    flat = flat[~np.isnan(flat)]
    out: List[float] = []
    for stat in statistics:
        if flat.size == 0:
            out.append(float("nan"))
            continue
        if stat == "mean":
            out.append(float(np.mean(flat)))
        elif stat == "median":
            out.append(float(np.median(flat)))
        elif stat == "mode":
            out.append(_mode_continuous(flat, mode_decimals))
        elif stat == "max":
            out.append(float(np.max(flat)))
        elif stat == "min":
            out.append(float(np.min(flat)))
        else:
            raise ValueError(f"Unknown statistic: {stat!r}")
    return np.asarray(out, dtype=np.float32)


def _slice_window(
    vol: np.ndarray, center: Sequence[int], half: Sequence[int]
) -> np.ndarray:
    """Crop ``vol`` around ``center`` with per-axis half extents ``half``,
    clipping at the volume borders (manuscript: voxels beyond the window /
    valid region are cropped, not padded)."""
    slices = []
    for axis, (c, h) in enumerate(zip(center, half)):
        lo = max(0, c - h)
        hi = min(vol.shape[axis], c + h + 1)
        slices.append(slice(lo, hi))
    return vol[tuple(slices)]


def sample_point_features(
    volumes: Sequence[np.ndarray],
    center: Sequence[int],
    half_window: Sequence[int],
    statistics: Sequence[str] = _STAT_FUNCS,
    mode_decimals: int = 2,
) -> np.ndarray:
    """Feature vector for a single center voxel across all channel ``volumes``.

    Returns a 1-D array of length ``len(volumes) * len(statistics)`` ordered as
    ``[stats(ch0), stats(ch1), ...]`` (Eq. 4).
    """
    feats = [
        window_statistics(
            _slice_window(vol, center, half_window), statistics, mode_decimals
        )
        for vol in volumes
    ]
    return np.concatenate(feats, axis=0)


def feature_names(channels: Sequence[str], statistics: Sequence[str]) -> List[str]:
    """Human-readable column names matching :func:`sample_point_features`."""
    return [f"{ch}_{stat}" for ch in channels for stat in statistics]


def sample_records(
    volumes: Sequence[np.ndarray],
    centers: Sequence[Sequence[int]],
    half_window: Sequence[int],
    statistics: Sequence[str] = _STAT_FUNCS,
    mode_decimals: int = 2,
) -> np.ndarray:
    """Stack :func:`sample_point_features` over many record centers -> (N, F)."""
    rows = [
        sample_point_features(volumes, c, half_window, statistics, mode_decimals)
        for c in centers
    ]
    if not rows:
        n_feat = len(volumes) * len(statistics)
        return np.empty((0, n_feat), dtype=np.float32)
    return np.stack(rows, axis=0)


def dense_feature_grid(
    volumes: Sequence[np.ndarray],
    half_window: Sequence[int],
    statistics: Sequence[str] = _STAT_FUNCS,
    mode_decimals: int = 2,
    valid_mask: np.ndarray | None = None,
    stride: Sequence[int] | int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the window features densely over the whole volume for
    pseudo-label generation (sliding box).

    Returns ``(coords, features)`` where ``coords`` is (M, 3) integer voxel
    indices and ``features`` is (M, F). Only voxels inside ``valid_mask`` (if
    given) and on the ``stride`` grid are returned, keeping memory bounded.
    """
    ref = volumes[0]
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    zz, yy, xx = (
        range(0, ref.shape[0], stride[0]),
        range(0, ref.shape[1], stride[1]),
        range(0, ref.shape[2], stride[2]),
    )
    coords: List[Tuple[int, int, int]] = []
    for z in zz:
        for y in yy:
            for x in xx:
                if valid_mask is not None and not valid_mask[z, y, x]:
                    continue
                coords.append((z, y, x))
    features = sample_records(
        volumes, coords, half_window, statistics, mode_decimals
    )
    return np.asarray(coords, dtype=np.int32), features
