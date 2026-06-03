"""Spatial alignment of multi-source data (Manuscript: Multi-source preprocess).

Two responsibilities:

1. Build the global transform ``Mg`` that maps mileage-based (linear-referenced)
   positions along the tunnel centerline into the project's absolute Cartesian
   coordinate system. Used to register volumetric TSP, raster DEM and vector
   tunnel-face records (TFR) into a common grid.
2. Derive the burial-depth distance field ``H(x, y, z) = z_DEM(x, y) - z_voxel``
   (vertical axis positive upward = overburden thickness), with negative/near
   zero values clipped or masked per the TSP valid volume.

The raw acquisition format is project-specific, so these helpers operate on
plain numpy arrays / callables and are intentionally I/O-free.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def discretize_centerline(
    mileage: np.ndarray, xyz: np.ndarray, step: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline centerline (given as ``mileage`` -> ``xyz`` control
    points) at a uniform ``step`` along chainage.

    Returns ``(sample_mileage, sample_xyz)``.
    """
    mileage = np.asarray(mileage, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    order = np.argsort(mileage)
    mileage, xyz = mileage[order], xyz[order]
    sample_m = np.arange(mileage[0], mileage[-1] + 1e-9, step)
    sample_xyz = np.stack(
        [np.interp(sample_m, mileage, xyz[:, i]) for i in range(3)], axis=1
    )
    return sample_m, sample_xyz


def global_transform(
    origin_xyz: np.ndarray,
    voxel_size: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assemble the 4x4 homogeneous transform ``Mg`` mapping voxel indices
    ``(i, j, k, 1)`` to absolute world coordinates.

    ``origin_xyz``  : world coordinate of voxel (0, 0, 0).
    ``voxel_size``  : (sx, sy, sz) spacing.
    ``rotation``    : optional 3x3 rotation (defaults to identity / axis-aligned).
    """
    R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    S = np.diag(np.asarray(voxel_size, dtype=np.float64))
    Mg = np.eye(4)
    Mg[:3, :3] = R @ S
    Mg[:3, 3] = np.asarray(origin_xyz, dtype=np.float64)
    return Mg


def apply_transform(Mg: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Map (N, 3) voxel indices to (N, 3) world coordinates with ``Mg``."""
    indices = np.asarray(indices, dtype=np.float64)
    homog = np.concatenate([indices, np.ones((indices.shape[0], 1))], axis=1)
    world = homog @ Mg.T
    return world[:, :3]


def burial_depth_field(
    grid_shape: Tuple[int, int, int],
    Mg: np.ndarray,
    dem_sampler: Callable[[np.ndarray, np.ndarray], np.ndarray],
    z_axis_world: int = 2,
    clip_negative: bool = True,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute ``H(x, y, z) = z_DEM(x, y) - z_voxel`` over the whole grid.

    ``dem_sampler(world_x, world_y) -> z_DEM`` returns terrain elevation at the
    horizontal world coordinates of each voxel. ``z_axis_world`` selects which
    world component is the vertical (up) axis.
    """
    zz, yy, xx = np.indices(grid_shape)
    idx = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    world = apply_transform(Mg, idx)
    horiz_axes = [a for a in range(3) if a != z_axis_world]
    z_dem = dem_sampler(world[:, horiz_axes[0]], world[:, horiz_axes[1]])
    H = (z_dem - world[:, z_axis_world]).reshape(grid_shape).astype(np.float32)
    if clip_negative:
        H = np.clip(H, 0.0, None)
    if valid_mask is not None:
        H = np.where(valid_mask, H, np.nan)
    return H
