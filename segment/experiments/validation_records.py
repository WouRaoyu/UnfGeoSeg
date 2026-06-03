"""Synthesize sparse validation records from dense labels.

Real tunnel-face records (TFR) and borehole/probe-hole logs are sparse 2-D faces
and 1-D trajectories through the volume. When only dense labels are available
(the smoke dataset), these helpers extract:

* **TFR faces**: one excavation-face section per sampled chainage; the face label
  is the majority foreground class on that section (or background).
* **Boreholes**: 1-D trajectories at fixed (y, x) columns walking along the axial
  (chainage) axis, with the per-voxel class along the trajectory.

If a real records CSV is available, prefer it; these synthesizers exist so the
fine-stage experiments are runnable end-to-end without bespoke field data.
All records carry the originating case id so held-out cases can be selected.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def faces_from_label(
    label: np.ndarray, axial_axis: int, n_sections: int, rng: np.random.Generator
) -> List[Dict]:
    """Sample ``n_sections`` excavation faces (chainage sections)."""
    n_ax = label.shape[axial_axis]
    sec_idx = np.linspace(0, n_ax - 1, min(n_sections, n_ax)).astype(int)
    records = []
    for s in sec_idx:
        face = np.take(label, s, axis=axial_axis)
        fg = face[face > 0]
        cls = int(np.bincount(fg).argmax()) if fg.size else 0
        records.append({"section": int(s), "class": cls})
    return records


def boreholes_from_label(
    label: np.ndarray, axial_axis: int, n_boreholes: int, rng: np.random.Generator
) -> List[Dict]:
    """Sample ``n_boreholes`` 1-D trajectories along the axial axis."""
    other_axes = [a for a in range(3) if a != axial_axis]
    shp = label.shape
    records = []
    for _ in range(n_boreholes):
        c0 = rng.integers(0, shp[other_axes[0]])
        c1 = rng.integers(0, shp[other_axes[1]])
        idx = [slice(None)] * 3
        idx[other_axes[0]] = c0
        idx[other_axes[1]] = c1
        traj = label[tuple(idx)]  # 1-D along axial
        records.append({"col": (int(c0), int(c1)), "labels": traj.astype(np.int16)})
    return records


def predicted_face_class(
    pred: np.ndarray, axial_axis: int, section: int, tolerance: int
) -> int:
    """Majority predicted class over a +/- tolerance band around a face."""
    n_ax = pred.shape[axial_axis]
    lo, hi = max(0, section - tolerance), min(n_ax, section + tolerance + 1)
    band = np.take(pred, range(lo, hi), axis=axial_axis)
    fg = band[band > 0]
    return int(np.bincount(fg).argmax()) if fg.size else 0


def predicted_trajectory(pred: np.ndarray, axial_axis: int, col) -> np.ndarray:
    idx = [slice(None)] * 3
    other_axes = [a for a in range(3) if a != axial_axis]
    idx[other_axes[0]] = col[0]
    idx[other_axes[1]] = col[1]
    return pred[tuple(idx)].astype(np.int16)
