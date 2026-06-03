"""Lightweight NIfTI / nnU-Net I/O helpers built on SimpleITK.

All volumes are returned as numpy arrays in ``(z, y, x)`` order (SimpleITK's
``GetArrayFromImage`` convention) together with their geometry so results can be
written back with an identical affine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


@dataclass
class Geometry:
    spacing: Tuple[float, ...]
    origin: Tuple[float, ...]
    direction: Tuple[float, ...]

    @classmethod
    def from_image(cls, img: sitk.Image) -> "Geometry":
        return cls(img.GetSpacing(), img.GetOrigin(), img.GetDirection())

    def apply_to(self, img: sitk.Image) -> sitk.Image:
        img.SetSpacing(self.spacing)
        img.SetOrigin(self.origin)
        img.SetDirection(self.direction)
        return img


def read_volume(path: str | Path) -> Tuple[np.ndarray, Geometry]:
    """Read a single NIfTI volume -> ``(array[z,y,x], geometry)``."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), Geometry.from_image(img)


def write_volume(
    array: np.ndarray, geometry: Geometry, path: str | Path, dtype=None
) -> None:
    """Write ``array[z,y,x]`` to ``path`` preserving ``geometry``."""
    if dtype is not None:
        array = array.astype(dtype)
    img = sitk.GetImageFromArray(array)
    geometry.apply_to(img)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def read_case(
    images_dir: str | Path, case_id: str, num_channels: int, file_ending: str = ".nii.gz"
) -> Tuple[np.ndarray, Geometry]:
    """Read an nnU-Net case ``<case_id>_0000..000N`` -> ``(array[C,z,y,x], geom)``."""
    images_dir = Path(images_dir)
    channels: List[np.ndarray] = []
    geom: Geometry | None = None
    for c in range(num_channels):
        arr, g = read_volume(images_dir / f"{case_id}_{c:04d}{file_ending}")
        channels.append(arr)
        geom = geom or g
    return np.stack(channels, axis=0), geom  # (C, z, y, x)


def resolve_label_path(
    labels_dir: str | Path,
    case_id: str,
    class_name: str | None = None,
    file_ending: str = ".nii.gz",
    strict_per_class: bool = False,
) -> Path:
    """Resolve the label file for one case and (independent) geology type.

    Each unfavorable-geology type is an independent 0/1 target, so labels are
    supplied as per-class binary masks ``labelsTr/<case>_<type>.nii.gz``. When a
    dataset carries only a single foreground type (e.g. the binary smoke
    dataset) the plain ``labelsTr/<case>.nii.gz`` is used as a fallback. Set
    ``strict_per_class=True`` for multi-type datasets so a missing per-class mask
    fails loudly instead of silently turning all nonzero old multiclass labels
    into foreground.
    """
    labels_dir = Path(labels_dir)
    if class_name is not None:
        per_class = labels_dir / f"{case_id}_{class_name}{file_ending}"
        if per_class.exists():
            return per_class
        if strict_per_class:
            raise FileNotFoundError(
                f"Missing per-class binary mask for class {class_name!r}: {per_class}"
            )
    return labels_dir / f"{case_id}{file_ending}"


def list_cases(images_dir: str | Path, file_ending: str = ".nii.gz") -> List[str]:
    """List nnU-Net case ids in an ``imagesTr``/``imagesTs`` folder."""
    images_dir = Path(images_dir)
    ids = set()
    for p in images_dir.glob(f"*_0000{file_ending}"):
        ids.add(p.name[: -len(f"_0000{file_ending}")])
    return sorted(ids)


def read_dataset_json(dataset_dir: str | Path) -> Dict:
    with open(Path(dataset_dir) / "dataset.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)


def nnunet_dirs() -> Dict[str, Path]:
    """Resolve the three nnU-Net base directories from the environment."""
    import os

    out = {}
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        val = os.environ.get(key)
        out[key] = Path(val) if val else None
    return out
