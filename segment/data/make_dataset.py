"""Assemble an nnU-Net dataset from aligned multi-source volumes.

The raw acquisition format (TSP traces, DEM rasters, TFR tables) is
project-specific; this module writes already-aligned per-tunnel volumes into the
nnU-Net ``imagesTr`` / ``labelsTr`` layout and emits a compatible
``dataset.json``. It mirrors the channel convention already used on disk by
``Dataset005_Hardness`` (``_0000=vp, _0001=vs, _0002=depth``).

Label encoding: unfavorable-geology types are INDEPENDENT 0/1 targets (a voxel
may belong to several types at once), so labels are written as **per-class binary
masks** ``labelsTr/<case>_<type>.nii.gz`` and the available types are declared in
``dataset.json`` under ``geology_classes``. The downstream pipeline runs one
binary problem per type (see ``segment.cli`` ``--class``). A dataset with a
single foreground type degenerates to a standard binary nnU-Net dataset with one
``labelsTr/<case>.nii.gz``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np

from ..io import Geometry, write_json, write_volume

LabelArg = Union[np.ndarray, Mapping[str, np.ndarray], None]


def write_dataset_json(
    dataset_dir: str | Path,
    channels: Sequence[str],
    classes: Sequence[str],
    num_training: int,
    file_ending: str = ".nii.gz",
) -> None:
    """Write ``dataset.json``. ``classes`` are the independent geology types."""
    channel_names = {str(i): name for i, name in enumerate(channels)}
    classes = list(classes)
    if len(classes) <= 1:
        # single foreground type -> standard binary nnU-Net dataset
        name = classes[0] if classes else "unfavorable"
        ds = {
            "channel_names": channel_names,
            "labels": {"background": 0, name: 1},
            "numTraining": int(num_training),
            "file_ending": file_ending,
        }
    else:
        # multiple independent types -> per-class binary masks on disk; the
        # authoritative type list is ``geology_classes`` (``labels`` keeps only
        # background since there is no single mutually-exclusive label map).
        ds = {
            "channel_names": channel_names,
            "labels": {"background": 0},
            "geology_classes": classes,
            "numTraining": int(num_training),
            "file_ending": file_ending,
        }
    write_json(ds, Path(dataset_dir) / "dataset.json")


def write_case(
    dataset_dir: str | Path,
    case_id: str,
    channel_volumes: Sequence[np.ndarray],
    label: LabelArg,
    geometry: Geometry,
    file_ending: str = ".nii.gz",
) -> None:
    """Write one case: ``imagesTr/<case>_000c`` channels + label mask(s).

    ``label`` may be ``None``, a single binary array (-> ``labelsTr/<case>``), or
    a mapping ``{type: binary_array}`` (-> ``labelsTr/<case>_<type>`` each 0/1).
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    for c, vol in enumerate(channel_volumes):
        write_volume(
            vol, geometry, images_dir / f"{case_id}_{c:04d}{file_ending}", dtype=np.float32
        )
    if label is None:
        return
    if isinstance(label, Mapping):
        for type_name, arr in label.items():
            write_volume(
                (np.asarray(arr) > 0).astype(np.uint8),
                geometry,
                labels_dir / f"{case_id}_{type_name}{file_ending}",
                dtype=np.uint8,
            )
    else:
        write_volume(
            (np.asarray(label) > 0).astype(np.uint8),
            geometry,
            labels_dir / f"{case_id}{file_ending}",
            dtype=np.uint8,
        )


def build_dataset(
    dataset_dir: str | Path,
    cases: Mapping[str, dict],
    channels: Sequence[str],
    classes: Sequence[str],
    file_ending: str = ".nii.gz",
) -> None:
    """Assemble a full dataset.

    ``cases`` maps ``case_id -> {"channels": [vp, vs, depth], "label": L,
    "geometry": Geometry}`` where ``L`` is ``None``, a single binary array, or a
    ``{type: binary_array}`` mapping (one per independent geology type).
    """
    n_train = 0
    for case_id, payload in cases.items():
        write_case(
            dataset_dir,
            case_id,
            payload["channels"],
            payload.get("label"),
            payload["geometry"],
            file_ending,
        )
        if payload.get("label") is not None:
            n_train += 1
    write_dataset_json(dataset_dir, channels, classes, n_train, file_ending)
