"""Build the confidence-augmented nnU-Net dataset for the fine stage.

Strategy (robust to nnU-Net version drift): instead of forking nnU-Net's
dataloader, the per-voxel confidence (soft label) is carried as an **extra input
channel**. nnU-Net then augments it spatially in lockstep with the image and
segmentation for free; the custom trainer splits this channel back off before
the network and feeds it to the confidence-constrained loss.

One dataset is assembled per independent geology type (e.g.
``Dataset0XX_GeologyCC_<type>``), each binary, whose:

* image channels are ``[vp, vs, depth, confidence]`` (confidence = ``conf_<case>``),
* labels are the **binary hard pseudo-labels** (0/1 for this type) produced by
  the coarse stage; ``dataset.json`` labels are ``{background:0, <type>:1}``.

It also patches the generated plans so the confidence channel uses
``NoNormalization`` (preserving its [0, 1] range) while the physical channels
keep z-score normalization.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from ..io import Geometry, list_cases, read_volume, write_json, write_volume

CONFIDENCE_CHANNEL_NAME = "confidence"


def build_cc_dataset(
    src_raw_dir: str | Path,
    pseudolabel_dir: str | Path,
    dst_raw_dir: str | Path,
    channels: Sequence[str],
    classes: Sequence[str],
    file_ending: str = ".nii.gz",
) -> None:
    """Create the 4-channel confidence-augmented dataset.

    ``src_raw_dir``        : existing nnU-Net raw dataset (vp/vs/depth images).
    ``pseudolabel_dir``    : dir holding ``<case>`` hard labels, ``prob_<case>``
                             foreground probabilities and ``conf_<case>`` hard-label confidence.
    ``dst_raw_dir``        : output dataset dir.
    """
    src_raw_dir = Path(src_raw_dir)
    pseudolabel_dir = Path(pseudolabel_dir)
    dst_raw_dir = Path(dst_raw_dir)
    (dst_raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst_raw_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    n_phys = len(channels)
    # only cases that actually have a (hard) pseudo-label are included as training
    # cases, so nnU-Net's images/labels correspondence stays consistent.
    pl_cases = sorted(
        p.name[: -len(file_ending)]
        for p in pseudolabel_dir.glob(f"*{file_ending}")
        if not p.name.startswith("prob_")
        and not p.name.startswith("conf_")
    )
    src_cases = set(list_cases(src_raw_dir / "imagesTr", file_ending))
    cases = [c for c in pl_cases if c in src_cases]
    n_train = 0
    for case in cases:
        # copy physical channels
        geom: Geometry | None = None
        ref_shape = None
        for c in range(n_phys):
            src = src_raw_dir / "imagesTr" / f"{case}_{c:04d}{file_ending}"
            dst = dst_raw_dir / "imagesTr" / f"{case}_{c:04d}{file_ending}"
            shutil.copy2(src, dst)
            if geom is None:
                arr, geom = read_volume(src)
                ref_shape = arr.shape
        # confidence channel
        conf_path = pseudolabel_dir / f"conf_{case}{file_ending}"
        legacy_prob_path = pseudolabel_dir / f"prob_{case}{file_ending}"
        if conf_path.exists():
            conf, _ = read_volume(conf_path)
        elif legacy_prob_path.exists():
            # Backward compatibility: older pseudo-label folders stored the
            # assigned hard-label confidence in prob_<case>.
            conf, _ = read_volume(legacy_prob_path)
        else:
            conf = np.ones(ref_shape, dtype=np.float32)
        write_volume(
            conf.astype(np.float32),
            geom,
            dst_raw_dir / "imagesTr" / f"{case}_{n_phys:04d}{file_ending}",
            dtype=np.float32,
        )
        # hard pseudo-label
        hard_path = pseudolabel_dir / f"{case}{file_ending}"
        if hard_path.exists():
            shutil.copy2(hard_path, dst_raw_dir / "labelsTr" / f"{case}{file_ending}")
            n_train += 1

    all_channels = list(channels) + [CONFIDENCE_CHANNEL_NAME]
    channel_names = {str(i): name for i, name in enumerate(all_channels)}
    labels = {"background": 0}
    for i, name in enumerate(classes, start=1):
        labels[name] = i
    write_json(
        {
            "channel_names": channel_names,
            "labels": labels,
            "numTraining": n_train,
            "file_ending": file_ending,
        },
        dst_raw_dir / "dataset.json",
    )


def patch_plans_no_norm_confidence(
    preprocessed_dataset_dir: str | Path,
    plans_name: str = "nnUNetPlans",
    confidence_channel_index: int | None = None,
) -> None:
    """Force the confidence (last) channel to ``NoNormalization`` in the plans.

    In nnU-Net v2 ``normalization_schemes`` lives under each
    ``configurations[<cfg>]``; patch every configuration that has the list.
    NOTE: run this BEFORE preprocessing so the cached ``*.npy`` use the
    un-normalized confidence channel.
    """
    plans_path = Path(preprocessed_dataset_dir) / f"{plans_name}.json"
    with open(plans_path, "r", encoding="utf-8") as fh:
        plans = json.load(fh)
    changed = False
    for cfg in plans.get("configurations", {}).values():
        norm = cfg.get("normalization_schemes")
        if not isinstance(norm, list):
            continue
        idx = confidence_channel_index if confidence_channel_index is not None else len(norm) - 1
        if 0 <= idx < len(norm):
            norm[idx] = "NoNormalization"
            cfg["normalization_schemes"] = norm
            changed = True
    if changed:
        write_json(plans, plans_path)
