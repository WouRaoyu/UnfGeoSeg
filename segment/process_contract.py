"""Python mirror of ``process/contract.h``.

The standalone VDB pipeline is the source of truth for coarse-stage feature
contracts. These helpers let the Python experiments validate and reproduce the
same column order for nnU-Net/NIfTI based smoke experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

FEATURE_GRIDS = ("vp", "vs", "depth")
STAT_SUFFIXES = ("Mean", "Q25", "Median", "Q75")
DISTRIBUTION_SUFFIXES = ("Std", "CV", "Skew", "IQR")
FIXED_SAMPLE_SUFFIXES = (
    "Center",
    "XMinus",
    "XPlus",
    "YMinus",
    "YPlus",
    "ZMinus",
    "ZPlus",
)
SAMPLE_VALID_RATIO_SUFFIXES = ("SampleValidRatio",)
SPATIAL_DISTRIBUTION_SUFFIXES = ("Std", "IQR")
SPATIAL_SUFFIXES = (
    "GradMagMean",
    "GradMagStd",
    "GradMagP75",
    "GradMagMax",
    "GradEnergyX",
    "GradEnergyY",
    "GradEnergyZ",
    "GradAnisoRatio",
    "RoughnessMean",
    "RoughnessStd",
    "RoughnessP75",
)


@dataclass(frozen=True)
class FeatureWindow:
    suffix: str
    size: Tuple[int, int, int]


def min_box(value: int) -> int:
    return max(3, int(value))


def is_baseline_feature_mode(mode: str) -> bool:
    return mode == "baseline"


def is_hybrid_feature_mode(mode: str) -> bool:
    return mode == "hybrid_spatial"


def is_spatial_v1_feature_mode(mode: str) -> bool:
    return mode == "spatial_v1"


def is_multiscale_feature_mode(mode: str) -> bool:
    return mode in {"multiscale", "directional"} or is_hybrid_feature_mode(mode)


def has_directional_features(mode: str) -> bool:
    return mode == "directional"


def is_supported_feature_mode(mode: str) -> bool:
    return mode in {
        "baseline",
        "multiscale",
        "directional",
        "hybrid_spatial",
        "spatial_v1",
    }


def feature_window_suffixes(mode: str) -> List[str]:
    suffixes = [""]
    if is_multiscale_feature_mode(mode):
        suffixes.extend(["Small", "Large"])
    if has_directional_features(mode):
        suffixes.extend(["Axial", "Lateral", "Vertical"])
    return suffixes


def feature_windows(base: Sequence[int], mode: str) -> List[FeatureWindow]:
    dz, dy, dx = (min_box(v) for v in base)
    windows = [FeatureWindow("", (dz, dy, dx))]
    if not is_multiscale_feature_mode(mode):
        return windows

    small = tuple(min_box(int(v) // 2) for v in base)
    large = tuple(min_box(int(v) * 2) for v in base)
    windows.extend(
        [
            FeatureWindow("Small", small),
            FeatureWindow("Large", large),
        ]
    )
    if has_directional_features(mode):
        windows.extend(
            [
                FeatureWindow("Axial", (large[0], small[1], small[2])),
                FeatureWindow("Lateral", (small[0], large[1], small[2])),
                FeatureWindow("Vertical", (small[0], small[1], large[2])),
            ]
        )
    return windows


def feature_columns(mode: str) -> List[str]:
    if not is_supported_feature_mode(mode):
        raise ValueError(f"unsupported process feature mode: {mode!r}")

    cols: List[str] = []
    for suffix in feature_window_suffixes(mode):
        for stat in STAT_SUFFIXES:
            for prop in FEATURE_GRIDS:
                cols.append(prop + suffix + stat)

    if has_directional_features(mode):
        for prop in FEATURE_GRIDS:
            cols.extend(
                [
                    prop + "GradMag",
                    prop + "GradX",
                    prop + "GradY",
                    prop + "GradZ",
                    prop + "Contrast",
                ]
            )

    if is_hybrid_feature_mode(mode):
        for suffix in DISTRIBUTION_SUFFIXES:
            for prop in FEATURE_GRIDS:
                cols.append(prop + suffix)
        for suffix in FIXED_SAMPLE_SUFFIXES:
            for prop in FEATURE_GRIDS:
                cols.append(prop + "Sample" + suffix)
        for suffix in SAMPLE_VALID_RATIO_SUFFIXES:
            for prop in FEATURE_GRIDS:
                cols.append(prop + suffix)

    if is_spatial_v1_feature_mode(mode):
        for suffix in SPATIAL_DISTRIBUTION_SUFFIXES:
            for prop in FEATURE_GRIDS:
                cols.append(prop + suffix)
        for prop in FEATURE_GRIDS:
            for suffix in SPATIAL_SUFFIXES:
                cols.append(prop + suffix)

    return cols
