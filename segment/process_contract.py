"""Python mirror of ``process/contract.h``.

The standalone VDB pipeline is the source of truth for coarse-stage feature
contracts. These helpers let the Python experiments validate and reproduce the
same column order for nnU-Net/NIfTI based smoke experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import List, Sequence, Tuple

FEATURE_GRIDS = ("vp", "vs", "depth")
STAT_FEATURE_GRIDS = ("vp", "vs")
STAT_SUFFIXES = ("Mean", "Q25", "Median", "Q75")
DISTRIBUTION_SUFFIXES = ("Std", "Skew")
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
VALUE_AXIS_SAMPLES = (("Minus", -1), ("Center", 0), ("Plus", 1))
PHYSICAL_DERIVED_FEATURE_COLUMNS = (
    "densityValue",
    "vpVsRatioValue",
    "poissonRatioValue",
    "shearModulusGPaValue",
    "bulkModulusGPaValue",
    "youngsModulusGPaValue",
    "lambdaModulusGPaValue",
    "pWaveModulusGPaValue",
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
    return mode == "enhanced"


def is_spatial_v1_feature_mode(mode: str) -> bool:
    return mode == "spatial"


def is_values_feature_mode(mode: str) -> bool:
    return mode == "values"


def is_physical_feature_mode(mode: str) -> bool:
    return mode == "physical"


def is_origin_feature_mode(mode: str) -> bool:
    return mode == "origin"


def is_mean_feature_mode(mode: str) -> bool:
    return mode == "mean"


def is_random_feature_mode(mode: str) -> bool:
    return mode == "random"


def is_point_value_feature_mode(mode: str) -> bool:
    return (
        is_origin_feature_mode(mode)
        or is_physical_feature_mode(mode)
        or is_random_feature_mode(mode)
    )


def is_multiscale_feature_mode(mode: str) -> bool:
    return mode in {"multiscale", "directional"} or is_hybrid_feature_mode(mode)


def has_directional_features(mode: str) -> bool:
    return mode == "directional"


def is_supported_feature_mode(mode: str) -> bool:
    return (
        mode in {"baseline", "multiscale", "directional"}
        or is_spatial_v1_feature_mode(mode)
        or is_values_feature_mode(mode)
        or is_mean_feature_mode(mode)
        or is_point_value_feature_mode(mode)
        or is_hybrid_feature_mode(mode)
    )


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


def value_sample_offset(size: int, axis_code: int) -> int:
    size = min_box(size)
    if axis_code < 0:
        return -(size // 2)
    if axis_code > 0:
        return size - (size // 2) - 1
    return 0


def value_sample_offsets(base: Sequence[int]) -> List[Tuple[str, Tuple[int, int, int]]]:
    offsets: List[Tuple[str, Tuple[int, int, int]]] = []
    for x_name, x_code in VALUE_AXIS_SAMPLES:
        for y_name, y_code in VALUE_AXIS_SAMPLES:
            for z_name, z_code in VALUE_AXIS_SAMPLES:
                offsets.append(
                    (
                        f"X{x_name}Y{y_name}Z{z_name}",
                        (
                            value_sample_offset(base[0], x_code),
                            value_sample_offset(base[1], y_code),
                            value_sample_offset(base[2], z_code),
                        ),
                    )
                )
    return offsets


def finite_or_zero(value: float) -> float:
    return float(value) if isfinite(value) else 0.0


def physical_feature_values(vp: float, vs: float, depth_mean: float) -> List[float]:
    values = [float(vp), float(vs), float(depth_mean)]
    if not isfinite(vp) or not isfinite(vs) or vp <= 0.0 or vs <= 0.0:
        return values + [0.0] * len(PHYSICAL_DERIVED_FEATURE_COLUMNS)

    vp2 = float(vp) * float(vp)
    vs2 = float(vs) * float(vs)
    density = 700.0 * (float(vp) * float(vs)) ** 0.08
    ratio = float(vp) / float(vs)

    poisson = 0.0
    denom = 2.0 * (vp2 - vs2)
    if vp > vs and abs(denom) > 1.0e-12:
        poisson = (vp2 - 2.0 * vs2) / denom
        poisson = min(max(poisson, 0.0), 0.49)

    pa_to_gpa = 1.0e-9
    shear = density * vs2
    bulk = density * (vp2 - (4.0 / 3.0) * vs2)
    youngs = 2.0 * shear * (1.0 + poisson)
    lambda_mod = density * (vp2 - 2.0 * vs2)
    p_wave = density * vp2
    return values + [
        finite_or_zero(density),
        finite_or_zero(ratio),
        finite_or_zero(poisson),
        finite_or_zero(max(0.0, shear) * pa_to_gpa),
        finite_or_zero(max(0.0, bulk) * pa_to_gpa),
        finite_or_zero(max(0.0, youngs) * pa_to_gpa),
        finite_or_zero(max(0.0, lambda_mod) * pa_to_gpa),
        finite_or_zero(max(0.0, p_wave) * pa_to_gpa),
    ]


def feature_columns(mode: str) -> List[str]:
    if not is_supported_feature_mode(mode):
        raise ValueError(f"unsupported process feature mode: {mode!r}")

    cols: List[str] = []
    if is_values_feature_mode(mode):
        for sample_name, _offset in value_sample_offsets((3, 3, 3)):
            for prop in STAT_FEATURE_GRIDS:
                cols.append(prop + "Value" + sample_name)
        cols.append("depthMean")
        return cols

    if is_mean_feature_mode(mode):
        return [prop + "Mean" for prop in FEATURE_GRIDS]

    if is_origin_feature_mode(mode):
        return [prop + "Value" for prop in FEATURE_GRIDS]

    if is_point_value_feature_mode(mode):
        cols.extend(prop + "Value" for prop in STAT_FEATURE_GRIDS)
        cols.append("depthMean")
        if is_physical_feature_mode(mode):
            cols.extend(PHYSICAL_DERIVED_FEATURE_COLUMNS)
        return cols

    for suffix in feature_window_suffixes(mode):
        for stat in STAT_SUFFIXES:
            for prop in STAT_FEATURE_GRIDS:
                cols.append(prop + suffix + stat)

    if has_directional_features(mode):
        for prop in STAT_FEATURE_GRIDS:
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
            for prop in STAT_FEATURE_GRIDS:
                cols.append(prop + suffix)

    if is_spatial_v1_feature_mode(mode):
        for suffix in SPATIAL_DISTRIBUTION_SUFFIXES:
            for prop in STAT_FEATURE_GRIDS:
                cols.append(prop + suffix)
        for prop in STAT_FEATURE_GRIDS:
            for suffix in SPATIAL_SUFFIXES:
                cols.append(prop + suffix)

    cols.append("depthMean")
    return cols
