#pragma once

#include <openvdb/openvdb.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace dtpipeline {

static const std::vector<std::string> kFeatureGrids = { "vp", "vs", "depth" };
static const std::vector<std::string> kStatFeatureGrids = { "vp", "vs" };
static const std::vector<std::string> kStatSuffixes = { "Mean", "Q25", "Median", "Q75" };
static const std::vector<std::string> kDistributionSuffixes = {
    "Std", "Skew"
};
static const std::vector<std::string> kSpatialDistributionSuffixes = {
    "Std", "IQR"
};
static const std::vector<std::string> kSpatialSuffixes = {
    "GradMagMean", "GradMagStd", "GradMagP75", "GradMagMax",
    "GradEnergyX", "GradEnergyY", "GradEnergyZ", "GradAnisoRatio",
    "RoughnessMean", "RoughnessStd", "RoughnessP75"
};
static const std::vector<std::pair<std::string, int>> kValueAxisSamples = {
    { "Minus", -1 },
    { "Center", 0 },
    { "Plus", 1 }
};
static const std::vector<std::string> kPhysicalDerivedFeatureColumns = {
    "densityValue",
    "vpVsRatioValue",
    "poissonRatioValue",
    "shearModulusGPaValue",
    "bulkModulusGPaValue",
    "youngsModulusGPaValue",
    "lambdaModulusGPaValue",
    "pWaveModulusGPaValue"
};

struct FeatureWindow
{
    std::string suffix;
    openvdb::Vec3I size;
};

inline int minBox(int value)
{
    return std::max(3, value);
}

static constexpr double kTunnelHeightMeters = 8.0;

inline int faceCenterZOffset(double voxelSizeZ)
{
    if (voxelSizeZ <= 0.0) return 0;
    return static_cast<int>(std::lround((kTunnelHeightMeters * 0.5) / voxelSizeZ));
}

inline bool isBaselineFeatureMode(const std::string& mode)
{
    return mode == "baseline";
}

inline bool isHybridFeatureMode(const std::string& mode)
{
    return mode == "enhanced";
}

inline bool isSpatialV1FeatureMode(const std::string& mode)
{
    return mode == "spatial";
}

inline bool isValuesFeatureMode(const std::string& mode)
{
    return mode == "values";
}

inline bool isPhysicalFeatureMode(const std::string& mode)
{
    return mode == "physical";
}

inline bool isOriginFeatureMode(const std::string& mode)
{
    return mode == "origin";
}

inline bool isMeanFeatureMode(const std::string& mode)
{
    return mode == "mean";
}

inline bool isRandomFeatureMode(const std::string& mode)
{
    return mode == "random";
}

inline bool isPointValueFeatureMode(const std::string& mode)
{
    return isOriginFeatureMode(mode) || isPhysicalFeatureMode(mode)
        || isRandomFeatureMode(mode);
}

inline bool isMultiscaleFeatureMode(const std::string& mode)
{
    return mode == "multiscale"
        || mode == "directional"
        || isHybridFeatureMode(mode);
}

inline bool hasDirectionalFeatures(const std::string& mode)
{
    return mode == "directional";
}

inline bool isSupportedFeatureMode(const std::string& mode)
{
    return mode == "baseline"
        || mode == "multiscale"
        || mode == "directional"
        || isSpatialV1FeatureMode(mode)
        || isValuesFeatureMode(mode)
        || isMeanFeatureMode(mode)
        || isPointValueFeatureMode(mode)
        || isHybridFeatureMode(mode);
}

inline std::vector<std::string> featureWindowSuffixes(const std::string& mode)
{
    std::vector<std::string> suffixes = { "" };
    if (isMultiscaleFeatureMode(mode)) {
        suffixes.push_back("Small");
        suffixes.push_back("Large");
    }
    if (hasDirectionalFeatures(mode)) {
        suffixes.push_back("Axial");
        suffixes.push_back("Lateral");
        suffixes.push_back("Vertical");
    }
    return suffixes;
}

inline std::vector<FeatureWindow> featureWindows(const openvdb::Vec3I& base,
    const std::string& mode)
{
    const int sx = minBox(base.x());
    const int sy = minBox(base.y());
    const int sz = minBox(base.z());

    std::vector<FeatureWindow> windows = {
        { "", openvdb::Vec3I(sx, sy, sz) },
    };

    if (!isMultiscaleFeatureMode(mode)) {
        return windows;
    }

    const int smallX = minBox(base.x() / 2);
    const int smallY = minBox(base.y() / 2);
    const int smallZ = minBox(base.z() / 2);
    const int largeX = minBox(base.x() * 2);
    const int largeY = minBox(base.y() * 2);
    const int largeZ = minBox(base.z() * 2);

    windows.push_back({ "Small", openvdb::Vec3I(smallX, smallY, smallZ) });
    windows.push_back({ "Large", openvdb::Vec3I(largeX, largeY, largeZ) });
    if (!hasDirectionalFeatures(mode)) {
        return windows;
    }
    windows.push_back({ "Axial", openvdb::Vec3I(largeX, smallY, smallZ) });
    windows.push_back({ "Lateral", openvdb::Vec3I(smallX, largeY, smallZ) });
    windows.push_back({ "Vertical", openvdb::Vec3I(smallX, smallY, largeZ) });
    return windows;
}

inline std::vector<FeatureWindow> featureWindows(const int depth, const int size,
    const std::string& mode)
{
    return featureWindows(openvdb::Vec3I(depth, size, size), mode);
}

inline int valueSampleOffset(const int size, const int axisCode)
{
    const int s = minBox(size);
    if (axisCode < 0) return -(s / 2);
    if (axisCode > 0) return s - (s / 2) - 1;
    return 0;
}

inline std::vector<std::pair<std::string, openvdb::Coord>> valueSampleOffsets(
    const openvdb::Vec3I& base)
{
    std::vector<std::pair<std::string, openvdb::Coord>> offsets;
    offsets.reserve(27);
    for (const auto& x : kValueAxisSamples) {
        for (const auto& y : kValueAxisSamples) {
            for (const auto& z : kValueAxisSamples) {
                offsets.push_back({
                    "X" + x.first + "Y" + y.first + "Z" + z.first,
                    openvdb::Coord(
                        valueSampleOffset(base.x(), x.second),
                        valueSampleOffset(base.y(), y.second),
                        valueSampleOffset(base.z(), z.second))
                });
            }
        }
    }
    return offsets;
}

inline float finiteOrZero(const double value)
{
    return std::isfinite(value) ? static_cast<float>(value) : 0.f;
}

inline std::vector<float> physicalFeatureValues(
    const float vp, const float vs, const float depthMean)
{
    std::vector<float> values = { vp, vs, depthMean };

    const double vpD = static_cast<double>(vp);
    const double vsD = static_cast<double>(vs);
    if (!std::isfinite(vpD) || !std::isfinite(vsD) || vpD <= 0.0 || vsD <= 0.0) {
        values.insert(values.end(), kPhysicalDerivedFeatureColumns.size(), 0.f);
        return values;
    }

    constexpr double kPaToGPa = 1.0e-9;
    const double vp2 = vpD * vpD;
    const double vs2 = vsD * vsD;
    const double density = 700.0 * std::pow(vpD * vsD, 0.08);
    const double ratio = vpD / vsD;

    double poisson = 0.0;
    const double poissonDenom = 2.0 * (vp2 - vs2);
    if (vpD > vsD && std::abs(poissonDenom) > 1.0e-12) {
        poisson = (vp2 - 2.0 * vs2) / poissonDenom;
        poisson = std::clamp(poisson, 0.0, 0.49);
    }

    const double shear = density * vs2;
    const double bulk = density * (vp2 - (4.0 / 3.0) * vs2);
    const double youngs = 2.0 * shear * (1.0 + poisson);
    const double lambda = density * (vp2 - 2.0 * vs2);
    const double pWave = density * vp2;

    values.push_back(finiteOrZero(density));
    values.push_back(finiteOrZero(ratio));
    values.push_back(finiteOrZero(poisson));
    values.push_back(finiteOrZero(std::max(0.0, shear) * kPaToGPa));
    values.push_back(finiteOrZero(std::max(0.0, bulk) * kPaToGPa));
    values.push_back(finiteOrZero(std::max(0.0, youngs) * kPaToGPa));
    values.push_back(finiteOrZero(std::max(0.0, lambda) * kPaToGPa));
    values.push_back(finiteOrZero(std::max(0.0, pWave) * kPaToGPa));
    return values;
}

inline std::vector<std::string> featureColumns(const std::string& mode)
{
    std::vector<std::string> cols;
    if (isValuesFeatureMode(mode)) {
        for (const auto& sample : valueSampleOffsets(openvdb::Vec3I(3, 3, 3))) {
            for (const auto& prop : kStatFeatureGrids) {
                cols.push_back(prop + "Value" + sample.first);
            }
        }
        cols.push_back("depthMean");
        return cols;
    }
    if (isMeanFeatureMode(mode)) {
        for (const auto& prop : kFeatureGrids) {
            cols.push_back(prop + "Mean");
        }
        return cols;
    }
    if (isOriginFeatureMode(mode)) {
        for (const auto& prop : kFeatureGrids) {
            cols.push_back(prop + "Value");
        }
        return cols;
    }
    if (isPointValueFeatureMode(mode)) {
        for (const auto& prop : kStatFeatureGrids) {
            cols.push_back(prop + "Value");
        }
        cols.push_back("depthMean");
        if (isPhysicalFeatureMode(mode)) {
            cols.insert(cols.end(), kPhysicalDerivedFeatureColumns.begin(),
                kPhysicalDerivedFeatureColumns.end());
        }
        return cols;
    }

    for (const auto& suffix : featureWindowSuffixes(mode)) {
        for (const auto& stat : kStatSuffixes) {
            for (const auto& prop : kStatFeatureGrids) {
                cols.push_back(prop + suffix + stat);
            }
        }
    }

    if (hasDirectionalFeatures(mode)) {
        for (const auto& prop : kStatFeatureGrids) {
            cols.push_back(prop + "GradMag");
            cols.push_back(prop + "GradX");
            cols.push_back(prop + "GradY");
            cols.push_back(prop + "GradZ");
            cols.push_back(prop + "Contrast");
        }
    }
    if (isHybridFeatureMode(mode)) {
        for (const auto& suffix : kDistributionSuffixes) {
            for (const auto& prop : kStatFeatureGrids) {
                cols.push_back(prop + suffix);
            }
        }
    }
    if (isSpatialV1FeatureMode(mode)) {
        for (const auto& suffix : kSpatialDistributionSuffixes) {
            for (const auto& prop : kStatFeatureGrids) {
                cols.push_back(prop + suffix);
            }
        }
        for (const auto& prop : kStatFeatureGrids) {
            for (const auto& suffix : kSpatialSuffixes) {
                cols.push_back(prop + suffix);
            }
        }
    }
    cols.push_back("depthMean");
    return cols;
}

inline std::vector<std::string> defaultFeatureColumns()
{
    return featureColumns("multiscale");
}

} // namespace dtpipeline
