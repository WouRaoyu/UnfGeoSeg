#pragma once

#include <openvdb/openvdb.h>

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace dtpipeline {

static const std::vector<std::string> kFeatureGrids = { "vp", "vs", "depth" };
static const std::vector<std::string> kStatSuffixes = { "Mean", "Q25", "Median", "Q75" };
static const std::vector<std::string> kDistributionSuffixes = {
    "Std", "CV", "Skew", "IQR"
};
static const std::vector<std::string> kFixedSampleSuffixes = {
    "Center", "XMinus", "XPlus", "YMinus", "YPlus", "ZMinus", "ZPlus"
};
static const std::vector<std::string> kSampleValidRatioSuffixes = {
    "SampleValidRatio"
};
static const std::vector<std::string> kSpatialDistributionSuffixes = {
    "Std", "IQR"
};
static const std::vector<std::string> kSpatialSuffixes = {
    "GradMagMean", "GradMagStd", "GradMagP75", "GradMagMax",
    "GradEnergyX", "GradEnergyY", "GradEnergyZ", "GradAnisoRatio",
    "RoughnessMean", "RoughnessStd", "RoughnessP75"
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

inline bool isBaselineFeatureMode(const std::string& mode)
{
    return mode == "baseline";
}

inline bool isHybridFeatureMode(const std::string& mode)
{
    return mode == "hybrid_spatial";
}

inline bool isSpatialV1FeatureMode(const std::string& mode)
{
    return mode == "spatial_v1";
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

inline std::vector<std::pair<std::string, openvdb::Coord>> fixedSampleOffsets(
    const openvdb::Vec3I& base)
{
    const int ox = std::max(1, minBox(base.x()) / 4);
    const int oy = std::max(1, minBox(base.y()) / 4);
    const int oz = std::max(1, minBox(base.z()) / 4);
    return {
        { "Center", openvdb::Coord(0, 0, 0) },
        { "XMinus", openvdb::Coord(-ox, 0, 0) },
        { "XPlus", openvdb::Coord(ox, 0, 0) },
        { "YMinus", openvdb::Coord(0, -oy, 0) },
        { "YPlus", openvdb::Coord(0, oy, 0) },
        { "ZMinus", openvdb::Coord(0, 0, -oz) },
        { "ZPlus", openvdb::Coord(0, 0, oz) },
    };
}

inline std::vector<std::string> featureColumns(const std::string& mode)
{
    std::vector<std::string> cols;
    for (const auto& suffix : featureWindowSuffixes(mode)) {
        for (const auto& stat : kStatSuffixes) {
            for (const auto& prop : kFeatureGrids) {
                cols.push_back(prop + suffix + stat);
            }
        }
    }

    if (hasDirectionalFeatures(mode)) {
        for (const auto& prop : kFeatureGrids) {
            cols.push_back(prop + "GradMag");
            cols.push_back(prop + "GradX");
            cols.push_back(prop + "GradY");
            cols.push_back(prop + "GradZ");
            cols.push_back(prop + "Contrast");
        }
    }
    if (isHybridFeatureMode(mode)) {
        for (const auto& suffix : kDistributionSuffixes) {
            for (const auto& prop : kFeatureGrids) {
                cols.push_back(prop + suffix);
            }
        }
        for (const auto& suffix : kFixedSampleSuffixes) {
            for (const auto& prop : kFeatureGrids) {
                cols.push_back(prop + "Sample" + suffix);
            }
        }
        for (const auto& suffix : kSampleValidRatioSuffixes) {
            for (const auto& prop : kFeatureGrids) {
                cols.push_back(prop + suffix);
            }
        }
    }
    if (isSpatialV1FeatureMode(mode)) {
        for (const auto& suffix : kSpatialDistributionSuffixes) {
            for (const auto& prop : kFeatureGrids) {
                cols.push_back(prop + suffix);
            }
        }
        for (const auto& prop : kFeatureGrids) {
            for (const auto& suffix : kSpatialSuffixes) {
                cols.push_back(prop + suffix);
            }
        }
    }
    return cols;
}

inline std::vector<std::string> defaultFeatureColumns()
{
    return featureColumns("multiscale");
}

} // namespace dtpipeline
