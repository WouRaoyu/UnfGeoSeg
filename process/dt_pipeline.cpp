/**
 * @ Author: WouRaoyu
 * @ Description: Standalone, self-contained pipeline tool combining two stages
 *   of the weakly-supervised workflow into a single executable. It has no
 *   dependency on the DTGeoStudio source tree; everything it needs (windowed
 *   feature contract, descriptive statistics) is inlined below.
 *
 *   Sub-commands
 *   ------------
 *   infer   Sliding-window inference that turns a coarse classifier into
 *           voxel-level SOFT pseudo-labels. For every `input_*.vdb` volume in a
 *           single flat folder it computes local statistical features, runs the
 *           trained model through the embedded Python interpreter
 *           (model_utils.execProb), and writes, per unfavorable-geology type, a
 *           soft probability grid (`prob_<type>`) plus a hard label grid
 *           (`<type>`) to `result_*.vdb` in the same folder. Per-volume metadata
 *           is recorded in dataset.json for the export stage.
 *
 *   export  Converts the inference result volumes (.vdb) to a nnU-Net / MONAI
 *           style nii.gz dataset for the downstream fine-grained 3D-TransUNet.
 *           For each volume listed in dataset.json it writes, per type:
 *             images{Tr,Ts}/<case>_0000|0001|0002.nii.gz   (vp / vs / depth)
 *             labels{Tr,Ts}/<case>.nii.gz                   (hard label, int)
 *             probs{Tr,Ts}/<case>.nii.gz                    (soft prob, float)
 *
 *   Unlike the original project tools, the inference stage no longer walks a
 *   project/site/TSP directory tree. All `input_*.vdb` volumes are expected to
 *   live directly inside a single `--input-dir` folder.
 *
 *   Usage
 *   -----
 *     dt_pipeline infer  --input-dir <dir> --model <pkl> [options]
 *     dt_pipeline export --dataset <dataset.json> --out <dir> --type <name|idx> [options]
 */

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Clip.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/ValueTransformer.h>

#include <tbb/parallel_for.h>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <vtkSmartPointer.h>
#include <vtkNIFTIImageWriter.h>
#include <vtkImageData.h>
#include <vtkNew.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace py = pybind11;
namespace fs = std::filesystem;

// ===========================================================================
// Inlined feature contract (was test/contract.h, namespace dtpipeline).
// Keeps the train / infer feature spaces identical without pulling in any
// project headers.
// ===========================================================================
namespace dtpipeline {

static const std::vector<std::string> kFeatureGrids = { "vp", "vs", "depth" };
static const std::vector<std::string> kStatSuffixes = { "Mean", "Q25", "Median", "Q75" };

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

inline bool isSupportedFeatureMode(const std::string& mode)
{
    return mode == "baseline" || mode == "directional_multiscale";
}

inline std::vector<std::string> featureWindowSuffixes(const std::string& mode)
{
    std::vector<std::string> suffixes = { "" };
    if (!isBaselineFeatureMode(mode)) {
        suffixes.push_back("Small");
        suffixes.push_back("Large");
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

    if (isBaselineFeatureMode(mode)) {
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
    windows.push_back({ "Axial", openvdb::Vec3I(largeX, smallY, smallZ) });
    windows.push_back({ "Lateral", openvdb::Vec3I(smallX, largeY, smallZ) });
    windows.push_back({ "Vertical", openvdb::Vec3I(smallX, smallY, largeZ) });
    return windows;
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

    if (!isBaselineFeatureMode(mode)) {
        for (const auto& prop : kFeatureGrids) {
            cols.push_back(prop + "GradMag");
            cols.push_back(prop + "GradX");
            cols.push_back(prop + "GradY");
            cols.push_back(prop + "GradZ");
            cols.push_back(prop + "Contrast");
        }
    }
    return cols;
}

// ---------------------------------------------------------------------------
// Inlined descriptive statistics (was component/SemanticStatistic + ToolStat).
// ---------------------------------------------------------------------------
struct SemanticStatistic final
{
    float mean   = 0.f; ///< Arithmetic mean
    float q25    = 0.f; ///< First quartile (25th percentile)
    float median = 0.f; ///< Median (50th percentile)
    float q75    = 0.f; ///< Third quartile (75th percentile)
    float iqr    = 0.f; ///< Inter-quartile range (q75 - q25)
    float std    = 0.f; ///< Standard deviation
    float cv     = 0.f; ///< Coefficient of variation (std / mean)
    float skew   = 0.f; ///< Skewness
};

inline void computeStatistic(SemanticStatistic& stats,
    const std::vector<float>& vals, bool hlevel)
{
    size_t n = vals.size();
    if (n == 0) return;

    auto data = vals;

    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    stats.mean = sum / n;

    // Quartiles need only three order statistics, so use partial selection
    // (nth_element, O(n) each) instead of a full O(n log n) sort.
    const size_t kQ25 = static_cast<size_t>(n * 0.25);
    const size_t kMed = static_cast<size_t>(n * 0.50);
    const size_t kQ75 = static_cast<size_t>(n * 0.75);

    std::nth_element(data.begin(), data.begin() + kQ25, data.end());
    stats.q25 = data[kQ25];
    std::nth_element(data.begin(), data.begin() + kMed, data.end());
    stats.median = data[kMed];
    std::nth_element(data.begin(), data.begin() + kQ75, data.end());
    stats.q75 = data[kQ75];
    stats.iqr = stats.q75 - stats.q25;

    if (!hlevel) return;

    float squareSumDiff = 0.0f;
    float cubicSumDiff = 0.0f;
    for (float val : data) {
        float diff = val - stats.mean;
        squareSumDiff += diff * diff;
        cubicSumDiff += diff * diff * diff;
    }

    float variance = squareSumDiff / n;
    stats.std = std::sqrt(variance);
    stats.cv = (stats.mean != 0) ? (stats.std / stats.mean) : 0.0f;

    if (stats.std > 1e-6) {
        float m3 = cubicSumDiff / n;
        stats.skew = m3 / (variance * stats.std);
    } else {
        stats.skew = 0.0f;
    }
}

} // namespace dtpipeline

using dtpipeline::SemanticStatistic;

static std::string pathToUtf8(const fs::path& path)
{
    auto value = path.u8string();
#if defined(__cpp_char8_t)
    return std::string(value.begin(), value.end());
#else
    return value;
#endif
}

// ===========================================================================
// Stage: infer (was test/wrapper.cpp)
// ===========================================================================
namespace infer {

// Const value accessor reused across the windowed lookups of a single voxel.
using FloatConstAccessor = openvdb::FloatGrid::ConstAccessor;

static SemanticStatistic sampleStat(const FloatConstAccessor& acc,
    const openvdb::Coord& center, const openvdb::Vec3I& size)
{
    SemanticStatistic stat;

    openvdb::Vec3I halfSize = size * 0.5;
    openvdb::Coord minCoord(center - halfSize);
    openvdb::Coord maxCoord(minCoord + size - openvdb::Coord(1, 1, 1));
    openvdb::CoordBBox bbox(minCoord, maxCoord);

    std::vector<float> values;
    values.reserve(static_cast<size_t>(size.x()) * size.y() * size.z());
    for (int x = bbox.min().x(); x <= bbox.max().x(); ++x) {
        for (int y = bbox.min().y(); y <= bbox.max().y(); ++y) {
            for (int z = bbox.min().z(); z <= bbox.max().z(); ++z) {
                openvdb::Coord ijk(x, y, z);
                if (acc.isValueOn(ijk)) {
                    values.push_back(acc.getValue(ijk));
                }
            }
        }
    }
    if (!values.empty()) {
        dtpipeline::computeStatistic(stat, values, false);
    }
    return stat;
}

static void appendFeatureRow(const std::array<FloatConstAccessor, 3>& accs,
    const openvdb::Coord& center, const openvdb::Vec3I& boxSize,
    const std::string& featureMode,
    py::detail::unchecked_mutable_reference<float, 2>& values,
    const size_t row)
{
    size_t col = 0;
    std::array<SemanticStatistic, 3> base{};
    for (const auto& win : dtpipeline::featureWindows(boxSize, featureMode)) {
        std::array<SemanticStatistic, 3> stats;
        for (size_t i = 0; i < accs.size(); ++i) {
            stats[i] = sampleStat(accs[i], center, win.size);
        }
        if (win.suffix.empty()) {
            base = stats;
        }
        for (const auto& stat : stats) values(row, col++) = stat.mean;
        for (const auto& stat : stats) values(row, col++) = stat.q25;
        for (const auto& stat : stats) values(row, col++) = stat.median;
        for (const auto& stat : stats) values(row, col++) = stat.q75;
    }

    if (dtpipeline::isBaselineFeatureMode(featureMode)) {
        return;
    }

    for (size_t i = 0; i < accs.size(); ++i) {
        const auto& acc = accs[i];
        float gx = 0.5f * (acc.getValue(center + openvdb::Coord(1, 0, 0))
            - acc.getValue(center - openvdb::Coord(1, 0, 0)));
        float gy = 0.5f * (acc.getValue(center + openvdb::Coord(0, 1, 0))
            - acc.getValue(center - openvdb::Coord(0, 1, 0)));
        float gz = 0.5f * (acc.getValue(center + openvdb::Coord(0, 0, 1))
            - acc.getValue(center - openvdb::Coord(0, 0, 1)));
        values(row, col++) = std::sqrt(gx * gx + gy * gy + gz * gz);
        values(row, col++) = gx;
        values(row, col++) = gy;
        values(row, col++) = gz;
        values(row, col++) = base[i].iqr;
    }
}

// Parallel feature extraction: each leaf owns a disjoint row range (via
// `indexMap`), so leaves are processed concurrently without synchronisation.
class FeatureExtractor
{
public:
    using LeafManagerType = openvdb::tree::LeafManager<const openvdb::FloatTree>;
    using PyBufferRef = py::detail::unchecked_mutable_reference<float, 2>;

    FeatureExtractor(
        const std::array<openvdb::FloatGrid::ConstPtr, 3>& grids,
        const LeafManagerType& leafs,
        const std::vector<size_t>& indexMap,
        const openvdb::Vec3I& boxSize,
        const std::string& featureMode,
        PyBufferRef& values,
        bool showProgress)
        : mGrids(grids)
        , mLeafs(leafs)
        , mIndexMap(indexMap)
        , mBoxSize(boxSize)
        , mFeatureMode(featureMode)
        , mValues(values)
        , mTotalLeaves(leafs.leafCount())
        , mShowProgress(showProgress)
        , mDone(std::make_shared<std::atomic<size_t>>(0))
        , mLastPct(std::make_shared<std::atomic<int>>(-1))
    {
    }

    void runParallel()
    {
        tbb::parallel_for(mLeafs.getRange(), *this);
        if (mShowProgress) {
            std::cout << "\r    features 100%   " << std::endl;
        }
    }

    inline void operator()(const typename LeafManagerType::RangeType& range) const
    {
        std::array<FloatConstAccessor, 3> accs = {
            mGrids[0]->getConstAccessor(),
            mGrids[1]->getConstAccessor(),
            mGrids[2]->getConstAccessor(),
        };

        for (size_t n = range.begin(); n < range.end(); ++n) {
            size_t row = mIndexMap[n];
            for (auto it = mLeafs.leaf(n).cbeginValueOn(); it; ++it) {
                appendFeatureRow(accs, it.getCoord(), mBoxSize, mFeatureMode,
                    mValues, row++);
            }
            reportProgress();
        }
    }

private:
    inline void reportProgress() const
    {
        if (!mShowProgress || mTotalLeaves == 0) return;
        size_t done = mDone->fetch_add(1, std::memory_order_relaxed) + 1;
        int pct = static_cast<int>(done * 100 / mTotalLeaves);
        int last = mLastPct->load(std::memory_order_relaxed);
        if (pct > last && mLastPct->compare_exchange_strong(last, pct)) {
            std::cout << "\r    features " << pct << "%   " << std::flush;
        }
    }

    const std::array<openvdb::FloatGrid::ConstPtr, 3>& mGrids;
    const LeafManagerType& mLeafs;
    const std::vector<size_t>& mIndexMap;
    const openvdb::Vec3I mBoxSize;
    const std::string mFeatureMode;
    PyBufferRef& mValues;
    const size_t mTotalLeaves;
    const bool mShowProgress;
    std::shared_ptr<std::atomic<size_t>> mDone;
    std::shared_ptr<std::atomic<int>> mLastPct;
};

// Writes, for one model output, the soft probability of the predicted class into
// `mProbs` and the hard class id into `mTypes`.
class VolumePetTAssigner
{
public:
    using LeafManagerType = openvdb::tree::LeafManager<openvdb::FloatTree>;
    using PyBufferRef = py::detail::unchecked_reference<float, 2>;

    VolumePetTAssigner(
        LeafManagerType& probs,
        LeafManagerType& types,
        const PyBufferRef& values,
        const size_t numClasses,
        const std::vector<size_t>& indexMap)
        : mProbs(probs)
        , mTypes(types)
        , mValues(values)
        , mNumClasses(numClasses)
        , mIndexMap(indexMap)
        , mVoxelsPerLeaf(openvdb::FloatTree::LeafNodeType::NUM_VOXELS)
        , mTrueNumbers(new size_t[mIndexMap.size()])
    {
    }

    void runParallel()
    {
        tbb::parallel_for(mProbs.getRange(), *this);
    }

    size_t trueNumber()
    {
        size_t totalNumber = 0;
        for (size_t i = 0; i < mIndexMap.size(); i++) {
            totalNumber += mTrueNumbers.get()[i];
        }
        return totalNumber;
    }

    inline void operator()(const typename LeafManagerType::RangeType& range) const
    {
        using openvdb::Index64;
        using ValueOnIter = typename openvdb::FloatTree::LeafNodeType::ValueOnIter;

        size_t index = 0;
        Index64 activeVoxels = 0;

        for (size_t n = range.begin(); n < range.end(); ++n) {
            index = mIndexMap[n]; mTrueNumbers.get()[n] = 0;
            ValueOnIter pit = mProbs.leaf(n).beginValueOn();
            ValueOnIter tit = mTypes.leaf(n).beginValueOn();
            activeVoxels = mProbs.leaf(n).onVoxelCount();

            if (activeVoxels <= mVoxelsPerLeaf) {
                for (; pit && tit; ++pit, ++tit) {
                    size_t hard = 0;
                    float positiveProb = 0.f;
                    if (mNumClasses == 2) {
                        positiveProb = mValues(index, 1);
                        hard = positiveProb >= 0.5f ? 1 : 0;
                    }
                    else {
                        size_t best = 0;
                        float bestProb = mValues(index, 0);
                        for (size_t c = 1; c < mNumClasses; ++c) {
                            float p = mValues(index, c);
                            if (p > bestProb) { bestProb = p; best = c; }
                        }
                        positiveProb = bestProb;
                        hard = best;
                    }
                    pit.setValue(positiveProb);
                    tit.setValue(static_cast<float>(hard));
                    if (hard != 0) {
                        mTrueNumbers.get()[n] += 1;
                    }
                    ++index;
                }
            }
            else {
                throw std::runtime_error("unknown error");
            }
        }
    }

private:
    LeafManagerType& mProbs;
    LeafManagerType& mTypes;
    const PyBufferRef& mValues;
    const size_t mNumClasses;
    const std::vector<size_t>& mIndexMap;
    const openvdb::Index64 mVoxelsPerLeaf;
    std::shared_ptr<size_t> mTrueNumbers;
};

py::array_t<float> fetchStatFromVolume(const openvdb::GridPtrVec& vec,
    openvdb::Vec3I boxSize,
    const std::string& featureMode,
    bool showProgress = true)
{
    int fetched = 0;
    size_t totalNum = 0;
    std::array<openvdb::FloatGrid::ConstPtr, 3> fgrids{ nullptr, nullptr, nullptr };

    for (const auto& grid : vec) {
        auto fptr = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
        if (fptr == nullptr) continue;

        auto numv = openvdb::tools::countActiveVoxels(fptr->tree());
        std::string name = fptr->getName();

        auto findItr = std::find(dtpipeline::kFeatureGrids.begin(),
            dtpipeline::kFeatureGrids.end(), name);
        if (findItr == dtpipeline::kFeatureGrids.end()) continue;

        auto idx = std::distance(dtpipeline::kFeatureGrids.begin(), findItr);

        if (totalNum == 0) {
            totalNum = numv;
        }
        else if (totalNum != numv) {
            throw std::runtime_error("error voxel numbers not equal, need transfer");
        }

        fgrids[idx] = fptr;
        fetched++;
    }

    if ((size_t)fetched != dtpipeline::kFeatureGrids.size()) {
        throw std::runtime_error("features number error");
    }

    const auto featureCols = dtpipeline::featureColumns(featureMode);
    auto result = py::array_t<float>({ totalNum, featureCols.size() });

    auto buf = result.mutable_unchecked<2>();

    using openvdb::tree::LeafManager;
    LeafManager<const openvdb::FloatTree> leafs(fgrids[0]->tree());

    std::vector<size_t> indexMap(leafs.leafCount());
    size_t row = 0;
    for (size_t l = 0, L = leafs.leafCount(); l < L; ++l) {
        indexMap[l] = row;
        row += leafs.leaf(l).onVoxelCount();
    }

    FeatureExtractor extractor(fgrids, leafs, indexMap, boxSize, featureMode,
        buf, showProgress);
    extractor.runParallel();

    return result;
}

openvdb::GridPtrVec assignForVolumePetT(const py::list& out,
    openvdb::FloatGrid::ConstPtr org,
    const std::vector<std::string>& typeNames,
    std::vector<float>& status)
{
    if (out.size() != typeNames.size()) {
        throw std::runtime_error("Result error: output count does not match class list");
    }

    using openvdb::tree::LeafManager;

    LeafManager<const openvdb::FloatTree> orgm(org->tree());
    openvdb::Index64 voxelsPerLeaf = openvdb::FloatTree::LeafNodeType::NUM_VOXELS;
    std::vector<size_t> indexMap(orgm.leafCount());
    size_t voxelCount = 0;
    for (size_t l = 0, L = orgm.leafCount(); l < L; ++l) {
        indexMap[l] = voxelCount;
        voxelCount += std::min(orgm.leaf(l).onVoxelCount(), voxelsPerLeaf);
    }

    openvdb::GridPtrVec grids;
    status.assign(typeNames.size(), 0.f);

    for (size_t i = 0; i < typeNames.size(); ++i) {
        auto ary = out[i].cast<py::array_t<float>>();
        auto buf = ary.unchecked<2>();
        const size_t numClasses = static_cast<size_t>(ary.shape(1));

        auto probGrid = org->deepCopy();
        LeafManager<openvdb::FloatTree> probm(probGrid->tree());

        auto typeGrid = org->deepCopy();
        LeafManager<openvdb::FloatTree> typem(typeGrid->tree());

        VolumePetTAssigner assigner(probm, typem, buf, numClasses, indexMap);
        assigner.runParallel();

        status[i] = voxelCount ? float(assigner.trueNumber()) / voxelCount : 0.f;

        probGrid->setName("prob_" + typeNames[i]);
        grids.push_back(probGrid);

        typeGrid->setName(typeNames[i]);
        grids.push_back(typeGrid);
    }

    return grids;
}

static bool runItemInference(const fs::path& mdl, const fs::path& feat, const fs::path& out,
    const openvdb::Vec3I& boxSize, const std::vector<std::string>& typeNames,
    const std::string& featureMode,
    const std::vector<std::string>& featureCols,
    std::vector<float>& status)
{
    try {
        py::module script = py::module::import("model_utils");

        openvdb::io::File f(feat.string());
        f.open(); auto grids = f.getGrids();

        py::array_t<float> input = fetchStatFromVolume(*grids, boxSize, featureMode);

        py::list strs;
        for (const auto& nm : featureCols) {
            strs.append(nm);
        }

        std::cout << "    running model..." << std::endl;
        py::object result = script.attr("execProb")(mdl.string(), input, strs);

        std::cout << "    writing " << out.filename().string() << "..." << std::endl;

        auto fptr = openvdb::gridPtrCast<openvdb::FloatGrid>(grids->front());

        const auto& buf = result.cast<py::list>();

        auto vlms = assignForVolumePetT(buf, fptr, typeNames, status);

        openvdb::io::File output(out.string());
        output.write(vlms);
        output.close();
        return true;
    }
    catch (py::error_already_set const& e) {
        std::cerr << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Inference failed for " << feat.string() << ": " << e.what() << std::endl;
    }
    return false;
}

// ---------------------------------------------------------------------------
// CLI / orchestration
// ---------------------------------------------------------------------------
struct InferOptions
{
    fs::path inputDir;                   ///< flat folder holding input_*.vdb volumes
    fs::path model;                      ///< trained .pkl model
    fs::path pyhome;                     ///< python home for the embedded interpreter
    fs::path scripts;                    ///< directory containing model_utils.py
    fs::path datasetOut;                 ///< dataset.json path (default: input-dir/dataset.json)
    int depth = 4;                       ///< window box size along the tunnel axis (voxels)
    int size = 32;                       ///< window box size on the cross-section (voxels)
    std::string featureMode = "directional_multiscale";
    std::vector<std::string> classes = { "fragment", "hardness", "watery" };
    std::set<std::string> heldOut;       ///< case ids reserved for validation
};

static std::vector<std::string> splitList(const std::string& s)
{
    std::vector<std::string> v; std::string cur;
    for (char c : s) { if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += c; }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

static bool parseInferArgs(int argc, char** argv, InferOptions& opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--input-dir") opt.inputDir = next("--input-dir");
        else if (a == "--model") opt.model = next("--model");
        else if (a == "--pyhome") opt.pyhome = next("--pyhome");
        else if (a == "--scripts") opt.scripts = next("--scripts");
        else if (a == "--dataset-out") opt.datasetOut = next("--dataset-out");
        else if (a == "--depth") opt.depth = std::stoi(next("--depth"));
        else if (a == "--size") opt.size = std::stoi(next("--size"));
        else if (a == "--feature-mode") opt.featureMode = next("--feature-mode");
        else if (a == "--classes") opt.classes = splitList(next("--classes"));
        else if (a == "--held-out") { for (auto& s : splitList(next("--held-out"))) opt.heldOut.insert(s); }
        else std::cerr << "Unknown option ignored: " << a << "\n";
    }
    if (opt.inputDir.empty() || opt.model.empty()) return false;
    if (opt.datasetOut.empty()) opt.datasetOut = opt.inputDir / "dataset.json";
    if (opt.scripts.empty() && !opt.pyhome.empty()) opt.scripts = opt.pyhome.parent_path() / "scripts";
    return true;
}

static fs::path featureContractPath(const fs::path& modelPath)
{
    return modelPath.parent_path() / (modelPath.stem().string() + ".features.json");
}

static void printVectorMismatch(const std::vector<std::string>& expected,
    const std::vector<std::string>& actual)
{
    if (expected.size() != actual.size()) {
        std::cerr << "  feature_columns size expected " << expected.size()
            << ", got " << actual.size() << "\n";
        return;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::cerr << "  first feature_columns mismatch at index " << i
                << ": expected " << expected[i] << ", got " << actual[i] << "\n";
            return;
        }
    }
}

static bool validateFeatureContract(const InferOptions& opt,
    const std::vector<std::string>& expectedColumns)
{
    if (!dtpipeline::isSupportedFeatureMode(opt.featureMode)) {
        std::cerr << "Unsupported feature mode: " << opt.featureMode
            << " (expected baseline or directional_multiscale)\n";
        return false;
    }

    const fs::path contract = featureContractPath(opt.model);
    if (!fs::exists(contract)) {
        std::cerr << "Feature contract sidecar is missing: " << contract.string() << "\n";
        return false;
    }

    nlohmann::json j;
    try {
        std::ifstream in(contract);
        in >> j;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to read feature contract " << contract.string()
            << ": " << e.what() << "\n";
        return false;
    }

    bool ok = true;
    const auto trainedMode = j.value("feature_mode", "");
    const auto trainedDepth = j.value("depth", -1);
    const auto trainedSize = j.value("size", -1);
    const auto trainedColumns = j.value("feature_columns", std::vector<std::string>{});
    const auto trainedLabels = j.value("label_columns", std::vector<std::string>{});

    if (trainedMode != opt.featureMode) {
        std::cerr << "Feature mode mismatch: trained " << trainedMode
            << ", infer " << opt.featureMode << "\n";
        ok = false;
    }
    if (trainedDepth != opt.depth) {
        std::cerr << "Depth mismatch: trained " << trainedDepth
            << ", infer " << opt.depth << "\n";
        ok = false;
    }
    if (trainedSize != opt.size) {
        std::cerr << "Size mismatch: trained " << trainedSize
            << ", infer " << opt.size << "\n";
        ok = false;
    }
    if (trainedColumns != expectedColumns) {
        printVectorMismatch(expectedColumns, trainedColumns);
        ok = false;
    }
    if (!trainedLabels.empty() && trainedLabels != opt.classes) {
        std::cerr << "Label/class order mismatch between model contract and --classes\n";
        ok = false;
    }

    if (!ok) {
        std::cerr << "Feature contract validation failed for "
            << contract.string() << "\n";
        return false;
    }

    std::cout << "Feature contract OK: " << contract.string()
        << " (" << expectedColumns.size() << " features)\n";
    return true;
}

static std::unordered_map<std::string, std::vector<float>> loadExistingStatusCache(
    const fs::path& datasetPath)
{
    std::unordered_map<std::string, std::vector<float>> cache;
    std::error_code ec;
    if (!fs::exists(datasetPath, ec) || fs::file_size(datasetPath, ec) == 0) {
        return cache;
    }

    try {
        nlohmann::json dataset;
        std::ifstream in(datasetPath);
        in >> dataset;
        for (const auto& item : dataset) {
            if (!item.contains("path") || !item.contains("status")) continue;
            cache[item["path"].get<std::string>()]
                = item["status"].get<std::vector<float>>();
        }
        std::cout << "Loaded " << cache.size()
            << " existing status entries from " << datasetPath.string() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to read existing dataset metadata "
            << datasetPath.string() << ": " << e.what() << "\n";
    }
    return cache;
}

static bool readExistingResultStatus(const fs::path& resultPath,
    const std::vector<std::string>& typeNames,
    std::vector<float>& status)
{
    status.assign(typeNames.size(), 0.f);
    bool ok = true;

    try {
        openvdb::io::File f(resultPath.string());
        f.open();
        auto grids = f.getGrids();
        if (!grids) {
            std::cerr << "    [warn] cannot read grids from "
                << resultPath.string() << "\n";
            return false;
        }

        for (size_t i = 0; i < typeNames.size(); ++i) {
            auto itr = std::find_if(grids->begin(), grids->end(),
                [&typeNames, i](const openvdb::GridBase::Ptr& grid) {
                    return grid && grid->getName() == typeNames[i];
                });
            if (itr == grids->end()) {
                std::cerr << "    [warn] existing result missing grid "
                    << typeNames[i] << ": " << resultPath.string() << "\n";
                ok = false;
                continue;
            }

            auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(*itr);
            if (!grid) {
                std::cerr << "    [warn] existing result grid is not FloatGrid "
                    << typeNames[i] << ": " << resultPath.string() << "\n";
                ok = false;
                continue;
            }

            size_t total = 0;
            size_t positive = 0;
            for (auto it = grid->cbeginValueOn(); it; ++it) {
                ++total;
                if (it.getValue() != 0.f) ++positive;
            }
            status[i] = total ? static_cast<float>(positive) / total : 0.f;
        }

        f.close();
    }
    catch (const std::exception& e) {
        std::cerr << "    [warn] failed to read existing result status from "
            << resultPath.string() << ": " << e.what() << "\n";
        return false;
    }

    return ok;
}

static void printStatusVector(const std::vector<float>& status)
{
    std::cout << "[";
    for (size_t i = 0; i < status.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << status[i];
    }
    std::cout << "]";
}

// Collects every `input_*.vdb` directly inside `dir` (non-recursive), sorted by
// file name for deterministic processing order.
static std::vector<fs::path> collectInputVolumes(const fs::path& dir)
{
    std::vector<fs::path> inputs;
    std::error_code ec;
    for (auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (name.rfind("input_", 0) != 0) continue;
        if (entry.path().extension() != ".vdb") continue;
        inputs.push_back(entry.path());
    }
    std::sort(inputs.begin(), inputs.end(),
        [](const fs::path& a, const fs::path& b) {
            return a.filename().string() < b.filename().string();
        });
    return inputs;
}

static int run(const InferOptions& opt)
{
    if (!fs::is_directory(opt.inputDir)) {
        std::cerr << "Input dir is not a directory: " << opt.inputDir.string() << "\n";
        return 2;
    }
    if (!fs::exists(opt.model)) {
        std::cerr << "Model not found: " << opt.model.string() << "\n";
        return 2;
    }
    const auto featureCols = dtpipeline::featureColumns(opt.featureMode);
    if (!validateFeatureContract(opt, featureCols)) {
        return 2;
    }

    openvdb::initialize();

    // Configure and launch the embedded Python interpreter.
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    if (!opt.pyhome.empty()) {
        PyConfig_SetString(&config, &config.home, opt.pyhome.wstring().c_str());
    }
    if (!opt.scripts.empty()) {
        PyConfig_SetString(&config, &config.pythonpath_env, opt.scripts.wstring().c_str());
    }
    py::scoped_interpreter guard{ &config };

    const openvdb::Vec3I boxSize(opt.depth, opt.size, opt.size);

    nlohmann::json jinfo = nlohmann::json::array();
    const auto existingStatus = loadExistingStatusCache(opt.datasetOut);

    const std::vector<fs::path> inputs = collectInputVolumes(opt.inputDir);
    std::cout << "Found " << inputs.size() << " input volume(s) in "
        << opt.inputDir.string() << "\n";

    size_t volumeIndex = 0;
    for (const auto& fpath : inputs) {
        const std::string namestr = fpath.filename().string();
        // input_<case>.vdb -> case id is everything between "input_" and ".vdb"
        const std::string caseId = fpath.stem().string().substr(6);

        ++volumeIndex;
        auto itemStart = std::chrono::steady_clock::now();
        std::cout << "[" << volumeIndex << "/" << inputs.size() << "] Processing "
            << fpath.string() << std::endl;

        const fs::path outpath = opt.inputDir / ("result_" + caseId + ".vdb");
        const bool heldOut = opt.heldOut.count(caseId) > 0;

        int iid = 0;
        try { iid = std::stoi(caseId); } catch (...) {}

        std::vector<float> status(opt.classes.size(), 0.f);
        if (fs::exists(outpath)) {
            std::cout << "    skipping existing " << outpath.filename().string() << std::endl;
            if (!readExistingResultStatus(outpath, opt.classes, status)) {
                const auto statusItr = existingStatus.find(pathToUtf8(outpath));
                if (statusItr == existingStatus.end()) {
                    std::fill(status.begin(), status.end(), 0.5f);
                }
                else {
                    status = statusItr->second;
                    status.resize(opt.classes.size(), 0.5f);
                }
            }
        }
        else {
            bool ok = runItemInference(opt.model, fpath, outpath,
                boxSize, opt.classes, opt.featureMode, featureCols, status);
            if (!ok) continue;
        }

        nlohmann::json element;
        element["case_id"] = caseId;
        element["iid"] = iid;
        element["held_out"] = heldOut;
        element["input"] = pathToUtf8(fpath);
        element["path"] = pathToUtf8(outpath);
        element["classes"] = opt.classes;
        element["status"] = status;
        element["label_source"] = "pseudo_label";
        element["probability_role"] = "soft_pseudo_label_confidence_prior";
        element["feature_mode"] = opt.featureMode;
        element["feature_columns"] = featureCols;

        jinfo.push_back(element);

        auto itemSecs = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - itemStart).count();
        std::cout << "  status ";
        printStatusVector(status);
        std::cout << " (" << itemSecs << "s)\n";
    }

    std::ofstream of(opt.datasetOut);
    if (!of) {
        std::cerr << "Cannot write dataset metadata: " << opt.datasetOut.string() << "\n";
        return 2;
    }
    of << jinfo.dump(2);
    of.close();
    std::cout << "Wrote " << jinfo.size() << " entries -> " << opt.datasetOut.string() << "\n";

    return jinfo.empty() ? 3 : 0;
}

static void usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << " infer"
        << " --input-dir <dir> --model <pkl> [--pyhome <dir>] [--scripts <dir>]\n"
           "          [--depth 4] [--size 32] [--classes fragment,hardness,watery]\n"
           "          [--feature-mode directional_multiscale|baseline]\n"
           "          [--held-out caseA,caseB] [--dataset-out <json>]\n";
}

} // namespace infer

// ===========================================================================
// Stage: export (was test/external.cpp)
// ===========================================================================
namespace exporter {

static const std::vector<std::string> kDefaultTypes = { "fragment", "hardness", "watery" };

openvdb::FloatGrid::Ptr clipProcess(openvdb::GridBase::ConstPtr input, const openvdb::math::Vec3i extent)
{
    if (!(extent > openvdb::Vec3i::zero())) return nullptr;

    auto fvptr = openvdb::gridConstPtrCast<openvdb::FloatGrid>(input);
    if (!fvptr) return nullptr;

    //! ONLY ACCEPT INSIDE THE ORIGIN VOLUME
    auto obox = fvptr->evalActiveVoxelBoundingBox();

    auto demiExt = extent * 0.5;

    openvdb::Coord min(obox.min().x(), -demiExt.y(), -demiExt.z());
    openvdb::Coord max(obox.min().x() + extent.x(), demiExt.y(), demiExt.z());
    openvdb::CoordBBox box(min, max);

    if (!obox.isInside(box)) return nullptr;

    auto mask = openvdb::MaskGrid::create();
    mask->fill(box, true, true);

    auto grid = openvdb::tools::clip_internal::doClip(*fvptr, *mask, true);

    grid->setTransform(fvptr->transform().copy());

    return grid;
}

openvdb::FloatGrid::Ptr filterProcess(openvdb::GridBase::ConstPtr input, float ratio, int width)
{
    float background = ratio > 0.5f ? 0.f : 1.f;

    auto grid = openvdb::gridConstPtrCast<openvdb::FloatGrid>(input);

    auto cpyGrid = grid->deepCopy(); //! Copy to new grid
    openvdb::tools::changeBackground(cpyGrid->tree(), background);
    openvdb::tools::Filter<openvdb::FloatGrid> f(*cpyGrid, nullptr);
    f.median(width);

    return cpyGrid;
}

// Serialise one OpenVDB grid to a NIfTI file. `vtkType` is VTK_FLOAT for the
// probability channel and VTK_INT for label / feature channels.
bool formatParse(openvdb::GridBase::ConstPtr grid, const fs::path& pth,
    float ratio, int width, int vtkType)
{
    std::cout << "Saveto " << pth.string();

    auto fptr = openvdb::gridConstPtrCast<openvdb::FloatGrid>(grid);

    if (fptr == nullptr || fptr->empty()) {
        std::cout << " Failed (empty grid)\n";
        return false;
    }

    auto size = fptr->voxelSize();
    auto bbox = fptr->evalActiveVoxelBoundingBox();

    //! account for the median-filter border
    bbox.min() += openvdb::Coord(width);
    bbox.max() -= openvdb::Coord(width);

    auto ext = bbox.extents();
    if (ext.x() <= 0 || ext.y() <= 0 || ext.z() <= 0) {
        std::cout << " Failed (degenerate extent)\n";
        return false;
    }

    auto dimension = ext.data();
    auto origin = bbox.min().asVec3d() * size;
    auto start = bbox.min();

    auto imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(dimension);
    imageData->SetSpacing(size.asV());
    imageData->SetOrigin(origin.asV());
    imageData->AllocateScalars(vtkType, 1);

    int* dims = imageData->GetDimensions();

    if (vtkType == VTK_FLOAT) {
        auto acc = fptr->getConstAccessor();
        for (int z = 0; z < dims[2]; z++) {
            for (int y = 0; y < dims[1]; y++) {
                for (int x = 0; x < dims[0]; x++) {
                    float* pixel = static_cast<float*>(imageData->GetScalarPointer(x, y, z));
                    pixel[0] = acc.getValue(start + openvdb::Coord(x, y, z));
                }
            }
        }
    }
    else if (vtkType == VTK_INT) {
        auto flt = filterProcess(fptr, ratio, width);
        auto acc = flt->getConstAccessor();
        for (int z = 0; z < dims[2]; z++) {
            for (int y = 0; y < dims[1]; y++) {
                for (int x = 0; x < dims[0]; x++) {
                    int* pixel = static_cast<int*>(imageData->GetScalarPointer(x, y, z));
                    pixel[0] = static_cast<int>(acc.getValue(start + openvdb::Coord(x, y, z)));
                }
            }
        }
    }
    else {
        throw std::runtime_error("unsupported VTK scalar type");
    }

    vtkNew<vtkNIFTIImageWriter> writer;
    writer->SetFileName(pth.string().c_str());
    writer->SetInputData(imageData);
    writer->Write();

    std::cout << " Succeeded\n";

    return true;
}

static openvdb::GridPtrVecPtr loadGrids(const fs::path& fpath)
{
    openvdb::io::File voxFile(fpath.string());
    voxFile.open();
    if (!voxFile.isOpen()) return nullptr;
    return voxFile.getGrids();
}

static openvdb::GridBase::Ptr findGrid(const openvdb::GridPtrVecPtr& grids, const std::string& name)
{
    if (!grids) return nullptr;
    auto itr = std::find_if(grids->begin(), grids->end(),
        [&name](const openvdb::GridBase::Ptr& g) { return g->getName() == name; });
    return itr == grids->end() ? nullptr : *itr;
}

static openvdb::GridBase::ConstPtr maybeClip(openvdb::GridBase::ConstPtr grid,
    bool clip, const openvdb::Vec3i& extent)
{
    if (!clip) return grid;
    auto clipped = clipProcess(grid, extent);
    if (!clipped) {
        std::cout << "  [warn] clip skipped (volume smaller than extent)\n";
        return grid;
    }
    return clipped;
}

// Export one volume entry for one geology type to the {Tr|Ts} split.
static bool exportItem(const fs::path& resultVdb, const fs::path& inputVdb,
    const fs::path& outDir, const std::string& typeName, const std::string& className,
    const std::string& caseId, float ratio, int width, bool clip,
    const openvdb::Vec3i& extent, bool isTest)
{
    const std::string split = isTest ? "Ts" : "Tr";
    const fs::path imagesDir = outDir / ("images" + split);
    const fs::path labelsDir = outDir / ("labels" + split);
    const fs::path probsDir = outDir / ("probs" + split);

    auto labelGrids = loadGrids(resultVdb);
    if (!labelGrids) {
        std::cerr << "  cannot open result volume: " << resultVdb.string() << "\n";
        return false;
    }

    auto typeGrid = findGrid(labelGrids, typeName);
    auto probGrid = findGrid(labelGrids, "prob_" + typeName);
    if (!typeGrid || !probGrid) {
        std::cerr << "  missing '" << typeName << "'/'prob_" << typeName
            << "' grid in " << resultVdb.string() << "\n";
        return false;
    }

    const std::string suffix = "_" + className;

    bool ok = true;
    ok &= formatParse(maybeClip(typeGrid, clip, extent),
        labelsDir / (caseId + suffix + ".nii.gz"), ratio, width, VTK_INT);
    ok &= formatParse(maybeClip(probGrid, clip, extent),
        probsDir / (caseId + suffix + ".nii.gz"), ratio, width, VTK_FLOAT);

    auto dataGrids = loadGrids(inputVdb);
    if (!dataGrids) {
        std::cerr << "  cannot open input volume: " << inputVdb.string() << "\n";
        return false;
    }

    for (auto grid : *dataGrids) {
        std::string imgSuffix;
        if (grid->getName() == "vp") imgSuffix = "_0000.nii.gz";
        else if (grid->getName() == "vs") imgSuffix = "_0001.nii.gz";
        else if (grid->getName() == "depth") imgSuffix = "_0002.nii.gz";
        else continue;

        ok &= formatParse(maybeClip(grid, clip, extent),
            imagesDir / (caseId + imgSuffix), ratio, width, VTK_FLOAT);
    }

    return ok;
}

struct ExportOptions
{
    fs::path dataset;          ///< dataset.json from the inference stage
    fs::path outDir;           ///< output dataset root
    std::string type = "fragment";
    std::string className;
    int width = 3;
    float minr = 0.3f;
    float maxr = 0.7f;
    bool clip = false;
    openvdb::Vec3i extent{ 0, 0, 0 };
};

static bool parseExportArgs(int argc, char** argv, ExportOptions& opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--dataset") opt.dataset = next("--dataset");
        else if (a == "--out") opt.outDir = next("--out");
        else if (a == "--type") opt.type = next("--type");
        else if (a == "--class-name") opt.className = next("--class-name");
        else if (a == "--width") opt.width = std::stoi(next("--width"));
        else if (a == "--minr") opt.minr = std::stof(next("--minr"));
        else if (a == "--maxr") opt.maxr = std::stof(next("--maxr"));
        else if (a == "--extent") {
            std::vector<int> v; std::string cur;
            std::string s = next("--extent");
            for (char c : s) { if (c == ',') { v.push_back(std::stoi(cur)); cur.clear(); } else cur += c; }
            if (!cur.empty()) v.push_back(std::stoi(cur));
            if (v.size() == 3) { opt.extent = openvdb::Vec3i(v[0], v[1], v[2]); opt.clip = true; }
        }
        else std::cerr << "Unknown option ignored: " << a << "\n";
    }
    return !opt.dataset.empty() && !opt.outDir.empty();
}

static void writeDatasetJson(const fs::path& outDir, const std::string& className,
    size_t numTraining)
{
    nlohmann::json ds;
    ds["channel_names"] = {
        { "0", "vp" },
        { "1", "vs" },
        { "2", "depth" }
    };
    ds["labels"] = {
        { "background", 0 },
        { className, 1 }
    };
    ds["numTraining"] = static_cast<int>(numTraining);
    ds["file_ending"] = ".nii.gz";
    ds["label_source"] = "pseudo_label";
    ds["probability_role"] = "soft_pseudo_label_confidence_prior";

    ds["geology_classes"] = { "fracture_zone", "soft_rock", "water_rich_zone" };

    std::ofstream out(outDir / "dataset.json");
    out << ds.dump(2);
}

static std::string resolveTypeName(const std::string& type)
{
    try {
        size_t pos = 0;
        int idx = std::stoi(type, &pos);
        if (pos == type.size() && idx >= 0 && idx < (int)kDefaultTypes.size()) {
            return kDefaultTypes[idx];
        }
    }
    catch (...) {}
    return type;
}

static int statusIndex(const nlohmann::json& item, const std::string& typeName)
{
    if (item.contains("classes")) {
        const auto& classes = item["classes"];
        for (size_t i = 0; i < classes.size(); ++i) {
            if (classes[i].get<std::string>() == typeName) return (int)i;
        }
    }
    auto itr = std::find(kDefaultTypes.begin(), kDefaultTypes.end(), typeName);
    return itr == kDefaultTypes.end() ? 0 : (int)std::distance(kDefaultTypes.begin(), itr);
}

// Derives a case id for output file names. Prefers the explicit `case_id`
// written by the infer stage, falling back to legacy `pid`_`iid`.
static std::string caseIdFor(const nlohmann::json& item)
{
    if (item.contains("case_id")) return item["case_id"].get<std::string>();
    return item.value("pid", std::string("p")) + "_"
        + std::to_string(item.value("iid", 0));
}

static int run(const ExportOptions& opt)
{
    openvdb::initialize();

    nlohmann::json dataset;
    {
        std::ifstream in(opt.dataset);
        if (!in) { std::cerr << "Cannot read dataset: " << opt.dataset.string() << "\n"; return 2; }
        in >> dataset;
    }

    const std::string typeName = resolveTypeName(opt.type);
    const std::string className = opt.className.empty() ? typeName : opt.className;

    for (const char* split : { "Tr", "Ts" }) {
        fs::create_directories(opt.outDir / ("images" + std::string(split)));
        fs::create_directories(opt.outDir / ("labels" + std::string(split)));
        fs::create_directories(opt.outDir / ("probs" + std::string(split)));
    }

    size_t exported = 0, exportedTrain = 0, skipped = 0;

    for (const auto& item : dataset) {
        if (!item.contains("path")) continue;

        const fs::path resultVdb = item["path"].get<std::string>();
        const fs::path inputVdb = item.contains("input")
            ? fs::path(item["input"].get<std::string>())
            : fs::path();

        const bool isTest = item.value("held_out", false);
        const int sIdx = statusIndex(item, typeName);

        float ratio = 0.f;
        if (item.contains("status") && sIdx < (int)item["status"].size()) {
            ratio = item["status"][sIdx].get<float>();
        }

        // Training samples are filtered to informative positive ratios; held-out
        // validation samples are always exported.
        if (!isTest && !(ratio > opt.minr && ratio < opt.maxr)) {
            ++skipped;
            continue;
        }

        const std::string caseId = caseIdFor(item);

        std::cout << "Exporting case " << caseId << " [" << className
            << " from " << typeName << "] "
            << (isTest ? "(test)" : "(train)") << " ratio=" << ratio << "\n";

        if (exportItem(resultVdb, inputVdb, opt.outDir, typeName, className,
                caseId, ratio, opt.width, opt.clip, opt.extent, isTest)) {
            ++exported;
            if (!isTest) ++exportedTrain;
        }
        else {
            ++skipped;
        }
    }

    writeDatasetJson(opt.outDir, className, exportedTrain);

    std::cout << "Exported " << exported << " case(s) for type '" << typeName
        << "', skipped " << skipped << " -> " << opt.outDir.string() << "\n";

    return exported > 0 ? 0 : 3;
}

static void usage(const char* prog)
{
    std::cerr << "Usage: " << prog << " export"
        << " --dataset <dataset.json> --out <dir> --type <name|idx>"
           " [--class-name <nnunet_label>] [--width 3]"
           " [--minr 0.3] [--maxr 0.7] [--extent x,y,z]\n";
}

} // namespace exporter

// ===========================================================================
// Top-level dispatch
// ===========================================================================
static void topUsage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << " <command> [options]\n\n"
           "Commands:\n"
           "  infer    Generate voxel-level soft pseudo-labels from input_*.vdb volumes.\n"
           "  export   Convert inference result .vdb volumes to a nii.gz dataset.\n\n";
    infer::usage(prog);
    exporter::usage(prog);
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    const char* prog = (argc > 0 && argv[0]) ? argv[0] : "dt_pipeline";

    if (argc < 2) {
        topUsage(prog);
        return 1;
    }

    const std::string command = argv[1];
    // Hand the sub-parsers an argv where argv[0] is the sub-command, so their
    // i=1 loops parse only that command's options.
    int subArgc = argc - 1;
    char** subArgv = argv + 1;

    auto start = std::chrono::system_clock::now();
    int rc = 0;

    if (command == "infer") {
        infer::InferOptions opt;
        try {
            if (!infer::parseInferArgs(subArgc, subArgv, opt)) {
                infer::usage(prog);
                return 1;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Argument error: " << e.what() << "\n";
            return 1;
        }
        rc = infer::run(opt);
    }
    else if (command == "export") {
        exporter::ExportOptions opt;
        try {
            if (!exporter::parseExportArgs(subArgc, subArgv, opt)) {
                exporter::usage(prog);
                return 1;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Argument error: " << e.what() << "\n";
            return 1;
        }
        rc = exporter::run(opt);
    }
    else {
        std::cerr << "Unknown command: " << command << "\n\n";
        topUsage(prog);
        return 1;
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "finished using time "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s\n";
    return rc;
}
