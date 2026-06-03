# dt_pipeline — standalone infer + export tool

A single, self-contained executable that merges two stages of the
weakly-supervised geology pipeline. It has **no dependency on the DTGeoStudio
source tree** — the feature contract and descriptive statistics that used to
come from project headers are inlined into `dt_pipeline.cpp`.

| Sub-command | Was            | Does                                                            |
|-------------|----------------|----------------------------------------------------------------|
| `infer`     | `test/wrapper.cpp`  | Runs a pre-trained model to produce voxel-level soft pseudo-labels (`result_*.vdb`) and a `dataset.json`. |
| `export`    | `test/external.cpp` | Converts the inference result `.vdb` volumes to a nnU-Net / MONAI style `.nii.gz` dataset. |

Unlike the originals, `infer` no longer walks a `project/site/TSP` directory
tree. All `input_*.vdb` volumes are read directly from a **single flat folder**
given by `--input-dir`, and `result_*.vdb` are written back into the same folder.

## Dependencies

- OpenVDB (with TBB)
- VTK 9 (CommonCore, CommonDataModel, IOImage — provides the NIFTI writer)
- pybind11 + a Python 3 development install (embedded interpreter)
- nlohmann_json
- At runtime: `numpy`, `pandas`, `joblib`, and the model's own deps (scikit-learn / xgboost)

## Build (Linux, Ninja preferred)

```bash
cd run
./build.sh                 # configures + builds into ./build
# or point at custom prefixes:
OPENVDB_ROOT=/opt/openvdb \
CMAKE_PREFIX_PATH=/opt/vtk:/opt/pybind11 \
  ./build.sh
```

The executable lands at `run/build/dt_pipeline`, with `model_utils.py` copied
alongside it.

## Usage

### infer

```bash
./build/dt_pipeline infer \
    --input-dir /data/volumes \
    --model /models/coarse.pkl \
    --pyhome /usr \
    --scripts ./build \
    --depth 4 --size 32 \
    --classes fragment,hardness,watery \
    --feature-mode directional_multiscale \
    --held-out 0007,0012 \
    --dataset-out /data/volumes/dataset.json
```

- Requires a feature-contract sidecar next to the model:
  `<model-stem>.features.json` (validates feature_mode / depth / size / columns).
- `--scripts` must point at the directory containing `model_utils.py`.
- Each `input_<case>.vdb` produces `result_<case>.vdb` in `--input-dir`.

### export

```bash
./build/dt_pipeline export \
    --dataset /data/volumes/dataset.json \
    --out /data/nnunet/Dataset001 \
    --type fragment \
    --class-name fracture_zone \
    --width 3 --minr 0.3 --maxr 0.7 \
    --extent 256,128,128
```

`--type` accepts a class name or a numeric index into the class list. Held-out
cases go to the `*Ts` split; the rest to `*Tr` (filtered by positive ratio).
