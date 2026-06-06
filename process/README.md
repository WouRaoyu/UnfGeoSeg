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
cd process
./build.sh # configures + builds into ./build
```

The executable lands at `process/build/dt_pipeline`, with `model_utils.py` copied
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
    --feature-mode multiscale \
    --held-out 0007,0012 \
    --dataset-out /data/volumes/dataset.json
```

- Requires a feature-contract sidecar next to the model:
  `<model-stem>.features.json` (validates feature_mode / depth / size / columns).
- `--scripts` must point at the directory containing `model_utils.py`.
- Each `input_<case>.vdb` produces `result_<case>.vdb` in `--input-dir`.
- `--feature-mode` follows the shared DTGeoStudio contract:
  `baseline`, `multiscale`, `directional`, `hybrid_spatial`, or `spatial_v1`.
- In sampling mode, each `input_<case>.vdb` uses the matching
  `mask_<case>.vdb` topology as the pseudo-label region; pass `--mask full` or
  `--full-volume` to infer every active voxel, or `--sampling-mask <vdb>` to
  point at a specific mask.
- Hard labels use per-class `recommended_thresholds` from the feature sidecar
  when present, with a 0.5 fallback.

### export

```bash
./build/dt_pipeline export \
    --dataset /data/volumes/dataset.json \
    --out /data/nnunet/Dataset001 \
    --type fragment \
    --class-name fracture_zone
```

`--type` accepts a class name or a numeric index into the class list. Held-out
cases go to the `*Ts` split; the rest go to `*Tr`.

Each export call writes one nnU-Net-ready binary dataset for the selected type:
`images*` contains `[vp, vs, depth, probfg]`, `labels*` contains
`<case>.nii.gz`, and `probs*` keeps a `probfg` sidecar for inspection. Export
each geology type to a separate dataset folder when training three independent
binary fine-stage models.

By default, export preserves the full VDB active extent for each case, performs
no positive-ratio filtering, and does not median-filter labels. These optional
cleaning / ROI controls are available when you explicitly want them:

```bash
./build/dt_pipeline export \
    --dataset /data/volumes/dataset.json \
    --out /data/nnunet/Dataset001_filtered \
    --type fragment \
    --class-name fracture_zone \
    --ratio-filter --minr 0.3 --maxr 0.7 \
    --width 3 \
    --extent 256,128,128
```
