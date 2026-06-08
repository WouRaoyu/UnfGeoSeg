# dt_pipeline - standalone VDB infer/export tool

`process/dt_pipeline` is the standalone C++ bridge between the coarse
classifier and the fine nnU-Net/3D-TransUNet stage. It has no dependency on the
DTGeoStudio source tree. The feature contract lives in `process/contract.h`, and
the embedded Python model runner lives in `process/model_utils.py`.

| Sub-command | Role | Main output |
|---|---|---|
| `infer` | Run a trained coarse model over flat `input_*.vdb` volumes. | `result_<case>.vdb` plus an inference `dataset.json`. |
| `export` | Convert the infer results to an nnU-Net/MONAI-style binary dataset. | `imagesTr/Ts`, `labelsTr/Ts`, `probsTr/Ts`, and `dataset.json`. |

The tool reads every `input_*.vdb` directly from one flat `--input-dir`; it does
not walk a nested `project/site/TSP` tree. For each case, the case id is the
filename part after `input_` and before `.vdb`.

## Dependencies

- CMake 3.18+
- C++17 compiler
- OpenVDB with TBB
- VTK 9 components `CommonCore`, `CommonDataModel`, `IOImage`
- pybind11 and a Python 3 development install
- nlohmann_json
- Runtime Python packages: `numpy`, `pandas`, `joblib`, plus the saved model's
  own dependencies such as scikit-learn or xgboost

## Build

Linux helper:

```bash
cd process
./build.sh
```

`build.sh` configures `process/build` with Ninja when available, falls back to
Unix Makefiles, and passes `-DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"`. The built
binary is `process/build/dt_pipeline`; `model_utils.py` is copied next to it.

Equivalent manual CMake from the repository root:

```bash
cmake -S process -B process/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build process/build --config Release --parallel
```

## Feature Contract

`infer` always requires a sidecar next to the model:

```text
<model-stem>.features.json
```

For `models/rf_fragment.joblib`, the required sidecar is
`models/rf_fragment.features.json`. The sidecar is written by
`segment coarse-train` when `configs/geology.yaml` sets `process.feature_mode`.
It is validated before inference:

- `feature_mode`
- `depth`
- `size`
- exact `feature_columns` order
- optional `label_columns` against `--classes`
- optional `recommended_thresholds` used for hard labels

Supported feature modes are defined in `process/contract.h` and mirrored by
`segment/process_contract.py`.

| Mode | Columns | Notes |
|---|---:|---|
| `baseline` | 9 | Base window statistics for `vp`/`vs` plus `depthMean`. |
| `multiscale` | 25 | Baseline plus small and large `vp`/`vs` windows. |
| `directional` | 59 | Multiscale plus axial/lateral/vertical windows and gradient/contrast features. |
| `enhanced` | 29 | Multiscale plus base-window `Std`/`Skew` for `vp`/`vs`. |
| `spatial` | 35 | Base stats plus local gradient energy, anisotropy, and roughness features. |
| `values` | 55 | 3x3x3 point samples for `vp`/`vs` plus `depthMean`. |
| `mean` | 3 | Window means for `vp`, `vs`, and `depth`. |
| `origin` | 3 | Center voxel values for `vp`, `vs`, and `depth`. |
| `physical` | 11 | Center `vp`/`vs`, `depthMean`, and derived elastic properties. |
| `random` | 3 | Center `vp`/`vs` plus `depthMean`. |

Hard labels are thresholded per class using `recommended_thresholds` from the
sidecar. Missing or invalid thresholds fall back to `0.5`.

## Infer

```bash
./build/dt_pipeline infer \
  --input-dir /data/flat_vdb \
  --model /models/rf_fragment.joblib \
  --scripts ./build \
  --depth 4 --size 64 \
  --feature-mode baseline \
  --classes fragment \
  --held-out 0007,0012 \
  --dataset-out /data/flat_vdb/dataset.json
```

Required:

- `--input-dir <dir>`: flat directory containing `input_<case>.vdb`.
- `--model <pkl|joblib>`: trained model loadable by `joblib.load`.

Common options:

- `--scripts <dir>`: directory containing `model_utils.py`. Usually
  `process/build`.
- `--pyhome <dir>`: optional Python home for the embedded interpreter. When
  `--scripts` is omitted and `--pyhome` is set, scripts defaults to
  `<pyhome>/../scripts`.
- `--depth <n>`: window size along the tunnel/advance axis. Default `4`.
- `--size <n>`: cross-section window size. Default `32`.
- `--feature-mode <mode>`: default `multiscale`.
- `--classes a,b,c`: class names and output grid names. Default
  `fragment,hardness,watery`.
- `--held-out caseA,caseB`: cases exported later to `imagesTs/labelsTs`.
- `--dataset-out <json>`: default `<input-dir>/dataset.json`.
- `--start-index <n>`: skip numeric case ids below `n`, useful for resuming.
- `--sampling-index <json>`: sampling manifest produced by the sample stage;
  preferred for masked inference.
- `--force-update`: recompute existing `result_<case>.vdb` files.

### Input VDBs

Each `input_<case>.vdb` must contain active FloatGrids named:

- `vp`
- `vs`
- `depth`

By default, inference only runs on the trained sampling region. The preferred
protocol is the sample-stage manifest:

- `--sampling-index <sampling_<mode>.json>`

The manifest stores each `input_<case>.vdb` path, the sampled axis voxels, and
the cross-section center; `infer` rebuilds the mask in memory and writes only
that region.

For older flat datasets, VDB masks are still supported:

- case-local mask: `mask_<case>.vdb` in `--input-dir`
- shared override: `--sampling-mask <mask.vdb>`

If neither a manifest entry nor a mask is present, the case is skipped. Use
`--mask full`, `--mask none`, `--mask all`, or `--full-volume` to infer every
active voxel in the input volume. `--mask sampling`, `--mask interval`, and
`--mask mask` select masked inference.

### Model Formats

`model_utils.py` supports:

- a scikit-learn-style `MultiOutputClassifier`, where `predict_proba` returns a
  list with one probability array per class;
- a single binary estimator blob saved by `segment coarse-train`, with keys such
  as `estimator`, `num_classes`, and optional `col_mean`;
- a single binary model whose `predict_proba` returns one probability array.

For the single-binary workflow, run `infer` with the matching one-item
`--classes <type>` and export that same type.

### Infer Outputs

For every processed case, `infer` writes:

```text
result_<case>.vdb
```

Each result VDB contains, for every class in `--classes`:

- `<class>`: hard 0/1 pseudo-label grid
- `prob_<class>`: positive-class probability grid

Existing `result_<case>.vdb` files are not recomputed. The tool tries to recover
their positive ratios from the VDB; if that fails, it uses cached `status` from
the existing `dataset.json`, then falls back to `0.5`.

The inference `dataset.json` is an array. Each entry includes:

- `case_id`
- `iid`
- `held_out`
- `input`
- `path`
- `classes`
- `status`: per-class positive voxel ratios
- `label_source`: `pseudo_label`
- `probability_role`: `positive_class_probability`
- `feature_mode`
- `feature_columns`
- `thresholds`
- `inference_region`: `sampling_interval` or `full_volume`

## Export

```bash
./build/dt_pipeline export \
  --dataset /data/flat_vdb/dataset.json \
  --out /data/nnunet/Dataset011_FragmentCC \
  --type fragment \
  --class-name fracture_zone
```

Required:

- `--dataset <json>`: inference metadata from `infer`.
- `--out <dir>`: output dataset root.

Common options:

- `--type <name|idx>`: result grid to export. Numeric defaults are
  `0=fragment`, `1=hardness`, `2=watery`; names may also be custom class names.
- `--class-name <label>`: label name written into the exported nnU-Net
  `dataset.json`. Defaults to `--type`.
- `--start-index <n>`: skip numeric case ids below `n`.
- `--width <n>`: median-filter width for labels. Default `0` means no filtering.
- `--ratio-filter --minr <a> --maxr <b>`: skip non-held-out training cases whose
  positive ratio is outside `(minr, maxr)`. Held-out cases are always exported.
- `--extent x,y,z`: optionally clip to a fixed VDB extent; if the volume is too
  small, clipping is skipped for that grid.

Export one geology type to one dataset folder. The project treats geology types
as independent binary targets, so overlapping types should not be merged into a
single multi-class label map.

### Export Outputs

For each entry in the inference metadata:

- `held_out=false` goes to `imagesTr`, `labelsTr`, `probsTr`.
- `held_out=true` goes to `imagesTs`, `labelsTs`, `probsTs`.

The output files are:

```text
imagesTr/<case>_0000.nii.gz   # vp
imagesTr/<case>_0001.nii.gz   # vs
imagesTr/<case>_0002.nii.gz   # depth
imagesTr/<case>_0003.nii.gz   # probfg, P(class=1)
labelsTr/<case>.nii.gz        # hard 0/1 pseudo-label
probsTr/<case>.nii.gz         # probfg sidecar for inspection/metrics
```

`imagesTs`, `labelsTs`, and `probsTs` use the same naming pattern.

The exported `dataset.json` contains:

- `channel_names`: `0=vp`, `1=vs`, `2=depth`, `3=probfg`
- `labels`: `background=0`, `<class-name>=1`
- `numTraining`
- `file_ending`: `.nii.gz`
- `label_source`: `pseudo_label`
- `probability_role`: `foreground_probability_soft_target`
- `geology_classes`: `[<class-name>]`

## End-to-End Pattern

Train one coarse binary model per geology type, infer on the flat VDB folder,
then export the selected type to a fine-stage nnU-Net dataset:

```bash
segment coarse-train \
  --dataset Dataset010_Geology \
  --class fracture_zone \
  --out models/rf_fracture_zone.joblib

process/build/dt_pipeline infer \
  --input-dir /data/flat_vdb \
  --model models/rf_fracture_zone.joblib \
  --scripts process/build \
  --depth 4 --size 64 \
  --feature-mode baseline \
  --classes fracture_zone \
  --dataset-out /data/flat_vdb/fracture_zone.dataset.json

process/build/dt_pipeline export \
  --dataset /data/flat_vdb/fracture_zone.dataset.json \
  --out "$nnUNet_raw/Dataset011_FractureZoneCC" \
  --type fracture_zone \
  --class-name fracture_zone
```

Repeat the same pattern for each independent geology type.
