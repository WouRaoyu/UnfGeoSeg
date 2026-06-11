# UnfavorSeg

Weakly-supervised **coarse → fine 3D semantic segmentation of unfavorable tunnel
geology**, implementing the method in `Manuscript.docx` on top of
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (data / preprocessing / training
/ inference) and a self-contained 3D-TransUNet-style fine-grained backbone.

```
P-wave (vp), S-wave (vs), burial depth
        │
        ▼  local statistical features (mean/median/mode/max/min over a window)
 Random Forest  ──►  coarse pseudo-labels + probfg weak direction
        │
        ▼  weak soft supervision + EMA consistency + edge-aware smoothness
 3D-TransUNet / WeakCC trainer  ──►  probabilistic 3D interpretation
```

The coarse classifier provides weak supervision from tunnel-face records; the
fine 3D-TransUNet is expected to learn spatial correlation, heterogeneity and
context from the 3D physical fields rather than merely copy pseudo-label
boundaries. The pseudo-label and `probfg` map are therefore treated as uncertain
weak constraints, and the refined interpretation is validated against **held-out
excavation faces**, **independent borehole/probe-hole logs**, or explicitly
marked fallback references.

## Layout

| Path | Purpose |
|---|---|
| `segment/data/` | alignment (`Mg`, burial-depth field), local statistical sampling (Eq. 2-4), leakage-controlled splits, nnU-Net dataset assembly |
| `segment/coarse/` | vectorized window features, Random-Forest classifier, sliding-box pseudo-label + foreground-probability generation |
| `segment/fine/` | `TransUNet3D` backbone, CC and WeakCC losses/trainers, soft-label dataset builder |
| `segment/experiments/` | metrics + `e1..e5`, fine-report, ablations, uncertainty (Table 6), inference time |
| `segment/cli.py` | `segment <command>` entry points |
| `scripts/run_all.ps1` | end-to-end reproduction |

## Setup

```powershell
# from repo root, in the dedicated env (see environment.yml)
pip install --prefer-binary nnunetv2==2.5.2 blosc2 einops
pip install -e .
segment install-trainers          # registers the custom nnU-Net trainers

$env:nnUNet_raw          = "D:\Datasets\Volumes\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\Datasets\Volumes\nnUNet_preprocessed"
$env:nnUNet_results      = "D:\Datasets\Volumes\nnUNet_results"
```

Data follows the standard nnU-Net layout. `Dataset005_Hardness` (channels
`0=vp, 1=vs, 2=depth`) is the included **binary** smoke dataset.

The manuscript's geology types (fracture zone / soft rock / water-rich zone) can
**spatially overlap** (a voxel may belong to several at once), so they are NOT a
single mutually-exclusive label map. Each type is treated as an **independent
0/1 binary problem** with its own model, pseudo-labels and foreground-probability
map. Supply `Dataset010_Geology` with the same channel convention and one
**per-class binary mask per type**:

```
imagesTr/<case>_0000.nii.gz  # vp
imagesTr/<case>_0001.nii.gz  # vs
imagesTr/<case>_0002.nii.gz  # depth
labelsTr/<case>_fracture_zone.nii.gz     # 0/1
labelsTr/<case>_soft_rock.nii.gz         # 0/1
labelsTr/<case>_water_rich_zone.nii.gz   # 0/1
dataset.json  ->  { ..., "geology_classes": ["fracture_zone","soft_rock","water_rich_zone"] }
```

Select the type to run with `--class <type>`. The single-class smoke dataset
needs no suffix (its `labelsTr/<case>.nii.gz` is used directly, with
`--class unfavorable`). For datasets declaring multiple `geology_classes`, the
pipeline requires the per-class files above and will not silently fall back to an
old mutually-exclusive `labelsTr/<case>.nii.gz` map.

## Pipeline

Run the pipeline **once per geology type** (pass `--class <type>`; below uses the
binary smoke dataset's `unfavorable`). For the multi-type dataset, repeat with
each of `fracture_zone` / `soft_rock` / `water_rich_zone` and suffix the
artifacts — or just use `scripts/run_all.ps1 -Classes ...`.

```powershell
# 1. process/dt_pipeline inference writes result_*.vdb + dataset.json
process/build/dt_pipeline infer `
  --input-dir D:\Volumes\flat_vdb `
  --model models/rf_unfavorable.joblib `
  --scripts process/build `
  --depth 4 --size 64 --feature-mode baseline `
  --classes unfavorable `
  --dataset-out D:\Volumes\flat_vdb\dataset.json

# 2. process export + nnU-Net preprocess/train
bash scripts/run_fine.sh `
  --dataset-json D:\Volumes\flat_vdb\dataset.json `
  --classes unfavorable `
  --cc-base HardnessCC `
  --cc-id-base 11
```

By default, `run_fine.sh`, `run_all.sh` and `run_all.ps1` train all four
fine-stage modes listed below so WeakCC can be compared against the nnU-Net,
plain TransUNet and stronger CC baselines under the same split.

`process/contract.h` is the source of truth for coarse features. When
`configs/geology.yaml` sets `process.feature_mode`, the Python coarse
experiments use the same feature columns and `segment coarse-train` writes the
`<model>.features.json` sidecar required by `dt_pipeline infer`. The legacy
`segment pseudolabel/build-cc` route is retained for smoke tests, but the
process route is the canonical workflow.

The exported `imagesTr/<case>_0003.nii.gz` channel is the foreground probability
`P(class=1)`. `probsTr/<case>.nii.gz` keeps the same map as an inspection and
uncertainty sidecar. The TransUNet trainers carry this map as the 4th nnU-Net
image channel only to keep it aligned with crops/augmentation; by default the
network wrapper drops it before forward, so the network sees only `[vp, vs,
depth]`. The loss reads `probfg` as a weak soft-target signal, not as a dense
ground-truth boundary.

### Fine-stage trainers

| Trainer | Purpose |
|---|---|
| `nnUNetTrainerUnfavorSeg` | nnU-Net architecture baseline with the project training schedule |
| `nnUNetTrainerTransUNet` | Plain 3D-TransUNet baseline without pseudo-probability constraint |
| `nnUNetTrainerTransUNetCC` | Probability-constrained baseline: Dice+CE plus `λ·KL(P‖Q)` |
| `nnUNetTrainerTransUNetWeakCC` | Default weak trainer: confidence-weighted soft supervision, weak KL warmup, EMA consistency and edge-aware TV |

`WeakCC` is the recommended default for the current research objective. It
intentionally reduces supervision pressure in low-confidence / mixed regions so
the model can deviate from coarse pseudo-labels when the 3D physical context
supports a different spatial distribution. Boundary error remains a diagnostic
metric; no hard boundary loss or morphology rule is used by default.

## Experiments (maps to manuscript tables)

| Command / module | Reproduces | Runs on |
|---|---|---|
| `segment e1` | Table 2 — held-out coarse classifier | CPU |
| `segment ab-burial` | Table 4 — with/without burial depth | CPU |
| `segment ab-window` | Fig. 12 — window-size sweep | CPU |
| `segment ab-tsp` | Vp/Vs ±3/5/10% coarse sensitivity | CPU |
| `experiments.e2_tfr_finegrained` | held-out TFR face consistency | needs predictions |
| `experiments.e3_borehole` | borehole/probe-hole + boundary error | needs predictions |
| `experiments.e4_pseudo_vs_refined` | raw pseudo-labels vs refined | needs predictions |
| `experiments.e5_lambda_sensitivity` | λ sweep table | needs predictions |
| `segment fine-report` | fine-stage method comparison + pseudo-copy/disagreement audit | needs predictions |
| `experiments.uncertainty` | Table 6 — SoftMax/Entropy/Variance | needs probabilities |
| `experiments.inference_time` | efficiency table | timing |

### Fine-stage experiments

After training, predict with probabilities and call the evaluators, e.g.:

```powershell
nnUNetv2_predict -i $env:nnUNet_raw/Dataset011_HardnessCC_unfavorable/imagesTr `
                 -o results/pred_weakcc_unfavorable -d 11 -c 3d_fullres -f 0 `
                 -tr nnUNetTrainerTransUNetWeakCC --save_probabilities

segment e4 --dataset Dataset011_HardnessCC_unfavorable `
           --pred results/pred_weakcc_unfavorable `
           --pseudolabels $env:nnUNet_raw/Dataset011_HardnessCC_unfavorable/labelsTr `
           --reference D:\ReferenceLabels\labelsTr
segment fine-report --dataset Dataset011_HardnessCC_unfavorable `
                    --method-pred WeakCC=results/pred_weakcc_unfavorable `
                    --method-pred CC=results/pred_cc_unfavorable `
                    --pseudolabels $env:nnUNet_raw/Dataset011_HardnessCC_unfavorable/labelsTr `
                    --reference D:\ReferenceLabels\labelsTr `
                    --data-source heldout_reference
segment uncertainty --dataset Dataset011_HardnessCC_unfavorable `
                    --method-pred WeakCC=results/pred_weakcc_unfavorable `
                    --pseudolabels $env:nnUNet_raw/Dataset011_HardnessCC_unfavorable/labelsTr
```

Results (CSV + Markdown) are written under `results/`. `fine-report` should be
used when judging whether the fine model learned useful spatial structure rather
than simply copying the pseudo-label: it reports case-macro Dice/IoU/F1,
calibration, pseudo-copy ratio, and disagreement audits by `probfg` confidence,
pseudo mixed-like regions and foreground-occupancy bucket.

### Training curves

nnU-Net writes `progress.png` in each fold output directory. To export the same
training history as CSV and a smoothed PNG:

```powershell
segment plot-training "$env:nnUNet_results/Dataset011_HardnessCC/nnUNetTrainerTransUNetWeakCC__nnUNetPlans__3d_fullres/fold_0" `
                         --out results/training_curves/weakcc --smooth 5
```

When curves are noisy, first compare `nnUNetTrainerTransUNet`, `TransUNetCC` and
`TransUNetWeakCC` under the same split. For WeakCC, tune
`UNFAVORSEG_WEAK_KL_MAX`, `UNFAVORSEG_CONSISTENCY_WEIGHT` and
`UNFAVORSEG_EDGE_TV_WEIGHT` before increasing model capacity; lower
`UNFAVORSEG_LR` if the smoothed validation loss still has no downward trend.

## Notes & assumptions

* **Independent binary types** — geology types may overlap, so each is run as a
  separate 0/1 problem (own model, pseudo-labels, probability map);
  results are never merged. `Dataset005_Hardness` is the single-type smoke case.
  Provide per-class binary masks (`labelsTr/<case>_<type>.nii.gz`) +
  `geology_classes` in `dataset.json` to populate the final manuscript numbers.
* **Deep supervision is disabled** for the TransUNet trainers so the single
  full-resolution output stays aligned with the probfg soft-target map.
* **Weak supervision is deliberate** — pseudo-labels and `probfg` are not treated
  as exact dense ground truth. WeakCC lowers supervision pressure in uncertain
  regions and uses consistency / soft spatial regularization to encourage
  learned 3D structure.
* **Fine-stage folds** — `segment make-splits` uses leave-one-tunnel-out when
  case ids indicate multiple tunnel/project groups. For a single-project dataset,
  it writes deterministic 5-fold case-level splits instead of unstable
  leave-one-volume-out folds. Override the count with `--folds`.
* **TFR / borehole records** — when real field records are unavailable, some
  fine-stage evaluators can synthesize sparse faces / 1-D trajectories from the
  reference labels (`experiments/validation_records.py`) so the pipeline is
  runnable end-to-end. These results are smoke/demo evidence only; publication
  or model-selection claims should use real sparse records or clearly marked
  held-out references.
* **Hardware** — developed/verified on an RTX 3060 (12 GB); the manuscript used
  an RTX 4090 (24 GB). Reduce the patch size in the plans if you hit OOM.
