# Reproduce the UnfavorSeg coarse->fine pipeline, run ONCE PER INDEPENDENT
# geology type (each is a 0/1 binary problem; types may overlap and are never
# merged). Per-class artifacts are suffixed with the type name.
#
# Usage (from the repo root, in the `segment` conda env):
#   # single-class smoke dataset:
#   .\scripts\run_all.ps1 -SrcDataset Dataset005_Hardness -Classes unfavorable -CCBase Dataset011_HardnessCC -CCIdBase 11
#   # multi-type geology dataset (one binary run per type):
#   .\scripts\run_all.ps1 -SrcDataset Dataset010_Geology -Classes fracture_zone,soft_rock,water_rich_zone -CCBase GeologyCC -CCIdBase 20
#
# Long-running steps (nnU-Net preprocess/train/predict) need a GPU. The coarse
# experiments (e1, ablations) run on CPU in minutes.

param(
    [string]$SrcDataset = "Dataset005_Hardness",
    [string[]]$Classes  = @("unfavorable"),
    [string]$CCBase     = "Dataset011_HardnessCC",
    [int]$CCIdBase      = 11,
    [string]$Config     = "configs/geology.yaml",
    [int]$Fold          = 0,
    [int]$Epochs        = 100,
    [double]$LR         = 0.0,
    [switch]$LambdaSweep,
    [string]$Py         = "python"
)

$ErrorActionPreference = "Stop"
$Results = "results"
New-Item -ItemType Directory -Force -Path $Results, "models" | Out-Null

function Get-PerClassDatasetName {
    param(
        [string]$Base,
        [int]$Id,
        [string]$ClassName
    )
    $DatasetPrefix = "Dataset{0:D3}" -f $Id
    if ($Base -match '^Dataset\d+_(.+)$') {
        $BaseSuffix = $Matches[1]
    } else {
        $BaseSuffix = $Base
    }
    return "${DatasetPrefix}_${BaseSuffix}_${ClassName}"
}

Write-Host "== 0. Register custom trainers ==" -ForegroundColor Cyan
& $Py -m segment.cli install-trainers

$ccIndex = 0
foreach ($cls in $Classes) {
    $CCId      = $CCIdBase + $ccIndex
    $CCDataset = Get-PerClassDatasetName -Base $CCBase -Id $CCId -ClassName $cls
    $ccIndex++

    Write-Host "######## Geology type: $cls (binary)  ->  $CCDataset (id $CCId) ########" -ForegroundColor Magenta

    Write-Host "== 1. Coarse RF + held-out metrics (Table 2) ==" -ForegroundColor Cyan
    & $Py -m segment.cli coarse-train --dataset $SrcDataset --class $cls --out "models/rf_$cls.joblib"      --config $Config
    & $Py -m segment.cli e1        --dataset $SrcDataset --class $cls --out "$Results/e1_coarse_$cls"      --config $Config
    & $Py -m segment.cli ab-burial --dataset $SrcDataset --class $cls --out "$Results/ab_burial_$cls"      --config $Config
    & $Py -m segment.cli ab-window --dataset $SrcDataset --class $cls --out "$Results/ab_window_$cls"      --config $Config
    & $Py -m segment.cli ab-tsp    --dataset $SrcDataset --class $cls --out "$Results/ab_tsp_$cls"         --config $Config

    Write-Host "== 2. Pseudo-labels + foreground probability/probfg ==" -ForegroundColor Cyan
    $PL = "$env:nnUNet_raw/_pseudolabels_${SrcDataset}_${cls}"
    & $Py -m segment.cli pseudolabel --dataset $SrcDataset --class $cls --model "models/rf_$cls.joblib" --out $PL --config $Config

    Write-Host "== 3. Build probability-augmented dataset ==" -ForegroundColor Cyan
    & $Py -m segment.cli build-cc --src $SrcDataset --class $cls --pseudolabels $PL --dst $CCDataset --config $Config

    Write-Host "== 4. nnU-Net plan -> patch probfg normalization -> preprocess ==" -ForegroundColor Cyan
    # fingerprint + planning first (no preprocessing yet)
    nnUNetv2_extract_fingerprint -d $CCId --verify_dataset_integrity
    nnUNetv2_plan_experiment -d $CCId
    # keep the probfg carrier channel un-normalized BEFORE the cached arrays are written
    & $Py -c "from segment.fine.dataset import patch_plans_no_norm_probfg as p; import os; p(os.path.join(os.environ['nnUNet_preprocessed'], '$CCDataset'))"
    nnUNetv2_preprocess -d $CCId -c 3d_fullres
    & $Py -m segment.cli make-splits --dataset $CCDataset

    Write-Host "== 5. Train: nnU-Net baseline, plain TransUNet, CC, WeakCC ==" -ForegroundColor Cyan
    $env:UNFAVORSEG_LAMBDA = "0.3"
    $env:UNFAVORSEG_EPOCHS = "$Epochs"
    if (-not $env:UNFAVORSEG_POS_ENCODING) { $env:UNFAVORSEG_POS_ENCODING = "sinusoidal_3d" }
    if (-not $env:UNFAVORSEG_CONFIDENCE_FLOOR) { $env:UNFAVORSEG_CONFIDENCE_FLOOR = "0.1" }
    if (-not $env:UNFAVORSEG_WEAK_KL_MAX) { $env:UNFAVORSEG_WEAK_KL_MAX = "0.1" }
    if (-not $env:UNFAVORSEG_CONSISTENCY) { $env:UNFAVORSEG_CONSISTENCY = "1" }
    if (-not $env:UNFAVORSEG_CONSISTENCY_WEIGHT) { $env:UNFAVORSEG_CONSISTENCY_WEIGHT = "0.2" }
    if (-not $env:UNFAVORSEG_CONSISTENCY_CONF) { $env:UNFAVORSEG_CONSISTENCY_CONF = "0.75" }
    if (-not $env:UNFAVORSEG_EMA_DECAY) { $env:UNFAVORSEG_EMA_DECAY = "0.99" }
    if (-not $env:UNFAVORSEG_EDGE_AWARE_TV) { $env:UNFAVORSEG_EDGE_AWARE_TV = "1" }
    if (-not $env:UNFAVORSEG_EDGE_TV_WEIGHT) { $env:UNFAVORSEG_EDGE_TV_WEIGHT = "0.02" }
    if ($LR -gt 0) {
        $env:UNFAVORSEG_LR = "$LR"
    }
    & $Py -m nnunetv2.run.run_training $CCId 3d_fullres $Fold -tr nnUNetTrainerUnfavorSeg
    & $Py -m nnunetv2.run.run_training $CCId 3d_fullres $Fold -tr nnUNetTrainerTransUNet
    & $Py -m nnunetv2.run.run_training $CCId 3d_fullres $Fold -tr nnUNetTrainerTransUNetCC
    & $Py -m nnunetv2.run.run_training $CCId 3d_fullres $Fold -tr nnUNetTrainerTransUNetWeakCC

    if ($LambdaSweep) {
        Write-Host "== 6. lambda sweep (Table X) ==" -ForegroundColor Cyan
        foreach ($lam in 0.0, 0.1, 0.5, 1.0) {
            $env:UNFAVORSEG_LAMBDA = "$lam"
            & $Py -m nnunetv2.run.run_training $CCId 3d_fullres $Fold -tr nnUNetTrainerTransUNetCC -o "lam_$lam"
        }
    }

    Write-Host "== 7. Predict (save probabilities) + fine experiments ==" -ForegroundColor Cyan
    Write-Host "  Run nnUNetv2_predict for $CCDataset into results/pred_<method>_$cls," -ForegroundColor Yellow
    Write-Host "  then call experiments.e2/e3/e4/uncertainty/e5/inference_time with those dirs (per class)." -ForegroundColor Yellow
    Write-Host "  See README 'Fine-stage experiments' for the exact predict + eval commands." -ForegroundColor Yellow
}

Write-Host "Done. Per-class tables under $Results/ (suffixed by geology type)." -ForegroundColor Green
