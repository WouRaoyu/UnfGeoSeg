#!/usr/bin/env bash
# Reproduce the UnfavorSeg coarse->fine pipeline on Linux, run ONCE PER
# independent geology type (each is a 0/1 binary problem; types may overlap and
# are never merged). Per-class artifacts are suffixed with the type name.
#
# Usage (from the repo root, in the `segment` conda env):
#   # single-class smoke dataset:
#   ./scripts/run_all.sh --src-dataset Dataset005_Hardness --classes unfavorable --cc-base Dataset011_HardnessCC --cc-id-base 11
#   # multi-type geology dataset (one binary run per type):
#   ./scripts/run_all.sh --src-dataset Dataset010_Geology --classes fracture_zone,soft_rock,water_rich_zone --cc-base GeologyCC --cc-id-base 20
#
# Long-running steps (nnU-Net preprocess/train/predict) need a GPU. The coarse
# experiments (e1, ablations) run on CPU in minutes.

set -Eeuo pipefail

SRC_DATASET="Dataset005_Hardness"
CLASSES=("unfavorable")
CC_BASE="Dataset011_HardnessCC"
CC_ID_BASE=11
CONFIG="configs/geology.yaml"
FOLD=0
FOLDS=5
EPOCHS=100
LR=0.0
LAMBDA_SWEEP=0
PY="${PYTHON:-python}"

if [[ -t 1 ]]; then
    CYAN=$'\033[36m'
    MAGENTA=$'\033[35m'
    YELLOW=$'\033[33m'
    GREEN=$'\033[32m'
    RESET=$'\033[0m'
else
    CYAN=""
    MAGENTA=""
    YELLOW=""
    GREEN=""
    RESET=""
fi

usage() {
    printf '%s\n' \
        "Usage:" \
        "  ./scripts/run_all.sh [options]" \
        "" \
        "Options:" \
        "  --src-dataset, -SrcDataset VALUE   Source nnU-Net dataset name" \
        "  --classes, -Classes VALUE          Comma-separated classes/geology types" \
        "  --cc-base, -CCBase VALUE           Base name for probability-augmented datasets" \
        "  --cc-id-base, -CCIdBase VALUE      First nnU-Net dataset id for generated datasets" \
        "  --config, -Config VALUE            YAML config path" \
        "  --fold, -Fold VALUE                nnU-Net fold" \
        "  --folds VALUE                      Number of case-level CV folds, default 5" \
        "  --epochs, -Epochs VALUE            Training epochs" \
        "  --lr, -LR VALUE                    Optional learning rate override" \
        "  --lambda-sweep, -LambdaSweep       Run lambda sweep" \
        "  --py, -Py VALUE                    Python executable" \
        "  --help, -h                         Show this help" \
        "" \
        "Examples:" \
        "  ./scripts/run_all.sh --src-dataset Dataset005_Hardness --classes unfavorable --cc-base Dataset011_HardnessCC --cc-id-base 11" \
        "  ./scripts/run_all.sh --src-dataset Dataset010_Geology --classes fracture_zone,soft_rock,water_rich_zone --cc-base GeologyCC --cc-id-base 20"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    [[ -n "$value" && "$value" != -* ]] || die "$flag requires a value"
}

parse_classes() {
    local raw="$1"
    local -a parts=()
    local part trimmed

    IFS=',' read -r -a parts <<< "$raw"
    CLASSES=()
    for part in "${parts[@]}"; do
        trimmed="${part#"${part%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
        [[ -n "$trimmed" ]] && CLASSES+=("$trimmed")
    done

    [[ ${#CLASSES[@]} -gt 0 ]] || die "--classes must contain at least one class"
}

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "environment variable $name is not set"
}

get_per_class_dataset_name() {
    local base="$1"
    local id="$2"
    local class_name="$3"
    local dataset_prefix base_suffix

    printf -v dataset_prefix 'Dataset%03d' "$id"
    if [[ "$base" =~ ^Dataset[0-9]+_(.+)$ ]]; then
        base_suffix="${BASH_REMATCH[1]}"
    else
        base_suffix="$base"
    fi
    printf '%s_%s_%s\n' "$dataset_prefix" "$base_suffix" "$class_name"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-dataset|-SrcDataset)
            require_value "$1" "${2:-}"
            SRC_DATASET="$2"
            shift 2
            ;;
        --classes|-Classes)
            require_value "$1" "${2:-}"
            parse_classes "$2"
            shift 2
            ;;
        --cc-base|-CCBase)
            require_value "$1" "${2:-}"
            CC_BASE="$2"
            shift 2
            ;;
        --cc-id-base|-CCIdBase)
            require_value "$1" "${2:-}"
            CC_ID_BASE="$2"
            shift 2
            ;;
        --config|-Config)
            require_value "$1" "${2:-}"
            CONFIG="$2"
            shift 2
            ;;
        --fold|-Fold)
            require_value "$1" "${2:-}"
            FOLD="$2"
            shift 2
            ;;
        --folds)
            require_value "$1" "${2:-}"
            FOLDS="$2"
            shift 2
            ;;
        --epochs|-Epochs)
            require_value "$1" "${2:-}"
            EPOCHS="$2"
            shift 2
            ;;
        --lr|-LR)
            require_value "$1" "${2:-}"
            LR="$2"
            shift 2
            ;;
        --lambda-sweep|-LambdaSweep)
            LAMBDA_SWEEP=1
            shift
            ;;
        --py|-Py)
            require_value "$1" "${2:-}"
            PY="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

command -v "$PY" >/dev/null 2>&1 || die "Python executable '$PY' was not found"
require_env nnUNet_raw
require_env nnUNet_preprocessed
require_env nnUNet_results
[[ "$FOLDS" =~ ^[0-9]+$ && "$FOLDS" -ge 2 ]] || die "--folds must be an integer >= 2"

mkdir -p results models
RESULTS="results"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

echo "${CYAN}== 0. Register custom trainers ==${RESET}"
"$PY" -m segment.cli install-trainers

cc_index=0
for cls in "${CLASSES[@]}"; do
    cc_id=$((CC_ID_BASE + cc_index))
    cc_dataset="$(get_per_class_dataset_name "$CC_BASE" "$cc_id" "$cls")"
    cc_index=$((cc_index + 1))

    echo "${MAGENTA}######## Geology type: $cls (binary)  ->  $cc_dataset (id $cc_id) ########${RESET}"

    echo "${CYAN}== 1. Coarse RF + held-out metrics (Table 2) ==${RESET}"
    "$PY" -m segment.cli coarse-train --dataset "$SRC_DATASET" --class "$cls" --out "models/rf_$cls.joblib" --config "$CONFIG"
    "$PY" -m segment.cli e1 --dataset "$SRC_DATASET" --class "$cls" --out "$RESULTS/e1_coarse_$cls" --config "$CONFIG"
    "$PY" -m segment.cli ab-burial --dataset "$SRC_DATASET" --class "$cls" --out "$RESULTS/ab_burial_$cls" --config "$CONFIG"
    "$PY" -m segment.cli ab-window --dataset "$SRC_DATASET" --class "$cls" --out "$RESULTS/ab_window_$cls" --config "$CONFIG"
    "$PY" -m segment.cli ab-tsp --dataset "$SRC_DATASET" --class "$cls" --out "$RESULTS/ab_tsp_$cls" --config "$CONFIG"

    echo "${CYAN}== 2. Pseudo-labels + foreground probability/probfg ==${RESET}"
    pl="${nnUNet_raw}/_pseudolabels_${SRC_DATASET}_${cls}"
    "$PY" -m segment.cli pseudolabel --dataset "$SRC_DATASET" --class "$cls" --model "models/rf_$cls.joblib" --out "$pl" --config "$CONFIG"

    echo "${CYAN}== 3. Build probability-augmented dataset ==${RESET}"
    "$PY" -m segment.cli build-cc --src "$SRC_DATASET" --class "$cls" --pseudolabels "$pl" --dst "$cc_dataset" --config "$CONFIG"

    echo "${CYAN}== 4. nnU-Net plan -> patch probfg normalization -> preprocess ==${RESET}"
    nnUNetv2_extract_fingerprint -d "$cc_id" --verify_dataset_integrity
    nnUNetv2_plan_experiment -d "$cc_id"
    CC_DATASET="$cc_dataset" "$PY" -c 'from segment.fine.dataset import patch_plans_no_norm_probfg as p; import os; p(os.path.join(os.environ["nnUNet_preprocessed"], os.environ["CC_DATASET"]))'
    nnUNetv2_preprocess -d "$cc_id" -c 3d_fullres
    "$PY" -m segment.cli make-splits --dataset "$cc_dataset" --protocol kfold --folds "$FOLDS"

    echo "${CYAN}== 5. Train: proposed (CC), plain TransUNet, nnU-Net baseline ==${RESET}"
    export UNFAVORSEG_LAMBDA="0.3"
    export UNFAVORSEG_EPOCHS="$EPOCHS"
    if [[ ! "$LR" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        die "--lr must be a non-negative number"
    fi
    if [[ ! "$LR" =~ ^0*([.]0*)?$ ]]; then
        export UNFAVORSEG_LR="$LR"
    fi
    "$PY" -m nnunetv2.run.run_training "$cc_id" 3d_fullres "$FOLD" -tr nnUNetTrainerTransUNetCC
    "$PY" -m nnunetv2.run.run_training "$cc_id" 3d_fullres "$FOLD" -tr nnUNetTrainerTransUNet
    "$PY" -m nnunetv2.run.run_training "$cc_id" 3d_fullres "$FOLD" -tr nnUNetTrainerUnfavorSeg

    if [[ "$LAMBDA_SWEEP" -eq 1 ]]; then
        echo "${CYAN}== 6. lambda sweep (Table X) ==${RESET}"
        for lam in 0.0 0.1 0.5 1.0; do
            export UNFAVORSEG_LAMBDA="$lam"
            "$PY" -m nnunetv2.run.run_training "$cc_id" 3d_fullres "$FOLD" -tr nnUNetTrainerTransUNetCC -o "lam_$lam"
        done
    fi

    echo "${CYAN}== 7. Predict (save probabilities) + fine experiments ==${RESET}"
    echo "${YELLOW}  Run nnUNetv2_predict for $cc_dataset into results/pred_<method>_$cls,${RESET}"
    echo "${YELLOW}  then call experiments.e2/e3/e4/uncertainty/e5/inference_time with those dirs (per class).${RESET}"
    echo "${YELLOW}  See README 'Fine-stage experiments' for the exact predict + eval commands.${RESET}"
done

echo "${GREEN}Done. Per-class tables under $RESULTS/ (suffixed by geology type).${RESET}"
