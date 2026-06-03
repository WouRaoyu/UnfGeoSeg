#!/usr/bin/env bash
# Export dt_pipeline inference results to nnU-Net raw datasets, then run the
# UnfavorSeg fine-stage preprocessing and training on Linux.
#
# Typical usage:
#   export nnUNet_raw=/data/nnUNet_raw
#   export nnUNet_preprocessed=/data/nnUNet_preprocessed
#   export nnUNet_results=/data/nnUNet_results
#
#   ./scripts/run_fine.sh \
#     --dataset-json /data/volumes/dataset.json \
#     --classes fragment,hardness,watery \
#     --class-names fracture_zone,soft_rock,water_rich_zone \
#     --cc-base GeologyCC \
#     --cc-id-base 20

set -Eeuo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR_PART="${SCRIPT_SOURCE%/*}"
if [[ "$SCRIPT_DIR_PART" == "$SCRIPT_SOURCE" ]]; then
    SCRIPT_DIR_PART="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR_PART" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_JSON=""
CLASSES=("unfavorable")
CLASS_NAMES=()
CC_BASE="Dataset011_HardnessCC"
CC_ID_BASE=11
FOLD=0
EPOCHS=100
LR=0.0
LAMBDA=0.3
CONFIGURATION="3d_fullres"
PY="${PYTHON:-python}"
DT_BIN="${REPO_ROOT}/process/build/dt_pipeline"
WIDTH=0
RATIO_FILTER=0
MINR=0.0
MAXR=1.0
EXTENT=""
EXPORT_START_INDEX=0
SKIP_EXPORT=0
SKIP_PREPROCESS=0
SKIP_TRAIN=0
LAMBDA_SWEEP=0
TRAINERS=("nnUNetTrainerTransUNetCC" "nnUNetTrainerTransUNet" "nnUNetTrainerUnfavorSeg")

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
        "  ./scripts/run_fine.sh --dataset-json PATH [options]" \
        "" \
        "Required:" \
        "  --dataset-json PATH              dataset.json written by dt_pipeline infer" \
        "" \
        "Dataset/export options:" \
        "  --classes VALUE                  Comma-separated dt_pipeline types to export" \
        "  --class-names VALUE              Optional comma-separated nnU-Net label names" \
        "  --cc-base VALUE                  Base name for generated nnU-Net datasets" \
        "  --cc-id-base VALUE               First nnU-Net dataset id" \
        "  --dt-bin PATH                    dt_pipeline binary, default process/build/dt_pipeline" \
        "  --width VALUE                    Export median-filter width, default 0" \
        "  --ratio-filter                   Enable export positive-ratio filtering" \
        "  --minr VALUE                     Ratio-filter min, default 0.0" \
        "  --maxr VALUE                     Ratio-filter max, default 1.0" \
        "  --extent X,Y,Z                   Optional export crop extent" \
        "  --export-start-index VALUE       Skip export cases with numeric id below VALUE" \
        "" \
        "Fine-stage options:" \
        "  --fold VALUE                     nnU-Net fold, default 0" \
        "  --epochs VALUE                   UNFAVORSEG_EPOCHS, default 100" \
        "  --lr VALUE                       Optional UNFAVORSEG_LR; 0 disables override" \
        "  --lambda VALUE                   UNFAVORSEG_LAMBDA, default 0.3" \
        "  --configuration VALUE            nnU-Net configuration, default 3d_fullres" \
        "  --trainer VALUE                  Trainer name; repeat or comma-separate" \
        "  --trainers VALUE                 Alias for --trainer" \
        "  --lambda-sweep                   Train TransUNetCC with lambda 0.0,0.1,0.5,1.0" \
        "  --py VALUE                       Python executable, default \$PYTHON or python" \
        "" \
        "Stage switches:" \
        "  --skip-export                    Reuse existing \$nnUNet_raw/DatasetXXX_*" \
        "  --skip-preprocess                Skip fingerprint/plan/patch/preprocess/splits" \
        "  --skip-train                     Stop after export/preprocess" \
        "  --help, -h                       Show this help" \
        "" \
        "Dataset naming:" \
        "  --cc-id-base 20 --cc-base GeologyCC --classes fragment,hardness" \
        "  -> \$nnUNet_raw/Dataset020_GeologyCC_fragment" \
        "  -> \$nnUNet_raw/Dataset021_GeologyCC_hardness"
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

parse_csv_into() {
    local raw="$1"
    local target_name="$2"
    local -a parts=()
    local part trimmed

    IFS=',' read -r -a parts <<< "$raw"
    eval "$target_name=()"
    for part in "${parts[@]}"; do
        trimmed="${part#"${part%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
        if [[ -n "$trimmed" ]]; then
            eval "$target_name+=(\"\$trimmed\")"
        fi
    done
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

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "${REPO_ROOT}/${path}"
    fi
}

validate_non_negative_number() {
    local flag="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "$flag must be a non-negative number"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-json)
            require_value "$1" "${2:-}"
            DATASET_JSON="$(abs_path "$2")"
            shift 2
            ;;
        --classes)
            require_value "$1" "${2:-}"
            parse_csv_into "$2" CLASSES
            shift 2
            ;;
        --class-names)
            require_value "$1" "${2:-}"
            parse_csv_into "$2" CLASS_NAMES
            shift 2
            ;;
        --cc-base)
            require_value "$1" "${2:-}"
            CC_BASE="$2"
            shift 2
            ;;
        --cc-id-base)
            require_value "$1" "${2:-}"
            CC_ID_BASE="$2"
            shift 2
            ;;
        --dt-bin)
            require_value "$1" "${2:-}"
            DT_BIN="$(abs_path "$2")"
            shift 2
            ;;
        --width)
            require_value "$1" "${2:-}"
            WIDTH="$2"
            shift 2
            ;;
        --ratio-filter)
            RATIO_FILTER=1
            shift
            ;;
        --minr)
            require_value "$1" "${2:-}"
            MINR="$2"
            RATIO_FILTER=1
            shift 2
            ;;
        --maxr)
            require_value "$1" "${2:-}"
            MAXR="$2"
            RATIO_FILTER=1
            shift 2
            ;;
        --extent)
            require_value "$1" "${2:-}"
            EXTENT="$2"
            shift 2
            ;;
        --export-start-index|--start-index)
            require_value "$1" "${2:-}"
            EXPORT_START_INDEX="$2"
            shift 2
            ;;
        --fold)
            require_value "$1" "${2:-}"
            FOLD="$2"
            shift 2
            ;;
        --epochs)
            require_value "$1" "${2:-}"
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            require_value "$1" "${2:-}"
            LR="$2"
            shift 2
            ;;
        --lambda)
            require_value "$1" "${2:-}"
            LAMBDA="$2"
            shift 2
            ;;
        --configuration)
            require_value "$1" "${2:-}"
            CONFIGURATION="$2"
            shift 2
            ;;
        --trainer|--trainers)
            require_value "$1" "${2:-}"
            parse_csv_into "$2" TRAINERS
            shift 2
            ;;
        --lambda-sweep)
            LAMBDA_SWEEP=1
            shift
            ;;
        --py)
            require_value "$1" "${2:-}"
            PY="$2"
            shift 2
            ;;
        --skip-export)
            SKIP_EXPORT=1
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=1
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
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

[[ -n "$DATASET_JSON" || "$SKIP_EXPORT" -eq 1 ]] || die "--dataset-json is required unless --skip-export is used"
[[ ${#CLASSES[@]} -gt 0 ]] || die "--classes must contain at least one type"
if [[ ${#CLASS_NAMES[@]} -gt 0 && ${#CLASS_NAMES[@]} -ne ${#CLASSES[@]} ]]; then
    die "--class-names count must match --classes count"
fi
[[ "$CC_ID_BASE" =~ ^[0-9]+$ ]] || die "--cc-id-base must be a non-negative integer"
[[ "$FOLD" =~ ^[0-9]+$ ]] || die "--fold must be a non-negative integer"
[[ "$WIDTH" =~ ^[0-9]+$ ]] || die "--width must be a non-negative integer"
[[ "$EXPORT_START_INDEX" =~ ^[0-9]+$ ]] || die "--export-start-index must be a non-negative integer"
validate_non_negative_number "--lr" "$LR"
validate_non_negative_number "--lambda" "$LAMBDA"
validate_non_negative_number "--minr" "$MINR"
validate_non_negative_number "--maxr" "$MAXR"
[[ ${#TRAINERS[@]} -gt 0 ]] || die "--trainer must contain at least one trainer"

require_env nnUNet_raw
require_env nnUNet_preprocessed
require_env nnUNet_results
command -v "$PY" >/dev/null 2>&1 || die "Python executable '$PY' was not found"

if [[ "$SKIP_EXPORT" -eq 0 ]]; then
    [[ -f "$DATASET_JSON" ]] || die "dataset metadata not found: $DATASET_JSON"
    [[ -f "$DT_BIN" ]] || die "dt_pipeline binary not found: $DT_BIN; build it with: cd process && ./build.sh"
    if [[ ! -x "$DT_BIN" ]]; then
        chmod +x "$DT_BIN" || die "cannot mark dt_pipeline executable: $DT_BIN"
    fi
    [[ -x "$DT_BIN" ]] || die "dt_pipeline is not executable: $DT_BIN"
fi

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "${CYAN}== 0. Register custom trainers ==${RESET}"
"$PY" -m segment.cli install-trainers

cc_index=0
for type_name in "${CLASSES[@]}"; do
    cc_id=$((CC_ID_BASE + cc_index))
    if [[ ${#CLASS_NAMES[@]} -gt 0 ]]; then
        class_name="${CLASS_NAMES[$cc_index]}"
    else
        class_name="$type_name"
    fi
    cc_dataset="$(get_per_class_dataset_name "$CC_BASE" "$cc_id" "$class_name")"
    raw_dataset_dir="${nnUNet_raw}/${cc_dataset}"
    cc_index=$((cc_index + 1))

    echo "${MAGENTA}######## Fine stage: $type_name -> $class_name -> $cc_dataset (id $cc_id) ########${RESET}"
    echo "${YELLOW}nnUNet raw dataset: $raw_dataset_dir${RESET}"

    if [[ "$SKIP_EXPORT" -eq 0 ]]; then
        echo "${CYAN}== 1. Export VDB results with dt_pipeline ==${RESET}"
        export_args=(
            export
            --dataset "$DATASET_JSON"
            --out "$raw_dataset_dir"
            --type "$type_name"
            --class-name "$class_name"
            --width "$WIDTH"
            --start-index "$EXPORT_START_INDEX"
        )
        if [[ "$RATIO_FILTER" -eq 1 ]]; then
            export_args+=(--ratio-filter --minr "$MINR" --maxr "$MAXR")
        fi
        if [[ -n "$EXTENT" ]]; then
            export_args+=(--extent "$EXTENT")
        fi
        "$DT_BIN" "${export_args[@]}"
    else
        echo "${YELLOW}== 1. Export skipped; reusing $raw_dataset_dir ==${RESET}"
    fi

    [[ -f "${raw_dataset_dir}/dataset.json" ]] || die "missing exported dataset.json: ${raw_dataset_dir}/dataset.json"
    [[ -d "${raw_dataset_dir}/imagesTr" ]] || die "missing imagesTr directory: ${raw_dataset_dir}/imagesTr"
    [[ -d "${raw_dataset_dir}/labelsTr" ]] || die "missing labelsTr directory: ${raw_dataset_dir}/labelsTr"

    if [[ "$SKIP_PREPROCESS" -eq 0 ]]; then
        echo "${CYAN}== 2. nnU-Net plan -> patch probfg normalization -> preprocess ==${RESET}"
        nnUNetv2_extract_fingerprint -d "$cc_id" --verify_dataset_integrity
        nnUNetv2_plan_experiment -d "$cc_id"
        CC_DATASET="$cc_dataset" "$PY" -c 'from segment.fine.dataset import patch_plans_no_norm_probfg as p; import os; p(os.path.join(os.environ["nnUNet_preprocessed"], os.environ["CC_DATASET"]))'
        nnUNetv2_preprocess -d "$cc_id" -c "$CONFIGURATION"
        "$PY" -m segment.cli make-splits --dataset "$cc_dataset"
    else
        echo "${YELLOW}== 2. Preprocess skipped ==${RESET}"
    fi

    if [[ "$SKIP_TRAIN" -eq 0 ]]; then
        echo "${CYAN}== 3. Train fine-stage models ==${RESET}"
        export UNFAVORSEG_LAMBDA="$LAMBDA"
        export UNFAVORSEG_EPOCHS="$EPOCHS"
        if [[ ! "$LR" =~ ^0*([.]0*)?$ ]]; then
            export UNFAVORSEG_LR="$LR"
        else
            unset UNFAVORSEG_LR || true
        fi

        for trainer in "${TRAINERS[@]}"; do
            "$PY" -m nnunetv2.run.run_training "$cc_id" "$CONFIGURATION" "$FOLD" -tr "$trainer"
        done

        if [[ "$LAMBDA_SWEEP" -eq 1 ]]; then
            echo "${CYAN}== 4. Lambda sweep for nnUNetTrainerTransUNetCC ==${RESET}"
            for lam in 0.0 0.1 0.5 1.0; do
                export UNFAVORSEG_LAMBDA="$lam"
                "$PY" -m nnunetv2.run.run_training "$cc_id" "$CONFIGURATION" "$FOLD" -tr nnUNetTrainerTransUNetCC -o "lam_$lam"
            done
        fi
    else
        echo "${YELLOW}== 3. Training skipped ==${RESET}"
    fi
done

echo "${GREEN}Done. Fine-stage export/preprocess/train completed for ${#CLASSES[@]} type(s).${RESET}"
