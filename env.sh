#!/usr/bin/env bash
# Environment configuration for UnfavorSeg on the Linux server.
#
# Usage:
#   source env.sh          # load into the current shell before run_fine.sh
#
# All data/model paths live under a single root so they are easy to relocate.

# ---- Root directory -------------------------------------------------------
# Change this single line to move every dataset/output tree at once.
export UNFGEOSEG_ROOT="/media/data2/wouraoyu"

# ---- nnU-Net base directories (required by run_fine.sh) -------------------
export nnUNet_raw="${UNFGEOSEG_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${UNFGEOSEG_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${UNFGEOSEG_ROOT}/nnUNet_results"

# Create the directories if they do not exist yet.
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# ---- conda environment ----------------------------------------------------
# Activate the dedicated `ugs` env if conda is available and it is not the
# currently active environment.
if command -v conda >/dev/null 2>&1; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "ugs" ]]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate ugs
    fi
fi

# ---- Optional: python executable used by run_fine.sh ----------------------
# export PYTHON=python

echo "[env.sh] UNFGEOSEG_ROOT     = ${UNFGEOSEG_ROOT}"
echo "[env.sh] nnUNet_raw          = ${nnUNet_raw}"
echo "[env.sh] nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "[env.sh] nnUNet_results      = ${nnUNet_results}"
echo "[env.sh] conda env           = ${CONDA_DEFAULT_ENV:-<none>}"
