#!/usr/bin/env bash
#
# Configure and build the standalone dt_pipeline tool on Linux.
# Prefers the Ninja generator; falls back to Unix Makefiles if ninja is absent.
#
# Usage:
#   ./build.sh [build-dir]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
else
    echo "[build.sh] ninja not found, falling back to Unix Makefiles" >&2
    GENERATOR="Unix Makefiles"
fi

CMAKE_ARGS=(
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
    -G "${GENERATOR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)

# Forward extra search prefixes for the remaining packages.
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="${CONDA_PREFIX}")
fi

echo "[build.sh] Configuring with generator '${GENERATOR}' -> ${BUILD_DIR}"
cmake "${CMAKE_ARGS[@]}"

echo "[build.sh] Building"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "$(nproc)"

echo "[build.sh] Done. Executable: ${BUILD_DIR}/dt_pipeline"
