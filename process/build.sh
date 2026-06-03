#!/usr/bin/env bash
#
# Configure and build the standalone dt_pipeline tool on Linux.
# Prefers the Ninja generator; falls back to Unix Makefiles if ninja is absent.
#
# Usage:
#   ./build.sh [build-dir]
#
# Useful environment variables / cmake hints (pass as -D... after editing or
# export before running):
#   OPENVDB_ROOT   install prefix of OpenVDB (if not on the default search path)
#   CMAKE_PREFIX_PATH  ':'-separated prefixes for VTK / pybind11 / TBB / json
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

# Forward an OpenVDB install prefix if the caller provided one.
if [[ -n "${OPENVDB_ROOT:-}" ]]; then
    CMAKE_ARGS+=(-DOPENVDB_ROOT="${OPENVDB_ROOT}")
fi

# Forward extra search prefixes for the remaining packages.
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}")
fi

echo "[build.sh] Configuring with generator '${GENERATOR}' -> ${BUILD_DIR}"
cmake "${CMAKE_ARGS[@]}"

echo "[build.sh] Building"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "$(nproc)"

echo "[build.sh] Done. Executable: ${BUILD_DIR}/dt_pipeline"
