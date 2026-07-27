#!/bin/bash
#HPC --partition=lab2rv
#HPC --cpu=4
#HPC --output=output/result_%j.log

set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build-riscv}
BUILD_JOBS=${BUILD_JOBS:-4}

cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --clean-first -j "${BUILD_JOBS}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

ITER_S1=${ITER_S1:-1000}
ITER_S2=${ITER_S2:-20}
ITER_S3=${ITER_S3:-10}
ITER_S4=${ITER_S4:-5}

run_case() {
    local name="$1"
    shift
    echo "===== ${name} ====="
    "./${BUILD_DIR}/lab2" "$@"
}

run_case S1 1 256 128 16 4 "${ITER_S1}"
run_case S2 1 1024 512 16 4 "${ITER_S2}"
run_case S3 128 256 128 16 4 "${ITER_S3}"
run_case S4 1024 512 128 512 2 "${ITER_S4}"
