#!/bin/bash
#HPC --partition=lab2
#HPC --cpu=16
#HPC --output=output/perf_%j.log
#SBATCH -C socket=1

set -euo pipefail

rm -rf build_perf
mkdir build_perf
cd build_perf
cmake ..
make -j 16
cd ..

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

ITER_S1=${ITER_S1:-100000}
ITER_S2=${ITER_S2:-50000}
ITER_S3=${ITER_S3:-20000}
ITER_S4=${ITER_S4:-1000}

run_perf_case() {
    local name="$1"
    shift

    echo "===== ${name}: perf stat --topdown ====="
    perf stat --topdown -- ./build_perf/lab2 "$@" --benchmark

    echo "===== ${name}: perf stat --topdown --td-level 2 ====="
    perf stat --topdown --td-level 2 -- ./build_perf/lab2 "$@" --benchmark

    echo "===== ${name}: perf stat --cycles,instructions,branches,branch-misses,cache-misses ====="
    perf stat -e cycles,instructions,branches,branch-misses,cache-misses -- ./build_perf/lab2 "$@" --benchmark
}

run_perf_case S1 1 256 128 16 4 "${ITER_S1}"
# run_perf_case S2 1 1024 512 16 4 "${ITER_S2}"
# run_perf_case S3 128 256 128 16 4 "${ITER_S3}"
# run_perf_case S4 1024 512 128 512 2 "${ITER_S4}"
