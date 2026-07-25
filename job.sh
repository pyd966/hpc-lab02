#!/bin/bash
#HPC --partition=lab2
#HPC --cpu=16
#HPC --output=output/result_%j.log
#SBATCH -C socket=1

set -euo pipefail

rm -rf build
mkdir build
cd build
cmake ..
make -j 16
cd ..

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

ITER_S1=${ITER_S1:-2000}
ITER_S2=${ITER_S2:-1000}
ITER_S3=${ITER_S3:-200}
ITER_S4=${ITER_S4:-5}

run_case() {
    local name="$1"
    shift
    echo "===== ${name} ====="
    ./build/lab2 "$@"
}

run_case S1 1 256 128 16 4 "${ITER_S1}"
run_case S2 1 1024 512 16 4 "${ITER_S2}"
run_case S3 128 256 128 16 4 "${ITER_S3}"
run_case S4 1024 512 128 512 2 "${ITER_S4}"
