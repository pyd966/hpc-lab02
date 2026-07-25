#!/bin/bash
#HPC --partition=lab2
#HPC --cpu=16
#HPC --output=output/result_%j.log
#SBATCH -C socket=1

rm -rf build
mkdir build && cd build
cmake ..
make -j 16

export OMP_NUM_THREADS=16
./lab2 128 256 128 16 4