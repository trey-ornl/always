#!/bin/bash
source ./env
module -t list
set -x
rm -f all iall
mpiCC -I. -std=gnu++1y -qsmp=omp -g -O -o all ../all.cc -L${CUDA_DIR}/lib64 -lcudart
mpiCC -I. -std=gnu++1y -qsmp=omp -g -O -o iall ../iall.cc -L${CUDA_DIR}/lib64 -lcudart

