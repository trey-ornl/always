#!/bin/bash
module -t list
set -x
rm -f hall
mpiCC -g -O -qsmp=omp -o hall ../hall.cc

