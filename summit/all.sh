#!/bin/bash
#BSUB -alloc_flags smt1 
#BSUB -J always

source ./env
module -t list
set -x
ulimit -c 0

ldd ./all

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./all contig 5 805306368
date

sleep 5

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./all strided 5 805306368
date
