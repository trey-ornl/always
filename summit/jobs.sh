#!/bin/bash
#BSUB -alloc_flags smt1 
#BSUB -W 25 
#BSUB -J always

source ./env
module -t list
set -x
ulimit -c 0

ldd ./all

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./all
date

sleep 5

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./all 0 strided
date

sleep 5

ldd ./iall

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./iall
date

sleep 5

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./iall 0 strided
date

