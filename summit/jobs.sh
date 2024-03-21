#!/bin/bash
#BSUB -alloc_flags smt1 
#BSUB -W 55 
#BSUB -J always

source ./env
module -t list
set -x
ulimit -c 0

ldd ./all

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./all contig 0 0 32768
date

sleep 5

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./all strided 0 0 32768
date

exit
sleep 5

ldd ./iall

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./iall
date

sleep 5

date
jsrun -a1 -c7 -g1 -r6 -brs --smpiargs="-gpu" ./iall 0 strided
date

