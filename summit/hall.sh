#!/bin/bash
#BSUB -alloc_flags smt1 
#BSUB -J halways

module -t list
set -x
ulimit -c 0

ldd ./hall

COUNTMAX=805306368

date
jsrun -a1 -c7 -g1 -r6 -brs ./hall contig 5 ${COUNTMAX}
date

sleep 5

date
jsrun -a1 -c7 -g1 -r6 -brs ./hall strided 5 ${COUNTMAX}
date
