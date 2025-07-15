#!/bin/bash
source ./env
module -t list
set -x
ulimit -c 0
export OMP_NUM_THREADS=7
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_OFI_CXI_COUNTER_REPORT=2
NODES=${1}
shift
TASKS=$(( NODES * 8 ))
TIME='5:00'
for EXE in $(ls aller-*)
do
ldd ./${EXE}
ALLER_USE_STRIDE=0 srun -t ${TIME} -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./${EXE} $@
done
