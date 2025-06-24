#!/bin/bash
source ./env
module -t list
set -x
export OMP_NUM_THREADS=7
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
export MPICH_OFI_CXI_COUNTER_REPORT=2
NODES=${1}
shift
TASKS=$(( NODES * 8 ))
#ldd ./aller-hsa
#srun -t 5:00 -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-hsa $@
ldd ./aller-alltoall
srun -t 5:00 -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-alltoall $@
exit
ldd ./aller-get
srun -t 5:00 -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-get $@
exit
ldd ./aller-isend
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-isend
ldd ./aller-rsend
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-rsend
