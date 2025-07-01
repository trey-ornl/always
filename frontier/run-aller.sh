#!/bin/bash
source ./env
module -t list
set -x
ulimit -c 0
export OMP_NUM_THREADS=7
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
export MPICH_OFI_CXI_COUNTER_REPORT=2
NODES=${1}
shift
TASKS=$(( NODES * 8 ))
#TASKS=$(( NODES * 4 ))
TIME='5:00'
ldd ./aller-alltoall
srun -t ${TIME} -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-alltoall $@
#exit
ldd ./aller-get
srun -t ${TIME} -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-get $@
#exit
ldd ./aller-2d
srun -q debug -t ${TIME} -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-2d $@
#exit
ldd ./aller-hsa
srun -t ${TIME} -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-hsa $@
#exit
ldd ./aller-isend
srun -t ${TIME} -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-isend $@
#exit
ldd ./aller-rsend
srun -t ${TIME} -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-rsend $@
#exit
ldd ./aller-put
srun -t ${TIME} -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ./aller-put $@
