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
NODES=${SLURM_JOB_NUM_NODES}
TASKS=$(( NODES * 8 ))
ldd aller-*

srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-alltoall ; sleep 5
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-2d ; sleep 5
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-get2d ; sleep 5

ALLER_USE_STRIDE=0 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-3d ; sleep 5
ALLER_USE_STRIDE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-3d ; sleep 5
ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-3d ; sleep 5
ALLER_USE_FARTHEST=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-3d ; sleep 5
ALLER_USE_ROTATE=2 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-3d ; sleep 5
ALLER_USE_ROTATE=3 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-3d ; sleep 5
ALLER_USE_ROTATE=${TASKS} srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-3d ; sleep 5

ALLER_USE_STRIDE=0 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-get ; sleep 5
ALLER_USE_STRIDE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-get ; sleep 5
ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-get ; sleep 5
ALLER_USE_FARTHEST=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-get ; sleep 5
ALLER_USE_ROTATE=2 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-get ; sleep 5
ALLER_USE_ROTATE=3 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-get ; sleep 5
ALLER_USE_ROTATE=${TASKS} srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered aller-get ; sleep 5
