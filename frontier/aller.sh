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

EXE=aller-alltoall
echo "## ${EXE}"
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5

EXE=aller-2d
echo "## ${EXE}"
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5

EXE=aller-get
echo "## ${EXE}"
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
ALLER_USE_FARTHEST=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
MPICH_RMA_MAX_PENDING=32 ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
MPICH_RMA_MAX_PENDING=128 ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5

EXE=aller-hsa
echo "## ${EXE}"
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
ALLER_USE_FARTHEST=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
MPICH_RMA_MAX_PENDING=32 ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
MPICH_RMA_MAX_PENDING=128 ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5

EXE=aller-3d
echo "## ${EXE}"
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
ALLER_USE_FARTHEST=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
MPICH_RMA_MAX_PENDING=32 ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
MPICH_RMA_MAX_PENDING=128 ALLER_USE_SHUFFLE=1 srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --exclusive --unbuffered ${EXE} ; sleep 5
