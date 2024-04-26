#!/bin/bash
#SBATCH -J allfish -t 55:00 --exclusive
source ./env
module -t list
set -x
export OMP_NUM_THREADS=7
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
ulimit -c 0
NODES=${SLURM_JOB_NUM_NODES}
TASKS=$(( NODES * 8 ))

ldd ./all

date
srun --exclusive -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ./all
date

sleep 5

date
srun --exclusive -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ./all strided
date

sleep 5

ldd ../../fishfry/fishfry
export ROCFFT_RTC_CACHE_PATH=/dev/null
NX=$(( $1 * 1024 ))
NY=$(( $2 * 1024 ))
NZ=$(( $3 * 1024 ))

date
srun --exclusive -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ../../fishfry/fishfry $1 $2 $3 $NX $NY $NZ 3
date

