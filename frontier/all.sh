#!/bin/bash
#SBATCH -J always --exclusive
source ./env
module -t list
set -x
export OMP_NUM_THREADS=7
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
export MPICH_OFI_CXI_COUNTER_REPORT=2
export PMI_MMAP_SYNC_WAIT_TIME=1800
ulimit -c 0
NODES=${SLURM_JOB_NUM_NODES}
TASKS=$(( NODES * 8 ))

ldd ./all

date
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ./all contig 5 2147483648
date

sleep 5

date
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ./all strided 5 2147483648
date
