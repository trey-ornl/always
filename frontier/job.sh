#!/bin/bash
#SBATCH -t 25:00 --exclusive
module load rocm
module load craype-accel-amd-gfx90a
module -t list

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export OMP_NUM_THREADS=7
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
export MPICH_OFI_NIC_VERBOSE=1
ulimit -c 0
NODES=${SLURM_JOB_NUM_NODES}
TASKS=$(( NODES * 8 ))
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ./all
sleep 5
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ./all 0 strided
sleep 5
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ./iall
sleep 5
srun -n${TASKS} -N${NODES} -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest --unbuffered ./iall 0 strided
