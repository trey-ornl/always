#!/bin/bash
MAX=${1}
N=1
STAMP=$(date +%s)
while [ ${N} -lt ${MAX} ]
do
  ID=$(printf %04d ${N})
  sbatch -o all.${STAMP}-${ID}.out -N${N} job.sh
  N=$(( N * 2 ))
done
ID=$(printf %04d ${MAX})
sbatch -o all.${STAMP}-${ID}.out -N${MAX} job.sh

