#!/bin/bash
MAX=$1
N=1
while [ ${N} -lt ${MAX} ]
do
  ID=$(printf %04d ${N})
  sbatch -o all.${ID}.out -N${N} job.sh
  sbatch -o iall.${ID}.out -N${N} ijob.sh
  N=$(( N * 2 ))
done
N=${MAX}
ID=$(printf %04d ${N})
sbatch -o all.${ID}.out -N${N} job.sh
sbatch -o iall.${ID}.out -N${N} ijob.sh

