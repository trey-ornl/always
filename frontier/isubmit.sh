#!/bin/bash
MAX=${1}
N=1
STAMP=$(date +%s)
while [ ${N} -lt ${MAX} ]
do
  ID=$(printf %04d ${N})
  sbatch -o iall.${STAMP}-${ID}.out -N${N} ijob.sh
  N=$(( N * 2 ))
done
ID=$(printf %04d ${MAX})
sbatch -o iall.${STAMP}-${ID}.out -N${MAX} ijob.sh

