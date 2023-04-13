#!/bin/bash
source ./env
module -t list
set -x
rm -f all iall
CC -fopenmp -g -O $(hipconfig -C) -o all ../all.cc -L${ROCM_PATH}/lib -lamdhip64
CC -fopenmp -g -O $(hipconfig -C) -o iall ../iall.cc -L${ROCM_PATH}/lib -lamdhip64
