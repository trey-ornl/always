#!/bin/bash
source ./env
module -t list
set -x
rm -f all iall
hipcc -g -O -I${CRAY_MPICH_DIR}/include -o all ../all.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a}
#CC -x hip -g -O -o all ../all.cc -L${ROCM_PATH}/lib -lamdhip64
#CC -fopenmp -g -O $(hipconfig -C) -o iall ../iall.cc -L${ROCM_PATH}/lib -lamdhip64
