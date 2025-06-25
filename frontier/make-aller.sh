#!/bin/bash
source ./env
module -t list
set -x

rm -f aller-alltoall
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_ALLTOALL -O -I${CRAY_MPICH_DIR}/include -o aller-alltoall ../aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

rm -f aller-hsa
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_HSA -O -I${CRAY_MPICH_DIR}/include -o aller-hsa ../aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} -lhsa-runtime64 || exit

rm -f aller-get
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_GET -O -I${CRAY_MPICH_DIR}/include -o aller-get ../aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

rm -f aller-isend
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_ISEND -O -I${CRAY_MPICH_DIR}/include -o aller-isend ../aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

rm -f aller-put
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_PUT -O -I${CRAY_MPICH_DIR}/include -o aller-put ../aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

rm -f aller-rsend
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_RSEND -O -I${CRAY_MPICH_DIR}/include -o aller-rsend ../aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit
