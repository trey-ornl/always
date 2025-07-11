#!/bin/bash
source ./env
module -t list
set -x
cd ..

rm -f frontier/aller-3d
hipcc -Wall -Wno-unused-function -g -DUSE_3D -DUSE_SHUFFLE -O -I${CRAY_MPICH_DIR}/include -o frontier/aller-3d aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} -lhsa-runtime64 || exit

exit

rm -f frontier/aller-alltoall
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_ALLTOALL -O -I${CRAY_MPICH_DIR}/include -o frontier/aller-alltoall aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

exit

rm -f frontier/aller-2d
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_2D -O -I${CRAY_MPICH_DIR}/include -o frontier/aller-2d aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

#exit

rm -f frontier/aller-hsa
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_HSA -O -I${CRAY_MPICH_DIR}/include -o frontier/aller-hsa aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} -lhsa-runtime64 || exit

#exit

rm -f frontier/aller-get
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_GET -O -I${CRAY_MPICH_DIR}/include -o frontier/aller-get aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

#exit

rm -f frontier/aller-isend
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_ISEND -O -I${CRAY_MPICH_DIR}/include -o frontier/aller-isend aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

#exit

rm -f frontier/aller-put
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_PUT -O -I${CRAY_MPICH_DIR}/include -o frontier/aller-put aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit

#exit

rm -f frontier/aller-rsend
hipcc -Wall -Werror -Wno-unused-function -g -DUSE_RSEND -O -I${CRAY_MPICH_DIR}/include -o frontier/aller-rsend aller.cc -L${CRAY_MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} || exit
