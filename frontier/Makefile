CXX := hipcc
CXXFLAGS := -Wall -Wno-unused-function -g -O -I$(CRAY_MPICH_DIR)/include
LDFLAGS := -L$(CRAY_MPICH_DIR)/lib $(PE_MPICH_GTL_DIR_amd_gfx90a) -lmpi $(PE_MPICH_GTL_LIBS_amd_gfx90a) -lhsa-runtime64
EXES := aller-a2ai aller-alltoall aller-3d aller-2d aller-get aller-get2d aller-hsa aller-node2d

default: $(EXES)

clean:
	rm -f $(EXES)

aller-2d: ../aller.cc ../Aller.2D.h
	$(CXX) $(CXXFLAGS) -DUSE_2D -o $@ ../aller.cc $(LDFLAGS)

aller-3d: ../aller.cc ../Aller.3D.h
	$(CXX) $(CXXFLAGS) -DUSE_3D -o $@ ../aller.cc $(LDFLAGS)

aller-a2ai: ../aller.cc ../Aller.a2ai.h
	$(CXX) $(CXXFLAGS) -DUSE_ALLTOALL_INIT -o $@ ../aller.cc $(LDFLAGS)

aller-alltoall: ../aller.cc ../Aller.MPI_Alltoall.h
	$(CXX) $(CXXFLAGS) -DUSE_ALLTOALL -o $@ ../aller.cc $(LDFLAGS)

aller-get: ../aller.cc ../Aller.MPI_Get.h
	$(CXX) $(CXXFLAGS) -DUSE_GET -o $@ ../aller.cc $(LDFLAGS)

aller-get2d: ../aller.cc ../Aller.get2D.h
	$(CXX) $(CXXFLAGS) -DUSE_GET2D -o $@ ../aller.cc $(LDFLAGS)

aller-hsa: ../aller.cc ../Aller.hsa.h
	$(CXX) $(CXXFLAGS) -DUSE_HSA -o $@ ../aller.cc $(LDFLAGS)

aller-node2d: ../aller.cc ../Aller.node2D.h
	$(CXX) $(CXXFLAGS) -DUSE_NODE2D -o $@ ../aller.cc $(LDFLAGS)
