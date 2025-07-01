#pragma once
#include <cassert>
#include <cstdio>
#include <mpi.h>
#include <sstream>
#include <vector>

#include "check.h"

static __global__ void copy(const unsigned count, const int rank, const int *const sharedRanks, const long *const *const sends, long *const recv)
{
  long *const __restrict__ p = recv+long(sharedRanks[blockIdx.y])*long(count);
  const long *const q = sends[blockIdx.y]+long(rank)*long(count);
  const unsigned stride = blockDim.x*gridDim.x;
  const unsigned ilo = threadIdx.x+blockIdx.x*blockDim.x;
  for (unsigned i = ilo; i < count; i += stride) p[i] = q[i];
}

struct Aller {

  static constexpr int block_ = 256;
  MPI_Comm comm_;
  std::string info_;
  long maxCount_;
  int maxThreads_;
  int rank_;
  long *recv_;
  long **sends_;
  MPI_Comm sharedComm_;
  int *sharedRanks_;
  int sharedSize_;
  hipStream_t stream_;
  std::vector<int> targets_;
  int targetSize_;
  MPI_Win win_;

  Aller(MPI_Comm const comm, long *const send, long *const recv, const size_t bytes):
    comm_(comm),
    maxCount_(0),
    maxThreads_(0),
    rank_(MPI_PROC_NULL),
    recv_(recv),
    sends_(nullptr),
    sharedComm_(MPI_COMM_NULL),
    sharedRanks_(nullptr),
    sharedSize_(0),
    targetSize_(0)
  {
    MPI_Comm_rank(comm_,&rank_);
    int size = 0;
    MPI_Comm_size(comm_,&size);

    int id = -1;
    CHECK(hipGetDevice(&id));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop,id));
    maxThreads_ = prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount;

    MPI_Comm_split_type(comm_,MPI_COMM_TYPE_SHARED,rank_,MPI_INFO_NULL,&sharedComm_);
    MPI_Comm_size(sharedComm_,&sharedSize_);
    CHECK(hipMalloc(&sharedRanks_,sharedSize_*sizeof(*sharedRanks_)));
    MPI_Allgather(&rank_,1,MPI_INT,sharedRanks_,1,MPI_INT,sharedComm_);

    std::vector<hsa_amd_ipc_memory_t> handles(sharedSize_);
    hsa_amd_ipc_memory_t handle;
    CHECK(hsa_amd_ipc_memory_create(send,bytes,&handle));
    MPI_Allgather(&handle,sizeof(hsa_amd_ipc_memory_t),MPI_BYTE,handles.data(),sizeof(hsa_amd_ipc_memory_t),MPI_BYTE,sharedComm_);

    CHECK(hipMalloc(&sends_,sharedSize_*sizeof(*sends_)));
    for (int i = 0; i < sharedSize_; i++) {
      if (sharedRanks_[i] == rank_) {
        sends_[i] = send;
      } else {
        void *p = nullptr;
        CHECK(hsa_amd_ipc_memory_attach(&handles[i],bytes,0,nullptr,&p));
        sends_[i] = static_cast<long*>(p);
      }
    }

    maxCount_ = bytes/(size*sizeof(*recv_));

    std::vector<bool> isRemote(size,true);
    for (int i = 0; i < sharedSize_; i++) isRemote.at(sharedRanks_[i]) = false;
    targetSize_ = size-sharedSize_;
    targets_.reserve(targetSize_);
    for (int i = 0; i < size; i++) {
      const int j = (rank_+i+1)%size;
      if (isRemote[j]) targets_.push_back(j);
    }
    assert(targets_.size() == targetSize_);

    CHECK(hipStreamCreateWithFlags(&stream_,hipStreamNonBlocking));
    CHECK(hipStreamSynchronize(stream_));

    MPI_Win_create(send,bytes,sizeof(*recv_),MPI_INFO_NULL,comm_,&win_);
    MPI_Win_fence(0,win_);

    std::stringstream info;
    info << __FILE__ << ": MPI_Get off node (" << targetSize_
      << " ranks), copy kernel on node (" << sharedSize_ << " ranks)";
    info_ = info.str();
  }

  ~Aller()
  {
    MPI_Win_fence(0,win_);
    MPI_Win_free(&win_);
    CHECK(hipStreamSynchronize(stream_));
    CHECK(hipStreamDestroy(stream_));
    targetSize_ = 0;
    for (int i = 0; i < sharedSize_; i++) {
      if (sharedRanks_[i] != rank_) CHECK(hsa_amd_ipc_memory_detach(sends_[i]));
    }
    sharedSize_ = 0;
    CHECK(hipFree(sharedRanks_));
    sharedRanks_ = nullptr;
    MPI_Comm_free(&sharedComm_);
    CHECK(hipFree(sends_));
    sends_ = nullptr;
    recv_ = nullptr;
    rank_ = MPI_PROC_NULL;
    maxCount_ = 0;
    comm_ = MPI_COMM_NULL;
  }

  const char *info() const
  {
    return info_.c_str();
  }

  void run(const int count)
  {
    assert(count <= maxCount_);
    const int threads = std::min<long>(maxThreads_,long(count)*long(sharedSize_));
    const dim3 grid{uint32_t((threads-1)/(block_*sharedSize_)+1),uint32_t(sharedSize_)};
    const MPI_Aint disp = long(count)*long(rank_);
    MPI_Win_fence(0,win_);
    copy<<<grid,block_,0,stream_>>>(count,rank_,sharedRanks_,sends_,recv_);
    for (int i = 0; i < targetSize_; i++) {
      const int target = targets_[i];
      long *const addr = recv_+(long(target)*long(count));
      MPI_Get(addr,count,MPI_LONG,target,disp,count,MPI_LONG,win_);
    }
    CHECK(hipStreamSynchronize(stream_));
    MPI_Win_fence(0,win_);
  }
};

