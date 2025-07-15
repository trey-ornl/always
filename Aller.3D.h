#pragma once
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <mpi.h>
#include <random>
#include <sstream>
#include <vector>

#include "check.h"

static void stop()
{
  fflush(stdout);
  CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  exit(0);
}

static __global__ void gatherFrom(const size_t count, const size_t sharedRank, const int *const ranksByNode, const long *const *const sends, long *const recv)
{
  const unsigned sendIndex = blockIdx.x;
  const unsigned fromIndex = blockIdx.y;
  const unsigned nodeIndex = blockIdx.z;
  const size_t fromRank = ranksByNode[fromIndex+gridDim.y*(sendIndex+gridDim.x*nodeIndex)];

  const long *const q = sends[sendIndex]+count*(fromIndex+gridDim.y*(sharedRank+gridDim.x*nodeIndex));
  long *const __restrict__ p = recv+count*fromRank;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) p[i] = q[i];
}

static __global__ void gatherTo(const size_t count, const int sharedRank, const int *const ranksByNode, const long *const *const sends, long *const recv)
{
  const unsigned sendIndex = blockIdx.x;
  const unsigned toIndex = blockIdx.y;
  const unsigned nodeIndex = blockIdx.z;
  const size_t toRank = ranksByNode[toIndex+gridDim.y*(sharedRank+nodeIndex*gridDim.x)];

  const long *const q = sends[sendIndex]+count*toRank;
  long *const __restrict__ p = recv+count*(sendIndex+gridDim.x*(toIndex+gridDim.y*nodeIndex));

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) p[i] = q[i];
}

struct Aller {

  static constexpr unsigned blockMax_ = 1024;
  MPI_Comm comm_;
  dim3 grid_;
  std::string info_;
  long maxCount_;
  size_t myNodeIndex_;
  int nGets_;
  std::vector<size_t> originOffsets_;
  bool punt_;
  int rank_;
  int *ranksByNode_;
  long *recv_;
  long *send_;
  long **sends_;
  MPI_Comm sharedComm_;
  int sharedRank_;
  int sharedSize_;
  hipStream_t stream_;
  std::vector<int> targetRanks_;
  MPI_Win win_;

  Aller(MPI_Comm const comm, long *const send, long *const recv, const size_t bytes):
    comm_(comm),
    grid_(0),
    maxCount_(0),
    myNodeIndex_(0),
    nGets_(0),
    punt_(false),
    rank_(MPI_PROC_NULL),
    ranksByNode_(nullptr),
    recv_(recv),
    send_(send),
    sends_(nullptr),
    sharedComm_(MPI_COMM_NULL),
    sharedRank_(MPI_PROC_NULL),
    sharedSize_(0),
    stream_(0),
    win_(MPI_WIN_NULL)
  {
    int size = 0;
    MPI_Comm_size(comm_,&size);
    maxCount_ = bytes/(size*sizeof(*recv_));

    MPI_Comm_rank(comm_,&rank_);

    MPI_Comm sharedComm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm_,MPI_COMM_TYPE_SHARED,rank_,MPI_INFO_NULL,&sharedComm);
    int sharedSize = 0;
    MPI_Comm_size(sharedComm,&sharedSize);
    int minSize = sharedSize;
    MPI_Allreduce(&sharedSize,&minSize,1,MPI_INT,MPI_MIN,comm_);
    int targetSize = 1;
    for (int i = 1; i <= minSize; i++) {
      if ((minSize%i == 0) && (sharedSize%i == 0) && (size%(i*i)  == 0)) targetSize = i;
    }
    MPI_Allreduce(&targetSize,&minSize,1,MPI_INT,MPI_MIN,comm_);

    std::stringstream info;
    info << __FILE__;

    if (minSize == 1) {
      punt_ = true;
      MPI_Comm_free(&sharedComm);
      info << ": one rank per node, falling back to MPI_Alltoall";
      info_ = info.str();
      return;
    }

    if (minSize == sharedSize) {
      sharedComm_ = sharedComm;
    } else {
      int sharedRank = MPI_PROC_NULL;
      MPI_Comm_rank(sharedComm,&sharedRank);
      MPI_Comm_split(sharedComm,sharedRank/minSize,sharedRank,&sharedComm_);
      MPI_Comm_free(&sharedComm);
    }
    MPI_Comm_size(sharedComm_,&sharedSize_);
    assert(sharedSize_ == minSize);

    nGets_ = size/(sharedSize_*sharedSize_);
    assert(nGets_*sharedSize_*sharedSize_ == size);

    grid_.x = sharedSize_;
    grid_.y = sharedSize_;
    grid_.z = nGets_;

    info << ": " << sharedSize_ << " x " << nGets_ << " x " << sharedSize_;

    MPI_Comm_rank(sharedComm_,&sharedRank_);

    std::vector<int> ranksByNode(size,MPI_PROC_NULL);

    int myNode = MPI_PROC_NULL;
    int nNodes = 0;
    {
      std::vector<int> ranksShared(sharedSize_,MPI_PROC_NULL);
      MPI_Allgather(&rank_,1,MPI_INT,ranksShared.data(),1,MPI_INT,sharedComm_);

      MPI_Comm crossComm = MPI_COMM_NULL;
      MPI_Comm_split(comm_,sharedRank_,ranksShared.front(),&crossComm);
      MPI_Comm_rank(crossComm,&myNode);
      MPI_Comm_size(crossComm,&nNodes);
      assert(sharedSize_*nNodes == size);
      myNodeIndex_ = myNode/sharedSize_;
      MPI_Allgather(ranksShared.data(),sharedSize_,MPI_INT,ranksByNode.data(),sharedSize_,MPI_INT,crossComm);
      MPI_Comm_free(&crossComm);

      {
        assert(myNode != MPI_PROC_NULL);
        int nodeCheck = MPI_PROC_NULL;
        MPI_Allreduce(&myNode,&nodeCheck,1,MPI_INT,MPI_MIN,sharedComm_);
        assert(myNode == nodeCheck);
      }

      std::vector<hsa_amd_ipc_memory_t> handles(sharedSize_);
      hsa_amd_ipc_memory_t handle;
      CHECK(hsa_amd_ipc_memory_create(send_,bytes,&handle));
      MPI_Allgather(&handle,sizeof(hsa_amd_ipc_memory_t),MPI_BYTE,handles.data(),sizeof(hsa_amd_ipc_memory_t),MPI_BYTE,sharedComm_);

      std::vector<long*> sends;
      sends.reserve(sharedSize_);
      for (int i = 0; i < sharedSize_; i++) {
        if (ranksShared.at(i) == rank_) {
          sends.push_back(send_);
        } else {
          void *p = nullptr;
          CHECK(hsa_amd_ipc_memory_attach(&handles[i],bytes,0,nullptr,&p));
          sends.push_back(static_cast<long*>(p));
        }
      }
      assert(sends.size() == sharedSize_);
      const size_t sendsBytes = sharedSize_*sizeof(*sends_);
      CHECK(hipMalloc(&sends_,sendsBytes));
      CHECK(hipMemcpyHtoD(sends_,sends.data(),sendsBytes));
    }

    const size_t ranksByNodeBytes = ranksByNode.size()*sizeof(*ranksByNode_);
    CHECK(hipMalloc(&ranksByNode_,ranksByNodeBytes));
    CHECK(hipMemcpyHtoD(ranksByNode_,ranksByNode.data(),ranksByNodeBytes));

    {
      std::vector<int> nodes;
      nodes.reserve(nGets_);

      if (getenv("ALLER_USE_FARTHEST")) {

        const int origin = sharedRank_+myNodeIndex_*sharedSize_;
        for (int i = 0; i < nGets_; i++) {
          const int step = (i%2) ? (i+1)/2 : -i/2;
          const int node = (origin+step*sharedSize_+nNodes)%nNodes;
          nodes.push_back(node);
        }
        std::reverse(nodes.begin(),nodes.end());
        info << ", " << nGets_ << " target nodes farthest to closest (ALLER_USE_FARTHEST)";

      } else if (getenv("ALLER_USE_SHUFFLE")) {

        for (int i = 0; i < nGets_; i++) {
          const int node = sharedRank_+i*sharedSize_;
          assert(node < nNodes);
          nodes.push_back(node);
        }
        std::shuffle(nodes.begin(),nodes.end(),std::default_random_engine(rank_+1));
        info << ", randomly shuffle " << nGets_ << " target nodes (ALLER_USE_SHUFFLE)";

      } else if (getenv("ALLER_USE_STRIDE")) {

        int stride = int(sqrt(double(nGets_)));
        const char *const useStrideStr = getenv("ALLER_USE_STRIDE");
        int value = stride;
        if ((sscanf(useStrideStr,"%d",&value) == 1) && (value > 0)) stride = value;
        for (int i = 0; i < stride; i++) {
          for (int j = i; j < nGets_; j += stride) {
            const int node = sharedRank_+j*sharedSize_;
            nodes.push_back(node);
          }
        }
        assert(nodes.size() == nGets_);
        std::reverse(nodes.begin(),nodes.end());
        info << ", stride " << stride << " through " << nGets_ << " target nodes (ALLER_USE_STRIDE)";

      } else {

        int useRotate = 3;
        const char *const useRotateStr = getenv("ALLER_USE_ROTATE");
        if (useRotateStr) {
          int value = 0;
          if ((sscanf(useRotateStr,"%d",&value) == 1) && (value > 0)) useRotate = value;
        }

        const int rotation = nGets_/useRotate+1;
        for (int i = 0; i < nGets_; i++) {
          const int node = (sharedRank_+(i+rotation)*sharedSize_)%nNodes;
          nodes.push_back(node);
        }
        info << ", rotate " << nGets_ << " target nodes by " << rotation << " (ALLER_USE_ROTATE=" << useRotate << ")";

      }

      info_ = info.str();

      const int targetSharedIndex = myNode%sharedSize_;
      originOffsets_.reserve(nGets_);
      targetRanks_.reserve(nGets_);

      for (int i = 0; i < nGets_; i++) {
        const int node = nodes.at(i);

        const size_t originOffset = node/sharedSize_;
        originOffsets_.push_back(originOffset);

        const int targetIndex = targetSharedIndex+node*sharedSize_;
        const int targetRank = ranksByNode.at(targetIndex);
        if (node == myNode) assert(targetRank == rank_);
        targetRanks_.push_back(targetRank);
      }

    }

    CHECK(hipStreamCreateWithFlags(&stream_,hipStreamNonBlocking));
    CHECK(hipDeviceSynchronize());

    MPI_Win_create(recv_,bytes,sizeof(*recv_),MPI_INFO_NULL,comm_,&win_);
    MPI_Win_fence(0,win_);
  }

  ~Aller()
  {
    if (punt_) {
      maxCount_ = 0;
      comm_ = MPI_COMM_NULL;
      return;
    }

    MPI_Win_free(&win_);
    CHECK(hipStreamDestroy(stream_));
    stream_ = 0;
    MPI_Comm_free(&sharedComm_);
    std::vector<long*> sends(sharedSize_,nullptr);
    CHECK(hipMemcpyDtoH(sends.data(),sends_,sharedSize_*sizeof(*sends_)));
    for (auto &send : sends) {
      if (send != send_) CHECK(hsa_amd_ipc_memory_detach(send));
      send = nullptr;
    }
    CHECK(hipFree(sends_));
    sends_ = nullptr;
    CHECK(hipFree(ranksByNode_));
    ranksByNode_ = nullptr;

    sharedSize_ = 0;
    sharedRank_ = MPI_PROC_NULL;
    send_ = nullptr;
    recv_ = nullptr;
    rank_ = MPI_PROC_NULL;
    nGets_ = 0;
    myNodeIndex_ = -1;
    maxCount_ = 0;
    comm_ = MPI_COMM_NULL;
  }

  const char *info() const
  {
    return info_.c_str();
  }

  void run(const int count)
  {
    if (false) {
      int size = 0;
      MPI_Comm_size(comm_,&size);
      for (int i = 0; i < size; i++) {
        MPI_Barrier(comm_);
        if (i == rank_) {
          printf("%d %d A:",rank_,sharedRank_);
          for (int j = 0; j < size; j++) {
            for (int k = 0; k < count; k++) {
              printf(" %ld",send_[k+j*count]);
            }
            printf(" |");
          }
          printf("\n");
          fflush(stdout);
        }
        MPI_Barrier(comm_);
      }
    }

    assert(count <= maxCount_);
    if (punt_) {
      MPI_Alltoall(send_,count,MPI_LONG,recv_,count,MPI_LONG,comm_);
      return;
    }

    MPI_Barrier(sharedComm_);

    const unsigned block = std::min<unsigned>(blockMax_,count);
    gatherTo<<<grid_,block,0,stream_>>>(count,sharedRank_,ranksByNode_,sends_,recv_);

    assert(long(count)*long(sharedSize_*sharedSize_) <= long(UINT_MAX));
    const size_t targetCount = count*sharedSize_*sharedSize_;
    const MPI_Aint targetDisp = myNodeIndex_*targetCount;

    CHECK(hipStreamSynchronize(stream_));
    MPI_Win_fence(0,win_);

    if (false) {
      int size = 0;
      MPI_Comm_size(comm_,&size);
      for (int i = 0; i < size; i++) {
        MPI_Barrier(comm_);
        if (i == rank_) {
          printf("%d %d B:",rank_,sharedRank_);
          for (int j = 0; j < size; j++) {
            for (int k = 0; k < count; k++) {
              printf(" %ld",recv_[k+j*count]);
            }
            printf(" |");
          }
          printf("\n");
          fflush(stdout);
        }
        MPI_Barrier(comm_);
      }
    }
    for (int i = 0; i < nGets_; i++) {
      long *const originAddr = send_+originOffsets_[i]*targetCount;
      if (targetRanks_[i] == rank_) {
        long *const src = recv_+myNodeIndex_*targetCount;
        const size_t bytes = sizeof(long)*targetCount;
        CHECK(hipMemcpyDtoDAsync(originAddr,src,bytes,stream_));
      } else {
        MPI_Get(originAddr,targetCount,MPI_LONG,targetRanks_[i],targetDisp,targetCount,MPI_LONG,win_);
      }
    }
    CHECK(hipStreamSynchronize(stream_));
    MPI_Win_fence(0,win_);

    if (false) {
      int size = 0;
      MPI_Comm_size(comm_,&size);
      for (int i = 0; i < size; i++) {
        MPI_Barrier(comm_);
        if (i == rank_) {
#if 0
          printf("%d %d BC ",rank_,sharedRank_);
          for (int j = 0; j < nGets_; j++) {
            printf(" %d[%lu] = %d[%lu]",rank_,originOffsets_[j],targetRanks_[j],myNodeIndex_);
          }
          printf("\n");
#endif
          printf("%d %d C:",rank_,sharedRank_);
          for (int j = 0; j < size; j++) {
            for (int k = 0; k < count; k++) {
              printf(" %ld",send_[k+j*count]);
            }
            printf(" |");
          }
          printf("\n");
          fflush(stdout);
        }
        MPI_Barrier(comm_);
      }
    }

    gatherFrom<<<grid_,block,0,stream_>>>(count,sharedRank_,ranksByNode_,sends_,recv_);
    CHECK(hipStreamSynchronize(stream_));
    MPI_Barrier(sharedComm_);
    if (false) {
      int size = 0;
      MPI_Comm_size(comm_,&size);
      for (int i = 0; i < size; i++) {
        MPI_Barrier(comm_);
        if (i == rank_) {
          printf("%d %d D:",rank_,sharedRank_);
          for (int j = 0; j < size; j++) {
            for (int k = 0; k < count; k++) {
              printf(" %ld",recv_[k+j*count]);
            }
            printf(" |");
          }
          printf("\n");
          fflush(stdout);
        }
        MPI_Barrier(comm_);
      }
    }
  }
};

