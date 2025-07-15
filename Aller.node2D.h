#pragma once
#include <cassert>
#include <climits>
#include <cmath>
#include <mpi.h>
#include <sstream>

#include "check.h"

__global__ static void transpose(const long count, const long nx, const long ny, const long *const from, long *const __restrict__ to)
{
  const long i = threadIdx.x+blockIdx.x*blockDim.x;
  if (i < count) {
    const long ix = blockIdx.y;
    const long iy = blockIdx.z;
    to[i+iy*count+ix*ny*count] = from[i+ix*count+iy*nx*count];
  }
}

struct Aller {

  MPI_Comm comm_;
  MPI_Comm commX_, commY_;
  std::string info_;
  long maxCount_;
  int rankX_, rankY_;
  long *recv_;
  long *send_;
  int sizeX_, sizeY_;
  hipStream_t stream_;

  Aller(MPI_Comm const comm, long *const send, long *const recv, const size_t bytes):
    comm_(comm),
    commX_(MPI_COMM_NULL), commY_(MPI_COMM_NULL),
    maxCount_(0),
    rankX_(MPI_PROC_NULL), rankY_(MPI_PROC_NULL),
    recv_(recv),
    send_(send),
    sizeX_(0), sizeY_(0)
  {
    int size = 0;
    MPI_Comm_size(comm,&size);
    maxCount_ = bytes/(long(size)*sizeof(*recv_));

    int rank = MPI_PROC_NULL;
    MPI_Comm_rank(comm,&rank);

    MPI_Comm sharedComm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm_,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&sharedComm);
    int sharedSize = 0;
    MPI_Comm_size(sharedComm,&sharedSize);
    int minSize = sharedSize;
    MPI_Allreduce(&sharedSize,&minSize,1,MPI_INT,MPI_MIN,comm_);
    int targetSize = 1;
    for (int i = 1; i <= minSize; i++) {
      if ((minSize%i == 0) && (sharedSize%i == 0)) targetSize = i;
    }
    MPI_Allreduce(&targetSize,&minSize,1,MPI_INT,MPI_MIN,comm_);

    if (minSize == sharedSize) {
      commX_ = sharedComm;
    } else {
      int sharedRank = MPI_PROC_NULL;
      MPI_Comm_rank(sharedComm,&sharedRank);
      MPI_Comm_split(sharedComm,sharedRank/minSize,sharedRank,&commX_);
      MPI_Comm_free(&sharedComm);
    }
    MPI_Comm_rank(commX_,&rankX_);
    MPI_Comm_size(commX_,&sizeX_);
    assert(sizeX_ == minSize);

    MPI_Comm_split(comm,rankX_,rank,&commY_);
    MPI_Comm_rank(commY_,&rankY_);
    MPI_Comm_size(commY_,&sizeY_);
    assert(sizeX_*sizeY_ == size);

    CHECK(hipStreamCreateWithFlags(&stream_,hipStreamNonBlocking));
    CHECK(hipStreamSynchronize(stream_));
    std::stringstream info;
    info << __FILE__ << ": " << sizeX_ << " x " << sizeY_ << " ranks";
    if ((sizeX_ == 1) || (sizeY_ == 1)) info << ", falling back to single MPI_Alltoall";
    info_ = info.str();
  }

  ~Aller()
  {
    CHECK(hipStreamSynchronize(stream_));
    CHECK(hipStreamDestroy(stream_));
    MPI_Comm_free(&commY_);
    MPI_Comm_free(&commX_);
    sizeX_ = sizeY_ = 0;
    recv_ = send_ = nullptr;
    rankX_ = rankY_ = MPI_PROC_NULL;
    maxCount_ = 0;
  }

  const char *info() const
  {
    return info_.c_str();
  }

  void run(const int count)
  {
    if ((sizeX_ == 1) || (sizeY_ == 1)) {
      MPI_Alltoall(send_,count,MPI_LONG,recv_,count,MPI_LONG,comm_);
      return;
    }

    assert(count <= maxCount_);
    assert(long(count)*long(sizeX_) <= long(INT_MAX));
    const int countY = count*sizeX_;
    MPI_Alltoall(send_,countY,MPI_LONG,recv_,countY,MPI_LONG,commY_);

    constexpr int block = 256;
    const dim3 gridY((count-1)/block+1,sizeX_,sizeY_);
    transpose<<<gridY,block,0,stream_>>>(count,sizeX_,sizeY_,recv_,send_);
    assert(long(count)*long(sizeY_) <= long(INT_MAX));
    const int countX = count*sizeY_;
    const dim3 gridX((count-1)/block+1,sizeY_,sizeX_);
    const long bytes = long(count)*long(sizeX_)*long(sizeY_)*sizeof(*recv_);
    CHECK(hipStreamSynchronize(stream_));

    MPI_Alltoall(send_,countX,MPI_LONG,recv_,countX,MPI_LONG,commX_);

    transpose<<<gridX,block,0,stream_>>>(count,sizeY_,sizeX_,recv_,send_);
    CHECK(hipMemcpyDtoDAsync(recv_,send_,bytes,stream_));
    CHECK(hipStreamSynchronize(stream_));
  }
};

