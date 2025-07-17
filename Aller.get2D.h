#pragma once
#include <cassert>
#include <climits>
#include <cmath>
#include <mpi.h>
#include <sstream>
#include <vector>

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
  std::vector<int> ranksX_, ranksY_;
  int rank_, rankX_, rankY_;
  long *recv_;
  long *send_;
  int sizeX_, sizeY_;
  hipStream_t stream_;
  MPI_Win winX_, winY_;

  Aller(MPI_Comm const comm, long *const send, long *const recv, const size_t bytes):
    comm_(comm),
    commX_(MPI_COMM_NULL), commY_(MPI_COMM_NULL),
    maxCount_(0),
    rank_(MPI_PROC_NULL), rankX_(MPI_PROC_NULL), rankY_(MPI_PROC_NULL),
    recv_(recv),
    send_(send),
    sizeX_(0), sizeY_(0),
    winX_(MPI_WIN_NULL), winY_(MPI_WIN_NULL)
  {
    int size = 0;
    MPI_Comm_size(comm,&size);
    maxCount_ = bytes/(long(size)*sizeof(*recv_));

    MPI_Comm_rank(comm,&rank_);

    const int sizeRoot = round(sqrt(double(size)));
    int nx = 1;
    for (int i = 1; i <= sizeRoot; i++) {
      if (size%i == 0) nx = i;
    }
    const int ny = size/nx;
    assert(nx*ny == size);

    MPI_Comm_split(comm,rank_/nx,rank_,&commX_);
    MPI_Comm_rank(commX_,&rankX_);
    MPI_Comm_size(commX_,&sizeX_);
    assert(sizeX_ == nx);

    std::vector<int> ranksX(sizeX_);
    MPI_Allgather(&rank_,1,MPI_INT,ranksX.data(),1,MPI_INT,commX_);
    const int strideX = int(std::sqrt(sizeX_));
    ranksX_.reserve(sizeX_);
    for (int i = 0; i < strideX; i++) {
      for (int j = i; j < sizeX_; j += strideX) {
        const int rank = (rankX_+j)%sizeX_;
        ranksX_.push_back(rank);
      }
    }
    assert(ranksX_.size() == sizeX_);
    std::reverse(ranksX_.begin(),ranksX_.end());

    MPI_Comm_split(comm,rank_%nx,rank_,&commY_);
    MPI_Comm_rank(commY_,&rankY_);
    MPI_Comm_size(commY_,&sizeY_);
    assert(sizeY_ == ny);

    std::vector<int> ranksY(sizeY_);
    MPI_Allgather(&rank_,1,MPI_INT,ranksY.data(),1,MPI_INT,commY_);
    const int strideY = int(std::sqrt(sizeY_));
    ranksY_.reserve(sizeY_);
    for (int i = 0; i < strideY; i++) {
      for (int j = i; j < sizeY_; j += strideY) {
        const int rank = (rankY_+j)%sizeY_;
        ranksY_.push_back(rank);
      }
    }
    assert(ranksY_.size() == sizeY_);
    std::reverse(ranksY_.begin(),ranksY_.end());

    CHECK(hipStreamCreateWithFlags(&stream_,hipStreamNonBlocking));
    CHECK(hipStreamSynchronize(stream_));

    MPI_Win_create(send_,bytes,sizeof(*recv_),MPI_INFO_NULL,commX_,&winX_);
    MPI_Win_fence(0,winX_);

    MPI_Win_create(send_,bytes,sizeof(*recv_),MPI_INFO_NULL,commY_,&winY_);
    MPI_Win_fence(0,winY_);

    std::stringstream info;
    info << __FILE__ << ": " << sizeX_ << " x " << sizeY_ << " ranks";
    if ((sizeX_ == 1) || (sizeY_ == 1)) info << ", falling back to single MPI_Alltoall";
    info_ = info.str();
  }

  ~Aller()
  {
    MPI_Win_free(&winY_);
    MPI_Win_free(&winX_);
    CHECK(hipStreamSynchronize(stream_));
    CHECK(hipStreamDestroy(stream_));
    MPI_Comm_free(&commY_);
    MPI_Comm_free(&commX_);
    sizeX_ = sizeY_ = 0;
    recv_ = send_ = nullptr;
    rank_ = rankX_ = rankY_ = MPI_PROC_NULL;
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

    MPI_Win_fence(0,winY_);
    assert(count <= maxCount_);
    assert(long(count)*long(sizeX_) <= long(INT_MAX));
    const size_t countY = count*sizeX_;
    const size_t bytesY = countY*sizeof(long);
    const MPI_Aint dispY = rankY_*countY;
    for (int i = 0; i < sizeY_; i++) {
      long *const originAddr = recv_+ranksY_[i]*countY;
      if (ranksY_[i] == rankY_) {
        long *const src = send_+rankY_*countY;
        CHECK(hipMemcpyDtoDAsync(originAddr,src,bytesY,stream_));
      } else {
        MPI_Get(originAddr,countY,MPI_LONG,ranksY_[i],dispY,countY,MPI_LONG,winY_);
      }
    }
    MPI_Win_fence(0,winY_);

    constexpr int block = 256;
    const dim3 gridY((count-1)/block+1,sizeX_,sizeY_);
    transpose<<<gridY,block,0,stream_>>>(count,sizeX_,sizeY_,recv_,send_);

    assert(long(count)*long(sizeY_) <= long(INT_MAX));
    const dim3 gridX((count-1)/block+1,sizeY_,sizeX_);
    const size_t bytes = size_t(count)*size_t(sizeX_)*size_t(sizeY_)*sizeof(*recv_);
    const size_t countX = count*sizeY_;
    const size_t bytesX = countX*sizeof(long);
    const MPI_Aint dispX = rankX_*countX;
    CHECK(hipStreamSynchronize(stream_));

    MPI_Win_fence(0,winX_);

    for (int i = 0; i < sizeX_; i++) {
      long *const originAddr = recv_+ranksX_[i]*countX;
      if (ranksX_[i] == rank_) {
        long *const src = send_+rankX_*countX;
        CHECK(hipMemcpyDtoDAsync(originAddr,src,bytesX,stream_));
      } else {
        MPI_Get(originAddr,countX,MPI_LONG,ranksX_[i],dispX,countX,MPI_LONG,winX_);
      }
    }
    MPI_Win_fence(0,winX_);

    transpose<<<gridX,block,0,stream_>>>(count,sizeY_,sizeX_,recv_,send_);
    CHECK(hipMemcpyDtoDAsync(recv_,send_,bytes,stream_));
    CHECK(hipStreamSynchronize(stream_));
  }
};

