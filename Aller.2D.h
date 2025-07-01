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

    const int sizeRoot = round(sqrt(double(size)));
    int nx = 1;
    for (int i = 1; i <= sizeRoot; i++) {
      if (size%i == 0) nx = i;
    }
    const int ny = size/nx;
    assert(nx*ny == size);

    MPI_Comm_split(comm,rank/nx,rank,&commX_);
    MPI_Comm_rank(commX_,&rankX_);
    MPI_Comm_size(commX_,&sizeX_);
    assert(sizeX_ == nx);

    MPI_Comm_split(comm,rank%nx,rank,&commY_);
    MPI_Comm_rank(commY_,&rankY_);
    MPI_Comm_size(commY_,&sizeY_);
    assert(sizeY_ == ny);

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
#if 0
    int rank = MPI_PROC_NULL;
    MPI_Comm_rank(comm_,&rank);
    int size = 0;
    MPI_Comm_size(comm_,&size);
    for (int i = 0; i < size; i++) {
      MPI_Barrier(comm_);
      if (rank == i) {
        printf("%d before |",rank);
        for (int j = 0; j < size; j++) {
          for (int k = 0; k < count; k++) printf(" %ld",send_[j*count+k]);
          printf(" |");
        }
        printf("\n");
        fflush(stdout);
      }
    }
    MPI_Alltoall(send_,count,MPI_LONG,recv_,count,MPI_LONG,comm_);
    for (int i = 0; i < size; i++) {
      MPI_Barrier(comm_);
      if (rank == i) {
        printf("%d expect |",rank);
        for (int j = 0; j < size; j++) {
          for (int k = 0; k < count; k++) printf(" %ld",recv_[j*count+k]);
          printf(" |");
        }
        printf("\n");
        fflush(stdout);
      }
      MPI_Barrier(comm_);
    }
#endif

    if ((sizeX_ == 1) || (sizeY_ == 1)) {
      MPI_Alltoall(send_,count,MPI_LONG,recv_,count,MPI_LONG,comm_);
      return;
    }

    assert(count <= maxCount_);
    assert(long(count)*long(sizeX_) <= long(INT_MAX));
    const int countY = count*sizeX_;
    MPI_Alltoall(send_,countY,MPI_LONG,recv_,countY,MPI_LONG,commY_);

#if 0
    for (int i = 0; i < size; i++) {
      MPI_Barrier(comm_);
      if (rank == i) {
        printf("%d pretra |",rank);
        for (int j = 0; j < size; j++) {
          for (int k = 0; k < count; k++) printf(" %ld",recv_[j*count+k]);
          printf(" |");
        }
        printf("\n");
        fflush(stdout);
      }
      MPI_Barrier(comm_);
    }
#endif

    constexpr int block = 256;
    const dim3 gridY((count-1)/block+1,sizeX_,sizeY_);
    transpose<<<gridY,block,0,stream_>>>(count,sizeX_,sizeY_,recv_,send_);
    assert(long(count)*long(sizeY_) <= long(INT_MAX));
    const int countX = count*sizeY_;
    const dim3 gridX((count-1)/block+1,sizeY_,sizeX_);
    const long bytes = long(count)*long(sizeX_)*long(sizeY_)*sizeof(*recv_);
    CHECK(hipStreamSynchronize(stream_));

#if 0
    for (int i = 0; i < size; i++) {
      MPI_Barrier(comm_);
      if (rank == i) {
        printf("%d postra |",rank);
        for (int j = 0; j < size; j++) {
          for (int k = 0; k < count; k++) printf(" %ld",send_[j*count+k]);
          printf(" |");
        }
        printf("\n");
        fflush(stdout);
      }
      MPI_Barrier(comm_);
    }
#endif

    MPI_Alltoall(send_,countX,MPI_LONG,recv_,countX,MPI_LONG,commX_);
    transpose<<<gridX,block,0,stream_>>>(count,sizeY_,sizeX_,recv_,send_);
    CHECK(hipMemcpyDtoDAsync(recv_,send_,bytes,stream_));
    CHECK(hipStreamSynchronize(stream_));

#if 0
    for (int i = 0; i < size; i++) {
      MPI_Barrier(comm_);
      if (rank == i) {
        printf("%d after  |",rank);
        for (int j = 0; j < size; j++) {
          for (int k = 0; k < count; k++) printf(" %ld",recv_[j*count+k]);
          printf(" |");
        }
        printf("\n");
        fflush(stdout);
      }
      MPI_Barrier(comm_);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    exit(0);
#endif

  }
};

