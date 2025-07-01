#pragma once
#include <cstdio>
#include <mpi.h>
#include <vector>

struct Aller {

  MPI_Comm comm_;
  long maxCount_;
  int rank_;
  long *recv_;
  int size_;
  std::vector<int> targets_;
  MPI_Win win_;

  Aller(MPI_Comm const comm, long *const send, long *const recv, const size_t bytes):
    comm_(comm),
    maxCount_(0),
    rank_(MPI_PROC_NULL),
    recv_(recv),
    size_(0)
  {
    MPI_Comm_rank(comm_,&rank_);
    MPI_Comm_size(comm_,&size_);

    MPI_Win_create(send,bytes,sizeof(*recv),MPI_INFO_NULL,comm_,&win_);
    MPI_Win_fence(0,win_);

    maxCount_ = bytes/(size_*sizeof(*recv));
    targets_.resize(size_,MPI_PROC_NULL);
    for (int i = 0; i < size_; i++) {
      targets_[i] = (rank_+i+1)%size_;
    }
  }

  ~Aller()
  {
    MPI_Win_fence(0,win_);
    MPI_Win_free(&win_);
    size_ = 0;
    recv_ = nullptr;
    rank_ = MPI_PROC_NULL;
    maxCount_ = 0;
    comm_ = MPI_COMM_NULL;
  }

  const char *info() const
  {
    return __FILE__ ": MPI_Get loop starting with rank+1";
  }

  void run(const int count)
  {
    assert(count <= maxCount_);
    const MPI_Aint disp = long(count)*long(rank_);
    MPI_Win_fence(0,win_);
    for (int i = 0; i < size_; i++) {
      const int target = targets_[i];
      long *const addr = recv_+(long(target)*long(count));
      MPI_Get(addr,count,MPI_LONG,target,disp,count,MPI_LONG,win_);
    }
    MPI_Win_fence(0,win_);
  }
};

