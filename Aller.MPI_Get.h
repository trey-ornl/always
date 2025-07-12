#pragma once
#include <cstdio>
#include <mpi.h>
#include <random>
#include <sstream>
#include <vector>

struct Aller {

  MPI_Comm comm_;
  std::string info_;
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
    std::stringstream info;
    info << __FILE__ << ": loop over MPI_Get";

    targets_.reserve(size_);

    if (getenv("ALLER_USE_FARTHEST")) {

      for (int i = 0; i < size_; i++) {
        const int step = (i%2) ? (i+1)/2 : -i/2;
        const int target = (rank_+step+size_)%size_;
        targets_.push_back(target);
      }
      std::reverse(targets_.begin(),targets_.end());
      info << ", " << size_ << " targets farthest to closest (ALLER_USE_FARTHEST)";

    } else if (getenv("ALLER_USE_SHUFFLE")) {

      for (int i = 0; i < size_; i++) {
        targets_.push_back(i);
      }
      std::shuffle(targets_.begin(),targets_.end(),std::default_random_engine(rank_+1));
      info << ", randomly shuffle " << size_ << " targets (ALLER_USE_SHUFFLE)";

    } else {

      int useRotate = 3;
      const char *const useRotateStr = getenv("ALLER_USE_ROTATE");
      if (useRotateStr) {
        int value = 0;
        if ((sscanf(useRotateStr,"%d",&value) == 1) && (value > 0)) useRotate = value;
      }

      const int rotation = size_/useRotate+1;
      for (int i = 0; i < size_; i++) {
        const int target = (rank_+i+rotation)%size_;
        targets_.push_back(target);
      }
      info << ", rotate " << size_ << " targets by " << rotation << " (ALLER_USE_ROTATE=" << useRotate << ")";

    }

    info_ = info.str();
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
    return info_.c_str();
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

