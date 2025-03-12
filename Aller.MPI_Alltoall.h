#pragma once
#include <mpi.h>

struct Aller {

  MPI_Comm comm_;
  long maxCount_;
  long *recv_;
  long *send_;

  Aller(const int color, const int key, long *const send, long *const recv, const size_t bytes):
    recv_(recv),
    send_(send)
  {
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&comm_);
    int size = 0;
    MPI_Comm_size(comm_,&size);
    maxCount_ = bytes/(size*sizeof(*recv_));
  }

  ~Aller()
  {
    MPI_Comm_free(&comm_);
    maxCount_ = 0;
    recv_ = send_ = nullptr;
  }

  int rank() const
  {
    int r = MPI_PROC_NULL;
    MPI_Comm_rank(comm_,&r);
    return r;
  }

  void run(const int count)
  {
    assert(count <= maxCount_);
    MPI_Alltoall(send_,count,MPI_LONG,recv_,count,MPI_LONG,comm_);
  }

  int size() const
  {
    int s = 0;
    MPI_Comm_size(comm_,&s);
    return s;
  }
};

