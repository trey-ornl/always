#pragma once
#include <mpi.h>

struct Aller {

  MPI_Comm comm_;
  long maxCount_;
  long *recv_;
  long *send_;

  Aller(MPI_Comm const comm, long *const send, long *const recv, const size_t bytes):
    comm_(comm),
    recv_(recv),
    send_(send)
  {
    int size = 0;
    MPI_Comm_size(comm_,&size);
    maxCount_ = bytes/(size*sizeof(*recv_));
  }

  ~Aller()
  {
    maxCount_ = 0;
    recv_ = send_ = nullptr;
    comm_ = MPI_COMM_NULL;
  }

  void run(const int count)
  {
    assert(count <= maxCount_);
    MPI_Alltoall(send_,count,MPI_LONG,recv_,count,MPI_LONG,comm_);
  }
};

