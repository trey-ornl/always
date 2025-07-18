#pragma once
#include <mpi.h>

struct Aller {

  MPI_Comm comm_;
  int count_;
  long maxCount_;
  long *recv_;
  MPI_Request req_;
  long *send_;

  Aller(MPI_Comm const comm, long *const send, long *const recv, const size_t bytes):
    comm_(comm),
    count_(0),
    recv_(recv),
    req_(MPI_REQUEST_NULL),
    send_(send)
  {
    int size = 0;
    MPI_Comm_size(comm_,&size);
    maxCount_ = bytes/(size*sizeof(*recv_));
  }

  ~Aller()
  {
    if (req_ != MPI_REQUEST_NULL) MPI_Request_free(&req_);
    count_ = 0;
    maxCount_ = 0;
    recv_ = send_ = nullptr;
    comm_ = MPI_COMM_NULL;
  }

  const char *info() const
  {
    return __FILE__ ": Persistent MPI_Alltoall_init";
  }

  void init(const int count)
  {
    assert(count <= maxCount_);
    count_ = count;
    if (req_ != MPI_REQUEST_NULL) MPI_Request_free(&req_);
    MPI_Alltoall_init(send_,count_,MPI_LONG,recv_,count_,MPI_LONG,comm_,MPI_INFO_NULL,&req_);
  }

  void run(const int count)
  {
    assert(count == count_);
    MPI_Start(&req_);
    MPI_Wait(&req_,MPI_STATUS_IGNORE);
  }
};

