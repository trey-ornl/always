#pragma once
#include <cstdio>
#include <mpi.h>
#include <vector>

struct Aller {

  MPI_Comm comm_;
  std::vector<int> dests_;
  long maxCount_;
  int rank_;
  long *recv_;
  std::vector<MPI_Request> rreqs_;
  long *send_;
  int size_;
  std::vector<int> sources_;
  static constexpr int tag_ = 0;

  Aller(MPI_Comm const comm, long *const send, long *const recv, const size_t bytes):
    comm_(comm),
    maxCount_(0),
    rank_(MPI_PROC_NULL),
    recv_(recv),
    send_(send),
    size_(0)
  {
    MPI_Comm_rank(comm_,&rank_);
    MPI_Comm_size(comm_,&size_);
    maxCount_ = bytes/(size_*sizeof(*recv_));
    rreqs_.resize(size_,MPI_REQUEST_NULL);
    dests_.resize(size_,MPI_PROC_NULL);
    sources_.resize(size_,MPI_PROC_NULL);
    for (int i = 0; i < size_; i++) {
      dests_[i] = (rank_+i+1)%size_;
      sources_[i] = (size_+rank_-i-1)%size_;
    }
  }

  ~Aller()
  {
    size_ = 0;
    recv_ = send_ = nullptr;
    rank_ = MPI_PROC_NULL;
    maxCount_ = 0;
    comm_ = MPI_COMM_NULL;
  }

  void run(const int count)
  {
    assert(count <= maxCount_);
    for (int i = 0; i < size_; i++) {
      const int source = sources_[i];
      long *const buf = recv_+(long(source)*long(count));
      MPI_Irecv(buf,count,MPI_LONG,source,tag_,comm_,&rreqs_[i]);
    }
    MPI_Barrier(comm_);
    for (int i = 0; i < size_; i++) {
      const int dest = dests_[i];
      const long *const buf = send_+(long(dest)*long(count));
      MPI_Rsend(buf,count,MPI_LONG,dest,tag_,comm_);
    }
    MPI_Waitall(size_,rreqs_.data(),MPI_STATUSES_IGNORE);
  }
};

