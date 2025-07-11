#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>

#include "check.h"

#ifdef USE_2D
#include "Aller.2D.h"
#endif

#ifdef USE_3D
#include "Aller.3D.h"
#endif

#ifdef USE_ALLTOALL
#include "Aller.MPI_Alltoall.h"
#endif

#ifdef USE_GET
#include "Aller.MPI_Get.h"
#endif

#ifdef USE_HSA
#include "Aller.hsa.h"
#endif

#ifdef USE_ISEND
#include "Aller.MPI_Isend.h"
#endif

#ifdef USE_PUT
#include "Aller.MPI_Put.h"
#endif

#ifdef USE_RSEND
#include "Aller.MPI_Rsend.h"
#endif

__global__ void init(const int rank, const int size, const int count, long *const recvD, long *const sendD)
{
  const long countAll = long(size)*long(count);
  const long origin = long(rank)*countAll;
  const long stride = blockDim.x*gridDim.x;
  const long ilo = threadIdx.x+blockIdx.x*blockDim.x;
  for (long i = ilo; i < countAll; i += stride) {
    recvD[i] = -1;
    sendD[i] = origin+i;
  }
}

__global__ void verify(const int rank, const int size, const long count, long *const recvD, unsigned long *const errorsD)
{
  const long countAll = long(size)*long(count);
  const long thatRank = blockIdx.x;
  const long origin = thatRank*countAll+long(rank)*long(count);
  const long offset = thatRank*long(count);
  const long stride = blockDim.x*gridDim.y;
  const long ilo = threadIdx.x+blockIdx.y*blockDim.x;
  for (long i = ilo; i < count; i += stride) {
    const long expected = origin+i;
    if (recvD[offset+i] != expected) atomicAdd(errorsD,1);
#if 0
    if (recvD[offset+i] != expected) {
      printf("ERROR: %d recvD[%ld] %ld != %ld\n",rank,offset+i,recvD[offset+i],expected);
    } else {
      printf("GOOOD: %d recvD[%ld] %ld == %ld\n",rank,offset+i,recvD[offset+i],expected);
    }
#endif
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);
  int worldSize = 0;
  MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
  const double perSize = 1.0/double(worldSize);
  int rank = MPI_PROC_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int nd = 0;
  CHECK(hipGetDeviceCount(&nd));
  assert(nd);
  int ndMax = 0;
  MPI_Allreduce(&nd,&ndMax,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if (ndMax > 1) {
    // Assign GPUs to MPI tasks on a node in a round-robin fashion
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
    int lrank = MPI_PROC_NULL;
    MPI_Comm_rank(local,&lrank);
    const int target = lrank%nd;
    CHECK(hipSetDevice(target));
    int myd = -1;
    CHECK(hipGetDevice(&myd));
    for (int i = 0; i < worldSize; i++) {
      if (rank == i) {
        printf("# MPI task %d with node rank %d using device %d (%d devices per node)\n",rank,lrank,myd,nd);
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  int strided = false;
  int iters = 3;
  long countAll = 1024L*1024L*1024L;
  int countLo = 1;
  if (rank == 0) {
    if (argc > 1) strided = ('s' == tolower(*argv[1]));
    long i = 0;
    if (argc > 2) sscanf(argv[2],"%ld",&i);
    if (i > 0) iters = i;
    i = 0;
    if (argc > 3) sscanf(argv[3],"%ld",&i);
    if (i > 0) countAll = i;
    countAll = (countAll/worldSize)*worldSize;
    i = 0;
    if (argc > 4) sscanf(argv[4],"%ld",&i);
    if (i > 0) countLo = i;
  }
  MPI_Bcast(&strided,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&iters,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&countAll,1,MPI_LONG,0,MPI_COMM_WORLD);

  const size_t bytes = countAll*sizeof(long);
  const long origin = long(rank)*countAll;

  long *recvD = nullptr;
  CHECK(hipMalloc(&recvD,bytes));
  assert(recvD);
  CHECK(hipMemset(recvD,0,bytes));

  long *sendD = nullptr;
  CHECK(hipMalloc(&sendD,bytes));
  assert(sendD);

  long *originsD = nullptr;
  CHECK(hipMalloc(&originsD,long(worldSize)*sizeof(*originsD)));
  assert(originsD);

  unsigned long *errorsD = nullptr;
  CHECK(hipMalloc(&errorsD,sizeof(*errorsD)));
  assert(errorsD);

  constexpr int block = 256;

  for (int targetSize = worldSize; targetSize > 0; targetSize /= 2) {

    const int stride = (worldSize-1)/targetSize+1;
    const int team = strided ? rank%stride : rank/targetSize;

    MPI_Comm subComm;
    MPI_Comm_split(MPI_COMM_WORLD,team,rank,&subComm);
    int subRank = MPI_PROC_NULL;
    MPI_Comm_rank(subComm,&subRank);
    int subSize = 0;
    MPI_Comm_size(subComm,&subSize);
    assert(subSize <= targetSize);
    const long countMax = std::min<long>(INT_MAX,countAll/long(targetSize));
    MPI_Allgather(&origin,1,MPI_LONG,originsD,1,MPI_LONG,subComm);

    {
      Aller aller(subComm,sendD,recvD,bytes);

      if (rank == 0) {
        printf("\n\n# %s\n",aller.info());
        printf("# %s Alltoall performance, %d tasks, %d iterations\n", strided ? "Strided" : "Contiguous",worldSize,iters);
        printf("# comms | comm size | count | GiB (in+out) | seconds (min, avg, max) | GiB/s (min, avg, max)\n");
        fflush(stdout);
      }

      for (int count = countMax; count >= countLo; count /= 2) {

        const size_t activeBytes = long(count)*long(subSize)*sizeof(long);
        assert(activeBytes <= bytes);
        const double gib = 2.0*double(activeBytes)/double(1024*1024*1024);
        const uint32_t gridX = subSize;
        const uint32_t gridY = std::min(1024,(count-1)/block+1);

        double timeMax = 0;
        double timeMin = 60.0*60.0*24.0*365.0;
        double timeSum = 0;

        for (int i = 0; i <= iters; i++) {

          init<<<gridX*gridY,block>>>(subRank,subSize,count,recvD,sendD);
          CHECK(hipDeviceSynchronize());
          MPI_Barrier(MPI_COMM_WORLD);
          const double before = MPI_Wtime();
          aller.run(count);
          const double after = MPI_Wtime();
          MPI_Barrier(MPI_COMM_WORLD);
          CHECK(hipMemset(errorsD,0,sizeof(*errorsD)));
          verify<<<dim3{gridX,gridY},block>>>(subRank,subSize,count,recvD,errorsD);
          CHECK(hipDeviceSynchronize());
          const unsigned long errors = *errorsD;
          MPI_Barrier(MPI_COMM_WORLD);
          if (errors > 0) {
            fprintf(stderr,"ERROR: rank %d subrank %d errors %lu of %ld\n",rank,subRank,errors,long(count)*long(subSize));
            MPI_Abort(subComm,errors);
          }
          const double myTime = after-before;
          double thisTimeMax, thisTimeMin, thisTimeSum;
          MPI_Reduce(&myTime,&thisTimeMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
          MPI_Reduce(&myTime,&thisTimeMin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
          MPI_Reduce(&myTime,&thisTimeSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          if (rank == 0) {
            if (i == 0) {
              printf("### 0 time min %g avg %g max %g (warmup, ignored)\n",thisTimeMin,thisTimeSum*perSize,thisTimeMax);
            } else {
              timeMax = std::max(timeMax,thisTimeMax);
              timeMin = std::min(timeMin,thisTimeMin);
              timeSum += thisTimeSum;
              printf("### %d time min %g avg %g max %g\n",i,thisTimeMin,thisTimeSum*perSize,thisTimeMax);
            }
            fflush(stdout);
          }
        }
        if (rank == 0) {
          const double timeAvg = timeSum*perSize/double(iters);
          printf("%d %d %d %g %g %g %g %g %g %g\n",stride,targetSize,count,gib,timeMin,timeAvg,timeMax,gib/timeMax,gib/timeAvg,gib/timeMin);
          fflush(stdout);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Comm_free(&subComm);
    }
  }
  if (rank == 0) { printf("# Done\n"); fflush(stdout); }

  CHECK(hipDeviceSynchronize());
  CHECK(hipFree(errorsD));
  errorsD = nullptr;
  CHECK(hipFree(originsD));
  originsD = nullptr;
  CHECK(hipFree(sendD));
  sendD = nullptr;
  CHECK(hipFree(recvD));
  recvD = nullptr;

  MPI_Finalize();
  return 0;
}

