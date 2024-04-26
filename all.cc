#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <mpi.h>

static void checkHip(const hipError_t err, const char *const file, const int line)
{
  if (err == hipSuccess) return;
  fprintf(stderr,"HIP ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,hipGetErrorName(err),hipGetErrorString(err));
  fflush(stderr);
  exit(err);
}

#define CHECK(X) checkHip(X,__FILE__,__LINE__)

__global__ void init(const int rank, const long countAll, long *const sendD)
{
  const long offset = countAll*long(rank);
  const long stride = blockDim.x*gridDim.x;
  const long ilo = threadIdx.x+blockIdx.x*blockDim.x;
  for (long i = ilo; i < countAll; i += stride) sendD[i] = i+offset;
}

__global__ void verify(const bool strided, const long myOffset, const long countAll, const long commSize, const long count, const int team, const int taskStride, long *const recvD)
{
  const long iStride = blockDim.x*gridDim.x;
  const long ilo = threadIdx.x+blockIdx.x*blockDim.x;
  const long ihi = count*commSize;
  for (long i = ilo; i < ihi; i += iStride) {
    const long task = i/count;
    const long j = i%count;
    const long worldTask = strided ? long(team+task*taskStride) : long(task+team*taskStride);
    const long offset = myOffset+worldTask*countAll;
    const long expected = j+offset;
    const long k = j+task*long(count);
    if (expected != recvD[k]) {
      printf("ERROR: task %ld element %ld expected %ld received %ld\n",task,k,expected,recvD[k]);
      abort();
    }
    recvD[k] = 0;
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);
  int worldSize = 0;
  MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
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
  int countLo = 1;
  int countHi = 32*1024;
  int iters = 3;
  if (rank == 0) {
    if (argc > 1) strided = ('s' == tolower(*argv[1]));
    int i = 0;
    if (argc > 2) sscanf(argv[2],"%d",&i);
    if (i > 0) iters = i;
    i = 0;
    if (argc > 3) sscanf(argv[3],"%d",&i);
    if (i > 0) countLo = i;
    i = 0;
    if (argc > 4) sscanf(argv[4],"%d",&i);
    if (i > 0) countHi = i;
  }
  MPI_Bcast(&strided,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&iters,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&countLo,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&countHi,1,MPI_INT,0,MPI_COMM_WORLD);

  const long countAll = long(worldSize)*long(countHi);
  const size_t bytes = countAll*sizeof(long);

  long *recvD = nullptr;
  CHECK(hipMalloc(&recvD,bytes));
  assert(recvD);
  CHECK(hipMemset(recvD,0,bytes));

  long *sendD = nullptr;
  CHECK(hipMalloc(&sendD,bytes));
  assert(sendD);

  const int block = 256;
  const int gridMax = 1024*1024;
  const int grid = std::min(long(gridMax),(countAll-1)/long(block)+1);
  init<<<grid,block>>>(rank,countAll,sendD);
  CHECK(hipDeviceSynchronize());

  for (int targetSize = worldSize; targetSize > 0; targetSize /= 2) {

    if (rank == 0) {
      printf("\n\n# %s Alltoall performance, %d tasks, %d iterations\n", strided ? "Strided" : "Contiguous",worldSize,iters);
      printf("# comms | comm size | count | GiB (in+out) | seconds (min, avg, max) | GiB/s (min, avg, max)\n");
      fflush(stdout);
    }

    const int stride = (worldSize-1)/targetSize+1;
    const int team = strided ? rank%stride : rank/targetSize;

    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD,team,rank,&comm);
    int actualSize = 0;
    MPI_Comm_size(comm,&actualSize);
    int id = MPI_PROC_NULL;
    MPI_Comm_rank(comm,&id);

    assert(actualSize <= targetSize);
    const long countMax = std::min<long>(INT_MAX,countAll/long(targetSize));
    const long taskStride = strided ? stride : actualSize;

    for (int count = countMax; count >= countLo; count /= 2) {

      const long activeAll = long(count)*long(actualSize);
      const size_t activeBytes = activeAll*sizeof(long);
      assert(activeBytes <= bytes);
      const double gib = 2.0*double(activeBytes)/double(1024*1024*1024);
      const long myOffset = long(count)*long(id);
      const int grid = std::min(long(gridMax),(activeAll-1)/long(block)+1);

      double timeMin = 60.0*60.0*24.0*365.0;
      double timeSum = 0;
      double timeMax = 0;

      for (int i = 0; i <= iters; i++) {

        MPI_Barrier(MPI_COMM_WORLD);
        const double before = MPI_Wtime();
        MPI_Alltoall(sendD,count,MPI_LONG,recvD,count,MPI_LONG,comm);
        const double after = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        verify<<<grid,block>>>(strided,myOffset,countAll,actualSize,count,team,taskStride,recvD);
        const double myTime = after-before;
        double time = 0;
        MPI_Reduce(&myTime,&time,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
        if (rank == 0) {
          if (i == 0) {
            printf("### 0 time %g (warmup, ignored)\n",time);
          } else {
            timeMin = std::min(timeMin,time);
            timeSum += time;
            timeMax = std::max(timeMax,time);
            printf("### %d time %g\n",i,time);
          }
          fflush(stdout);
        }
        CHECK(hipDeviceSynchronize());
      }
      if (rank == 0) {
        const double timeAvg = timeSum/double(iters);
        printf("%d %d %d %g %g %g %g %g %g %g\n",stride,targetSize,count,gib,timeMin,timeAvg,timeMax,gib/timeMax,gib/timeAvg,gib/timeMin);
        fflush(stdout);
      }
    }
    MPI_Comm_free(&comm);
  }
  if (rank == 0) { printf("# Done\n"); fflush(stdout); }

  CHECK(hipFree(sendD));
  sendD = nullptr;
  CHECK(hipFree(recvD));
  recvD = nullptr;

  MPI_Finalize();
  return 0;
}

