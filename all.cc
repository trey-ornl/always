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
  int countHi = 40*1024;
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

  long *host = nullptr;
  CHECK(hipHostMalloc(&host,bytes,hipHostMallocDefault));
  assert(host);
  
  const long offset = countAll*long(rank);
#pragma omp parallel for
  for (long i = 0; i < countAll; i++) host[i] = i+offset;

  long *recvD = nullptr;
  CHECK(hipMalloc(&recvD,bytes));
  assert(recvD);
  CHECK(hipMemset(recvD,0,bytes));

  long *sendD = nullptr;
  CHECK(hipMalloc(&sendD,bytes));
  assert(sendD);
  CHECK(hipMemcpy(sendD,host,bytes,hipMemcpyHostToDevice));

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

    for (int count = countMax; count >= countLo; count /= 2) {

      const size_t activeBytes = size_t(count)*size_t(actualSize)*sizeof(long);
      assert(activeBytes <= bytes);
      const double gib = 2.0*double(activeBytes)/double(1024*1024*1024);
      const long myOffset = long(count)*long(id);

      double timeMin = 60.0*60.0*24.0*365.0;
      double timeSum = 0;
      double timeMax = 0;

      for (int i = 0; i <= iters; i++) {

        MPI_Barrier(MPI_COMM_WORLD);
        const double before = MPI_Wtime();
        MPI_Alltoall(sendD,count,MPI_LONG,recvD,count,MPI_LONG,comm);
        const double after = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK(hipMemcpy(host,recvD,activeBytes,hipMemcpyDeviceToHost));
        bool failed = false;
#pragma omp parallel for collapse(2) shared(failed)
        for (long task = 0; task < actualSize; task++) {
          for (long j = 0; j < count; j++) {
            if (failed) continue;
            const long worldTask = strided ? long(team+task*stride) : long(task+team*targetSize);
            const long offset = myOffset+worldTask*countAll;
            const long expected = j+offset;
            const long k = j+task*long(count);
            if (expected != host[k]) {
              failed = true;
              fprintf(stderr,"ERROR: rank %d task %ld element %ld expected %ld received %ld\n",rank,task,k,expected,host[k]);
              fflush(stderr);
            }
          }
        }
        if (failed) {
          fprintf(stderr,"FAILURE: rank %d aborting\n",rank);
          fflush(stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (failed) MPI_Abort(MPI_COMM_WORLD,rank+1);
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

  CHECK(hipHostFree(host));
  host = nullptr;
  MPI_Finalize();
  return 0;
}

