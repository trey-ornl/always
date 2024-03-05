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

  int myCountHi = 0;
  {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CHECK(hipMemGetInfo(&freeBytes,&totalBytes));
    const size_t targetBytes = freeBytes/8; // half memory, half that for MPI, bi-dir
    const size_t targetCount = targetBytes/(worldSize*sizeof(long));
    myCountHi = std::min<long>(INT_MAX,targetCount);
  }

  int strided = false;
  int countLo = 1;
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
    if (i > 0) myCountHi = std::min(myCountHi,i);
  }
  MPI_Bcast(&strided,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&countLo,1,MPI_INT,0,MPI_COMM_WORLD);

  int countHi = 0;
  MPI_Allreduce(&myCountHi,&countHi,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);

  const long countAll = long(worldSize)*long(countHi);
  const size_t bytes = countAll*sizeof(long);
  if (rank == 0) {
    printf("# %s Alltoall performance\n", strided ? "Strided" : "Contiguous");
    printf("# %d tasks, counts %d..%d, max aggregate message size %g GB\n",worldSize,countLo,countHi,double(bytes)/double(1024*1024*1024));
    fflush(stdout);
  }

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

  if (rank == 0) {
    printf("# comm size | count (longs) | seconds (min, avg, max) | bi-dir GiB/s (min, avg, max)\n");
    fflush(stdout);
  }
  for (int targetSize = worldSize; targetSize > 0; targetSize /= 2) {

    const int stride = worldSize/targetSize;
    const int team = strided ? rank%stride : rank/targetSize;

    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD,team,rank,&comm);
    int actualSize = 0;
    MPI_Comm_size(comm,&actualSize);
    int id = MPI_PROC_NULL;
    MPI_Comm_rank(comm,&id);

    for (int count = countHi; count >= countLo; count /= 2) {

      const size_t activeBytes = size_t(count)*size_t(actualSize)*sizeof(long);
      const double gib = 2.0*double(activeBytes)/double(1024*1024*1024);
      const long myOffset = long(count)*long(id);

      for (int i = 0; i <= iters; i++) {

        MPI_Barrier(MPI_COMM_WORLD);
        const double before = MPI_Wtime();
        MPI_Alltoall(sendD,count,MPI_LONG,recvD,count,MPI_LONG,comm);
        const double after = MPI_Wtime();

        CHECK(hipMemcpy(host,recvD,activeBytes,hipMemcpyDeviceToHost));
        int fails = 0;
#pragma omp parallel for collapse(2) reduction(+:fails)
        for (int task = 0; task < actualSize; task++) {
          for (int j = 0; j < count; j++) {
            const long worldTask = strided ? team+task*stride : task+team*targetSize;
            const long offset = myOffset+worldTask*countAll;
            const long expected = j+offset;
            const long k = j+task*count;
            if (expected != host[k]) {
              fprintf(stderr,"ERROR %d: %d %ld %ld %ld\n",rank,task,k,expected,host[k]);
              fflush(stderr);
              fails++;
            }
          }
        }
        MPI_Barrier(comm);
        if (fails) MPI_Abort(MPI_COMM_WORLD,fails);

        const double time = after-before;
        double timeMin = 0;
        double timeMax = 0;
        double timeSum = 0;
        MPI_Reduce(&time,&timeMin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
        MPI_Reduce(&time,&timeSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&time,&timeMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

        if (rank == 0) {
          const double timeAvg = timeSum/double(worldSize);
          if (i == 0) printf("\n\n# ");
          printf("%d %d %g %g %g %g %g %g",targetSize,count,timeMin,timeAvg,timeMax,gib/timeMax,gib/timeAvg,gib/timeMin);
          if (i == 0) printf(" # warm up");
          printf("\n");
          fflush(stdout);
        }
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

