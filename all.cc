#include <cassert>
#include <cctype>
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

#ifdef VERBOSE
static constexpr bool verbose = true;
#else
static constexpr bool verbose = false;
#endif

int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);
  int worldSize = 0;
  MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
  int rank = MPI_PROC_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int n = 640*640*640;
  int strided = false;
  if (rank == 0) {
    int n0 = 0;
    if (argc > 1) sscanf(argv[1],"%d",&n0);
    if (n0 > 0) n = n0;
    if (argc > 2) strided = ('s' == tolower(*argv[2]));
  }
  MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&strided,1,MPI_INT,0,MPI_COMM_WORLD);
  const size_t bytes = size_t(n)*sizeof(double);
  if (rank == 0) {
    printf("# %s Alltoall performance\n", strided ? "Strided" : "Contiguous");
    fflush(stdout);
  }

  int nd = 0;
  CHECK(hipGetDeviceCount(&nd));
  assert(nd);
  if (nd > 1) {
    // Assign GPUs to MPI tasks on a node in a round-robin fashion
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
    int lrank = MPI_PROC_NULL;
    MPI_Comm_rank(local,&lrank);
    const int target = lrank%nd;
    CHECK(hipSetDevice(target));
    int myd = -1;
    CHECK(hipGetDevice(&myd));
    for (int i = 0; i < size; i++) {
      if (rank == i) {
        printf("# MPI task %d with node rank %d using Hip device %d (%d devices per node)\n",rank,lrank,myd,nd);
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  double *host = nullptr;
  CHECK(hipHostMalloc(&host,bytes,hipHostMallocDefault));
  assert(host);
  if (verbose && (rank == 0)) {
    printf("# Setting array\n");
    fflush(stdout);
  }
#pragma omp parallel for
  for (int i = 0; i < n; i++) host[i] = double(i+rank);

  double *recvD = nullptr;
  CHECK(hipMalloc(&recvD,bytes));
  assert(recvD);
  CHECK(hipMemset(recvD,0,bytes));
  double *sendD = nullptr;
  CHECK(hipMalloc(&sendD,bytes));
  assert(sendD);
  CHECK(hipMemcpy(sendD,host,bytes,hipMemcpyHostToDevice));

  if (rank == 0) {
    printf("# tasks steps count MiB/task/step GiB/task seconds GiB/s/task\n");
    fflush(stdout);
  }
  for (int targetSize = worldSize; targetSize > 0; targetSize /= 2) {

    if (verbose && (rank == 0)) {
      printf("# Running with %d %s tasks\n",targetSize,(strided ? "strided" : "contiguous"));
      fflush(stdout);
    }
    const int stride = worldSize/targetSize;
    const int team = strided ? rank%stride : rank/targetSize;

    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD,team,rank,&comm);
    int actualSize = 0;
    MPI_Comm_size(comm,&actualSize);
    int id = MPI_PROC_NULL;
    MPI_Comm_rank(comm,&id);

    const int countMax = n/targetSize;
    const int twoMax = 2*countMax;

    for (int twoStep = 1; twoStep < twoMax; twoStep += twoStep) {

      // Run single step twice, first is warmup
      const int steps = (twoStep > 1) ? twoStep/2 : 1;

      const int count = countMax/steps;
      if (verbose && (rank == 0)) {
        printf("# Splitting into %d steps, %d count, %d tasks\n",steps,count,targetSize);
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      const double before = MPI_Wtime();
      for (int step = 0; step < steps; step++) {
        const int offset = step*targetSize*count;
        MPI_Alltoall(sendD+offset,count,MPI_DOUBLE,recvD+offset,count,MPI_DOUBLE,comm);
      }
      const double after = MPI_Wtime();

      if (verbose && (rank == 0)) {
        printf("# Checking results\n");
        fflush(stdout);
      }
      CHECK(hipMemcpy(host,recvD,bytes,hipMemcpyDeviceToHost));
      int fails = 0;
#pragma omp parallel for collapse(3)
      for (int step = 0; step < steps; step++) {
        for (int task = 0; task < actualSize; task++) {
          for (int j = 0; j < count; j++) {
            const int i = j+count*(id+targetSize*step);
            const double expected = strided ? double(i+team+task*stride) : double(i+task+team*targetSize);
            const int k = j+count*(task+targetSize*step);
            if (expected != host[k]) {
              fprintf(stderr,"ERROR %d: %d %d %d %g %g\n",rank,step,task,j,expected,host[k]);
              fflush(stderr);
              fails++;
            }
          }
        }
      }
      MPI_Barrier(comm);
      if (fails) MPI_Abort(MPI_COMM_WORLD,fails);

      const double time = after-before;
      double timeMax = 0;
      MPI_Allreduce(&time,&timeMax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

      if (rank == 0) {
        const size_t sendRecvBytes = size_t(2)*targetSize*count*sizeof(double);
        const double gib = double(sendRecvBytes)/double(1024*1024*1024);
        const double mib = gib*double(1024);
        const double tgib = gib*steps;
        const double gibps = tgib/timeMax;
        if (twoStep == 1) printf("\n\n# ");
        printf("%d %d %d %g %g %g %g",targetSize,steps,count,mib,tgib,timeMax,gibps);
        if (twoStep == 1) printf(" # warm up");
        printf("\n");
        fflush(stdout);
      }
      static constexpr double timeLimit = 2;
      if ((twoStep > 2) && (timeMax > timeLimit)) break;
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

