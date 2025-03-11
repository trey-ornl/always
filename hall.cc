#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <mpi.h>

void init(const int rank, const long countAll, long *const sendD)
{
  const long offset = countAll*long(rank);
#pragma omp parallel for simd
  for (long i = 0; i < countAll; i++) sendD[i] = i+offset;
}

void verify(const bool strided, const long myOffset, const long countAll, const long commSize, const long count, const int team, const int taskStride, long *const recvD)
{
  const long ihi = count*commSize;
#pragma omp parallel for
  for (long i = 0; i < ihi; i++) {
    const long task = i/count;
    const long j = i%count;
    const long worldTask = strided ? long(team+task*taskStride) : long(task+team*taskStride);
    const long offset = myOffset+worldTask*countAll;
    const long expected = j+offset;
    const long k = j+task*long(count);
    if (expected != recvD[k]) {
      printf("ERROR: task %ld element %ld expected %ld received %ld\n",task,k,expected,recvD[k]);
      assert(false);
    }
    recvD[k] = 0;
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

  int strided = false;
  int iters = 3;
  long countAll = 2L*1024L*1024L*1024L;
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

  long *const recvD = (long*)malloc(bytes);
  assert(recvD);
  memset(recvD,0,bytes);

  long *const sendD = (long*)malloc(bytes);
  assert(sendD);

  init(rank,countAll,sendD);

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

      double timeMax = 0;
      double timeMin = 60.0*60.0*24.0*365.0;
      double timeSum = 0;

      for (int i = 0; i <= iters; i++) {

        MPI_Barrier(MPI_COMM_WORLD);
        const double before = MPI_Wtime();
        MPI_Alltoall(sendD,count,MPI_LONG,recvD,count,MPI_LONG,comm);
        const double after = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        verify(strided,myOffset,countAll,actualSize,count,team,taskStride,recvD);
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
    MPI_Comm_free(&comm);
  }
  if (rank == 0) { printf("# Done\n"); fflush(stdout); }

  free(sendD);
  free(recvD);

  MPI_Finalize();
  return 0;
}

