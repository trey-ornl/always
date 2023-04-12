# Always
These bencharks measure the performance of `MPI_Alltoall` and `MPI_Ialltoall` using a triply nested loop of calls.

```
outer: loop over communicator sizes, i = size of MPI_COMM_WORLD .. 1
  middle: loop over number of calls in inner loop, j = 1 .. number that exceeds time limit
     inner: loop over j all-to-all calls, k = 1..j, each with total/j of the buffer
```

The outer loop iterates over increasing numbers of smaller subcommunicators, starting with `MPI_COMM_WORLD` and ending at one communicator per task. The number of communicators doubles with each iteration. The benchmark runs all-to-alls across all subcommunicators simultaneously, so every task in `MPI_COMM_WORLD` is busy.

The middle loop iterates over increasing numbers of calls with smaller message sizes, starting with one call with a large message and ending when the time for all calls of a given message size exceeds a maximum time limit (currently one second). The default total message size is 640x640x640 doubles. An optional first argument specifies a different size, in number of `double`s. A non-positive value for this argument results in the default.

The inner loop is over multiple calls with equal-sized fractions of the buffer. The first iteration of the middle loop does a single all-to-all with the full buffer size. The second middle-loop iteration does an inner loop over two all-to-alls, each with half the full size. The number of inner-loop calls increases and the size of the messages decreases by powers of two with each middle-loop iteration, until the time for all the calls in the inner loop exceeds the time limit.

The standard output is text that can be used with Gnuplot. The output includes a block of lines for each outer-loop iteration, with a line for each middle-loop iteration. Results for a warm-up run for each outer-loop iteration appear in a comment line.

Each middle-loop line lists the number of tasks in each subcommunicator, the number of all-to-all calls in the inner loop, the `count` argument for each call, the buffer size used for each call (MiB), the total buffer size for all calls across that inner loop (GiB), the total time for the calls (seconds), and the per-task communication bandwidth.

The inner loop of `all` makes blocking calls to `MPI_Alltoall`, while the inner loop of `iall` makes nonblocking calls to `MPI_Ialltoall`, with a single `MPI_Waitall` after that inner loop.

The default assignment of tasks to subcommunicators uses contiguous blocks. For example, the second iteration of the outer loop splits `MPI_COMM_WORLD` into a communicator with the first half of the tasks, as numbered by rank, and a communicator with the second half. An optional second argument beginning with 's' requests a strided assignment, where tasks within each communicator are spread out. For example, the second iteration of the outer loop splits `MPI_COMM_WORLD` into a communictor with the odd-rank tasks and one with the even-rank tasks. The penultimate iteration uses many two-task communicators, where the task pairs have `MPI_COMM_WORLD` ranks that differ by half the size of `MPI_COMM_WORLD`. A second argument that begins with anything other than 's' or 'S' results in the default contiguous partioning of tasks.

The benchmarks use OpenMP on the host to initialize values and check results. Multiple OpenMP threads may shorten overall runtime, but they should not affect the performance measurements.

## Examples
Run with the default per-task buffer size and the default subcommunicator partitioning of contiguous tasks:
```
export OMP_NUM_THREADS=7
srun ... -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest ./[i]all
```
Run with a non-default per-task buffer size of 128 doubles:
```
export OMP_NUM_THREADS=7
srun ... -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest ./[i]all 128
```
Run with the default per-task buffer size but with strided partitioning of tasks into subcommunicators:
```
export OMP_NUM_THREADS=7
srun ... -c${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest ./[i]all 0 strided
```

