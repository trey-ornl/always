# Always
These bencharks measure the performance of all-to-all and concurrent many-to-many communications using a triply nested loop of calls.

- outer: loop over sizes of concurrent subcommunicators, `size = <size of MPI_COMM_WORLD>..1`, halved each time
  - middle: loop over message size, `count = <max allowed by buffers and communicator size>..1`, halved each time
    - inner: loop over timed iterations, plus one warmup iteration

The outer loop iterates over increasing numbers of smaller subcommunicators, starting with `MPI_COMM_WORLD` and ending at one communicator per task. The number of communicators doubles with each iteration. The benchmark runs all-to-alls across all subcommunicators simultaneously, so every task in `MPI_COMM_WORLD` is busy.

The middle loop iterates over decreasing message sizes, starting with the maximum size allowed by the buffers, down to one `long` value per rank in the communicator. The default total message size is 1 GiB, or 128x1024x1024 `long` values. An optional third argument specifies a different size, in number of longs. A non-positive value for this argument results in the default.

The inner loop is over timed all-to-all calls, where the time of the first iteration is ignored as a warmup. The default number of timed iterations is 3, and an optional second argument specifies a different number. A non-positive value for this argument results in the default.

The standard output is text that can be used with Gnuplot. The output includes a block of lines for each outer-loop iteration, with a data line for each middle-loop iteration. Each middle-loop line lists the number of concurrent communicators, the number of tasks in each communicator, the `count` argument for the all-to-all operation, the total message size (input + output) in GiB, the min, average, and max runtimes across all tasks and inner-loop iterations, and the min, average, and max effective bandwidths across all tasks and inner-loop iterations.

The default assignment of tasks to subcommunicators uses contiguous blocks. For example, the second iteration of the outer loop splits `MPI_COMM_WORLD` into a communicator with the first half of the tasks, as numbered by rank, and a communicator with the second half. An optional second argument beginning with 's' requests a strided assignment, where tasks within each communicator are spread out. For example, the second iteration of the outer loop splits `MPI_COMM_WORLD` into a communictor with the odd-rank tasks and a communicator with the even-rank tasks. The penultimate iteration uses many two-task communicators, where the task pairs have `MPI_COMM_WORLD` ranks that differ by half the size of `MPI_COMM_WORLD`. A first argument that begins with anything other than 's' or 'S' results in the default contiguous partioning of tasks.

## Examples
Run with the default per-task buffer size and the default subcommunicator partitioning of contiguous tasks:
```
srun ... --gpus-per-task=1 --gpu-bind=closest ./aller
```
Run with a non-default per-task buffer size of 128 longs:
```
srun ... --gpus-per-task=1 --gpu-bind=closest ./aller 0 0 128
```
Run with strided partitioning of tasks into subcommunicators:
```
srun ... --gpus-per-task=1 --gpu-bind=closest ./aller strided
```

