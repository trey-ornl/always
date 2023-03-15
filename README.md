# Always
These bencharks measure the performance of `MPI_Alltoall` and `MPI_Ialltoall` using a triply nested loop of calls.

The outer loop iterates over increasing numbers of smaller subcommunicators, starting with `MPI_COMM_WORLD` and ending at one communicator per task. The number of communicators doubles with each iteration. The benchmark runs all-to-alls across all subcommunicators simultaneously.

The middle loop iterates over increasing numbers of calls with smaller message sizes, starting with one call with a large message and ending when the time for all calls of a given message size exceeds a maximum time limit (currently one second). The default total message size is 640x640x640 doubles. An optional single argument specifies a different size, in number of `double`s. 

The inner loop is over multiple calls with the same buffer size. The first iteration of the middle loop does a single all-to-all with the full message size. The second middle-loop iteration does an inner loop over two all-to-alls, each with half the full size. The number of inner-loop calls increases and the size of the message buffers decreases by powers of two with each middle-loop iteration, until the time for all the calls in the inner loop exceeds the time limit.

The standard output is text that can be used with Gnuplot. The output includes a block of lines for each outer-loop iteration, with a line for each middle-loop iteration. Results for a warm-up run for each outer-loop iteration appear in a comment line.

Each middle-loop line lists the number of tasks in each subcommunicator, the number of all-to-all calls in the inner loop, the `count` argument for each call, the buffer size used for each call (MiB), the total buffer size for all calls across that inner loop (GiB), the total time for the calls (seconds), and the per-task communication bandwidth.

The inner loop of `all` makes blocking calls to `MPI_Alltoall`, while the inner loop of `iall` makes nonblocking calls to `MPI_Ialltoall`, with a single `MPI_Waitall` after that inner loop.
