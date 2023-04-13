Run `make.sh` to build the `MPI_Alltoall` benchmark (`all`) and the `MPI_Ialltoall` benchmark (`iall`).

To run `all`, submit the `job.sh` batch script and specify the number of nodes. The job will run `all` twice with the default buffer size, once with contiguous subcommunicators and once with strided subcommunicators.
```
$ sbatch -N<nodes> job.sh
```
Similarly for `iall` and `ijob.sh`.
```
$ sbatch -N<nodes> ijob.sh
```

The scripts `[i]submit.sh` will submit `[i]all` jobs on powers-of-two nodes up to and including some maximum node count. So this:
```
$ ./[i]submit.sh 100
```
... will submit jobs on 1, 2, 4, 8, 16, 32, 64, and 100 nodes for `[i]all`. The jobs will store output in files `[i]all.<timestamp>-<nodes>.out`.
