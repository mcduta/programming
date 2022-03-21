# MPI communicaton in a ring

The demos implement MPI message sending in a ring. MPI process 0 sends to process 1, process 1 send to 2, and so on to the last rank, which messages back to process 0. The messages sent are double precision arrays of random numbers. The mean values of the arrays are computed before the send and after messaging as a means of messaging validation.

There are two equivalent implementations

  * ``ring_mpi.c``
  * ``ring_mpi.F90``

with the difference that the C version takes command line arguments in a different way. Thus, for the C version, run with
```
mpirun -np <num_processes> ring_mpi -s <message_size> [-l <num_loops> -f <print_out_freq>
```
where all options are integers
  * ``message_size`` -- the mesage size (number of doubles allocated);
  * ``num_loops`` -- optional (default is 1): the number of repeats (loops) for the ring communication;
  * ``print_out_freq`` -- optional (default is 10): frequency of reporting (every other specified number of repeats).

The Fortran version call is
```
mpirun -np <num_processes> ring_mpi <message_size> [<num_loops> <print_out_freq>]
```
in which the meaning of the arguments is the same as for the C version.

The messaging is achieved via pairs of non-blocking ``MPI_Isend`` and ``MPI_Irecv`` calls per each process, combined with ``MPI_Wait`` on the messaging.
