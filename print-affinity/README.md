# Program to print process and thread affinity info

## Background
Written to test and verify affinity control via ``srun`` in Slurm.

## Build
Prerequisites are a C99 compiler (``gcc`` only was tested), an MPI software stack (``mpicc`` is assumed available in command search path) and CUDA.

The build is via ``CMake``, e.g.
```
cmake -S. -Bbuild -GNinja
cmake --build build

```

An old ``makefile`` also exists.

The executables created are:
  * ``paff``          -- single process, OpenMP multi-threaded;
  * ``paff_cuda``     -- single process, OpenMP multi-threaded with CUDA-controlled GPU affinity;
  * ``paff_mpi``      -- multiple processes, OpenMP multi-threaded;
  * ``paff_mpi_cuda`` -- multiple processes, OpenMP multi-threaded with CUDA-controlled GPU affinity.


## Usage
Run on its own, ``paff`` and variants report process/thread affinity. For example
```
OMP_NUM_THREADS=2 mpirun -np 4 paff_mpi
```
reports something like
```
Host=cs05r-sc-gpu01-34 MPIrank=0 OMPthread=0 CPU=1 NUMAnode=0 Affinity=0-1
Host=cs05r-sc-gpu01-34 MPIrank=0 OMPthread=1 CPU=0 NUMAnode=0 Affinity=0-1
Host=cs05r-sc-gpu01-34 MPIrank=1 OMPthread=0 CPU=2 NUMAnode=0 Affinity=2-3
Host=cs05r-sc-gpu01-34 MPIrank=1 OMPthread=1 CPU=3 NUMAnode=0 Affinity=2-3
Host=cs05r-sc-gpu01-34 MPIrank=2 OMPthread=0 CPU=5 NUMAnode=0 Affinity=4-5
Host=cs05r-sc-gpu01-34 MPIrank=2 OMPthread=1 CPU=4 NUMAnode=0 Affinity=4-5
Host=cs05r-sc-gpu01-34 MPIrank=3 OMPthread=0 CPU=7 NUMAnode=0 Affinity=6-7
Host=cs05r-sc-gpu01-34 MPIrank=3 OMPthread=1 CPU=7 NUMAnode=0 Affinity=6-7
```

The affinity control of processes and threads in the ``paff`` and ``paff_mpi`` is external. It can be changed with tools such as ``numactl`` or ``taskset``, for example
```
OMP_NUM_THREADS=2 mpirun -np 4 numactl --cpunodebind=1 paff_mpi
```

In addition to CPU affinity, the ``_cuda`` version also prints the GPU selected by each process or thread. This selection is done by enac MPI rank or by each thread (if single process), with a default of GPU 0, in the case of single-threaded process. The selection is done in a round robin fashion on all the GPUs available. Arguably, the ``_cuda`` version is the least useful variant.
