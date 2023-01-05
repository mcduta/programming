# Program to print process and thread affinity info

## Background
Written to test affinity control via ``srun`` in Slurm.

## Build
Prerequisites are a C99 compiler (``gcc`` only was tested) and an MPI software stack (``mpicc`` is assumed available in command search path).

``make`` or ``make all``. Three executables are created:
  * ``paff`` -- single threaded single process
  * ``paff_omp``  -- OpenMP threaded single process
  * ``paff_mpi`` -- single threaded, multiple processes
  * ``paff_mpi_omp`` -- OpenMP threaded, multiple processes

## Usage
Run on its own, ``paff`` and variants report process/thread affinity. For example
```
OMP_NUM_THREADS=2 mpirun -np 4 paff_mpi_omp
```
reports something like
```
hostname=com01-02 MPIrank=1 OMPthread=0 CPU=27 NUMAnode=1 affinity=24-47
hostname=com01-02 MPIrank=1 OMPthread=1 CPU=30 NUMAnode=1 affinity=24-47
hostname=com01-02 MPIrank=2 OMPthread=0 CPU=3 NUMAnode=0 affinity=0-23
hostname=com01-02 MPIrank=2 OMPthread=1 CPU=4 NUMAnode=0 affinity=0-23
hostname=com01-02 MPIrank=3 OMPthread=0 CPU=26 NUMAnode=1 affinity=24-47
hostname=com01-02 MPIrank=3 OMPthread=1 CPU=29 NUMAnode=1 affinity=24-47
hostname=com01-02 MPIrank=0 OMPthread=0 CPU=0 NUMAnode=0 affinity=0-23
hostname=com01-02 MPIrank=0 OMPthread=1 CPU=1 NUMAnode=0 affinity=0-23
```

If needed, affinity can be controlled via tools such as ``numactl`` or ``taskset``, for example
```
mpirun -np 4 taskset -c 0,1,2,6,24,25,26,30 paff_mpi
mpirun -np 4 numactl --cpunodebind=1 paff_mpi
```

However, where this tool is probably useful is at checking "black-box" affinity control, such as through ``srun``.
