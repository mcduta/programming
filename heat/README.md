# Various implementations of an implicit finite difference solver for the heat equation

## Implementations

1D, 2D and 3D implemented in C with several parallelisation options
  * sequential;
  * MPI;
  * OpenMP;
  * CUDA;
  * hybrid (MPI/OpenMP).

The 2D version is also written in Python to illustrate various possibilities for the core compute load
  * python (using ``for`` loops);
  * numpy;
  * numba;
  * blitz;
  * inline;
  * fortran (Fortran library);
  * ctypes (C library);
  * cython.

The purpose is two-fold: to illustrate parallel implementations and to provide a toy code for profiling and tracing tools.


## Old notes regarding profiling

### IPM

Load a module for OpenMPI and make PAPI loadable
> export LD_LIBRARY_PATH=/system/software/arcus/lib/papi/4.1.1/lib:$LD_LIBRARY_PATH

Then, preload the IPM library, so that IPM intercepts the MPI calls of an already-linked executable:
> export LD_PRELOAD=/system/software/arcus/ipm/0.983/openmpi-1.6.5__intel-2013/lib/libipm.so

Run the code
> mpirun -np 32 -hostfile hosts__32.txt ./heat_mpi 401 1000

Finally,
> export LD_PRELOAD=

The above produces a file called something like ``mihai.1400147974.550356.0``. This file can be used to produce an html directory, containing all the findings

> export IPM_KEYFILE=/path/to/ipm_key
> ipm_parse -html mihai.1400147974.550356.0

``ipm_parse`` has to have ``ploticus`` (http://ploticus.sourceforge.net) installed.


### PARAVER

> module load extrae/2.2.1_intel
> module load openmpi/1.6.2__intel-2012

Run serial program
> cp ${EXTRAE_HOME}/share/example/SEQ/extrae.xml .
> ${EXTRAE_HOME}/bin/ompitrace -config extrae.xml <program>

Run mpi program
> cp ${EXTRAE_HOME}/share/example/MPI/ld-preload/trace.sh .
> cp ${EXTRAE_HOME}/share/example/MPI/extrae.xml .

Adjust in ``trace.sh`` ``EXTRAE_HOME`` and ``EXTRAE_CONFIG_FILE``. Then
> mpirun -np 4 ./trace.sh heat_mpi

Obtain the final trace:
> ${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -e heat_mpi -o trace.prv

Lastly, ``module load paraver`` and use ``wxparaver.bin``.
