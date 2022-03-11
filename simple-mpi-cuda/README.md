# Simple MPI + CUDA example

## Purpose
  * Test process placement on a multi-GPU setting;
  * Illustrate the need for explicit device set in programming.

## Build
Dependencies:
  * CUDA
  * MPI
  * CMake (> 3.8)

Build
```
mkdir build && cd build
cmake ../
```

Target CUDA arch capabilities can be specified via ``CMAKE_CUDA_ARCHITECTURES`` and installation prefix can be set via ``CMAKE_INSTALL_PREFIX``. For instance
```
cmake ../ -D CMAKE_CUDA_ARCHITECTURES=70 -D CMAKE_INSTALL_PREFIX=./bin
```

## Run
The executable is ``simple-mpi-cuda`` and can take two command line options:
  * ``-g`` to report the GPU ID which each MPI rank uses;
  * ``-s`` to select the above utilisation in a round-robin fashion over the GPU cards available.

Example:
```
mpirun -np 8 bin/simple-mpi-cuda -g -s
```