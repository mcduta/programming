# Simple MPI + CUDA example

## Purpose
  * Test process placement on a multi-GPU setting;
  * Illustrate the need for explicit device set in programming.

## Versions
Two similar versions are available: one written in C++ and the other in Python. The C++ version is written with MPI programming [[1]](#1) and uses the CUDA libraries [[2]](#2). The python version uses ``mpi4py``[[3]](#3)  and ``cupy``[[4]](#4).

## Build
Dependencies for the C++ version are:
  * CUDA
  * MPI
  * CMake (> 3.8)

There is no build for the Python version, which should run with an exising Python appropriate distribution/installation, such as Anaconda. The package ``mpi4py`` may already come with the MPI launcher ``mpirun``, and that can be used for single host experiments. CUDA must be present in the environment, _e.g._ library paths.

Building the C++ version follows the steps
```
mkdir build && cd build
cmake ../
```

Target CUDA arch capabilities can be specified via ``CMAKE_CUDA_ARCHITECTURES`` and installation prefix can be set via ``CMAKE_INSTALL_PREFIX``. For instance
```
cmake ../ -D CMAKE_CUDA_ARCHITECTURES=70 -D CMAKE_INSTALL_PREFIX=./bin
```

## Run
The C++ executable is ``simple-mpi-cuda`` and can take two command line options:
  * ``-g`` to report the GPU ID which each MPI rank uses;
  * ``-s`` to select the above utilisation in a round-robin fashion over the GPU cards available.

Example:
```
mpirun -np 8 bin/simple-mpi-cuda -g -s
```

Similarly, the Python version is run with
```
mpirun -np 8 python simple-mpi4py-cupy.py -g -s
```

## References
<a id="1">[1]</a>
https://en.wikipedia.org/wiki/Message_Passing_Interface

<a id="2">[2]</a>
https://developer.nvidia.com/cuda-zone

<a id="3">[3]</a>
https://mpi4py.readthedocs.io/en/stable/

<a id="4">[4]</a>
https://docs.cupy.dev/en/stable/index.html
