# OpenMP device offload examples

## Background

OpenMP offload constructs were introduced in OpenMP 4.0 and further enhanced in later versions.


## The code examples

There are two source files
  * ``omp_loop_offload.c``
  * ``omp_loop_offload_metadirective.c``

The first example implements a Jacobi-like update [[1]](#1) on a 2D array and offloads it via OpenMP to device execution. For validation, the result of the offloaded update is compared with that of the equivalent single threaded updated on the host. The update is carried out in single precision (default), unless ``-D REAL=double`` is added to the precompiler flags in ``makefile``.

The second example is very similar but manages the host to device data transfers via a conditional metadirective.


## Build and run

Code is generated via the command ``make build`` and both examples can be run with ``make run``.

The ``makefile`` assumes the ``nvc`` compiler is available from Nvidia's HPC SDK [[2]](#2). OpenMP offloading is also supported by GNU compilers and LLVM.


## Notes on OpenMP offloading

Documentation with examples on OpenMP offloading are not abundant, yet good introductory material does exist, _e.g._ [[3]](#3). For the purpose of this project, it is worth mentioning the following.

The classic loop parallelisation construct in OpenMP is ``pragma omp parallel for`` and targets thread execution. On a GPU device, this targeting would utilise only a single Streaming MultiProcessors (SM). To avoid this, loop parallelisation for offloaded execution uses ``pragma omp target teams``, which creates multiple master threads inside a target region with the following properties
  * each master thread has the ability to spawn its own team of threads within a parallel region;
  * threads in different teams cannot synchronise with each other;
  * barriers, critical regions, locks, and atomics only apply to the threads within a team.

The contruct used in the examples to expose parallalelism on offloaded operations is therefore ``pragma omp target teams``, working in combination with a following ``distribute parallel for`` to further distribute iterations across threads within each multiprocessor. Note that, by default, the host thread blocks until the target region is completed (this can be changed using a ``nowait`` clause).

Moving data between the host and device is the expensive part of offloading. The dafault (from version 4.5 on) on entering the ``target`` region is
  * scalars referenced in the target construct are treated as firstprivate (new copy on device, initialised with value on host);
  * static arrays are copied to the device on entry and back to the host on exit.

Movement of data for heap allocated arrays receive special treatment using the ``map`` clause. In ``omp_loop_offload.c``, the oute-most loop (on ``iter``) is within a ``target`` region in which the arrays ``x`` and ``x2`` are transferred from host the the on entry and back to host on exit but no data movement takes place across iterations.


## Further reading

  * https://enccs.github.io/openmp-gpu/

  * http://www.archer.ac.uk/training/virtual/2019-08-28-OpenMP-GPUs/OpenMPTargetOffload.pdf

  * https://www.nersc.gov/users/training/past-training-events/2022/introduction-to-openmp-offload-aug-sep-2022/


## References
<a id="1">[1]</a>
https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-finite-difference-docs-laplacian_part1/

<a id="2">[2]</a>
https://developer.nvidia.com/hpc-sdk

<a id="3">[3]</a>
https://www.openmp.org/wp-content/uploads/2021-10-20-Webinar-OpenMP-Offload-Programming-Introduction.pdf
