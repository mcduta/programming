# OpenMP GPU Offloading

## Overview
OpenMP GPU offloading is an extension of the OpenMP standard that allows C, C++, and Fortran programs to execute portions of code on accelerators (GPUs) using compiler directives. This has been part of OpenMP since version 4.0 (2013), with major improvements in 4.5, 5.0, and later.

The purpose of this project is to illustrate OpenMP GPU offloading and to highlight the pros and cons of OpenMP programming (compared to native models, such as CUDA) in terms of ease of use, portability and performance.

## OpenMP GPU offloading at a glance
The core idea is to write code using high-level OpenMP pragmas and let the compiler map it to CPUs or GPUs. The key construct in doing so is `target`, which annotates a region of code to be offloaded to a device (e.g. GPU). Thus, instead of writing CUDA or HIP kernels explicitly, the region that `target` marks is targeted by the compiler for device execution:

```c
#pragma omp target
{
    // runs on GPU
}
```

To control the offload, two aspects are to be specified: how the parallel workload is scheduled and what memory transfers or allocations are needed to support the workload.

First, to specify parallelism, `target` is used together with `teams` to map work to the GPU thread blocks:
```c
#pragma omp target teams
parallel for (or distribute)
```
To distribute the parallel work further between the thread blocks and all the threads within a block on the device, the following completes the loop parallelisation
```c
#pragma omp target teams distribute parallel for
for (int i = 0; i < N; i++) {
    A[i] += B[i];
}
```
This roughly corresponds to:
* `teams` map to thread blocks;
* `parallel` maps to threads.

Further to the above the `simd` construct can be used to indicate the loop can be transformed into a SIMD loop, which maps well to the GPU SIMT (Single Instruction, Multiple Threads) model. While GPUs do not have traditional SIMD units (like CPUs), they execute instructions in warp-level parallelism, in which a single instruction is applied to multiple data elements simultaneously.

Second, the `map` clause controls memory movement between host and device. For instance,
```c
#pragma omp target map(to: A[0:N]) map(from: B[0:N])
```
instructs the compiler that the variable `A` is copied from host to device and `B` from device to host. The options are:
* `to` from host to device;
* `from` from device to host;
* `tofrom` both ways;
* `alloc` device allocation only.

A good introduction to OpenMP GPU offloading is [this webinar](https://www.openmp.org/wp-content/uploads/2021-10-20-Webinar-OpenMP-Offload-Programming-Introduction.pdf).

## The codes

The principal codes are contained in the directories
* `saxpy` with `saxpy.cpp`, implementing the [BLAS SAXPY vector operation](https://www.netlib.org/blas/) and
* `jacobi` with `jacobi.cpp`, implementing a 2D Jacobi update.

Both codes use C++ `std::vector` (there are plenty of examples using C-style arrays, far fewer with C++ vectors) of type `float` (to allow execution on GPUs that do not support performance double precision execution).

> [!NOTE]
> `saxpy` is purely illustrative and the typical do-not-do-this-at-home; if this functionality is needed, use the appropriate library (`cuBLAS` or `hipBLAS`) instead of rewriting. `jacobi` is reasonably close to real-life stencil-based algorithm implementastion to be practically useful as a coding starting point.

In addition to the above, `saxpy` also contains a C-style code `saxpy.c-style.cpp` using `new` allocated arrays that might be useful. More importantly, `jacobi` contains a CUDA implementation of the Jacobi stencil-based algorithm in `jacobi-update-cuda.cu`, which is the reference performance point for the OpenMP offloading.

Also present, in the `utils` directory, are a few simple OpenMP examples to illustrate the detection of GPUs hardware.

> [!NOTE]
> On AMD GPUs, `jacobi-update-cuda.cu` is "hipified" (a source code transformation using the `hipify-perl` tool) to be built on AMD hardware.

All codes are built using the `make` command. Study the `makefile` source for hints to compiler options. The `saxpy` executable runs with vectors of fixed length and takes no command line arguments. The Jacobi executable `jac`, on the other hand, takes 5 optional arguments: the 2 sizes of the updated matrix, the number of iterations and two booleans to indicate CPU runs (0) or GPU offloading (1) and writing the initial and final solutions to disk (for correctness checks). For example,
```bash
./jac 30000 20000 500 1
```
runs 500 iterations on a 30k by 20k matrix on the GPU.

## Remarks
Both `saxpy` and `jac` measure the execution time of the offloaded region and report it along with a GFLOP score.

> [!NOTE]
> Neither code provides a high enough arithmetic intensity to be remotely close to the theoretical peak performance of GPU cards. Both codes are memory bound.

A performance comparison of the Jacobi update between the OpenMP offloaded variant and the CUDA implementation highlights the pros and cons of OpenMP offloading:

| GPU         | OpenMP offloading | CUDA/HIP acceleration |
|-------------|-------------------|-----------------------|
| NVIDIA A100 |         90        |      2530             |
| AMD MI300A  |        450        |      2300             |

OpenMP is thus excellent in terms of ease of programming and code reusability. The usual OpenMP incremental development (adding pragmas to existing CPU code) leads to portable code (same code runs on CPU and GPU), with standardised, vendor-neutral coding. There is a clear performance gap in comparison with native (CUDA/HIP) programming, compiler support is usably mature but still varies and debugging tooling is less capable and usable than CUDA/HIP tooling. Yet, OpenMP can prove very useful in the case of existing OpenMP CPU code ported to GPU, especially in the case of HPC kernels with regular data parallelism and in a multi-vendor environment.

## Resources

1. An extended workshop presentation:
 * [day 1](https://www.olcf.ornl.gov/wp-content/uploads/2021/08/ITOpenMP_Day1.pdf);
 * [day 2](https://www.olcf.ornl.gov/wp-content/uploads/2021/08/ITOpenMPO_Day2.pdf).

2. [AMD notes](https://rocm.blogs.amd.com/high-performance-computing/jacobi/README.html) on the Jacobi iteration.

