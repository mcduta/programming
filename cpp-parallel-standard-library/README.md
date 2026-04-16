# C++

## Overview
The C++17 Standard introduced high-level parallelism features that allow programmers to request parallelization of Standard Template Library (STL) algorithms. Thus, C++17 STL parallelism consists of:
* parallel execution policies (`seq`, `par`, `unseq`, `par_unseq`);
* parallel-enabled algorithms in `<algorithm>` and `<numeric>` (e.g. `std::for_each`) and
* bulk parallel numeric primitives (e.g. `std::reduce` and `std::transform_reduce`).

This project illustrates the capability of the STL parallelism to generate code from a single source that can run in parallel on either CPUs or GPUs. (The emphasis is on the GPOU target.)

## C++ STL parallelism at a glance

C++17 introduced parallel algorithms, implemented in functions that can optionally run sequentially, in parallel (multithreaded), vectorised (SIMD) or both in parallel and vectorised. The execution is controlled via the following execution policies, passed to the functions as the first argument:
* `std::execution::seq` -- forces sequential execution;
* `sd::execution::par` -- allows parallel execution across multiple threads;
* `std::execution::par_unseq` -- allows parallel multithreaded execution and vectorisation (SIMD);
* `std::execution::unseq` -- allows SIMD vectorisation but not multithreading.

The above policies are passed to the standard library algorithms (which internally decides how to act on them) and then the compiler optimises the code the standard library ends up instantiating. If the instantiation requires parallel execution, the compiler thus generates code that 
* spawns worker threads;
* uses a thread pool;
* splits the input range into chunks and
* uses SIMD instructions for `unseq` and `par_unseq`.

## Parallel implementations
The Parallel Standard Template Library (PSTL) standard defines the API and semantic constraints, but does not require an implementation to actually run in parallel. To honour the promise that PSTL code can target both CPU and GPU, real-world parallelism comes from different sources.

### CPU parallel execution
First, CPU-side parallelism depends almost entirely on the standard library implementation (the GCC `libstdc++` and the LLVM `libc++`) and its backend library (usually Intel TBB), not directly on the compiler. As such, a compiler may fully support PSTL syntactically but the implementation may be fully or partially parallel, may fall back on sequential execution or may be dependent entirely on optional backend libraries (e.g. TBB).

The LLVM `libc++` historically lagged behind the GCC `libstdc++`, being typically focused on correctness and conformance first, with conservative parallel execution support. The consequence for parallel performance is `libstdc++` is preferred for both `g++` and `clang++`.

Intel Threading Building Blocks (TBB), now oneTBB, is a C++ library for task-based parallelism. It isd often used as the backend for PSTL implementations in `libstdc++` and `libc++` configurations, as it provides good performance scaling onn multi-core CPUs.

### GPU parallel execution
Second, GPU-level parallelism is vendor-specific.

The NVIDIA `nvc++` compiler (from the NVIDIA HPC SDK) accelerates C++ PSTL by offloading eligible parallel algorithms to the GPU when they are invoked with execution policies like `std::execution::par`. In practice, the NVIDIA HPC SDK ships a modified/extended standard library implementation (and toolchain integration) where calls such as
```c++
std::sort(std::execution::par, begin, end);
```
are intercepted and mapped to GPU kernels, typically via the NVIDIA parallel algorithm infrastructure (often leveraging Thrust primitives under the hood) and replacing the `libstdc++` typical CPU backends (typically, TBB). Thus, PSTL becomes a frontend, and `nvc++` provides a GPU execution backend that replaces TBB execution with CUDA kernel execution.

The AMD `clang++` accelerates C++ PSTL on AMD GPUs primarily through the HIP/ROCm offload toolchain, using the same general model as LLVM OpenMP target offload.

## Code examples
There are two C++ code examples:
* `saxpy`, with `saxpy.cpp`, implementing the [BLAS SAXPY vector operation](https://www.netlib.org/blas/) and
* `jacobi` with `jacobi.cpp`, implementing a 2D Jacobi update.

Both codes use C++ `std::vector` of type `float` (to allow execution on GPUs that do not support performance double precision execution).

> [!NOTE]
> `saxpy` is purely illustrative and the typical do-not-do-this-at-home; if this functionality is needed, use the appropriate library (`cuBLAS` or `hipBLAS`) instead of rewriting. `jacobi` is reasonably close to real-life stencil-based algorithm implementastion to be practically useful as a coding starting point.

Both codes are built using the `make` command and a compiler option that is either `nvhpc` (for NVIDIA GPUs) or `rocm` (for AMD GPUs). For example,
```bash
COMPILER=nvhpc make
```
builds an executable that offloads to NVIDIA GPUs.

The `saxpy` executable runs with vectors of fixed length and takes no command line arguments. The Jacobi executable `jac`, on the other hand, takes 4 optional arguments: the 2 sizes of the updated matrix, the number of iterations and one boolean to trigger writing the initial and final solution to disk (for correctness checks). For example,
```bash
./jac 30000 20000 500
```
runs 500 iterations on a 30k by 20k matrix.

## Observations
Typical run performances in GFlops are

| GPU         | SAXPY | Jacobi |
|-------------|-------|--------|
| NVIDIA A100 |  195  |   340  |
| AMD MI300A  |  135  |   480  |

The `makefile` can be modified to target CPU execution rather than GPU offloading. For example, the flags to `nvc++` can be changed from `-stdpar=gpu` to `-stdpar=multicore` to achieve that.

## References
https://docs.nvidia.com/hpc-sdk/archive/22.2/pdf/hpc222c++_par_alg.pdf
