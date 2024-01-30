//
// --- header guards to prevent multiple inclusions of the same header file
//     (equivalent to the non-standard # pragma once preprocessor directive)
//
# ifndef _TEST_UTILS_CUDA_H_
# define _TEST_UTILS_CUDA_H_

# include <stdio.h>

// default real is single precision
# ifndef REAL
# define REAL float
# endif

//
// --- CUDA error handling macro
//
# define CUDA_SAFE_CALL(cudaResult) \
  if ((cudaResult) != cudaSuccess) { \
    fprintf(stderr, " *** error: CUDA error code: %s\n", cudaGetErrorString(cudaResult)); \
    assert((cudaResult) == cudaSuccess); \
  }


//
// --- CUDA kernel
//
template <class T> __global__ void cudaKernel (T *x, T *y, const size_t N, const size_t offset);


//
// --- include the template implementation source file to instantiate
//
# include "test-utils-cuda.cu"

# endif
