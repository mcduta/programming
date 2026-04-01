/*
  File:     saxpy.cpp
  Synopsis: Illustrates the implementation of the BLAS AXPY functionality using
             * OpenMP CPU multithreading and
             * OpenMP GPU offloading
  Details:
             * Implemented using std::vector, e.g.
                 std::vector<float> x(N)

             * OpenMP parallel regions access std::vector data via std::span, e.g.
                 std::span<float> xs(x);

             * OpenMP target region access std::vector data via a pointer to vector.data(), e.g.
                 float *xd = x.data();
  Build:
             * make COMPILER=<gcc|llvm|nvhpc>
  Run:
             * ./saxpy
 */

# include <stdio.h>
# include <stdlib.h>


//
// ----- helper functions
//       * extracted from <CUDA_SAMPLES>/Common/helper_cuda.h
//
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

# define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


//
// ----- CUDA setup
//
# define CUDA_idx_fuse(NX,NY, iX,iY) __mul24((iX),(NY)) + (iY)
# define DIM_BLOCK_X 64
# define DIM_BLOCK_Y 64


//
// ----- optimised kernel for Jacobi update
//
__global__ void jac_iter_cuda_kernel (const size_t NX, const size_t NY, const float * __restrict__ ud, float * __restrict__ ud2) {

  // indices
  size_t k; // global linear mem index
  size_t ks,ksn,kss,ksw,kse; // shared memory linear mem indices
  // active thread check
  bool  active;

  // GPU shared memory variable
  //For devices of compute capability 8.0 (i.e., A100 GPUs) shared memory capacity per SM is 164 KB
  __shared__ float us[(DIM_BLOCK_X+2)*(DIM_BLOCK_Y+2)];

  // shared memory size
  size_t NXs = DIM_BLOCK_X + 2;
  size_t NYs = DIM_BLOCK_Y + 2;

  // global indices
  size_t iX = threadIdx.x + blockIdx.x*blockDim.x;
  size_t iY = threadIdx.y + blockIdx.y*blockDim.y;

  // shared memory indices
  size_t iXs = threadIdx.x + 1;
  size_t iYs = threadIdx.y + 1;

  // active thread?
  active = iX < NX && iY < NY;

  // copy to shared mem (if index within limits)
  if (active) {

    //
    // --- indices
    //
    // index for central node
    k  =  CUDA_idx_fuse(NX,NY,iX,iY);       // global linear mem index
    ks =  CUDA_idx_fuse(NXs,NYs,iXs,iYs);   // shared memory linear mem index
    // indices for nodes up, down, etc.: north, south, west, east
    ksn = CUDA_idx_fuse(NXs,NYs,iXs-1,iYs); // shared memory linear mem index
    kss = CUDA_idx_fuse(NXs,NYs,iXs+1,iYs); // shared memory linear mem index
    ksw = CUDA_idx_fuse(NXs,NYs,iXs,iYs-1); // shared memory linear mem index
    kse = CUDA_idx_fuse(NXs,NYs,iXs,iYs+1); // shared memory linear mem index
 
    //
    // --- central node treatment
    //     * each thread copies "central" node into shared mem
    //     * copy from variable ud
    //
    us[ks] = ud[k];

    //
    // --- halo treatment
    //     * iY index changes fastest, so coallesced mem transfer along iY
    //     * copy from variable ud
    //
    if (iXs==1 && iX>0) {
      us[ksn] = ud[k-NY];
    }

    if (iXs==NXs-2 && iX<NX-1) {
      us[kss] = ud[k+NY];
    }

    if (iYs==1 && iY>0) {
      us[ksw] = ud[k-1];
    }

    if (iYs==NY-2 && iY<NY-1) {
      us[kse] = ud[k+1];
    }
  }

  __syncthreads();


  //
  // ----- update central node using shared memory
  //
  if (active) {
    ud2[k] = 0.25f * ( us[ksn] + us[kss] + us[kse] + us[ksw] );
  }

  __syncthreads();

}


//
// ----- non-optimised kernel for Jacobi update
//
__global__ void jac_iter_cuda_kernel_naive (const size_t NX, const size_t NY, const float * __restrict__ ud, float * __restrict__ ud2) {

  // global indices
  size_t iX = threadIdx.x + blockIdx.x*blockDim.x;
  size_t iY = threadIdx.y + blockIdx.y*blockDim.y;
  //For devices of compute capability 8.0 (i.e., A100 GPUs) shared memory capacity per SM is 164 KB

  // computation
  if (iX>0 && iX<NX-1 && iY>0 && iY<NY-1) {
    // update solution
    ud2[ CUDA_idx_fuse (NX,NY, iX,iY) ] = 0.25f * ( ud[ CUDA_idx_fuse(NX,NY, iX+1,iY) ] + ud[ CUDA_idx_fuse(NX,NY, iX-1,iY) ]
                                                  + ud[ CUDA_idx_fuse(NX,NY, iX,iY+1) ] + ud[ CUDA_idx_fuse(NX,NY, iX,iY-1) ] );
  }

}


//
// ----- jacobi solution iteration in CUDA
//
void jac_iter_cuda (const size_t NX, const size_t NY, const size_t NT, float *u) {

  float *ud, *ud2;

  // kernel execution configuration
  dim3 dimBlock ( DIM_BLOCK_X, DIM_BLOCK_Y );
  dim3 dimGrid  ( 1 + (NX-1) / DIM_BLOCK_X, 1 + (NY-1) / DIM_BLOCK_Y );


  // allocate device memory
  checkCudaErrors ( cudaMalloc((void **) &ud,  NX*NY * sizeof(float)) );
  checkCudaErrors ( cudaMalloc((void **) &ud2, NX*NY * sizeof(float)) );

  // copy data contents of std::vector u to device
  checkCudaErrors ( cudaMemcpy(ud, u, NX*NY*sizeof(float), cudaMemcpyHostToDevice) );

  // Jacobi iterations
  for (auto iT=0; iT<NT; iT++) {
    // Jacobi update
    jac_iter_cuda_kernel_naive <<<dimGrid, dimBlock>>> (NX,NY, ud,ud2);
    // swap pointers
    auto ud_tmp = ud; ud = ud2; ud2 = ud_tmp;
  }

  // copy u2 to u, if u2 is latest updated
  if (NT%2) {
    ud = ud2;
  }

  // copy device data back to std::vector u
  checkCudaErrors ( cudaMemcpy(u, ud, NX*NY*sizeof(float), cudaMemcpyDeviceToHost) );

  // free device memory
  checkCudaErrors (cudaFree (ud));
  checkCudaErrors (cudaFree (ud2));

}
