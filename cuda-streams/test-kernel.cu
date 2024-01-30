/*

  Simple CUDA kernel test that times one kernel execution:
  * overall timing, inc. memory transfers and
  * kernel timing only.

 */

# include <stdio.h>
# include <assert.h>
# include <time.h>
# include "test-utils.h"
# include "test-utils-cuda.h"

# ifndef REAL
# define REAL float
# endif


//
// --- the MAIN
//
int main (int argc, char **argv)
{
  // total amount of memory used on GPU (1GB default)
  size_t totalBytes = NUM_GIGA;
  if (argc > 1) totalBytes = parseArgv (argc, argv);

  // sizes
  const size_t totalSize = totalBytes / sizeof(REAL);
  
  // fixed parameters
  const size_t blockSize = 256;

  // report size
  printSize (totalBytes);

  // device ID
  int devId = 0;

  // cudaDeviceProp struct reference
  cudaDeviceProp devProp;
  CUDA_SAFE_CALL( cudaGetDeviceProperties(&devProp, devId) );
  assert (totalBytes < devProp.totalGlobalMem);
  printf(" found device %s with %fGB\n", devProp.name, ((REAL) devProp.totalGlobalMem)/((REAL) (NUM_GIGA)));
  CUDA_SAFE_CALL( cudaSetDevice(devId) );

  // allocate pinned host memory and device memory
  REAL *xCPU, *yCPU, *xGPU, *yGPU;
  CUDA_SAFE_CALL( cudaMallocHost((void**) &xCPU, totalBytes) );   // host pinned
  CUDA_SAFE_CALL( cudaMallocHost((void**) &yCPU, totalBytes) );   // host pinned
  CUDA_SAFE_CALL( cudaMalloc    ((void**) &xGPU, totalBytes) );   // device
  CUDA_SAFE_CALL( cudaMalloc    ((void**) &yGPU, totalBytes) );   // device

  // elapsed time in milliseconds (has to be float)
  float elapsedTime, elapsedTimeAllOps;


  //
  // ..... create events and streams
  //
  cudaEvent_t startEvent,       stopEvent,
              startEventAllOps, stopEventAllOps;
  CUDA_SAFE_CALL( cudaEventCreate(&startEvent) );
  CUDA_SAFE_CALL( cudaEventCreate(&startEventAllOps) );
  CUDA_SAFE_CALL( cudaEventCreate(&stopEvent) );
  CUDA_SAFE_CALL( cudaEventCreate(&stopEventAllOps) );


  //
  // ===== baseline case - sequential transfer and execute
  //
  srand(time(NULL));
  randVec <REAL> (xCPU, totalSize);
  CUDA_SAFE_CALL( cudaEventRecord(startEventAllOps, 0) );
  CUDA_SAFE_CALL( cudaMemcpy(xGPU, xCPU, totalBytes, cudaMemcpyHostToDevice) );

  CUDA_SAFE_CALL( cudaEventRecord(startEvent, 0) );
  cudaKernel <REAL> <<<totalSize/blockSize, blockSize>>> (xGPU, yGPU, totalSize, 0);
  CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
  CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );

  CUDA_SAFE_CALL( cudaMemcpy(yCPU, yGPU, totalBytes, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaEventRecord(stopEventAllOps, 0) );
  CUDA_SAFE_CALL( cudaEventSynchronize(stopEventAllOps) );

  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsedTime,       startEvent,       stopEvent) );
  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsedTimeAllOps, startEventAllOps, stopEventAllOps) );
  printf(" \n timing transfer and execute\n");
  printf("    ... time (overall) = %f ms\n", elapsedTimeAllOps);
  printf("    ... time (kernel)  = %f ms\n", elapsedTime);
  printf("    ... error          = %g\n", maxError <REAL> (yCPU, totalSize));


  //
  // ===== cleanup
  //
  CUDA_SAFE_CALL( cudaEventDestroy(startEvent) );
  CUDA_SAFE_CALL( cudaEventDestroy(startEventAllOps) );
  CUDA_SAFE_CALL( cudaEventDestroy(stopEvent) );
  CUDA_SAFE_CALL( cudaEventDestroy(stopEventAllOps) );

  cudaFree(xGPU);
  cudaFree(yGPU);
  cudaFreeHost(xCPU);
  cudaFreeHost(yCPU);

  return EXIT_SUCCESS;
}
