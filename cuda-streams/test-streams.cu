/*

  Demo for CUDA using streams

  Demo adapted from original source
    https://github.com/NVIDIA-developer-blog/code-samples/tree/master/series/cuda-cpp/overlap-data-transfers

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

  // fixed parameters
  const size_t blockSize = 256;
  const size_t streamNum = 4;

  // sizes
  const size_t totalSize   = totalBytes / sizeof(REAL);
  const size_t blockNum    = totalSize / (streamNum * blockSize );
  const size_t streamSize  = blockNum * blockSize;
  const size_t streamBytes = streamSize * sizeof(REAL);

  assert (streamNum * streamBytes == totalBytes);

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
  REAL *xCPU, *xGPU;
  CUDA_SAFE_CALL( cudaMallocHost((void**) &xCPU, totalBytes) );   // host pinned
  CUDA_SAFE_CALL( cudaMalloc    ((void**) &xGPU, totalBytes) );   // device

  // elapsed time in milliseconds (has to be float)
  float elapsedTime;


  //
  // ..... create events and streams
  //
  cudaEvent_t startEvent, stopEvent;
  cudaStream_t stream[streamNum];
  CUDA_SAFE_CALL( cudaEventCreate(&startEvent) );
  CUDA_SAFE_CALL( cudaEventCreate(&stopEvent) );
  for (int i = 0; i < streamNum; ++i)
    CUDA_SAFE_CALL( cudaStreamCreate(&stream[i]) );


  //
  // ===== baseline case - sequential transfer and execute
  //
  srand(time(NULL));
  randVec(xCPU, totalSize);
  CUDA_SAFE_CALL( cudaEventRecord(startEvent,0) );
  CUDA_SAFE_CALL( cudaMemcpy(xGPU, xCPU, totalBytes, cudaMemcpyHostToDevice) );
  //  cudaKernel <<<streamNum*blockNum*sizeof(REAL), blockSize>>>(xGPU, 0);
  cudaKernel <REAL> <<<totalSize/blockSize, blockSize>>>(xGPU, xGPU, totalSize, 0);
  CUDA_SAFE_CALL( cudaMemcpy(xCPU, xGPU, totalBytes, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
  CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent) );
  printf(" \n sequential transfer and execute\n");
  printf("    ... time  = %f ms\n", elapsedTime);
  printf("    ... error = %g\n", maxError(xCPU, totalSize));


  //
  // ===== asynchronous version 1: loop over {copy, kernel, copy}
  //
  srand(time(NULL));
  randVec(xCPU, totalSize);
  CUDA_SAFE_CALL( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < streamNum; ++i) {
    int offset = i * streamSize;
    CUDA_SAFE_CALL( cudaMemcpyAsync(&xGPU[offset], &xCPU[offset], 
				    streamBytes, cudaMemcpyHostToDevice, 
				    stream[i]) );
    cudaKernel <REAL> <<<blockNum, blockSize, 0, stream[i]>>> (xGPU, xGPU, totalSize, offset);
    CUDA_SAFE_CALL( cudaMemcpyAsync(&xCPU[offset], &xGPU[offset], 
                                    streamBytes, cudaMemcpyDeviceToHost,
                                    stream[i]) );
  }
  CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
  CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent) );
  printf(" \n asynchronous V1 transfer and execute\n");
  printf("    ... time  = %f ms\n", elapsedTime);
  printf("    ... error = %g\n", maxError(xCPU, totalSize));


  //
  // ===== asynchronous version 2: 
  //       loop over copy, loop over kernel, loop over copy
  //
  srand(time(NULL));
  randVec(xCPU, totalSize);
  CUDA_SAFE_CALL( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < streamNum; ++i)
  {
    int offset = i * streamSize;
    CUDA_SAFE_CALL( cudaMemcpyAsync(&xGPU[offset], &xCPU[offset], 
                                    streamBytes, cudaMemcpyHostToDevice,
                                    stream[i]) );
  }
  for (int i = 0; i < streamNum; ++i)
  {
    int offset = i * streamSize;
    cudaKernel <REAL> <<<blockNum, blockSize, 0, stream[i]>>> (xGPU, xGPU, totalSize, offset);
  }
  for (int i = 0; i < streamNum; ++i)
  {
    int offset = i * streamSize;
    CUDA_SAFE_CALL( cudaMemcpyAsync(&xCPU[offset], &xGPU[offset],
                                    streamBytes, cudaMemcpyDeviceToHost,
                                    stream[i]) );
  }
  CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
  CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent) );
  printf(" \n asynchronous V2 transfer and execute\n");
  printf("    ... time  = %f ms\n", elapsedTime);
  printf("    ... error = %g\n", maxError(xCPU, totalSize));


  // cleanup
  CUDA_SAFE_CALL( cudaEventDestroy(startEvent) );
  CUDA_SAFE_CALL( cudaEventDestroy(stopEvent) );
  for (int i = 0; i < streamNum; ++i)
    CUDA_SAFE_CALL( cudaStreamDestroy(stream[i]) );

  cudaFree(xGPU);
  cudaFreeHost(xCPU);

  return EXIT_SUCCESS;
}
