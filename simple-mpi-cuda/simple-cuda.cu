
/*
   simple-cuda.cu -- CUDA kernel and driver function
 */

# include <iostream>
# include "simple-mpi-cuda.hpp"


//
// --- CUDA error handling macro
# define CUDA_SAFE_CALL(CUDA_CALL) \
  if((CUDA_CALL) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    std::cerr << " *** error: CUDA call: \""#CUDA_CALL"\", error code:" << err << std::endl; \
    simple_mpi_abort (-1); \
  }


//
// --- GPU device check
void mpi_set_device (const int mpi_proc_num, const int mpi_proc_id) {

  int dev_num = 0;
  CUDA_SAFE_CALL ( cudaGetDeviceCount (&dev_num) );
  CUDA_SAFE_CALL ( cudaSetDevice ( mpi_proc_id % dev_num ) );

}


//
// --- GPU device check
void mpi_get_device (const int mpi_proc_num, const int mpi_proc_id) {

  int dev_id;
  CUDA_SAFE_CALL ( cudaGetDevice ( &dev_id ) );
  std::cout << " MPI GET device: rank " << mpi_proc_id << " : device " << dev_id << std::endl;

}


//
// --- CPU data initialisation
void data_cpu_init (const int size, REAL *data) {
  for (int i = 0; i < size; i++) {
    data[i] = (REAL) rand() / RAND_MAX;
  }
}


//
// --- GPU data processing
__global__ void data_dev_process (const REAL *data_in, REAL *data_out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  data_out[tid] = sqrt(data_in[tid]);
}


//
// --- MPI process function (called from main() and handles all computation)
void mpi_data_process (const int mpi_rank, const int block_size, const int grid_size) {

  // data size is so that data maps nicely onto the CUDA "grid of blocks" SIMD model
  int data_size = block_size * grid_size;


  int dev_id;
  cudaDeviceProp dev_prop;
  CUDA_SAFE_CALL ( cudaGetDevice ( &dev_id ) );
  CUDA_SAFE_CALL ( cudaGetDeviceProperties( &dev_prop, dev_id ) );

  // total GPU memory available has to accommodate 4 times (2 arrays of equal size)
  int size  = dev_prop.totalGlobalMem / (8 * sizeof(REAL));
  size      = size / data_size;
  data_size = size * data_size;

  // allocate data on CPU memory and initialise
  REAL *data_cpu = new REAL[data_size];
  data_cpu_init (data_size, data_cpu);

  // allocate data on GPU memory
  REAL *data_dev_in = NULL,
       *data_dev_out = NULL;

  CUDA_SAFE_CALL ( cudaMalloc ((void **) &data_dev_in,  data_size * sizeof(REAL)) );
  CUDA_SAFE_CALL ( cudaMalloc ((void **) &data_dev_out, data_size * sizeof(REAL)) );

  // host to device memory copy
  CUDA_SAFE_CALL ( cudaMemcpy (data_dev_in, data_cpu, data_size * sizeof(REAL), cudaMemcpyHostToDevice) );

  // run device kernel
  data_dev_process <<<grid_size, block_size>>> (data_dev_in, data_dev_out);

  // device to host memory copy
  CUDA_SAFE_CALL ( cudaMemcpy (data_cpu, data_dev_out, data_size *sizeof(REAL), cudaMemcpyDeviceToHost) );

  // free host memory
  delete [] data_cpu;

  // free device memory
  CUDA_SAFE_CALL ( cudaFree (data_dev_in) );
  CUDA_SAFE_CALL ( cudaFree (data_dev_out) );

  ///////// print sucess from process mpi_rank on cudaGetDevice ( int* device )

}
