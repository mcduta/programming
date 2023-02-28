# include "paff_utils.h"


/*
  cuda_set_device_id -- set GPU device based on input
    * set GPU on 0 if single process
    * set GPU on (mpi_process_id % total_number_of_gpus) if multiple MPI processes
 */
int cuda_set_device_id (const int task) {
  int ngpu=0;
  cudaError_t cuda_err;
  cuda_err = cudaGetDeviceCount (&ngpu);  if (cuda_err != cudaSuccess || ngpu == 0) return -1;
  cuda_err = cudaSetDevice (task % ngpu); if (cuda_err != cudaSuccess) return -1;
  return 0;
}


/*
  cuda_get_device_id -- get GPU device ID
 */

int cuda_get_device_id () {
  int gpu;
  cudaError_t cuda_err;
  cuda_err = cudaGetDevice (&gpu);
  if (cuda_err == cudaSuccess) {
    return gpu;
  } else {
    return -1;
  }
}
