
# include "gpu-mem-common.h"


//
// ----- main ----- //
//
int main(int narg, char** varg) {

  // ----- main variables
  float *xCPU=NULL, *xGPU=NULL;

  // ----- default values
  size_t N = 1<<28;   // default size of allocation (2**28*sizeof(float)/1024**3 = 1GB)
  uint T = 1000;      // default number of iterations
  uint G = 1;         // default number of GB to allocate

  // ----- process arguments
  int option=0;
  while ( (option = getopt(narg, varg, "g:t:")) != -1 ) {
    switch (option) {
    case 'g': G = (uint) atoi(optarg);
      break;
    case 't': T = (uint) atoi(optarg);
      break;
    default:  fprintf(stderr, " *** usage: %s -g size(GB) -t iterations\n"); exit (EXIT_FAILURE);
    }
  }
  N *= (size_t) G;

  // ----- allocate memory
  printf (" allocating %f GB ...\n", N*sizeof(float)/((float) GIGA));

  xCPU = new float[N];
  if (xCPU==NULL) {
    fprintf (stderr, " *** error: failed to allocate host memory!\n");
    exit (EXIT_FAILURE);
  }

  cudaError_t err = cudaMalloc ((void **) &xGPU, N*sizeof(float));
  if (err != cudaSuccess) {
    fprintf (stderr, " *** error: failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
    exit (EXIT_FAILURE);
  }

  // ----- init CPU array
  printf (" initialising data ...\n");
  array_init (xCPU, N);

  // ----- copy array: Host to Device
  printf (" copying data (H2D) ...\n");
  err = cudaMemcpy (xGPU, xCPU, N*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf (stderr, " *** error: failed to copy memory H2D (error code %s)!\n", cudaGetErrorString(err));
    exit (EXIT_FAILURE);
  }

  // ------ CUDA kernel
  printf (" processing data ...\n");
  uint threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  array_calcs <<<blocksPerGrid, threadsPerBlock>>> (xGPU, N, T);

  // ----- sync (wait for GPU to finish before accessing on host)
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf (stderr, " *** error: failed to synchronize (error code %s)!\n", cudaGetErrorString(err));
    exit (EXIT_FAILURE);
  }

  // ----- copy array:  Device to Host
  printf (" copying data (D2H) ...\n");
  err = cudaMemcpy (xCPU, xGPU, N*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf (stderr, " *** error: failed to copy memory D2H (error code %s)!\n", cudaGetErrorString(err));
    exit (EXIT_FAILURE);
  }

  // ----- check calculations
  bool pass = array_check (xCPU, N, 1.e-6);
  if (pass) {
    printf (" ... passed.\n");
  } else {
    printf (" ... failed.\n");
  }

  // ... free memory
  delete[] xCPU;
  err = cudaFree(xGPU);
  if (err != cudaSuccess) {
    fprintf(stderr, " *** error: failed to free device memory (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
