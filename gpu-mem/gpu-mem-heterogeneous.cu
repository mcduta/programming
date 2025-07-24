
# include "gpu-mem-common.h"


//
// ----- main ----- //
//
int main(int narg, char** varg) {

  // ----- main variables
  float *xGPU=NULL;

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

  xGPU = new float[N];
  if (xGPU==NULL) {
    fprintf (stderr, " *** error: failed to allocate host memory!\n");
    exit (EXIT_FAILURE);
  }

  // ----- init CPU array
  printf (" initialising data ...\n");
  array_init (xGPU, N);

  // ------ CUDA kernel
  printf (" processing data ...\n");
  uint threadsPerBlock = 256;
  size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  array_calcs <<<blocksPerGrid, threadsPerBlock>>> (xGPU, N, T);

  // ----- sync (wait for GPU to finish before accessing on host)
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf (stderr, " *** error: failed to synchronize (error code %s)!\n", cudaGetErrorString(err));
    exit (EXIT_FAILURE);
  }

  // ----- check calculations
  bool pass = array_check (xGPU, N, 1.e-6);
  if (pass) {
    printf (" ... passed.\n");
  } else {
    printf (" ... failed.\n");
  }

  // ... free memory
  delete[] xGPU;

  exit(EXIT_SUCCESS);
}
