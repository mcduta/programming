// ... basic includes
# include <stdio.h>
# include <cuda_runtime.h>
# include <getopt.h>

// ... sizes
# define MEGA (1<<20)
# define GIGA (1<<30)


//
// ----- main ----- //
//
int main(int narg, char** varg) {

  // ----- default values
  size_t sizeBytes = 1<<30;   // default size of each individual allocation (2**30 = 1GB)
  size_t numTimes = 1;        // default number of allocations

  // ----- process arguments
  int option=0;
  while ( (option = getopt(narg, varg, "m:g:t:")) != -1 ) {
    switch (option) {
    case 'm': sizeBytes = ((size_t) atoi(optarg)) * MEGA;
      break;
    case 'g': sizeBytes = ((size_t) atoi(optarg)) * GIGA;
      break;
    case 't': numTimes  = atoi(optarg);
      break;
    default:  fprintf(stderr, " *** usage: gpumemfill -g size(GB) -t repeats\n"); exit (EXIT_FAILURE);
    }
  }

  // ----- report parameters
  printf(" allocating ");
  if (sizeBytes < GIGA) {
    printf(" %6.2f MB ", ((double) sizeBytes)/((double) MEGA));
  } else {
    printf(" %6.2f GB ", ((double) sizeBytes)/((double) GIGA));
  }
  printf("%d times ...\n", numTimes);

  // ----- allocate size
  cudaError_t err = cudaSuccess;                              // error code for CUDA calls
  int t;                                                      // times counter
  void **d = (void **) malloc (numTimes * sizeof (void *));   // array of pointers to a

  for (t=0; t < numTimes; t++) {
    err = cudaMalloc((void **) &d[t], sizeBytes);
  //err = cudaMallocManaged ((void **) &d[t], sizeBytes);

    if (err == cudaSuccess) {
      printf (" ... %d\n", t+1);
    } else {
      fprintf(stderr, " *** error: failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit (EXIT_FAILURE);
    }
  }

  // Free device global memory
  for (t=0; t < numTimes; t++) {
    err = cudaFree(d[t]);

    if (err != cudaSuccess) {
      fprintf(stderr, " *** error: failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  printf (" ... finished.\n");
  exit(EXIT_SUCCESS);
}
