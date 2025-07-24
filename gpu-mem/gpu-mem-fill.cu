// ... basic includes
# include <stdio.h>
# include <getopt.h>
# include <unistd.h>
# include <cuda_runtime.h>

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
  unsigned int numSecs = 0;   // default wait between allocation and deallocation

  // ----- process arguments
  int option=0;
  while ( (option = getopt(narg, varg, "m:g:i:s:")) != -1 ) {
    switch (option) {
    case 'm': sizeBytes = ((size_t) atoi(optarg)) * MEGA;
      break;
    case 'g': sizeBytes = ((size_t) atoi(optarg)) * GIGA;
      break;
    case 'i': numTimes  = atoi(optarg);
      break;
    case 's': numSecs   = ((unsigned int) atoi(optarg));
      break;
    default:  fprintf(stderr, " *** usage: gpumemfill -g size(GB) -i repeats -s seconds (sleep)\n"); exit (EXIT_FAILURE);
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

  // ----- 1. allocate numTimes memory of sizeBytes
  //       2. wait numSecs
  //       3. deallocate 
  cudaError_t err = cudaSuccess;                              // error code for CUDA calls
  int i;                                                      // times counter
  void **d = (void **) malloc (numTimes * sizeof (void *));   // array of pointers to a

  // ... allocate device global memory
  for (i=0; i < numTimes; i++) {
    err = cudaMalloc((void **) &d[i], sizeBytes);
  //err = cudaMallocManaged ((void **) &d[t], sizeBytes);

    if (err == cudaSuccess) {
      printf (" ... %d\n", i+1);
    } else {
      fprintf(stderr, " *** error: failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit (EXIT_FAILURE);
    }
  }

  // ... wait
  if (numSecs) {
    printf (" ... waiting %d seconds\n", numSecs);
    sleep (numSecs);
  }

  // ... free device global memory
  for (i=0; i < numTimes; i++) {
    err = cudaFree(d[i]);

    if (err != cudaSuccess) {
      fprintf(stderr, " *** error: failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  printf (" ... finished.\n");
  exit(EXIT_SUCCESS);
}
