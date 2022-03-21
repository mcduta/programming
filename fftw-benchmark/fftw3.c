
/*

  name:     fftw3.c
  synopsis: demonstrate Fast Fourier Transform (1/2/3D) using the
            fftw library version 3 and interleaved input/output
            arrays of fftw_complex.
  verify:   using the Matlab script fftw3.m

 */

//
// ... specific definitions
// maximum data size for file output
# define __FFTW3_MAX_OUTPUT_SIZE 2097152
// poutput to file if within the above size
# ifndef FFTW3_FILE_OUTPUT
  # define FFTW3_FILE_OUTPUT
# endif


//
// ... general definitions
# include "fftw3_aux.h"




int main (int narg, char *varg[]) {

  __FFTW3_COMPLEX *fftIN,*fftOUT;    // DFT input/output
  __FFTW3_PLAN    plan;              // DFT plan

  int             ndim,              // DFT number of dimensions
                  vdim[3],           // DFT dimensions
                  M,N,L,             // DFT dimensions M=vim[0], etc.
                  nthreads,          // number of threads
                  err,               // error flag
                  idim;

  double          time_start, time_end;


  // process arguments
  err = fftw3_args(narg,varg, &ndim,vdim, &nthreads);

  if (err < 0) {
    printf(" *** error in fftw3_args, aborting...\n");
    return -1;
  }

  printf(" %d dimensional DFT\n", ndim);
  printf(" size = %d", vdim[0]);
  for (idim=1; idim<ndim; idim++) { printf("x%d", vdim[idim]); }
  putchar('\n');
  printf(" %d threads\n", nthreads);


  // dimensions
  M = vdim[0];
  N = vdim[1];
  L = vdim[2];


# ifdef FFTW3_OMP
  // initialise threads
  __FFTW3_INIT_THREADS();
  // multi-threaded plan
  __FFTW3_PLAN_WITH_NTHREADS(nthreads);
# endif


  // allocate
  fftIN   = (__FFTW3_COMPLEX *) __FFTW3_MALLOC(M*N*L * sizeof(__FFTW3_COMPLEX));
  fftOUT  = (__FFTW3_COMPLEX *) __FFTW3_MALLOC(M*N*L * sizeof(__FFTW3_COMPLEX));


  // initialise
  err = fftw3_init(ndim,vdim, fftIN, NULL,NULL);

  if (err < 0) {
    printf(" *** error in fftw3_init, aborting...\n");
    return -2;
  }


  // timing
  time_start = __FFTW3_WTIME();

  // create plan
  switch (ndim) {
  case 1:
    plan = __FFTW3_PLAN_DFT_1D(M,     fftIN,fftOUT, FFTW_FORWARD,FFTW_ESTIMATE);
    break;
  case 2:
    plan = __FFTW3_PLAN_DFT_2D(M,N,   fftIN,fftOUT, FFTW_FORWARD,FFTW_ESTIMATE);
    break;
  case 3:
    plan = __FFTW3_PLAN_DFT_3D(M,N,L, fftIN,fftOUT, FFTW_FORWARD,FFTW_ESTIMATE);
    break;
  }


  // execute plan
  __FFTW3_EXECUTE(plan);


  // destroy plan
  __FFTW3_DESTROY_PLAN(plan);


# ifdef FFTW3_OMP
  // clean threads
  __FFTW3_CLEANUP_THREADS();
# endif


  // timing
  time_end = __FFTW3_WTIME();


  // output to file
# ifdef FFTW3_FILE_OUTPUT
  if (M*N*L < __FFTW3_MAX_OUTPUT_SIZE) {
    err = fftw3_out("fftw3.out", ndim,vdim, fftIN,fftOUT, NULL,NULL, NULL,NULL);

    if (err < 0) {
      printf(" *** error in fftw3_out, aborting...\n");
      return -3;
    }
  }
# endif

  // free memory
  __FFTW3_FREE(fftIN);
  __FFTW3_FREE(fftOUT);


  // timing
  printf (" time elapsed [s] = %14.8e\n", time_end - time_start);



  return 0;

}
