
/*

  name:     fftw3s.c
  synopsis: demonstrate Fast Fourier Transform (1/2/3D) using the
            fftw library version 3 and split input/output
            arrays of float/double.
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

  __FFTW3_REAL    *fftINr, *fftINi,  // DFT input
                  *fftOUTr,*fftOUTi; // DFT output
  fftw_iodim      dims[3];           // dimension info
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
  fftINr  = (__FFTW3_REAL *)    __FFTW3_MALLOC(M*N*L * sizeof(__FFTW3_REAL));
  fftINi  = (__FFTW3_REAL *)    __FFTW3_MALLOC(M*N*L * sizeof(__FFTW3_REAL));
  fftOUTr = (__FFTW3_REAL *)    __FFTW3_MALLOC(M*N*L * sizeof(__FFTW3_REAL));
  fftOUTi = (__FFTW3_REAL *)    __FFTW3_MALLOC(M*N*L * sizeof(__FFTW3_REAL));


  // initialise
  err = fftw3_init(ndim,vdim, NULL, fftINr,fftINi);

  if (err < 0) {
    printf(" *** error in fftw3_init, aborting...\n");
    return -2;
  }


  // timing
  time_start = __FFTW3_WTIME();

  // create plan
  for (idim=ndim-1; idim>=0; idim--) {
    dims[idim].n  = vdim[idim];
    if (idim == ndim-1) {
      dims[idim].is = 1;
      dims[idim].os = 1;
    } else {
      dims[idim].is = vdim[idim+1]*dims[idim+1].is;
      dims[idim].os = dims[idim].is;
    }
  }

  plan = __FFTW3_PLAN_GURU_SPLIT_DFT(ndim, dims, 0, (fftwf_iodim *) 0,
                                     fftINr,fftINi, fftOUTr,fftOUTi,
                                     FFTW_ESTIMATE);


  // execute plan
  __FFTW3_EXECUTE_SPLIT_DFT(plan, fftINr,fftINi, fftOUTr,fftOUTi);


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
    err = fftw3_out("fftw3s.out", ndim,vdim, NULL,NULL, fftINr,fftINi, fftOUTr,fftOUTi);

    if (err < 0) {
      printf(" *** error in fftw3_out, aborting...\n");
      return -3;
    }
  }
# endif

  // free memory
  __FFTW3_FREE(fftINr);
  __FFTW3_FREE(fftINi);
  __FFTW3_FREE(fftOUTr);
  __FFTW3_FREE(fftOUTi);

  // timing
  printf (" time elapsed [s] = %14.8e\n", time_end - time_start);



  return 0;

}
