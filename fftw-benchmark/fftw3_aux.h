
/*
  name:     fftw3_aux.h
  synopsis: auxiliary functions

 */

# include "fftw3-macros.h"


//
// ----- OpenMP macros
//
# if ( defined(FFTW3_OMP) )
  # define __FFTW3_WTIME  omp_get_wtime
# else
  # define __FFTW3_WTIME  fftw3_wtime
# endif


//
// ----- auxiliary functions
//
int fftw3_args (int narg, char *varg[],
                int *Pndim, int vdim[3],
                int *Pnthreads);
int fftw3_init (int ndim, int *vdim,
                __FFTW3_COMPLEX *fftIN,
                __FFTW3_REAL *fftINr, __FFTW3_REAL *fftINi);
int fftw3_out (char *file_out,
               int ndim, int *vdim,
               __FFTW3_COMPLEX *fftIN,   __FFTW3_COMPLEX *fftOUT,
               __FFTW3_REAL    *fftINr,  __FFTW3_REAL    *fftINi,
               __FFTW3_REAL    *fftOUTr, __FFTW3_REAL    *fftOUTi);
double fftw3_wtime (void);


/*                              E  N  D                               */
