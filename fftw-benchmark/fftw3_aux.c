
/*
  name:     fftw3_aux.c
  synopsis: auxiliary functions

 */

# include "fftw3-macros.h"


// -------------------------------------------------------------------- //
//                                                                      //
//  fftw3_args -- process line arguments                                //
//                                                                      //
// -------------------------------------------------------------------- //

int fftw3_args (int narg, char *varg[],
                int *Pndim, int vdim[3],
                int *Pnthreads)
{

  int ndim,idim, iarg, nthreads;

  // defaults
  nthreads = 1;
  vdim[0]  = 1;
  vdim[1]  = 1;
  vdim[2]  = 1;

  // process options
  iarg =  1;  // second arg, first is name of executable
  ndim = -1;  // undefined dimensionality

  if (narg <= 1) {
    printf(" *** error: no arguments suppplied\n");
    return -1;
  }

  while (iarg < narg) {

    if (strcmp(varg[iarg], "-d") == 0) {
      if (iarg+1 <= narg-1) {
        ndim = atoi(varg[iarg+1]);
        iarg = iarg+2;
        if (ndim<1 || ndim>3) {
          printf(" *** error: unexpected DFT dimensionality\n");
          return -2;
        }
      } else {
        printf(" *** error: no dimensionality arguments\n");
        return -3;
      }
    } else if (strcmp(varg[iarg], "-n") == 0) {
      if (ndim == -1) {
        printf(" *** error: undefined DFT dimensionality\n");
        return -4;
      } else {
        if (iarg+ndim <= narg-1) {
          for (idim=0; idim<ndim; idim++) {
            vdim[idim] = atoi(varg[iarg+idim+1]);
            if (vdim[idim] < 1) {
              printf(" *** error: wrong dimension value\n");
              return -5;
            }
          }
          iarg = iarg+ndim+1;
        } else {
          printf(" *** error: insufficient dimension arguments\n");
          return -6;
        }
      }
# ifdef FFTW3_OMP
    } else if (strcmp(varg[iarg], "-t") == 0) {
      if (iarg+1 <= narg-1) {
        nthreads = atoi(varg[iarg+1]);
        iarg = iarg+2;
      } else {
        printf(" *** error: insufficient arguments\n");
        return -6;
      }
# endif
    } else {
      printf(" *** error: incorrect arguments\n");
      printf("     usage: fftw3 -d 3 -n M N L");
# ifdef FFTW3_OMP
      printf(" -t nthreads\n");
# else
      printf("\n");
# endif
      return -7;
    }

  }

  // return values
  *Pndim     = ndim;
  *Pnthreads = nthreads;

  return 0;

}


// -------------------------------------------------------------------- //
//                                                                      //
//  fftw3_wtime -- compute wall time based on gettimeofday()            //
//                                                                      //
// -------------------------------------------------------------------- //

double fftw3_wtime (void) {
# define MILLION 1000000.0
  double secs;
  struct timeval tp;

  gettimeofday (&tp,NULL);
  secs = (MILLION * (double) tp.tv_sec + (double) tp.tv_usec) / MILLION;
  return secs;
}


// -------------------------------------------------------------------- //
//                                                                      //
//  fftw3_init -- initialise fftw data                                  //
//                                                                      //
// -------------------------------------------------------------------- //

int fftw3_init (int ndim, int *vdim,
                __FFTW3_COMPLEX *fftIN,
                __FFTW3_REAL *fftINr, __FFTW3_REAL *fftINi)
{

  int             M,N,L, m,n,l, k,   // indices
                  split;
  __FFTW3_REAL    wnum[3],           // wave numbers
                  pi,                // number PI
                  dx,dy,dz,          // grid discretisation
                  zz,zr,zi;          // ancillary variables

  // determine case: interleaved/split
  split = fftIN == NULL;

  if (split && (fftINr == NULL || fftINi == NULL)) {
    printf(" *** error: null pointers to split DFT data\n");
    return -1;
  }
  if (!split && (fftINr != NULL || fftINi != NULL)) {
    printf(" *** error: ambiguous pointers to split DFT data\n");
    return -2;
  }

  // dimensions
  M = vdim[0];
  N = vdim[1];
  L = vdim[2];

  // pi
  pi = 4.0*atan(1.0);

  // wave numbers
  wnum[0] = 2.0;
  wnum[1] = 4.6;
  wnum[2] = 3.2;

  // domain discretisation
  switch (ndim) {
    case 1:
      dx = 2.0*pi/((__FFTW3_REAL) M-1);
      dy = 0.0;
      dz = 0.0;
      break;
    case 2:
      dx = 2.0*pi/((__FFTW3_REAL) M-1);
      dy = 2.0*pi/((__FFTW3_REAL) N-1);
      dz = 0.0;
      break;
    case 3:
      dx = 2.0*pi/((__FFTW3_REAL) M-1);
      dy = 2.0*pi/((__FFTW3_REAL) N-1);
      dz = 2.0*pi/((__FFTW3_REAL) L-1);
      break;
  }

  // initialise (NB: to use OMP threads, the init loop has to change)
  for (m=0; m<M; m++) {
    for (n=0; n<N; n++) {
      for (l=0; l<L; l++) {
        zz = pi*(wnum[0]*m*dx + wnum[1]*n*dy + wnum[2]*l*dz);
	if        (ndim == 2) {
          k = ROW_MAJOR_INDEX_2D(M,N,m,n);
        } else if (ndim == 3) {
          k = ROW_MAJOR_INDEX_3D(M,N,L,m,n,l);
        } else {
          k = m;
        }
        if (split) {
          fftINr[k]   = cos(zz);
          fftINi[k]   = sin(zz);
        } else {
          fftIN[k][0] = cos(zz);
          fftIN[k][1] = sin(zz);
        }
      }
    }
  }

  return 0;

}


// -------------------------------------------------------------------- //
//                                                                      //
//  fftw3_out -- write fftw data to ascii file                          //
//                                                                      //
// -------------------------------------------------------------------- //

int fftw3_out (char *file_out,
               int ndim, int *vdim,
               __FFTW3_COMPLEX *fftIN,   __FFTW3_COMPLEX *fftOUT,
               __FFTW3_REAL    *fftINr,  __FFTW3_REAL    *fftINi,
               __FFTW3_REAL    *fftOUTr, __FFTW3_REAL    *fftOUTi)
{

  int             M,N,L, m,n,l, k, idim, split;
  FILE            *fid;

  // determine case: interleaved/split
  split = fftIN == NULL;

  if (split) {
    if (fftOUT != NULL) {
      printf(" *** error: ambiguous pointer to DFT data\n");
      return -1;
    }
    if (fftINr  == NULL || fftINi  == NULL
     || fftOUTr == NULL || fftOUTr == NULL) {
      printf(" *** error: null pointers to split DFT dat\na");
      return -2;
    }
  } else {
    if (fftOUT == NULL) {
      printf(" *** error: ambiguous pointer to DFT data\n");
      return -1;
    }
    if (fftINr  != NULL || fftINi != NULL
     || fftOUTr != NULL || fftOUTr != NULL) {
      printf(" *** error: ambiguous pointers to split DFT data\n");
      return -2;
    }
  }

  // dimensions
  M = vdim[0];
  N = vdim[1];
  L = vdim[2];

  // write data
  fid = fopen (file_out, "w");

  fprintf (fid, "ndim = %d\n", ndim);
  fprintf (fid, "size = %d", vdim[0]);
  for (idim=1; idim<ndim; idim++) { fprintf(fid, " %d", vdim[idim]); }
  fprintf (fid, "\n");

  for (m=0; m<M; m++) {
    for (n=0; n<N; n++) {
      for (l=0; l<L; l++) {
        if        (ndim == 2) {
          k = ROW_MAJOR_INDEX_2D(M,N,m,n);
        } else if (ndim == 3) {
          k = ROW_MAJOR_INDEX_3D(M,N,L,m,n,l);
        } else {
          k = m;
        }
        if (split) {
          fprintf (fid, "%24.18e %24.18e %24.18e %24.18e\n",
                        fftINr[k], fftINi[k],
                        fftOUTr[k],fftOUTi[k]);
        } else {
          fprintf (fid, "%24.18e %24.18e %24.18e %24.18e\n",
                        fftIN[k][0], fftIN[k][1],
                        fftOUT[k][0],fftOUT[k][1]);
        }
      }
    }
  }

  fclose (fid);

  return 0;

}


/*                              E  N  D                               */
