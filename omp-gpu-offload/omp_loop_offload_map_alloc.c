/* 

   OpenMP loop offloading on device.


 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include <string.h>


//
// --- real precision is single by default
//
# ifndef REAL
# define REAL float
# endif


//
// ===== device smoother
//

void smooth_device ( REAL *restrict x, REAL *restrict x2,
       	             const REAL w0, const REAL w1, const REAL w2,
                     const int n, const int m, const int niters )
{
  int i, j, iter;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      x2[i*m+j] = 0.0;
    }
  }

  # pragma omp target enter data map(to:x[:n*m]) map(to:x2[:n*m])
  {
  # pragma omp target teams distribute parallel for collapse(2)
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      x2[i*m+j] = 0.0;
    }
  }
  for( iter = 1; iter <= niters; iter++ ){
    # pragma omp target teams distribute parallel for collapse(2)
    for (i = 1; i < n-1; i++) {
      for (j = 1; j < m-1; j++) {
        x2[i*m+j] = w0 *   x[i*m+j]
                  + w1 * ( x[(i-1)*m+j]   + x[(i+1)*m+j]   + x[i*m+j-1]     + x[i*m+j+1]     )
                  + w2 * ( x[(i-1)*m+j-1] + x[(i-1)*m+j+1] + x[(i+1)*m+j-1] + x[(i+1)*m+j+1] );
      }
    }
    # pragma omp target teams distribute parallel for collapse(2)
    for (i = 1; i < n-1; i++) {
      for (j = 1; j < m-1; j++) {
        x[i*m+j] = x2[i*m+j];
      }
    }
  }
  }
  # pragma omp target exit data map(from: x[:n*m])
}


/* void smooth_device ( REAL *restrict x, REAL *restrict x2, */
/*        	             const REAL w0, const REAL w1, const REAL w2, */
/*                      const int n, const int m, const int niters ) */
/* { */
/*   int i, j, iter; */
/*   REAL *tmp; */
/*   int xp2x=1; */

/*   # pragma omp target enter data map(to:x[:n*m]) map(alloc:x2[:n*m]) */
/*   # pragma omp target teams distribute parallel for */
/*   for (i = 0; i < n*m; i++) { */
/*     x2[i*m+j] = 0.0; */
/*   } */
/*   for( iter = 1; iter <= niters; ++iter ){ */
/*     # pragma omp target teams distribute parallel for collapse(2) */
/*     for (i = 1; i < n-1; ++i) { */
/*       for (j = 1; j < m-1; ++j) { */
/*         x2[i*m+j] = w0 *   x[i*m+j] */
/*                   + w1 * ( x[(i-1)*m+j]   + x[(i+1)*m+j]   + x[i*m+j-1]     + x[i*m+j+1]     ) */
/*                   + w2 * ( x[(i-1)*m+j-1] + x[(i-1)*m+j+1] + x[(i+1)*m+j-1] + x[(i+1)*m+j+1] ); */
/*       } */
/*     } */
/*     # pragma omp master */
/*     { */
/*       tmp = x2;  x2 = x;  x = tmp; xp2x = !xp2x; */
/*       printf("iter%2=%d, xp2x=%d\n", iter%2, xp2x); */
/*     } */
/*   } */
/*   if (! xp2x) { */
/*     tmp = x2;  x2 = x;  x = tmp; */
/*   } */
/*   /\* # pragma omp target teams distribute parallel for collapse(2) *\/ */
/*   /\* for (i = 1; i < n-1; ++i) { *\/ */
/*   /\*   for (j = 1; j < m-1; ++j) { *\/ */
/*   /\*     x[i*m+j] = x2[i*m+j]; *\/ */
/*   /\*   } *\/ */
/*   /\* } *\/ */
/* # pragma omp target exit data map(from: x[:n*m]) */
/* } */


//
// ===== host smoother (the gold standard)
//

void smooth_host( REAL *restrict x, REAL *restrict x2,
                  const REAL w0, const REAL w1, const REAL w2,
                  const int n, const int m, const int niters )
{
  int i, j, iter;
  REAL* tmp;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      x2[i*m+j] = 0.0;
    }
  }

  for( iter = 1; iter <= niters; iter++ ){
    for (i = 1; i < n-1; i++) {
      for (j = 1; j < m-1; j++) {
        x2[i*m+j] = w0 *   x[i*m+j]
                  + w1 * ( x[(i-1)*m+j]   + x[(i+1)*m+j]   + x[i*m+j-1]     + x[i*m+j+1]     )
                  + w2 * ( x[(i-1)*m+j-1] + x[(i-1)*m+j+1] + x[(i+1)*m+j-1] + x[(i+1)*m+j+1] );
      }
    }
    tmp = x;  x = x2;  x2 = tmp;
  }
}


//
// ===== main
//

int main( int argc, char* argv[] )
{
  REAL *x_D, *y_D, *x_H, *y_H;
  int i,j,k;
  REAL w0, w1, w2;
  int n, m, nerr, niters;
  REAL dif, rdif, tol;
  double t1, t2, t3;

  // --- init sizes
  n = 0;
  m = 0;
  niters = 0;

  if( argc > 1 ){
    n = atoi( argv[1] );
    if( argc > 2 ){
      m = atoi( argv[2] );
      if( argc > 3 ){
        niters = atoi( argv[3] );
      }
    }
  }

  if( n <= 0 ) n = 1000;
  if( m <= 0 ) m = n;
  if( niters <= 0 ) niters = 10;

  // --- info
  if (omp_get_num_devices() > 0) {
    printf(" running on the GPU...\n");
  } else {
    printf(" running on the Host...\n");
  }

  // --- init data
  //     x_D, y_D -- device data
  //     x_H, y_H -- host data
  x_D = (REAL*) malloc ( sizeof(REAL) * n * m );
  x_H = (REAL*) malloc ( sizeof(REAL) * n * m );
  if (omp_get_num_devices() > 0) {
  y_D = (REAL *) omp_target_alloc( sizeof(REAL) * n * m, omp_get_device_num() );
  } else {
  y_D = (REAL*) malloc ( sizeof(REAL) * n * m );
  }
  y_H = (REAL*) malloc ( sizeof(REAL) * n * m );
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      k = i*m+j;
      x_D[k] = ((REAL) (i-n/2) * (j-m/2)) / ((REAL) n+m);
      x_H[k] = x_D[k];
    }
  }
  w0 = 0.5;
  w1 = 0.3;
  w2 = 0.2;


  // --- iterations
  t1 = omp_get_wtime();
  smooth_device ( x_D, y_D, w0, w1, w2, n, m, niters );
  t2 = omp_get_wtime();
  smooth_host   ( x_H, y_H, w0, w1, w2, n, m, niters );
  t3 = omp_get_wtime();


  // --- report
  printf (" matrix %d x %d, %d iterations\n", n, m, niters);
  printf (" device: %f seconds\n", t2-t1);
  printf (" host:   %f seconds\n", t3-t2);

  // --- verify
  nerr = 0; // no errors initially
  tol = 0.000005; // set tolerance
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      k = i*m+j;
      rdif = dif = fabsf(x_D[k] - x_H[k]);
      if( x_H[k] ) rdif = fabsf(dif / x_H[k]);
      if( rdif > tol ) { // we have an error
        ++nerr;
        if (nerr == 1) // print header on first error
          printf( "    i,   j:        device,          host,         error\n");
        if ( nerr < 10 ) // print top 10 errors
          printf( " %4d,%4d: %12.7e, %12.7e, %12.7e\n", i, j, (double)x_D[k], (double) x_H[k], (double)dif );
      }
    }
  }


  /* printf( "\n\n === DEV         ===HST\n"); */
  /* for (i = 18; i < 22; ++i) { */
  /*   for (j = 18; j < 22; ++j) { */
  /*     k = i*m+j; */
  /*     printf( " %4d,%4d: %12.7e; %12.7e;\n", i, j, (double)x_D[k], (double) x_H[k] ); */
  /*   } */
  /* } */


  free(x_D);
  free(x_H);
  if (omp_get_num_devices() > 0) {
  omp_target_free(y_D, omp_get_device_num());
  } else {
  free(y_D);
  }
  free(y_H);

  if ( nerr == 0 ) {
    printf( " test PASSED\n" );
    return 0;
  } else {
    printf( " %d ERRORS found\n", nerr );
    printf( " test FAILED\n" );
    return 1;
  }
}
