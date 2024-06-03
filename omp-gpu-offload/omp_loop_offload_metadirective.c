/* 

   OpenMP loop offloading on device.

   Modification if the basic example using a "pragma omp metadirective" construct.

 */

# include <stdio.h>
# include <stdlib.h>
# include <assert.h>
# include <omp.h>
# include <math.h>
# include <string.h>
# include <stdbool.h>


//
// --- real precision is single by default
//
# ifndef REAL
# define REAL float
# endif


//
// ===== global bool to control gpu offload
//
bool use_gpu=true;


//
// ===== device smoother
//

void smooth_device ( REAL*restrict x, REAL*restrict x2,
       	             const REAL w0, const REAL w1, const REAL w2,
                     const int n, const int m, const int niters )
{
  int i, j, iter;
  REAL* tmp;

  for( iter = 1; iter <= niters; iter++ ){
    # pragma omp metadirective when(user={condition(use_gpu)}: target teams loop collapse(2) map(tofrom:x[:n*m],x2[:n*m])) default(parallel for collapse(2))
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
// ===== host smoother (the gold standard)
//

void smooth_host( REAL *restrict x, REAL *restrict x2,
                  const REAL w0, const REAL w1, const REAL w2,
                  const int n, const int m, const int niters )
{
  int i, j, iter;
  REAL* tmp;

  for( iter = 1; iter <= niters; iter++ ){
    for (i = 1; i < n-1; i++) {
      for (j = 1; j < m-1; j++) {
        x2[i*m+j] = w0 * x[i*m+j]
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
  y_D = (REAL*) malloc ( sizeof(REAL) * n * m );
  y_H = (REAL*) malloc ( sizeof(REAL) * n * m );
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      k = i*m+j;
      x_D[k] = 0.;
      x_H[k] = 0.;
      y_D[k] = ((REAL) (i-n/2) * (j-m/2)) / ((REAL) n+m);
      y_H[k] = y_D[k];
    }
  }
  w0 = 0.5;
  w1 = 0.3;
  w2 = 0.2;


  // --- iterations
  t1 = omp_get_wtime();
  # pragma omp metadirective when(user={condition(use_gpu)}: target data map(tofrom:x_D[:n*m], y_D[:n*m])) 
  {
  smooth_device ( x_D, y_D, w0, w1, w2, n, m, niters );
  }
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
          printf( " %4d,%4d: %12.7e, %12.7e, %12.7e\n", i, j, x_D[k], x_H[k], dif );
      }
    }
  }


  //
  // --- clean up and report
  //
  free(x_D);
  free(x_H);
  free(y_D);
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
