/*
further ideas:
* introduce extra command line param to say how many times run something
* express performance in terms of gflops
* based on that, explore possibilities (vectorisation, optimisation)
 */


# include <stdio.h>
# include <stdlib.h>

# define MAX_ARRAY_SIZE 1000000000
# define MAX_NUM_CALLS  10

# include "cfunc.h"

void init (int n, double *x, double *y) {
  int i;
  for (i = 0; i < n; i++) {
    x[i] = ((double) i+1)/((double) n);
    y[i] = ((double) n-i)/((double) n);
  }
}


// ==================================================================== //
//                                                                      //
//     system_timer -- precision walltime function, computes            //
//                     walltime based on the gettimeofday() function    //
//                                                                      //
// ==================================================================== //

double system_timer (void) {

# include <stdlib.h>
# include <sys/time.h>

  struct timeval time;

  gettimeofday (&time, NULL);

  return time.tv_sec + time.tv_usec / 1000000.0;

}





//--------------------------------------------------------------
//---------------------- START OF main()  ----------------------
//--------------------------------------------------------------
int main (int narg, char** varg) {
  // array size
  int n;
  // iterator
  int i;
  // three arrays
  double *x, *y, *z;
  // timing variables
  double wtime_start,
         wtime_stop;

  //
  // --- command line
  //
  // size of arrays can be command line argument
  if (narg >= 2) {
    // command line argument value
    n = atoi(varg[1]);
    if (n < 1 || n > MAX_ARRAY_SIZE) {
      printf(" *** error: argument %d out of range (1 -- %d)\n", 1,MAX_ARRAY_SIZE);
      exit (EXIT_FAILURE);
    }
  } else {
    // default value
    n = 1024;
  }


  //
  // --- initialise arrays
  //
  // allocate
  x  = (double *) malloc(n * sizeof(double));
  y  = (double *) malloc(n * sizeof(double));
  z  = (double *) malloc(n * sizeof(double));


  //
  // --- calculations
  //
  init (n, x,y);
  wtime_start = system_timer ();
  for (i=0; i<MAX_NUM_CALLS; i++) mult (n, x,y,z);
  wtime_stop = system_timer ();
  printf(" elapsed time (C aliased)  = %f [s]\n", (wtime_stop - wtime_start)/((double) MAX_NUM_CALLS));

  init (n, x,y);
  wtime_start = system_timer ();
  for (i=0; i<MAX_NUM_CALLS; i++) mult2 (n, x,y,z);
  wtime_stop = system_timer ();
  printf(" elapsed time (C restrict) = %f [s]\n", (wtime_stop - wtime_start)/((double) MAX_NUM_CALLS));

  init (n, x,y);
  wtime_start = system_timer ();
  for (i=0; i<MAX_NUM_CALLS; i++) fmult_ (&n, x,y,z);
  wtime_stop = system_timer ();
  printf(" elapsed time (Fortran)    = %f [s]\n", (wtime_stop - wtime_start)/((double) MAX_NUM_CALLS));


  //
  // --- wrap-up
  //
  free(x);
  free(y);
  free(z);

  return EXIT_SUCCESS;
}

//--------------------------------------------------------------
//----------------------- END OF main()  -----------------------
//--------------------------------------------------------------
