
/*

   name:    vsl_mc
   purpose: Demo of a Monte Carlo simulation employing RNG from the
            mkl/vsl library.  MT2203 is used as the basic generator
            and a uniform distribution between 0 and 1.
   compile: Makefile

 */

# include <stdio.h>
# include <math.h>
# include <mpi.h>
# include <getopt.h>
# include <stdlib.h>
# include "mkl_vsl.h"


// define basic RNG
# define BRNG VSL_BRNG_MT2203
# define SEED 7777777


int main(int argc, char** argv) {

  // MPI variables
  int ip,np;

  // basic RNG parameters
  int brng,seed,
      method,errcode;

  // random numbers
  int     N;   // length of RN vector
  double *r;   // vector of RNs

  VSLStreamStatePtr stream;

  // numbers between 0 and 1
  double a=0.0, b=1.0;

  // extra
  int    i,NN;
  double res;
  int    fdm=0;   // no dumping to file is default
  char   fnm[32]; // file name
  FILE   *fid;    // file handle
  int option=0;   // getopt option


  // init MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &np);
  MPI_Comm_rank (MPI_COMM_WORLD, &ip);

  // default values
  N = 1000;

  // command line args
  while ((option = getopt(argc, argv,"n:o:")) != -1) {
    switch (option) {
       case 'n' :
         N = atoi(optarg);
         break;
       case 'o' :
         fdm = 1;
         strcpy(fnm, optarg);
         break;
    }
  }

  // allocate
  r = (double *) malloc(N * sizeof(double));

  // RNG parameters
  seed   = SEED;       // NB: seed is not (but could be) different for each process
  brng   = BRNG + ip;  // NB: this is different for each process
  method = 0;

  // initialise stream
  errcode = vslNewStream(&stream, brng, seed);

  // generate RNs
  errcode = vdRngUniform(method, stream, N, r, a, b);

  // work on each RN from the stream
  for (i=0; i<N; i++) {
    res = useful(r[i]);
  }

  // write numbers to file
  if (fdm) {
    sprintf (fnm, "%s.%d", fnm, ip);
    fid = fopen (fnm, "w");
    for (i=0; i<N; i++) {
      fprintf(fid, "%20.16e\n",r[i]);
    }
    fclose(fid);
  }

  // close stream
  errcode = vslDeleteStream(&stream);

  // free
  free(r);

  // finalise MPI
  MPI_Finalize();

  return EXIT_SUCCESS;
}


/*
  routine that does something useful
 */
double useful (double r) {
  double res;
  res = cos(r);
  return res;
}



/*
  end
*/
