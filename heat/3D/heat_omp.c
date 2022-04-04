
/*

  name:     HEAT.C

  synopsis: 3D time-dependent heat equation solved using
            explicit finite-differencing

  version:  serial

  doc:      heat.pdf

  compile:  icc -O2 -o heat heat.c -limf
            gcc -O2 -o heat heat.c -lm

 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

/*
 ================================================================
     FDIN -- finite difference matrix index to linear memory index
 ================================================================
*/
static inline int fdin(const int I,
                       const int J,
                       const int K,
                       const int i,
                       const int j,
                       const int k)
{
  return (k*J + j)*I + i;
}


/*
 ================================================================
     MIN -- C - macro for minimum
 ================================================================
*/
# define MIN(a, b) \
  ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

# define MAX(a, b) \
  ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


/*
 ================================================================
     MAIN
 ================================================================
*/

int main ( int argc, char** argv )
{
  // number of discrete points (x,y,z and t)
  int     I,J,K, N;
  // wave numbers (x,y,z);
  int     wnx,wny,wnz;
  // 3D indices (x,y,z and t)
  int     i,j,k, n;
  // linear memory indices
  int     s,                    // (i,j,k) --> s
          sim,sip,              // (i-1,j,k) --> sim, (i+1,j,k) --> sip
          sjm,sjp,              // (i,j-1,k) --> sjm, (i,j+1,k) --> sjp
          skm,skp;              // (i,j,k-1) --> skm, (i,j,k+1) --> skp
  // space and solution arrays
  double  *x,*y,*z,             // coordinates
          *u1,*u2,*u3,          // solution storage
          *u,*uo;               // pointers to solution (u "current" and u "old")
  // other variables
  double  dx,dy,dz, dt,         // discretisation steps
          pi,                   // pi
          T,                    // final simulated time
          nu,                   // nu = dt/(dx*dx)
          de,
          rms,                  // rms error
          fac;
  // walltime
  double  wtime_sol,            // solution time
          wtime_tot;            // total time
  // number of OpenMP threads
  int     nt;
  // iteration output control
  int     Nout;
  // extras
  FILE    *fileid;
  int     fileIO;


  // start total time
  wtime_tot = omp_get_wtime ();

  # ifdef DEBUG
  // print header
  printf("\n");
  printf(" 3D heat equation demo\n");
  printf("\n");
  # endif

  // process arguments
  if (argc >= 3) {
    // discretisation parameters
    I = atoi(argv[1]); 
    N = atoi(argv[2]);

    // write solution to file?
    fileIO = 0;
    if (argc >= 4) {
      fileIO = atoi(argv[3]);
    }

    // number of OMP threads
    if (argc == 5) {
      nt = atoi(argv[4]);
      omp_set_num_threads (nt);
    }
    nt = omp_get_max_threads ();

  } else {
    printf(" *** error: insufficient arguments, two expected\n");
    return -1;
  }

  // check arguments
  if (I < 2) {
    printf(" *** error: number of space intervals too small\n");
    return -2;
  }

  if (N < 1) {
    printf(" *** error: number of time intervals too small\n");
    return -3;
  }


  # ifdef DEBUG
  // scheme parameters
  printf(" number of space nodes = %d x %d x %d\n", I,I,I);
  printf(" number of time levels = %d\n", N);
  printf(" writing solution to file = %c\n", fileIO?'y':'n');
  printf(" number of OpenMP threads = %d\n", nt);
  printf("\n");
  # endif

 // scheme parameter (<= 1/6 for stability)
  nu = 0.15;

  // distance between space nodes
  J  = I;
  K  = I;
  dx = 1.0 / ((double) I - 1);
  dy = dx;
  dz = dx;

  // wave numbers
  wnx = 1;
  wny = 4;
  wnz = 2;

  // compute pi
  pi = 4.0*atan(1.0);

  // allocate memory
  x  = (double *) malloc( I*J*K * sizeof (double) );
  y  = (double *) malloc( I*J*K * sizeof (double) );
  z  = (double *) malloc( I*J*K * sizeof (double) );
  u1 = (double *) malloc( I*J*K * sizeof (double) );
  u2 = (double *) malloc( I*J*K * sizeof (double) );

  // pointers to solution storage
  u  = u1;
  uo = u2;

  // initial condition
  # pragma omp parallel for default(none) \
               shared(I,J,K, dx,dy,dz, wnx,wny,wnz, pi, x,y,z,u) \
               private(i,j,k,s)
  for (k=0; k<K; k++) {
    for (j=0; j<J; j++) {
      for (i=0; i<I; i++) {
	s    = fdin(I,J,K, i,j,k);
	x[s] = i*dx;
	y[s] = j*dy;
	z[s] = k*dz;
	u[s] = sin(((double) wnx)*pi*x[s])*sin(((double) wny)*pi*y[s])*sin(((double) wnz)*pi*z[s]);
      }
    }
  }

  // output every Nout iterations
  Nout = MIN( MAX(N/10, 1), 100);

  // start solution time
  wtime_sol = omp_get_wtime ();

  // time loop
  # pragma omp parallel default(none) \
               shared(N,Nout, I,J,K, u,uo,u3, nu) \
               private(n, i,j,k, s,sim,sip,sjm,sjp,skm,skp)
  {
  for (n=0; n<N; n++) {
    // output
    # pragma omp master
    {
      if (n % Nout == 0) printf (" iteration %10d\n", n);
    }

    // swap pointers to solution storage (u points to updates, uo to previous step solution)
    # pragma omp master
    {
    u3 = u;
    u  = uo;
    uo = u3;
    }
    # pragma omp barrier

    // finite difference scheme
    # pragma omp for
    for (k=1; k<K-1; k++) {
      for (j=1; j<J-1; j++) {
	for (i=1; i<I-1; i++) {
	  s    = fdin(I,J,K, i,j,k);      // (i,j,k) <-- centre point
          sim  = s - 1;                   // (i-1,j,k)
          sip  = s + 1;                   // (i+1,j,k)
          sjm  = s - I;                   // (i,j-1,k)
          sjp  = s + I;                   // (i,j+1,k)
          skm  = s - I*J;                 // (i,j,k-1)
          skp  = s + I*J;                 // (i,j,k+1)

          u[s] = uo[s] + nu * ( uo[sip] - 2.0*uo[s] + uo[sim]
                              + uo[sjp] - 2.0*uo[s] + uo[sjm]
                              + uo[skp] - 2.0*uo[s] + uo[skm] );
        }
      }
    }
  }
  }

  // stop solution time
  wtime_sol = omp_get_wtime () - wtime_sol;

  // measure error
  T   = N*nu*dx*dx;             // final simulated time
  rms = 0.0;                    // rms error
  fac = ((double) wnx)*((double) wnx)
      + ((double) wny)*((double) wny)
      + ((double) wnz)*((double) wnz);
  fac = exp(-fac*pi*pi*T);      // exponential factor in analytic solution

  # pragma omp parallel for default(none) \
               shared(I,J,K, x,y,z,u, fac,wnx,wny,wnz, pi) \
               private(i,j,k, s, de) \
               reduction(+:rms)
  for (k=1; k<K-1; k++) {
    for (j=0; j<J; j++) {
      for (i=0; i<I; i++) {
	s    = fdin(I,J,K, i,j,k);
        de   = u[s] - fac*sin(((double) wnx)*pi*x[s])*sin(((double) wny)*pi*y[s])*sin(((double) wnz)*pi*z[s]);
        rms += de*de;
      }
    }
  }

  // write to file
  if (fileIO) {
    fileid = fopen ("heat_omp.out", "w");
    for (k=0; k<K; k++) {
      for (j=0; j<J; j++) {
        for (i=0; i<I; i++) {
          s = fdin(I,J,K, i,j,k);
          fprintf (fileid, "%20.16e %20.16e %20.16e %20.16e\n", x[s],y[s],z[s], u[s]);
        }
      }
    }
    fclose(fileid);
  }

  // free memory
  free (x);
  free (y);
  free (z);
  free (u);
  free (uo);

  // stop total time
  wtime_tot = omp_get_wtime () - wtime_tot;

  // report
  printf("\n");
  printf(" wall clock elapsed time (solution) = %f sec\n", wtime_sol);
  printf(" wall clock elapsed time (total)    = %f sec\n", wtime_tot);
  printf(" error rms                          = %10.6e\n", sqrt(rms) * dx*dy*dz);
  printf("\n");

  return 0;
}

/*
  end
*/
