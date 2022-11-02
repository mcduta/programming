
/*

  name:     HEAT_OMP

  synopsis: 1D time-dependent heat equation solved using
            explicit finite-differencing, OpenMP version

  model:    The PDE is

                 du/dt = d2u/dx2

            defined on the interval [0, 1] and with the boundary conditions

                 u(0,t) = uA(t) = 0
                 u(1,t) = uB(t) = 0
                 u(x,0) = u0(x) = sin(pi*x)

            The anaytic solution is

                 u(x,t) = sin(pi*x) * exp(-pi**2*t)

	    The finite difference stencil is:

            time step n+1              u(j,n+1)
                                           |
                                           |
            time step n  u(j-1,n) ----- u(j,n) ----- u(j+1,n)

            The solution scheme is

                 u(j,n+1) = u(j,n) + nu*(u(j-1,n)-2*u(j,n)+u(j+1,n))

            with the stability condition

                 nu = dt/dx**2 <= 1/2

  compile:  icc -O2 -openmp  -o heat_omp heat_omp.c -limf
            gcc -O2 -fopenmp -o heat_omp heat_omp.c -lm

 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>
# include <string.h>


int main ( int argc, char **argv )
{
  // scheme variables
  int        J=1<<28, N=100,
             j,n;
  double     pi,T,
             *u,*uo,
             dx,nu,xj;
  double     de,rms,fac;
  double     *orig, *dest, *temp;

  // file output (debug)
  FILE       *fid;
  char       outfile_name[32];
  int        file_output=0;

  // timing variables
  double     t_init, t_iter;

  // print header
  printf("\n 1D heat equation (explicit) demo using OpenMP\n");

  // scheme variables
  if (argc > 1) J = atoi(argv[1]);
  if (argc > 2) N = atoi(argv[2]);
  if (argc > 3) { strcpy(outfile_name, argv[3]); file_output = 1; }

  // scheme parameter (<=0.5 for stability)
  nu = 0.5;

  // distance between space nodes
  dx = 1.0 / ((double) J - 1);

  // compute pi
  pi = 4.0*atan(1.0);

  // allocate memory
  u  = (double *) malloc( J * sizeof (double) );
  uo = (double *) malloc( J * sizeof (double) );


  //
  // ... initialisation
  //
  t_init = omp_get_wtime ();

  u[0] = 0.0;

  # pragma omp parallel for \
           default(none) \
           shared(J,pi,dx,u) \
           private(j,xj)
  for (j=1; j<J-1; j++) {
    xj   = ((double) j)*dx;
    u[j] = sin(pi*xj);
  }
  u[J-1] = 0.0;

  t_init = omp_get_wtime () - t_init;


  //
  // ... iterations
  //
  // start iteration timing
  t_iter = omp_get_wtime ();

  // solution copy
  # pragma omp parallel for \
           default(none) \
           shared(J,u,uo) \
           private(j)
  for (j=0; j<J; j++) {
    uo[j] = u[j];
  }

  // work pointers
  orig = u;
  dest = uo;

  // time loop
  # pragma omp parallel \
           default(none) \
           shared(N,J,u,uo, nu, orig,dest,temp) \
           private(n,j)
  for (n=0; n<N; n++) {

    // finite difference scheme
    dest[0] = 0.0;
    # pragma omp for
    for (j=1; j<J-1; j++) {
      dest[j] = orig[j] + nu*(orig[j-1]-2.0*orig[j]+orig[j+1]);
    }
    dest[J-1] = 0.0;

    // swap pointers
    # pragma omp single
    {
      temp = orig;
      orig = dest;
      dest = temp;
    }
  }

  // copy to u, if needed
  if (dest == u) {
    # pragma omp parallel for \
             default(none) \
             shared(J,u,uo) \
             private(j)
    for (j=0; j<J; j++) {
      u[j] = uo[j];
    }
  }

  // stop iteration timing
  t_iter = omp_get_wtime () - t_iter;


  //
  // ... solution to file
  //
  if (file_output) {
    fid = fopen (outfile_name, "w");
    for (j=0; j<J; j++) {
      fprintf (fid, " %20.16e\n", u[j]);
    }
    fclose(fid);
  }


  //
  // ... error
  //
  T   = N*nu*dx*dx;    // final simulated time
  rms = 0.0;           // rms error
  fac = exp(-pi*pi*T); // exponential factor in analytic soln

  # pragma omp parallel for \
           default(none) \
           shared(J,u, pi,dx,fac) \
           private(j,de,xj) \
           reduction(+:rms)
  for (j=0; j<J; j++) {
    xj   = ((double) j)*dx;
    de   = u[j] - fac*sin(pi*xj);
    rms += de*de;
  }

  // free memory
  free (u);
  free (uo);


  //
  // ... report
  //
  // report time
  printf("\twall clock elapsed time\t\t = %10.6f sec (initialisation)\n", t_init);
  printf("\t\t\t\t\t = %10.6f sec (iterations total)\n", t_iter);
  printf("\tfinal simulated time\t\t = %10.6e\n", T);
  printf("\terror rms\t\t\t = %10.6e\n", sqrt(rms/((double) J)));

  return EXIT_SUCCESS;
}

/*
  end
*/
