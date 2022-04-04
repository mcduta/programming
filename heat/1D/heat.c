
/*

  name:     HEAT

  synopsis: 1D time-dependent heat equation solved using
            explicit finite-differencing, serial version

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

  compile:  icc -O2 -o heat heat.c -limf
            gcc -O2 -o heat heat.c -lm

 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>


int main ( int argc, char *argv[] )
{
  // scheme variables
  int        J,N,
             j,n;
  double     pi,T,
             *x,*u,*uo,
             dx,dt,nu;
  double     de,rms,fac;

  // other variables
  FILE       *fid;
  clock_t    wtime;

  // print header
  printf("\n");
  printf(" 1D heat equation MPI (explicit) demo\n");
  printf("\n");

  // scheme variables
  printf(" number of space nodes = "); scanf("%d", &J);
  printf(" number of time levels = "); scanf("%d", &N);
  printf("\n");
  nu = 0.5; // scheme parameter (<=0.5 for stability)

  // start time
  wtime = clock();

  // distance between space nodes
  dx = 1.0 / ((double) J - 1);

  // compute pi
  pi = 4.0*atan(1.0);

  // allocate memory
  x  = (double *) malloc( J * sizeof (double) );
  u  = (double *) malloc( J * sizeof (double) );
  uo = (double *) malloc( J * sizeof (double) );

  // initial condition
  x[0] = 0.0;
  u[0] = 0.0;
  for (j=1; j<J-1; j++) {
    x[j] = j*dx;
    u[j] = sin(pi*x[j]);
  }
  x[J-1] = 1.0;
  u[J-1] = 0.0;

  // time loop
  for (n=0; n<N; n++) {
    // store solution
    for (j=0; j<J; j++) {
      uo[j] = u[j];
    }

    // finite difference scheme
    u[0] = 0.0;
    for (j=1; j<J-1; j++) {
      u[j] = uo[j] + nu*(uo[j-1]-2.0*uo[j]+uo[j+1]);
    }
    u[J-1] = 0.0;
  }

  // output to disk
  fid = fopen ("heat.out", "w");
  for (j=0; j<J; j++) {
    fprintf (fid, " %20.16e %20.16e\n", x[j],u[j]);
  }
  fclose(fid);

  // measure error
  T   = N*nu*dx*dx;    // final simulated time
  rms = 0.0;           // rms error
  fac = exp(-pi*pi*T); // exponential factor in analytic soln
  for (j=0; j<J; j++) {
    de   = u[j] - fac*sin(pi*x[j]);
    rms += de*de;
  }

  // free memory
  free (x);
  free (u);
  free (uo);

  // report time
  wtime = clock() - wtime;
  printf("\n");
  printf(" wall clock elapsed time = %f sec\n", ((double) wtime) / CLOCKS_PER_SEC );
  printf(" final simulated time    = %10.6e\n", T);
  printf(" error rms               = %10.6e\n", sqrt(rms/J));
  printf("\n");

  return 0;
}
/*
  end
*/
