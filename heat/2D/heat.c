
/*

  name:     HEAT

  synopsis: 2D time-dependent heat equation solved using
            explicit finite-differencing

  version:  serial

  model:    see heat.tex or heat.pdf

  compile:  icc -O2 -o heat heat.c -limf
            gcc -O2 -o heat heat.c -lm

 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>

// finite difference matrix indexing
# define fdin(I,J,i,j) j*I + i


int main ( int argc, char *argv[] )
{
  // scheme variables
  int        I,J,N,
             wnx,wny,
             i,j,k,n,
             kn,ks,ke,kw;
  double     pi,T,
             *x,*y,*u,*uo,
             dx,dy,dt,nu;
  double     de,rms,fac;

  // other variables
  FILE       *fid;
  clock_t    wtime;

  // print header
  printf("\n");
  printf(" 2D heat equation (explicit) demo\n");
  printf("\n");

  // scheme parameters
  printf(" number of space nodes = "); scanf("%d", &I);
  printf(" number of time levels = "); scanf("%d", &N);
  printf("\n");
  nu = 0.25; // scheme parameter (<=0.25 for stability)

  // start time
  wtime = clock();

  // distance between space nodes
  J  = I;
  dx = 1.0 / ((double) I - 1);
  dy = dx;

  // wave numbers
  wnx = 1;
  wny = 2;

  // compute pi
  pi = 4.0*atan(1.0);

  // allocate memory
  x  = (double *) malloc( I*J * sizeof (double) );
  y  = (double *) malloc( I*J * sizeof (double) );
  u  = (double *) malloc( I*J * sizeof (double) );
  uo = (double *) malloc( I*J * sizeof (double) );

  // initial condition
  for (j=0; j<J; j++) {
    for (i=0; i<I; i++) {
      k    = fdin(I,J,i,j);
      x[k] = i*dx;
      y[k] = j*dy;
      u[k] = sin(((double) wnx)*pi*x[k])*sin(((double) wny)*pi*y[k]);
    }
  }


  // time loop
  for (n=0; n<N; n++) {
    // store old solution
    for (k=0; k<I*J; k++) {
      uo[k] = u[k];
    }

    // finite difference scheme
    for (j=1; j<J-1; j++) {
      for (i=1; i<I-1; i++) {
	k    = fdin(I,J,i,j);    // centre point
	kn   = k + I;            // north point
        ks   = k - I;            // south point
        ke   = k + 1;            // east point
	kw   = k - 1;            // west point

	u[k] = uo[k] + nu*(uo[kn]+uo[ks]+uo[ke]+uo[kw]-4.0*uo[k]);
      }
    }
  }


  // write to file
  fid = fopen ("heat.out", "w");
  for (j=0; j<J; j++) {
    for (i=0; i<I; i++) {
      k = fdin(I,J,i,j);
      fprintf (fid, "%20.16e %20.16e %20.16e\n", x[k],y[k],u[k]);
    }
  }
  fclose(fid);

  // measure error
  T   = N*nu*dx*dx;        // final simulated time
  rms = 0.0;               // rms error
  fac = ((double) wnx)*((double) wnx) + ((double) wny)*((double) wny);
  fac = exp(-fac*pi*pi*T); // exponential factor in analytic soln

  for (j=0; j<J; j++) {
    for (i=0; i<I; i++) {
      k    = fdin(I,J,i,j);
      de   = u[k] - fac*sin(((double) wnx)*pi*x[k])*sin(((double) wny)*pi*y[k]);
      rms += de*de;
    }
  }

  // free memory
  free (x);
  free (y);
  free (u);
  free (uo);


  // report time
  wtime = clock() - wtime;
  printf("\n");
  printf(" wall clock elapsed time = %f sec\n", ((double) wtime) / CLOCKS_PER_SEC );
  printf(" final simulated time    = %10.6e\n", T);
  printf(" error rms               = %10.6e\n", sqrt(rms)/((double) I));
  printf("\n");

  return 0;
}
/*
  end
*/
