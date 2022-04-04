
/*

  name:     HEAT_MPI

  synopsis: 2D time-dependent heat equation solved using
            explicit finite-differencing

  version:  parallel processing using MPI,
            the division of work is along the y coordinate,
            each process performing computations on points
            with indices i=1,...,I and j=j1,...,j2.

  model:    see heat.tex or heat.pdf

  compile:  mpicc -O2 -o heat_mpi heat_mpi.c -limf
            mpicc -O2 -o heat_mpi heat_mpi.c -lm

  run:      sh heat_mpi.run num_procs
            ( script HEAT_MPI.RUN runs the executable and
              joins the output files into a single one )

 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <mpi.h>

// finite difference matrix indexing
static inline int fdin(const int I,
                       const int J,
                       const int i,
                       const int j)
{
  return j*I + i;
}


int main ( int argc, char *argv[] )
{
  // MPI variables
  int        ip,np,
             ip_lo,ip_hi;

  // scheme variables
  int        I,J,N,
             wnx,wny,
             i,j,k,n,
             j1,j2,
             Jp,
             kn,ks,ke,kw, koff,
             ksnd,krcv;
  double     pi,T,
             *x,*y,*u ,*uo,
             dx,dy,dt,nu;
  double     de,rms,rmsp,fac;

  // other variables
  MPI_Status status;
  FILE       *fid;
  char       fnm[32];
  double     wtime;


  // init MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &np);
  MPI_Comm_rank (MPI_COMM_WORLD, &ip);

  // print header
  if (ip == 0) {
    printf("\n");
    printf(" 2D heat equation MPI (explicit) demo\n");
    printf("\n");
  }

  // start time
  if (ip == 0) {
    wtime = MPI_Wtime ( );
  }

  // scheme variables
  if (ip == 0) {
    putchar('\n');
    printf(" number of space nodes = "); scanf("%d", &I);
    printf(" number of time levels = "); scanf("%d", &N);
    printf("\n\n");
    nu = 0.25; // scheme parameter (<=0.25 for stability)
  }

  // broadcast common variables
  MPI_Bcast (&I,  1,MPI_INT,   0,MPI_COMM_WORLD);
  MPI_Bcast (&N,  1,MPI_INT,   0,MPI_COMM_WORLD);
  MPI_Bcast (&nu, 1,MPI_DOUBLE,0,MPI_COMM_WORLD);

  // distance between space nodes
  J  = I;
  dx = 1.0 / ((double) I - 1);
  dy = dx;

  // wave numbers
  wnx = 1;
  wny = 2;

  // compute pi
  pi = 4.0*atan(1.0);

  // global index limits
  j1 = (J * ip)   / np;
  j2 =  J *(ip+1) / np - 1;

  // process array size
  Jp = j2 - j1 + 1;

  // allocate memory
  x    = (double *) malloc( I*(Jp+2) * sizeof (double) );
  y    = (double *) malloc( I*(Jp+2) * sizeof (double) );
  u    = (double *) malloc( I*(Jp+2) * sizeof (double) );
  uo   = (double *) malloc( I*(Jp+2) * sizeof (double) );

  // initial condition
  for (j=1; j<=Jp; j++) {
    for (i=0; i<I; i++) {
      k    = fdin(I,Jp+2,i,j);
      x[k] =       i     *dx;
      y[k] = (j1 + j - 1)*dy;
      u[k] = sin(((double) wnx)*pi*x[k])*sin(((double) wny)*pi*y[k]);
    }
  }

  // outer halo (around boundary)
  koff = I*(Jp+1);
  for (i=0; i<I; i++) {
    u[i]      = 0.0;
    u[i+koff] = 0.0;
  }

  // establish process ids for communication
  if (ip == 0) {
    ip_lo = MPI_PROC_NULL;
  } else {
    ip_lo = ip - 1;
  }

  if (ip == np-1) {
    ip_hi = MPI_PROC_NULL;
  } else {
    ip_hi = ip + 1;
  }


  // time loop
  for (n=0; n<N; n++) {

    //
    // ----- exchange halos (process inter-communication)
    //
    // send/receive from/to partition low
    ksnd = I;
    krcv = 0;

    MPI_Sendrecv(&u[ksnd], I,MPI_DOUBLE, ip_lo, 0,
                 &u[krcv], I,MPI_DOUBLE, ip_lo, 0,
                 MPI_COMM_WORLD, &status);


    // send/receive from/to partition high
    ksnd = I* Jp;
    krcv = I*(Jp+1);

    MPI_Sendrecv(&u[ksnd], I,MPI_DOUBLE, ip_hi, 0,
                 &u[krcv], I,MPI_DOUBLE, ip_hi, 0,
                 MPI_COMM_WORLD, &status);

    // store old solution
    for (k=1; k<I*(Jp+2); k++) {
      uo[k] = u[k];
    }

    // finite difference scheme
    for (j=1; j<=Jp; j++) {
      for (i=1; i<=I; i++) {
        k    = fdin(I,Jp+2,i,j);     // centre point
        kn   = k + I;                // north point
        ks   = k - I;                // south point
        ke   = k + 1;                // east point
        kw   = k - 1;                // west point

        u[k] = uo[k] + nu*(uo[kn]+uo[ks]+uo[ke]+uo[kw]-4.0*uo[k]);
      }
    }

    // homogeneous Dirichlet boundary conditions
    koff = I-1;
    for (j=1; j<=Jp; j++) {
      k         = j*I;
      u[k]      = 0.0;
      u[k+koff] = 0.0;
    }

    if (ip == 0) {
      koff = I;
      for (i=0; i<I; i++) {
        k    = i + koff;
        u[k] = 0.0;
      }
    }

    if (ip == np-1) {
      koff = Jp*I;
      for (i=0; i<I; i++) {
        k    = i + koff;
        u[k] = 0.0;
      }
    }

  }


/*/////
  // write to file
  sprintf (fnm, "heat_mpi.out_p%d", ip);

  fid = fopen (fnm, "w");
  for (j=1; j<=Jp; j++) {
    for (i=0; i<I; i++) {
      k = fdin(I,Jp+2,i,j);
      fprintf (fid, "%20.16e %20.16e %20.16e\n", x[k],y[k],u[k]);
    }
  }
  fclose(fid);
/////*/


  // measure error
  T    = N*nu*dx*dx;        // final simulated time
  rmsp = 0.0;               // rms error
  fac  = ((double) wnx)*((double) wnx) + ((double) wny)*((double) wny);
  fac  = exp(-fac*pi*pi*T); // exponential factor in analytic soln

  for (j=1; j<=Jp; j++) {
    for (i=0; i<I; i++) {
      k     = fdin(I,Jp+2,i,j);
      de    = u[k] - fac*sin(((double) wnx)*pi*x[k])*sin(((double) wny)*pi*y[k]);
      rmsp += de*de;
    }
  }

  // add up all local rmsp into global rms
  MPI_Reduce (&rmsp,&rms, 1,MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);

  // free memory
  free (x);
  free (y);
  free (u);
  free (uo);

  // report time
  if (ip == 0) {
    wtime = MPI_Wtime() - wtime;
    printf("\n");
    printf(" wall clock elapsed time = %f sec\n", wtime );
    printf(" final simulated time    = %10.6e\n", T);
    printf(" error rms               = %10.6e\n", sqrt(rms)/((double) I));
    printf("\n");
  }

  // finalise MPI
  MPI_Finalize ( );

  return 0;
}

/*
  end
*/
