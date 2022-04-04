
/*

  name:     HEAT_MPI

  synopsis: 1D time-dependent heat equation solved using
            explicit finite-differencing, MPI version

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

  compile:  mpicc -O2 -o heat_mpi heat_mpi.c -limf  (icc)
            mpicc -O2 -o heat_mpi heat_mpi.c -lm    (gcc)


  validate: matlab

            t = ...; % input from output of heat_mpi
            d = load('heat_mpi.out');
            [d2,i] = sort(d(:,2));
            x   = d(i,2);
            u   = d(i,3);
            ua  = sin(pi*x).*exp(-pi^2*t);
            plot(x,ua,'g-', x,u,'b-'); legend('analytic', 'computed');
            rms = norm(u-ua)/sqrt(size(u,1));

 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <mpi.h>


int main ( int argc, char *argv[] )
{
  // MPI variables
  int        id, np;

  // scheme variables
  int        J,N,
             j,n,
             j1,j2,Jp;
  double     pi,T,
             *x,*u,*uo,
             dx,dt,nu;
  double     de,rms,rmsp,fac;

  // other variables
  int        id_lo,id_hi,
             tag_lo=0,tag_hi=1;
  MPI_Status status;
  FILE       *fid;
  double     wtime;
/*   MPI_File   fid; */


  // init MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &np);
  MPI_Comm_rank (MPI_COMM_WORLD, &id);

  // start time
  if ( id == 0 ) {
    wtime = MPI_Wtime ( );
  }

  // print header
  if ( id == 0 ) {
    printf("\n");
    printf(" 1D heat equation MPI (explicit) demo\n");
    printf("\n");
  }

  // scheme variables
  if ( id == 0 ) {
    printf(" number of space nodes = "); scanf("%d", &J);
    printf(" number of time levels = "); scanf("%d", &N);
    printf("\n");
    nu = 0.5; // scheme parameter (<=0.5 for stability)
  }

  // broadcast common variables
  MPI_Bcast (&J,  1,MPI_INT,   0,MPI_COMM_WORLD);
  MPI_Bcast (&N,  1,MPI_INT,   0,MPI_COMM_WORLD);
  MPI_Bcast (&nu, 1,MPI_DOUBLE,0,MPI_COMM_WORLD);

  // empty output file
  if ( id == 0 ) {
    fid = fopen ("heat_mpi.out", "w");
    fclose (fid);
  }

  // distance between space nodes
  dx = 1.0 / ((double) J - 1);

  // compute pi
  pi = 4.0*atan(1.0);

  // global index limits
  j1 = (J * id)   / np;
  j2 =  J *(id+1) / np - 1;

  // process array size
  Jp = j2 - j1 + 1;

  // allocate memory
  x  = (double *) malloc( (Jp + 2) * sizeof (double) );
  u  = (double *) malloc( (Jp + 2) * sizeof (double) );
  uo = (double *) malloc( (Jp + 2) * sizeof (double) );

  // initial condition
  for (j=1; j<=Jp; j++) {
    x[j] = (j1 + j - 1)*dx;
    u[j] = sin(pi*x[j]);
  }

  // establish process id's for communication
  if (id == 0) {
    id_lo = MPI_PROC_NULL;
  } else {
    id_lo = id-1;
  }

  if (id == np-1) {
    id_hi = MPI_PROC_NULL;
  } else {
    id_hi = id+1;
  }

  // time loop
  for (n=0; n<N; n++) {

/*     // send u[1] to process (id-1) */
/*     MPI_Send (&u[1],1,    MPI_DOUBLE, id_lo, 0, MPI_COMM_WORLD); */

/*     // send u[J] to process (id+1) */
/*     MPI_Send (&u[Jp],1,   MPI_DOUBLE, id_hi, 0, MPI_COMM_WORLD); */

/*     // receive u[0] from process (id-1) */
/*     MPI_Recv (&u[0],1,    MPI_DOUBLE, id_lo, 0, MPI_COMM_WORLD, &status); */

/*     // receive u[J+1] from process (id+1) */
/*     MPI_Recv (&u[Jp+1],1, MPI_DOUBLE, id_hi, 0, MPI_COMM_WORLD, &status); */


    // send u[1] to process (id-1) and receive u[0] from process (id-1)
    MPI_Sendrecv(&u[1],   1,MPI_DOUBLE, id_lo, 0,
                 &u[0],   1,MPI_DOUBLE, id_lo, 0,
                 MPI_COMM_WORLD, &status);

    // send u[J] to process (id+1) and receive u[J+1] from process (id+1)
    MPI_Sendrecv(&u[Jp],  1,MPI_DOUBLE, id_hi, 0,
                 &u[Jp+1],1,MPI_DOUBLE, id_hi, 0,
                 MPI_COMM_WORLD, &status);


    // store solution
    for (j=0; j<=Jp+1; j++) {
      uo[j] = u[j];
    }

    // finite difference scheme
    for (j=1; j<=Jp; j++) {
      u[j] = uo[j] + nu*(uo[j-1]-2.0*uo[j]+uo[j+1]);
    }

    // boundary conditions
    if (id ==    0) u[1]  = 0.0;
    if (id == np-1) u[Jp] = 0.0;

    // barrier
    MPI_Barrier (MPI_COMM_WORLD);

  }

  // output to disk
  fid = fopen ("heat_mpi.out", "a");
  for (j=1; j<=Jp; j++) {
    fprintf (fid, "%6d %20.16e %20.16e\n", id,x[j],u[j]);
  }
  fclose(fid);

  // measure error
  T    = N*nu*dx*dx;    // final simulated time
  rmsp = 0.0;           // rms error
  fac  = exp(-pi*pi*T); // exponential factor in analytic soln

  for (j=1; j<=Jp; j++) {
    de    = u[j] - fac*sin(pi*x[j]);
    rmsp += de*de;
  }

  // add up all local rmsp into global rms
  MPI_Reduce (&rmsp,&rms, 1,MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);


  // free memory
  free (x);
  free (u);
  free (uo);

  // report time
  if ( id == 0 ) {
    wtime = MPI_Wtime ( ) - wtime;
    printf("\n");
    printf(" wall clock elapsed time = %f sec\n", wtime );      
    printf(" final simulated time    = %10.6e\n", T);
    printf(" error rms               = %10.6e\n", sqrt(rms/J));
    printf("\n");
  }

  // finalise MPI
  MPI_Finalize ( );

  return 0;
}
/*
  end
*/
