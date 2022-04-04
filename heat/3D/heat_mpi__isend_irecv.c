
/*

  name:     HEAT_MPI.C

  synopsis: 3D time-dependent heat (difussion) equation solved
            using explicit finite-differencing

  version:  parallel processing using MPI,
            the division of work is along the z coordinate,
            each process performing computations on points
            with indices i=1,...,I, j=1,...,J and k=k1,...,k2

  doc:      heat.pdf

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
          *u,*uo;               // solution (u "current" and u "old")
  // other variables
  double  dx,dy,dz, dt,         // discretisation steps
          pi,                   // pi
          T,                    // final simulated time
          nu,                   // nu = dt/(dx*dx)
          de,
          rms,                  // rms error
          rmsp,                 // rms error (process local)
          fac;
  // walltime
  double  wtime_sol,            // solution time
          wtime_tot;            // total time
  // iteration output control
  int     Nout;
  // extras
  FILE    *fileid;
  int     fileIO;
  char    filename[32];
  // extra k indices
  int     k1,k2, Kp;
  // MPI variables
  int     ip,np,                // process id, total number of processes
          ip_lo,ip_hi,          // id of neighbouring processes
          s_send,s_recv;        // halo send/receive indices
  MPI_Status  status[4];
  MPI_Request request[4];

  // init MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &np);
  MPI_Comm_rank (MPI_COMM_WORLD, &ip);

  // start total time
  if (ip == 0) {
    wtime_tot = MPI_Wtime ( );
  }

  # ifdef DEBUG
  // print header
  if (ip == 0) {
    printf("\n");
    printf(" 3D heat equation demo\n");
    printf("\n");
  }
  # endif

  // process arguments
  if (argc >= 3) {
    // discretisation parameters
    I = atoi(argv[1]);
    N = atoi(argv[2]);

    // write solution to file?
    fileIO = 0;
    if (argc == 4) {
      fileIO = atoi(argv[3]);
    }
  } else {
    if (ip == 0) printf(" *** error: insufficient arguments, two expected\n");
    MPI_Abort (MPI_COMM_WORLD, -1);
  }

  // check arguments
  if (I < 2) {
    if (ip == 0) printf(" *** error: number of space intervals too small\n");
    MPI_Abort (MPI_COMM_WORLD, -2);
  }

  if (N < 1) {
    if (ip == 0) printf(" *** error: number of time intervals too small\n");
    MPI_Abort (MPI_COMM_WORLD, -3);
  }

  if (I/np < 2) {
    if (ip == 0) printf(" *** error: MPI k-partition too thin\n");
    MPI_Abort (MPI_COMM_WORLD, -4);
  }


  # ifdef DEBUG
  // scheme parameters
  if (ip == 0) {
    printf(" number of space nodes = %d x %d x %d\n", I,I,I);
    printf(" number of time levels = %d\n", N);
    printf(" writing solution to file = %c\n", fileIO?'y':'n');
    printf("\n");
  }
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

  // global index limits
  k1 = (K * ip)   / np;
  k2 =  K *(ip+1) / np - 1;

  // process array size
  Kp = k2 - k1 + 1;

  // allocate memory (dimension is IxJx(Kp+2), with 2 extra halo points in z)
  x    = (double *) malloc( I*J*(Kp+2) * sizeof (double) );
  y    = (double *) malloc( I*J*(Kp+2) * sizeof (double) );
  z    = (double *) malloc( I*J*(Kp+2) * sizeof (double) );
  u1   = (double *) malloc( I*J*(Kp+2) * sizeof (double) );
  u2   = (double *) malloc( I*J*(Kp+2) * sizeof (double) );

  // pointers to solution storage
  u  = u1;
  uo = u2;

  // initial condition
  for (k=1; k<=Kp; k++) {
    for (j=0; j<J; j++) {
      for (i=0; i<I; i++) {
  	s    = fdin(I,J,Kp+2, i,j,k);
  	x[s] = i*dx;
  	y[s] = j*dy;
  	z[s] = (k1+k-1)*dz;
  	u[s] = sin(((double) wnx)*pi*x[s])*sin(((double) wny)*pi*y[s])*sin(((double) wnz)*pi*z[s]);
      }
    }
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

  // output every Nout iterations
  Nout = MIN( MAX(N/10, 1), 100);

  // establish finite difference limits along z
  k1 = 1;
  k2 = Kp;

  // first and last process do not update the domain boundary
  if (ip==0)      k1 = 2;
  if (ip==(np-1)) k2 = Kp-1;

  // start solution time
  if (ip == 0) {
    wtime_sol = MPI_Wtime ( );
  }

  // time loop
  for (n=0; n<N; n++) {
    // output
    if (n % Nout == 0 && ip == 0) printf (" iteration %10d\n", n);

    //
    // ----- exchange halos (process inter-communication)
    //

    // non-blocking send (halo data)
    MPI_Isend (&u[I*J    ],    I*J, MPI_DOUBLE, ip_lo, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Isend (&u[I*J* Kp],    I*J, MPI_DOUBLE, ip_hi, 0, MPI_COMM_WORLD, &request[1]);

    // non-blocking receive (halo data)
    MPI_Irecv (&u[0],          I*J, MPI_DOUBLE, ip_lo, 0, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv (&u[I*J*(Kp+1)], I*J, MPI_DOUBLE, ip_hi, 0, MPI_COMM_WORLD, &request[3]);

    // swap pointers to solution storage (u points to updates, uo to previous step solution)
    u3 = u;
    u  = uo;
    uo = u3;

    //
    // ----- apply scheme
    //
    // finite difference scheme (core of the domain)
    for (k=k1+1; k<k2; k++) {
      for (j=1; j<J-1; j++) {
	for (i=1; i<I-1; i++) {
	  s    = fdin(I,J,Kp+2, i,j,k);   // (i,j,k) <-- centre point
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

    // communication sync point
    MPI_Waitall (4, request, status);

    // finite difference scheme (first k slice)
    k = k1;
    for (j=1; j<J-1; j++) {
      for (i=1; i<I-1; i++) {
        s    = fdin(I,J,Kp+2, i,j,k);   // (i,j,k) <-- centre point
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

    // finite difference scheme (last k slice)
    k = k2;
    for (j=1; j<J-1; j++) {
      for (i=1; i<I-1; i++) {
        s    = fdin(I,J,Kp+2, i,j,k);   // (i,j,k) <-- centre point
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

  // stop solution time
  if (ip == 0) {
    wtime_sol = MPI_Wtime ( ) - wtime_sol;
  }

  // measure error
  T    = N*nu*dx*dx;            // final simulated time
  rmsp = 0.0;                   // rms error (process local)
  fac = ((double) wnx)*((double) wnx)
      + ((double) wny)*((double) wny)
      + ((double) wnz)*((double) wnz);
  fac = exp(-fac*pi*pi*T);      // exponential factor in analytic solution

  for (k=1; k<=Kp; k++) {
    for (j=0; j<J; j++) {
      for (i=0; i<I; i++) {
	s    = fdin(I,J,Kp+2, i,j,k);
        de   = u[s] - fac*sin(((double) wnx)*pi*x[s])*sin(((double) wny)*pi*y[s])*sin(((double) wnz)*pi*z[s]);
        rmsp+= de*de;
      }
    }
  }

  // add up all local rmsp into global rms
  MPI_Reduce (&rmsp,&rms, 1,MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);

  // write to file
  if (fileIO) {
    sprintf (filename, "heat_mpi.out_p%d", ip);
    fileid = fopen (filename, "w");
    for (k=1; k<=Kp; k++) {
      for (j=0; j<J; j++) {
        for (i=0; i<I; i++) {
          s    = fdin(I,J,Kp+2, i,j,k);
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
  if (ip == 0) {
    wtime_tot = MPI_Wtime ( ) - wtime_tot;
  }

  // report
  if (ip == 0) {
    printf("\n");
    printf(" wall clock elapsed time (solution) = %f sec\n", wtime_sol);
    printf(" wall clock elapsed time (total)    = %f sec\n", wtime_tot);
    printf(" error rms                          = %10.6e\n", sqrt(rms) * dx*dy*dz);
    printf("\n");
  }

  // finalise MPI
  MPI_Finalize ( );

}

/*
  end
*/
