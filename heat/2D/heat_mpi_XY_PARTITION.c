
/*

  name:     HEAT_MPI

  synopsis: 2D time-dependent heat equation solved using
            explicit finite-differencing, serial version

            NB: this version of HEAT_MPI partitions the
            2D grid in both X and Y directions

  model:    The PDE is

                       2      2
                 du   d u    d u
                 -- = ---2 + ---2
                 dt   d x    d x

            defined on the unit square [0, 1] X [0, 1]
            and with the boundary conditions

                 u(0,y,t) = u(1,y,t) = 0
                 u(x,0,t) = u(x,1,t) = 0
                 u(x,y,0) = sin(wnx*pi*x)*sin(wny*pi*y)

            The anaytic solution is

                 u(x,y,t) = sin(wnx*pi*x)*sin(wny*pi*y)
                          * exp(-(wnx**2 + wny**2)*pi**2*t)

            The two wave numbers wnx, wny break the x/y
            symmetry of the problem.

            The finite difference discretisation is on a
            grid with same number of points in both coordinates
            (dx = dy).  The finite difference stencil is:

            time step n+1                 u(i,j,n+1)
                                              |
                                              | u(i,j+1,n)
                                              |    _/
                                              |  _/
                                              | /
            time step n  u(i-1,j,n) ----- u(i,j,n) ----- u(i+1,j,n)
                                           _/
                                         _/
                                        /
                                  u(i,j-1,n)

            The solution scheme is

                 u(i,j,n+1) = u(i,j,n)
                            + nu * ( u(i-1,j,n) + u(i+1,j,n)
                                   + u(i,j-1,n) + u(i,j+1,n)
                                   - 4*u(i,j,n) )

            with the stability condition

                 nu = dt/dx**2 <= 1/4


  compile:  mpicc -O2 -o heat_mpi heat_mpi.c -limf
            mpicc -O2 -o heat_mpi heat_mpi.c -lm

  run:      sh heat_mpi.run num_procs
            The bash script HEAT_MPI.RUN runs the executable and
            joins the output files into a single one.

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

// local functions
void part2 (const int, int *,int *);


int main ( int argc, char *argv[] )
{
  // MPI variables
  int        ip,ipx,ipy, np,npx,npy;

  // scheme variables
  int        I,J,N,
             wnx,wny,
             i,j,k,n,
             i1,i2,j1,j2,
             Ip,Jp,
             ipn,ips,ipe,ipw,
             kn,ks,ke,kw, koff,
             ksnd,krcv;
  double     pi,T,
             *x,*y,*u,*uo,
             *bsnd,*brcv,
             dx,dy,dt,nu;
  double     de,rms,rmsp,fac;

  // other variables
  MPI_Status status;
  FILE       *fid;
  char       fnm[32];
  clock_t    wtime;


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

  // 2D data partition size
  part2(np, &npx,&npy);

  if ((npx*npy) == 0) {
    if (ip == 0) printf(" ***error: cannot partition %d processes\n", np);
    MPI_Finalize ( );
    return -1;
  }

  if (ip == 0) {
    printf(" 2D process partition %d = %d X %d\n", np,npx,npy);
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

/*   // empty output file */
/*   if (ip == 0) { */
/*     fid = fopen ("heat_mpi.out", "w"); */
/*     fclose (fid); */
/*   } */

  // distance between space nodes
  J  = I;
  dx = 1.0 / ((double) I - 1);
  dy = dx;

  // wave numbers
  wnx = 1;
  wny = 2;

  // compute pi
  pi = 4.0*atan(1.0);

  // 2D data block id
  ipx = ip % npx;
  ipy = ip / npx;

  // global index limits
  i1 = (I * ipx)   / npx;
  i2 =  I *(ipx+1) / npx - 1;

  j1 = (J * ipy)   / npy;
  j2 =  J *(ipy+1) / npy - 1;

  // process array size
  Ip = i2 - i1 + 1;
  Jp = j2 - j1 + 1;

  printf(" process %d: indices are (%d-%d)x(%d-%d)\n",ip,i1,i2,j1,j2);

  // barrier
  MPI_Barrier (MPI_COMM_WORLD);

  // allocate memory
  x    = (double *) malloc( (Ip+2)*(Jp+2) * sizeof (double) );
  y    = (double *) malloc( (Ip+2)*(Jp+2) * sizeof (double) );
  u    = (double *) malloc( (Ip+2)*(Jp+2) * sizeof (double) );
  uo   = (double *) malloc( (Ip+2)*(Jp+2) * sizeof (double) );

  bsnd = (double *) malloc(  Jp           * sizeof (double) );
  brcv = (double *) malloc(  Jp           * sizeof (double) );


  // initial condition
  for (j=1; j<=Jp; j++) {
    for (i=1; i<=Ip; i++) {
      k    = fdin(Ip+2,Jp+2,i,j);
      x[k] = (i1 + i - 1)*dx;
      y[k] = (j1 + j - 1)*dy;
      u[k] = sin(((double) wnx)*pi*x[k])*sin(((double) wny)*pi*y[k]);
    }
  }

  // outer halo (around boundary)
  koff = (Jp+1)*(Ip+2);
  for (i=0; i<Ip+2; i++) {
    u[i]      = 0.0;
    u[i+koff] = 0.0;
  }

  koff = Ip+1;
  for (j=1; j<=Jp; j++) {
    k         = j*(Ip+2);
    u[k]      = 0.0;
    u[k+koff] = 0.0;
  }


  // establish process ids for communication
  if (ipx == 0) {
    ipw = MPI_PROC_NULL;
  } else {
    ipw = fdin(npx,npy,ipx-1,ipy);
  }

  if (ipx == npx-1) {
    ipe = MPI_PROC_NULL;
  } else {
    ipe = fdin(npx,npy,ipx+1,ipy);
  }

  if (ipy == 0) {
    ips = MPI_PROC_NULL;
  } else {
    ips = fdin(npx,npy,ipx,ipy-1);
  }

  if (ipy == npy-1) {
    ipn = MPI_PROC_NULL;
  } else {
    ipn = fdin(npx,npy,ipx,ipy+1);
  }


  // time loop
  for (n=0; n<N; n++) {

    //
    // ----- exchange halos (process inter-communication)
    //
    // send/receive from/to partition south
    krcv = fdin(Ip+2,Jp+2,1,0);
    ksnd = fdin(Ip+2,Jp+2,1,1);

    MPI_Sendrecv(&u[ksnd], Ip,MPI_DOUBLE, ips, 0,
                 &u[krcv], Ip,MPI_DOUBLE, ips, 0,
                 MPI_COMM_WORLD, &status);


    // send/receive from/to partition north
    krcv = fdin(Ip+2,Jp+2,1,Jp+1);
    ksnd = fdin(Ip+2,Jp+2,1,Jp);

    MPI_Sendrecv(&u[ksnd], Ip,MPI_DOUBLE, ipn, 0,
                 &u[krcv], Ip,MPI_DOUBLE, ipn, 0,
                 MPI_COMM_WORLD, &status);

    // send/receive from/to partition west
    for (j=1; j<=Jp; j++) {
      k = fdin(Ip+2,Jp+2,1,j);
      bsnd[j-1] = u[k];
    }

    MPI_Sendrecv(bsnd, Jp,MPI_DOUBLE, ipw, 0,
                 brcv, Jp,MPI_DOUBLE, ipw, 0,
                 MPI_COMM_WORLD, &status);

    for (j=1; j<=Jp; j++) {
      k = fdin(Ip+2,Jp+2,0,j);
      u[k] = brcv[j-1];
    }

    // send/receive from/to partition east
    for (j=1; j<=Jp; j++) {
      k = fdin(Ip+2,Jp+2,Ip,j);
      bsnd[j-1] = u[k];
    }

    MPI_Sendrecv(bsnd, Jp,MPI_DOUBLE, ipe, 0,
                 brcv, Jp,MPI_DOUBLE, ipe, 0,
                 MPI_COMM_WORLD, &status);

    for (j=1; j<=Jp; j++) {
      k = fdin(Ip+2,Jp+2,Ip+1,j);
      u[k] = brcv[j-1];
    }

    // store old solution
    for (k=1; k<(Ip+2)*(Jp+2); k++) {
      uo[k] = u[k];
    }

    // finite difference scheme
    for (j=1; j<=Jp; j++) {
      for (i=1; i<=Ip; i++) {
        k    = fdin(Ip+2,Jp+2,i,j);  // centre point
        kn   = k + Ip + 2;           // north point
        ks   = k - Ip - 2;           // south point
        ke   = k + 1;                // east point
        kw   = k - 1;                // west point

        u[k] = uo[k] + nu*(uo[kn]+uo[ks]+uo[ke]+uo[kw]-4.0*uo[k]);
      }
    }

    // homogeneous Dirichlet boundary conditions
    if (ipx == 0) {
      //      for (j=2; j<=Jp-1; j++) {
      for (j=1; j<=Jp; j++) {
        k    = fdin(Ip+2,Jp+2,1,j);
        u[k] = 0.0;
      }
    }

    if (ipx == npx-1) {
      //      for (j=2; j<=Jp-1; j++) {
      for (j=1; j<=Jp; j++) {
        k    = fdin(Ip+2,Jp+2,Ip,j);
        u[k] = 0.0;
      }
    }

    if (ipy == 0) {
      for (i=1; i<=Ip; i++) {
        k    = fdin(Ip+2,Jp+2,i,1);
        u[k] = 0.0;
      }
    }

    if (ipy == npy-1) {
      for (i=1; i<=Ip; i++) {
        k    = fdin(Ip+2,Jp+2,i,Jp);
        u[k] = 0.0;
      }
    }

  }


  // write to file
  sprintf (fnm, "heat_mpi.out_p%d", ip);

  fid = fopen (fnm, "w");
  for (j=1; j<=Jp; j++) {
    for (i=1; i<=Ip; i++) {
      k = fdin(Ip+2,Jp+2,i,j);
      fprintf (fid, "%20.16e %20.16e %20.16e\n", x[k],y[k],u[k]);
    }
  }
  fclose(fid);

  // measure error
  T    = N*nu*dx*dx;        // final simulated time
  rmsp = 0.0;               // rms error
  fac  = ((double) wnx)*((double) wnx) + ((double) wny)*((double) wny);
  fac  = exp(-fac*pi*pi*T); // exponential factor in analytic soln

  for (j=1; j<=Jp; j++) {
    for (i=1; i<=Ip; i++) {
      k     = fdin(Ip+2,Jp+2,i,j);
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
  free (bsnd);
  free (brcv);

  // report time
  if (ip == 0) {
    wtime = clock() - wtime;
    printf("\n");
    printf(" wall clock elapsed time = %f sec\n", ((double) wtime) / CLOCKS_PER_SEC );
    printf(" final simulated time    = %10.6e\n", T);
    printf(" error rms               = %10.6e\n", sqrt(rms)/((double) I));
    printf("\n");
  }

  // finalise MPI
  MPI_Finalize ( );

  return 0;
}


// ====================================================================
//
//     PART2 - compute size of 2D partitions
//
// ====================================================================

void part2 (const int np, int *npx, int *npy) {

  switch (np) {
  case  1:
    *npx =  1; *npy =  1; break;
  case  2:
    *npx =  1; *npy =  2; break;
  case  4:
    *npx =  2; *npy =  2; break;
  case  6:
    *npx =  2; *npy =  3; break;
  case  8:
    *npx =  2; *npy =  4; break;
  case  9:
    *npx =  3; *npy =  3; break;
  case 12:
    *npx =  3; *npy =  4; break;
  case 16:
    *npx =  4; *npy =  4; break;
  case 20:
    *npx =  4; *npy =  5; break;
  case 32:
    *npx =  4; *npy =  8; break;
  case 64:
    *npx =  8; *npy =  8; break;
  default:
    *npx =  0; *npy =  0; break;
  }

}

/*
  end
*/
