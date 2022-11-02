
/*

  name:     HEAT.C

  synopsis: 3D time-dependent heat equation solved using
            explicit finite-differencing

  version:  OpenMP

  doc:      heat.pdf

  compile:  icc -O2 -o heat heat.c -limf
            gcc -O2 -o heat heat.c -lm

 */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>
# include <string.h>


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
     MAIN
 ================================================================
*/

int main ( int argc, char** argv ) {

  // number of discrete points (x,y,z and t)
  int     I=650,J,K, N=100;
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
  double  *u,*uo,               // solution storage
          *orig, *dest, *temp;  // pointers to manage storage
  // other variables
  double  dx,dy,dz, dt,         // discretisation steps
          pi,                   // pi
          T,                    // final simulated time
          nu,                   // nu = dt/(dx*dx) = dt/(dy*dy) = dt/(dz*dz)
          xs,ys,zs,             // current x,y,z coords
          de,
          rms,                  // rms error
          fac;
  // walltime
  double  t_init, t_iter;
  // extras
  FILE    *fid;
  char    outfile_name[32];
  int     file_output=0;


  // print header
  printf("\n 3D heat equation demo using OpenMP\n");

  // process arguments
  if (argc > 1) I = atoi(argv[1]);
  if (argc > 2) N = atoi(argv[2]);
  if (argc > 3) { strcpy(outfile_name, argv[3]); file_output = 1; }

  // scheme parameter (<= 1/6 for stability)
  nu = 1.0/6.0;

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
  u  = (double *) malloc( I*J*K * sizeof (double) );
  uo = (double *) malloc( I*J*K * sizeof (double) );


  //
  // ... initialisation
  //
  t_init = omp_get_wtime ();

  # pragma omp parallel for default(none) \
               shared(I,J,K, dx,dy,dz, wnx,wny,wnz, pi, u) \
               private(i,j,k,s, xs,ys,zs)
  for (k=0; k<K; k++) {
    for (j=0; j<J; j++) {
      for (i=0; i<I; i++) {
	s    = fdin(I,J,K, i,j,k);
	xs   = i*dx;
	ys   = j*dy;
	zs   = k*dz;
	u[s] = sin(((double) wnx)*pi*xs)*sin(((double) wny)*pi*ys)*sin(((double) wnz)*pi*zs);
      }
    }
  }

  t_init = omp_get_wtime () - t_init;


  //
  // ... iterations
  //

  // start solution time
  t_iter = omp_get_wtime ();

  // solution copy
  # pragma omp parallel for \
           default(none) \
           shared(u,uo,I,J,K,) \
           private(i)
  for (i=0; i<I*J*K; i++) {
    uo[i] = u[i];
  }

  // work pointers
  orig = u;
  dest = uo;

  // time loop
  for (n=0; n<N; n++) {

    // finite difference scheme (no boundary conditions)
    # pragma omp parallel for \
             default(none) \
             shared(I,J,K, nu,  orig,dest,temp) \
             private(i,j,k, s,sim,sip,sjm,sjp,skm,skp)
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

          dest[s] = orig[s] + nu * ( orig[sip] - 2.0*orig[s] + orig[sim]
                                   + orig[sjp] - 2.0*orig[s] + orig[sjm]
                                   + orig[skp] - 2.0*orig[s] + orig[skm] );
        }
      }
    }

    // swap pointers
    temp = orig;
    orig = dest;
    dest = temp;
  }


  // copy to u, if needed
  if (dest == u) {
    # pragma omp parallel for \
             default(none) \
             shared(u,uo,I,J,K,) \
             private(i)
    for (i=0; i<I*J*K; i++) {
      u[i] = uo[i];
    }
  }

  // stop solution time
  t_iter = omp_get_wtime () - t_iter;


  //
  // ... solution to file
  //
  if (file_output) {
    fid = fopen (outfile_name, "w");
    for (k=0; k<K; k++) {
      for (j=0; j<J; j++) {
        for (i=0; i<I; i++) {
          s = fdin(I,J,K, i,j,k);
          fprintf (fid, "%20.16e\n", u[s]);
        }
      }
    }
    fclose(fid);
  }


  //
  // ... error
  //
  T   = N*nu*dx*dx;             // final simulated time
  rms = 0.0;                    // rms error
  fac = ((double) wnx)*((double) wnx)
      + ((double) wny)*((double) wny)
      + ((double) wnz)*((double) wnz);
  fac = exp(-fac*pi*pi*T);      // exponential factor in analytic solution

  # pragma omp parallel for default(none) \
               shared(I,J,K, u, fac, dx,dy,dz, wnx,wny,wnz, pi) \
               private(i,j,k, s, de, xs,ys,zs) \
               reduction(+:rms)
  for (k=1; k<K-1; k++) {
    for (j=0; j<J; j++) {
      for (i=0; i<I; i++) {
	s    = fdin(I,J,K, i,j,k);
	xs   = i*dx;
	ys   = j*dy;
	zs   = k*dz;
        de   = u[s] - fac*sin(((double) wnx)*pi*xs)*sin(((double) wny)*pi*ys)*sin(((double) wnz)*pi*zs);
        rms += de*de;
      }
    }
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
  printf("\terror rms\t\t\t = %10.6e\n", sqrt(rms/((double) I*J*K)));

  return EXIT_SUCCESS;
}

/*
  end
*/
