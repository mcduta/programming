/*

  project: rd -- finite difference solution of a reaction-diffusion equation
  file:    solution.cpp -- solution class

 */


// ----- headers
# include "solution.hpp"




//====================================================================//
//                                                                    //
//     contructor, destructor, initialisation                         //
//                                                                    //
//====================================================================//



// ================================================================== //
// ----- constructor (default)
// ================================================================== //

solution :: solution () {
  // number of nodes in x and y
  M = 0;
  N = 0;

  // solution storage
  u = NULL;
  v = NULL;
}


// ================================================================== //
// ----- destructor
// ================================================================== //

solution :: ~solution( ) {
  if (u)  delete [] u;
}


// ================================================================== //
// ----- initialiser
// ================================================================== //

void solution :: init (const std::string configFileName) {

  // extra initialisation values
  REAL us, vs;     // values used to initialise solution with spots of const value
  std::size_t ns;  // number of spots

  // read initialisation parameters from file
  std::ifstream configFile (configFileName);

  if (configFile.is_open ()) {
    configFile >> M;
    configFile >> N;
    configFile >> Du;
    configFile >> Dv;
    configFile >> alpha;
    configFile >> beta;
    configFile >> niter;
    configFile >> us;
    configFile >> vs;
    configFile >> ns;
    configFile.close ();

  } else {
    throw " *** error: file " + configFileName + " cannot be opened";
  }

  // random number generation
  std::random_device rand_dev;
  std::mt19937 rand_gen (rand_dev());
  std::normal_distribution <REAL> rand_distr_sol (0, 1);   // normal distribution, mean=0, sigma=1
  std::uniform_real_distribution <REAL> rand_distr_init (0, 1); // unifirm distribution between 0 and 1

  // common indices
  std::size_t m,n,k;

  // step 0: allocate memory
  u = new REAL[M*N];
  v = new REAL[M*N];

  // step 1: random solution everywhere
  # pragma omp parallel shared(u,v,M,N) private(m,n,k)
  for (n=0; n<N; n++) {
      for (m=0; m<M; m++) {
        k  = COL_MAJOR_INDEX_2D(M,N,m,n);
        u[k] = 1.0 + 0.02 * rand_distr_sol (rand_gen);
        v[k] = 0.0 + 0.02 * rand_distr_sol (rand_gen);
      }
  }

  // extra indices
  std::size_t is,     // current spot
              mc, nc, // 2D indices for current spot centre
              ss,     // spot size
              ss2;    // ss**2/4
  // step 2: go over all spots and initialise
  for (is=0; is<ns; is++) {
    // spot centre
    mc = 1 + std::round ( (M-2)*rand_distr_init (rand_gen) ); mc = MIN (M-2, mc);
    nc = 1 + std::round ( (N-2)*rand_distr_init (rand_gen) ); nc = MIN (N-2, nc);
    // spot size
    ss = 1 + std::round ( (MIN(M,N)/20-1)*rand_distr_init (rand_gen) );
    // initialise spot
    ss2 = 1 + ss * ss / 4;
    for (n=MAX(0, nc-ss); n<MIN(N-1,nc+ss); n++) {
      for (m=MAX(0, mc-ss); m<MIN(M-1,mc+ss); m++) {
        if ( (m-mc)*(m-mc)+(n-nc)*(n-nc) < ss2) {
          k = COL_MAJOR_INDEX_2D(M,N,m,n);
          u[k] = us;
          v[k] = vs;
        }
      }
    }
  }

}



//====================================================================//
//                                                                    //
//     data access utils                                              //
//                                                                    //
//====================================================================//



// ================================================================== //
// ----- dump solution to file
// ================================================================== //

void solution :: dump (std::string filename) {

  auto s = sizeof (REAL);
  std::ofstream file;
  file.open  (filename, std::ios::binary | std::ios::out);
  file.write ((char *) &s, sizeof s);
  file.write ((char *) &M, sizeof M);
  file.write ((char *) &N, sizeof N);
  file.write ((char *)  u, sizeof (REAL) * M * N);
  file.write ((char *)  v, sizeof (REAL) * M * N);
  file.close ();

  /*
  file.write ((char *) &M, sizeof M);
  file.write ((char *) &N, sizeof N);
  file.write ((char *) &u, sizeof u);
  file.write ((char *) &v, sizeof v);
  */

  /*
  std::ofstream file;
  file.open(filename);
  file.precision(16);
  file.setf(std::ios::scientific);

  for (std::size_t int k=0; k<M*N; k++) {
  file << u[k] << ", " << v[k] << std::endl;
  }
  file.close();
  */
}



//====================================================================//
//                                                                    //
//     solution numerical manipulation functions                      //
//                                                                    //
//====================================================================//


// ================================================================== //
// ----- solution iterate
// ================================================================== //

void solution :: iterate (std::size_t titer) {

  // indices
  std::size_t m,n,k,kx,i;

  // temp vars
  REAL uvv, Lu,Lv;

  // allocate extra memory
  REAL *ux = new REAL[N*M];
  REAL *vx = new REAL[N*M];

  // extra pointers
  REAL *u1, *v1, *u2, *v2;


  // initially...
  //    (u1,v1) point to (u,v)
  //    (u2,v2) point to (ux,vx)
  u1 = u;  v1 = v;
  u2 = ux; v2 = vx;


  // omp parallel region
  # pragma omp parallel default(none) \
    shared(M,N,u1,v1,u2,v2,titer)     \
    private(i,m,n,k,kx,Lu,Lv,uvv)
  {

    // time iterations
    for (i=0; i<titer; i++) {

      // step 1: solution update
      # pragma omp for
      for (n=1; n<N-1; n++) {
        for (m=1; m<M-1; m++) {
          // central index
          k  = COL_MAJOR_INDEX_2D(M,N,m,n);

          // Laplacian values
          Lu = u1[k+1] + u1[k-1] + u1[k+M] + u1[k-M] - 4.0*u1[k];
          Lv = v1[k+1] + v1[k-1] + v1[k+M] + v1[k-M] - 4.0*v1[k];

          // u*v*v nonlinear term
          uvv = u1[k]*v1[k]*v1[k];

          // update (u2,v2) from (u1,v1)
          u2[k] = u1[k] + Du*Lu - uvv + alpha * (1.0 - u1[k]);
          v2[k] = v1[k] + Dv*Lv + uvv - (alpha + beta) * v1[k] ;

        }
      } // step 1

      // step 2: solution boundary conditions
      # pragma omp for nowait
      for (m=0; m<M; m++) {
        // u[0, :] = u[-2, :]
        k  = COL_MAJOR_INDEX_2D(M,N,m,0);
        kx = k+M*(N-2);
        u2[k] = u2[kx];
        v2[k] = v2[kx];
        // u[-1, :] = u[1, :]
        k  = COL_MAJOR_INDEX_2D(M,N,m,1);
        kx = k+M*(N-2);
        u2[kx] = u2[k];
        v2[kx] = v2[k];
      }
      # pragma omp for
      for (n=0; n<N; n++) {
        // u[:, 0] = u[:, -2]
        k  = COL_MAJOR_INDEX_2D(M,N,0,n);
        kx = k+M-2;
        u2[k] = u2[kx];
        v2[k] = v2[kx];
        // u[:, -1] = u[:, 1]
        k  = COL_MAJOR_INDEX_2D(M,N,1,n);
        kx = k+M-2;
        u2[kx] = u2[k];
        v2[kx] = v2[k];
      } // step 2

      // swap (u1,v1) and (u2,v2)
      # pragma omp single
      {
        REAL *uswp = u2;
        REAL *vswp = v2;
        u2 = u1;
        v2 = v1;
        u1 = uswp;
        v1 = vswp;
      }

    } // time iterations

  } // omp parallel region

  // free extra memory
  if (ux) delete [] ux;
  if (vx) delete [] vx;

}


/*
    // u_dot (from Maini paper)
    // "Two-stage Turing model for generating pigment patterns on the leopard and the jaguar"
    // by R. T. Liu, S. S. Liaw, and P. K. Maini
    # pragma omp for
    for (n=0; n<N; n++) {
      for (m=0; m<M; m++) {
       k  = COL_MAJOR_INDEX_2D(M,N,m,n);

       u1u2   = u1[k]*u2[k];
       u1u2u2 = u1u2*u2[k];

       du1[k] = D1*Lu1[k] + alpha*u1[k] +      u2[k] - r2*u1u2 - alpha*r3*u1u2u2;
       du2[k] = D2*Lu2[k] + gamma*u1[k] + beta*u2[k] + r2*u1u2 + alpha*r3*u1u2u2;
      }
    }
*/


/* end */
