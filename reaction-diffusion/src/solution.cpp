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
              ss;     // spot size

  // step 2: go over all spots and initialise
  for (is=0; is<ns; is++) {
    // spot centre
    mc = 1 + std::round ( (M-2)*rand_distr_init (rand_gen) ); mc = MIN (M-2, mc);
    nc = 1 + std::round ( (N-2)*rand_distr_init (rand_gen) ); nc = MIN (N-2, nc);
    // spot size
    ss = 1 + std::round ( (MIN(M,N)/20-1)*rand_distr_init (rand_gen) );
    // initialise spot
    for (n=MAX(0, nc-ss); n<MIN(N-1,nc+ss); n++) {
      for (m=MAX(0, mc-ss); m<MIN(M-1,mc+ss); m++) {
        k = COL_MAJOR_INDEX_2D(M,N,m,n);
        u[k] = us;
        v[k] = vs;
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

  std::ofstream file;
  file.open  (filename, std::ios::binary | std::ios::out);
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
  std::size_t m,n,k,i;

  // temp vars
  REAL uvv, Lu,Lv;

  // omp parallel region
  # pragma omp parallel default(none) \
                        shared(M,N,u,v,titer) \
                        private(i,m,n,k,Lu,Lv,uvv)
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
          Lu = u[k+1] + u[k-1] + u[k+M] + u[k-M] - 4.0*u[k];
          Lv = v[k+1] + v[k-1] + v[k+M] + v[k-M] - 4.0*v[k];

          // u*v*v nonlinear term
          uvv = u[k]*v[k]*v[k];

          // update
          u[k] += Du*Lu - uvv + alpha * (1.0 - u [k]);
          v[k] += Dv*Lv + uvv - (alpha + beta) * v[k] ;

        }
      } // step 1

      // step 2: solution boundary conditions
      # pragma omp for
      for (m=0; m<M; m++) {
        // u[0, :] = u[-2, :]
        k = COL_MAJOR_INDEX_2D(M,N,m,0);
        u[k] = u[k+M*(N-2)];
        v[k] = v[k+M*(N-2)];
        // u[-1, :] = u[1, :]
        k = COL_MAJOR_INDEX_2D(M,N,m,1);
        u[k+M*(N-2)] = u[k];
        v[k+M*(N-2)] = v[k];
      }
      # pragma omp for
      for (n=0; n<N; n++) {
        // u[:, 0] = u[:, -2]
        k = COL_MAJOR_INDEX_2D(M,N,0,n);
        u[k] = u[k+M-2];
        v[k] = v[k+M-2];
        // u[:, -1] = u[:, 1]
        k = COL_MAJOR_INDEX_2D(M,N,1,n);
        u[k+M-2] = u[k];
        v[k+M-2] = v[k];
      } // step 2

    } // time iterations

  } // omp parallel region

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
