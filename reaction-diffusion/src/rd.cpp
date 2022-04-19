/*

  rd -- finite difference solution of a reaction-diffusion equation
     -- pure compute version

 */


//====================================================================//
//                                                                    //
//                             includes                               //
//                                                                    //
//====================================================================//

// general
# include "rd.hpp"
# include "solution.hpp"
# include "timer.hpp"



//====================================================================//
//                                                                    //
//                            M  A  I  N                              //
//                                                                    //
//====================================================================//


int main (int narg, char **varg) {

  // configuration file
  std::string config_file = "config.in";
  if (narg > 1) config_file = varg[1];

  // ----- new solution
  class solution *sol = new solution;

  // initialise solution
  try {
    sol -> init (config_file);
  } catch (const std::string errMsg) {
    std::cerr << errMsg << std::endl;
    return EXIT_FAILURE;
  }

  // time iterate solution
  auto timer = get_time_start ();
  sol -> iterate (sol->niter);
  auto wtime = get_time_elapsed (timer);

  // count flops
  auto gflops = 24.e-9*(sol->M-2)*(sol->N-2)*(sol->niter);
  std::cout << " ... " << sol->niter << " iterations, " << gflops / wtime << " gflops, " << wtime << " seconds." << std::endl;

/*
  // write solution to file
  std::string filename="sol";
  # ifdef _HDF5
  filename.append (".h5");
  # else
  filename.append (".dat");
  # endif
  sol -> dump (filename);
*/

  // cleanup
  delete sol;

  return EXIT_SUCCESS;

}
