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
  std::cout << " *** starting " << sol->niter << " iterations" << std::endl;
  double wtime = omp_get_wtime ();
  sol -> iterate (sol->niter);
  wtime = omp_get_wtime () - wtime;
  std::cout << " *** finished " << sol->niter << " iterations in " << wtime << std::endl;

  // write solution to file
  sol -> dump ("sol.dat");

  // cleanup
  delete sol;

  return EXIT_SUCCESS;

}
