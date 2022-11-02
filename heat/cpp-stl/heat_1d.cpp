
/*

  name:     HEAT

  synopsis: 1D time-dependent heat equation solved using
            explicit finite-differencing, serial version

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

  compile:  icc -O2 -o heat heat.c -limf
            gcc -O2 -o heat heat.c -lm

 */

# include <execution>
# include <memory>
# include <algorithm>
# include <numeric>
# include <chrono>
# include <iostream>
# include <fstream>
# include <cmath>

# include "iterator.hpp"


int main ( int argc, char *argv[] )
{
  // scheme variables (index_t defined in iterator.hpp
  index_t    J=1<<28, N=100;
  double     pi,T,
             *u,*uo,
             dx,nu;
  double     rms,fac;
  double     *orig, *dest;

  // file output (debug)
  bool        file_output=false;
  std::string outfile_name;

  // set execution policy for STL
  auto std_exec_policy = std::execution::par_unseq;



  // print header
  std::cout << std::endl << " 1D heat equation (explicit) demo using C++ STL" << std::endl;

  // scheme variables
  if (argc > 1) J = std::stoi(argv[1]);
  if (argc > 2) N = std::stoi(argv[2]);
  if (argc > 3) { outfile_name = argv[3]; file_output = true; }

  nu = 0.5; // scheme parameter (<=0.5 for stability)

  // distance between space nodes
  dx = 1.0 / ((double) J - 1);

  // compute pi
  pi = 4.0*std::atan(1.0);

  // allocate memory
  u  = new double [J];
  uo = new double [J];


  //
  // ... initialisation
  //
  auto t_start = std::chrono::high_resolution_clock::now();
  std::for_each (std_exec_policy,
                 array_iterator(0), array_iterator(J),
                 [=](const int j) {
                   auto xj = double(j)*dx;
                   u[j] = sin(pi*xj);
                 });
  auto t_stop = std::chrono::high_resolution_clock::now();
  auto t_init = std::chrono::duration_cast<std::chrono::duration<double>>(t_stop - t_start);


  //
  // ... iterations
  //
  // start iteration timing
  t_start = std::chrono::high_resolution_clock::now();

  // solution copy
  std::copy (std_exec_policy, u, u+J, uo);

  // work pointers
  orig = u;
  dest = uo;


  // time loop
  for (auto n=0; n<N; n++) {
    // finite difference scheme
    u[0] = 0.0;
    std::for_each (std_exec_policy,
                   array_iterator(1), array_iterator(J-1),
                   [=](const int j) {
                     dest[j] = orig[j] + nu*(orig[j-1]-2.0*orig[j]+orig[j+1]);
                   });
    u[J-1] = 0.0;

    // swap pointers
    std::swap (orig, dest);
  }

  // copy to u, if needed
  if (dest == u) {
    std::copy (std_exec_policy, uo, uo+J, u);
  }

  // stop iteration timing
  t_stop = std::chrono::high_resolution_clock::now();
  auto t_iter = std::chrono::duration_cast<std::chrono::duration<double>>(t_stop - t_start);


  //
  // ... solution to file
  //
  if (file_output) {
    std::ofstream outfile (outfile_name);
    if (outfile.is_open()) {
      std::copy (u, u+J, std::ostream_iterator<float>(outfile, "\n"));
      outfile.close();
    } else {
      std::cerr << " *** error: unable to open output file";
    }
  }


  //
  // ... error
  //
  T   = N*nu*dx*dx;    // final simulated time
  fac = exp(-pi*pi*T); // exponential factor in analytic soln

  rms = std::transform_reduce (std_exec_policy,
                               array_iterator(0), array_iterator(J),
                               0.0, std::plus<double>(),
                               [=](const int j) {
                                 auto xj = double(j)*dx;
                                 auto de = u[j] - fac*sin(pi*xj);
                                 return de*de;
                               });

  // free memory
  delete [] u;
  delete [] uo;


  //
  // ... report
  //
  std::cout << "\twall clock elapsed time\t= " << t_init.count() << " sec (initialisation)" << std::endl;
  std::cout << "\t\t\t\t\t= " << t_iter.count() << " sec (iterations total)" << std::endl;
  std::cout << "\tfinal simulated time\t\t= " << T << std::endl;
  std::cout << "\terror rms\t\t\t= " << std::sqrt(rms/((double) J)) << std::endl;

  return EXIT_SUCCESS;
}
/*
  end
*/
