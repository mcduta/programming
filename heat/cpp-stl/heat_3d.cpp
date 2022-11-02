
/*

  name:     HEAT.C

  synopsis: 3D time-dependent heat equation solved using
            explicit finite-differencing

  version:  C++ STL

  doc:      heat.pdf

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


/*
 ================================================================
     MAIN
 ================================================================
*/

int main ( int argc, char *argv[] )
{
  // number of discrete points (x,y,z and t) (index_t defined in iterator.hpp
  index_t     I=650,J,K, N=100;
  // wave numbers (x,y,z);
  int         wnx,wny,wnz;
  // space and solution arrays
  double      *u,*uo,               // solution storage
              *orig, *dest;         // pointers to manage storage
  // other
  double      dx,dy,dz,             // discretisation stepspi,T,
              pi,                   // pi
              T,                    // final simulated time
              nu,                   // nu = dt/(dx*dx) = dt/(dy*dy) = dt/(dz*dz)
              rms,                  // rms error
              fac;

  // file output (debug)
  bool        file_output=false;
  std::string outfile_name;

  // set execution policy for STL
  auto        std_exec_policy = std::execution::par_unseq;


  // print header
  std::cout << std::endl << " 3D heat equation demo using C++ STL" << std::endl;

  // process arguments
  if (argc > 1) I = std::stoi(argv[1]);
  if (argc > 2) N = std::stoi(argv[2]);
  if (argc > 3) { outfile_name = argv[3]; file_output = true; }

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
  pi = 4.0*std::atan(1.0);

  // allocate memory
  u  = new double [I*J*K];
  uo = new double [I*J*K];


  //
  // ... initialisation
  //
  auto t_start = std::chrono::high_resolution_clock::now();

  std::for_each (std_exec_policy,
                 array_iterator(0), array_iterator(I*J*K),
                 [=](const int s) {
                   // linear index s to 3D indices (i,j,k)
                   auto i = s % I;
                   auto j = s / I;
                   auto k = j / J;
                   j = j % J;
                   // coodinates (xs,ys,zs) at point s (i,j,k)
                   auto xs = double(i)*dx;
                   auto ys = double(j)*dy;
                   auto zs = double(k)*dz;
                   // initial field value at point s
                   u[s] = sin(double(wnx)*pi*xs)
                        * sin(double(wny)*pi*ys)
                        * sin(double(wnz)*pi*zs);
                 });

  auto t_stop = std::chrono::high_resolution_clock::now();
  auto t_init = std::chrono::duration_cast<std::chrono::duration<double>>(t_stop - t_start);


  //
  // ... iterations
  //
  // start iteration timing
  t_start = std::chrono::high_resolution_clock::now();

  // solution copy
  std::copy (std_exec_policy, u, u+I*J*K, uo);

  // work pointers
  orig = u;
  dest = uo;


  // time loop
  for (auto n=0; n<N; n++) {
    // finite difference scheme
    std::for_each (std_exec_policy,
                   array_iterator(0), array_iterator(I*J*K),
                   [=](const int s) {
                     // linear index s to 3D indices (i,j,k)
                     auto i = s % I;
                     auto j = s / I;
                     auto k = j / J;
                     j = j % J;
                     // update
                     if ((i > 0 && i < I-1) && (j > 0 && j < J-1) && (k > 0 && k < K-1)) {
                       auto sim  = s - 1;                   // (i-1,j,k)
                       auto sip  = s + 1;                   // (i+1,j,k)
                       auto sjm  = s - I;                   // (i,j-1,k)
                       auto sjp  = s + I;                   // (i,j+1,k)
                       auto skm  = s - I*J;                 // (i,j,k-1)
                       auto skp  = s + I*J;                 // (i,j,k+1)
                       dest[s] = orig[s] + nu * ( orig[sip] - 2.0*orig[s] + orig[sim]
                                                + orig[sjp] - 2.0*orig[s] + orig[sjm]
                                                + orig[skp] - 2.0*orig[s] + orig[skm] );
                     }
                   });

    // swap pointers
    std::swap (orig, dest);
  }

  // copy to u, if needed
  if (dest == u) {
    std::copy (std_exec_policy, uo, uo+I*J*K, u);
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
      std::copy (u, u+I*J*K, std::ostream_iterator<float>(outfile, "\n"));
      outfile.close();
    } else {
      std::cerr << " *** error: unable to open output file";
    }
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

  rms = std::transform_reduce (std_exec_policy,
                               array_iterator(0), array_iterator(I*J*K),
                               0.0, std::plus<double>(),
                               [=](const int s) {
                                 // linear index s to 3D indices (i,j,k)
                                 auto i = s % I;
                                 auto j = s / I;
                                 auto k = j / J;
                                 j = j % J;
                                 // coodinates (xs,ys,zs) at point s (i,j,k)
                                 auto xs = double(i)*dx;
                                 auto ys = double(j)*dy;
                                 auto zs = double(k)*dz;
                                 // analytic solution at point s (i,j,k)
                                 auto us = fac * sin(double(wnx)*pi*xs)
                                               * sin(double(wny)*pi*ys)
                                               * sin(double(wnz)*pi*zs);
                                 // difference between analytic and computed solution
                                 auto de = u[s] - us;
                                 // return square of difference
                                 return de*de;
                               });

  // free memory
  delete [] u;
  delete [] uo;


  //
  // ... report
  //
  std::cout << "\twall clock elapsed time\t\t= " << t_init.count() << " sec (initialisation)" << std::endl;
  std::cout << "\t\t\t\t\t= " << t_iter.count() << " sec (iterations total)" << std::endl;
  std::cout << "\tfinal simulated time\t\t= " << T << std::endl;
  std::cout << "\terror rms\t\t\t= " << std::sqrt(rms/((double) J)) << std::endl;

  return EXIT_SUCCESS;
}
/*
  end
*/
