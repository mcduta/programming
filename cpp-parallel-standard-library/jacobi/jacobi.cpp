/*
  File:     jacobi.cpp
  Synopsis: Illustrates the implementation of the Jacobi iteration of a 2D vector
            ISO C++ stl parallelism
              * std::for_each
              * std::execution::par_unseq
            It is a demo for the capability of the STL C++17 (and newer) to generate
            code from a single source that can run in parallel on either CPUs or GPUs.
  Details:
             * Implemented using std::vector, e.g.
                 std::vector<float> x(N)
  Build:
             * COMPILER=<nvhpc|rocm> make
  Run:
             * CPU multithreading
               $ ./jac 30000 20000 500 0
                ... jacobi iterations: CPU multithreading
                ... 30000X20000 matrix:    2.23517GB
                ... initialisation:        0.230902 sec
                ... iterations:           81.6217 sec @ 14.6995 Gflops
 
               * GPU offloading
               $ ./jac 30000 20000 500 1
                ... jacobi iterations: GPU offloading
                ... 30000X20000 matrix:    2.23517GB
                ... initialisation:        0.877695 sec
                ... iterations:           13.1077 sec @ 91.5338 Gflops




  module load intel/tbb/2025 gcc/14.2.0__cuda-12
  g++ -I $TBBROOT/include --std=c++23 -o jac jacobi.cpp -Wl,-rpath=$TBBROOT/lib/intel64/gcc4.8 -L $TBBROOT/lib/intel64/gcc4.8 -ltbb

next steps:
* figure out accuracy, why only even numsteps match
* keep jacobi1d implemented woith spans as well as with vector.data()
* implement jacobi in 2d, probably using pointers to vector.data(), just as in the example below
* demo targeting GPU with that

inspired by https://developer.nvidia.com/blog/multi-gpu-programming-with-standard-parallel-c-part-1/
*/


# include <vector>
# include <execution>
# include <memory>
# include <algorithm>
# include <ranges>
# include <chrono>
# include <cmath>
# include <iostream>
# include <fstream>


void jac_init (const size_t NX, const size_t NY, std::vector<float> &u) {

  // sanity check
  if (NX*NY != u.size()) {
    throw std::invalid_argument("jac_init: argument mismatch.");
  }

  // pointers to access matrix data
  float *orig = u.data();

  // index mapping
  // * idx_fuse: map 2D matrix indexing (iX,iY) to linear index i
  // * idx_split: maps linear index i to 2D matrix indexing (iX,iY)
  auto idx_fuse  = [NY](size_t iX, size_t iY) { return iY + NY * iX; };
  auto idx_split = [NY](size_t i) { return std::make_tuple(i / NY, i % NY); };

  // matrix initialisation
  // * u[iX,iY] = min(iX,NX-iX-1) * min(iY,NY-iY-1) / (NX*NY)
  auto index = std::views::iota((size_t) 0, NX*NY);
  std::for_each (std::execution::par_unseq, index.begin(), index.end(),
                 [=](size_t i) {
                   auto [iX, iY] = idx_split(i);
                   orig[idx_fuse(iX,iY)] = float(std::min(iX,NX-iX-1))*float(std::min(iY,NY-iY-1))/float(NX*NY);
		 });

}


//
// ----- jacobi solution iteration
//
void jac_iter (const size_t NX, const size_t NY, const size_t NT, std::vector<float> &u) {

  // sanity check
  if (NX*NY != u.size()) {
    throw std::invalid_argument("jac_init: argument mismatch.");
  }

  // extra vector for copies
  // * at each time step iT, u2 gets updated from u
  //   then the two are swapped
  std::vector<float> u2(NX*NY);

  // pointers to access matrix data
  // * C++ algorithms use the pointers to implement the 4 point stencil of the Jacobi iterations
  float* orig = u.data();
  float* dest = u2.data();

  // index mapping
  // * idx_fuse: map 2D matrix indexing (iX,iY) to linear index i
  // * idx_split: maps linear index i to 2D matrix indexing (iX,iY)
  auto idx_fuse  = [NY](size_t iX, size_t iY) { return iY + NY * iX; };
  auto idx_split = [NY](size_t i) { return std::make_tuple(i / NY, i % NY); };

  // work indices: avoid first and last row of u
  auto index = std::views::iota((size_t) idx_fuse(1,0), idx_fuse(NX-2,NY-1));
  // auto index = std::views::iota(NY, NY -1 + NY*(NX-2));

  // Jacobi iterations
  for (auto iT=0; iT<NT; iT++) {
    // Jacobi update
    std::for_each (std::execution::par_unseq, index.begin(), index.end(),
                   [=](size_t i) {
                   auto [iX, iY] = idx_split(i);
                   if (iY>0 && iY<NY-1) {   // if not at boundary
                     dest[idx_fuse(iX,iY)] = 0.25f * ( orig[idx_fuse(iX+1,iY)] + orig[idx_fuse(iX-1,iY)]
                                                     + orig[idx_fuse(iX,iY+1)] + orig[idx_fuse(iX,iY-1)] );
                   }
       	           });
    // swap pointers
    std::swap (orig, dest);
  }

  // copy u2 to u, if u2 is latest updated
  if (NT%2) {
    std::copy(std::execution::par_unseq, u2.begin(), u2.end(), u.begin());
  }

}


//
// ----- jacobi output
//
void jac_to_disk (const std::string &file_name, const std::vector<float> &data) {
  
  std::ofstream file_handle (file_name);
  if (file_handle.is_open()) {
    std::copy (data.begin(), data.end(), std::ostream_iterator<float>(file_handle, "\n"));
    file_handle.close();
  } else {
    std::cerr << " unable to open file";
  }
}


//
// ----- jacobi output
//

int main (int argc, char** argv) {

  // default sizes
  // * matrix of NX*NY
  // * NT Jacobi iterations
  size_t NX = 20000; // default values mean NX*NY = 320M floats
  size_t NY = 16000; // amounting to 1.19GB
  size_t NT =   100;
  // default option (do not write solution to disk)
  bool dump2disk = 0;

  // set execution policy for STL
  //  auto std_exec_policy = std::execution::par_unseq;

  // optional input sizes
  if (argc > 1) NX = std::stoi(argv[1]);
  if (argc > 2) NY = std::stoi(argv[2]);
  if (argc > 3) NT = std::stoi(argv[3]);
  if (argc > 4) dump2disk = std::stoi(argv[4])==1;

  // vector declaration
  // * matrices are 1D allocated
  // * caution: std::vector<std::vector<type>> is not memory contiguous (performance hit)
  // * matrix u contains the actual data, u2 is a copy used during iterations
  std::vector<float> u(NX*NY);

  // report memory usage
  std::cout << " ... " << NX << "X" << NY << " matrix (" << float(sizeof(float) * u.size()) / float(1024*1024*1024) << "GB)" << std::endl;

  // matrix initialisation
  auto t_start = std::chrono::high_resolution_clock::now();
  jac_init (NX, NY, u);
  auto t_stop = std::chrono::high_resolution_clock::now();
  auto t_elap = std::chrono::duration_cast<std::chrono::duration<double>>(t_stop - t_start);
  std::cout << " ... initialisation:\t" << t_elap.count() << " sec" << std::endl;

  // write to disk for debug purposes
  if (dump2disk) jac_to_disk ("sol_init.txt", u);



  //
  // --- Jacobi iterations
  //

  // start time
  t_start = std::chrono::high_resolution_clock::now();

  // Jacobi iterations
  jac_iter (NX, NY, NT, u);

  // stop time
  t_stop = std::chrono::high_resolution_clock::now();
  // report performance
  t_elap = std::chrono::duration_cast<std::chrono::duration<double>>(t_stop - t_start);
  std::cout << " ... iterations:\t" << t_elap.count() << " sec" << " @ " << 4.0f*NT*(NX-2)*(NY-2)*1.e-9/t_elap.count() << " Gflops" << std::endl;

  // write to disk for debug purposes
  if (dump2disk) jac_to_disk ("sol_iter.txt", u);

  return EXIT_SUCCESS;
}
