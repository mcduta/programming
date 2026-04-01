/*
  File:     jacobi.cpp
  Synopsis: Illustrates the implementation of the Jacobi iteration of a 2D vector using:
              * OpenMP CPU multithreading and
              * OpenMP GPU offloading
  Details:
              * Implemented using std::vector, e.g.
                  std::vector<float> x(N)

              * OpenMP parallel regions access std::vector data via std::span, e.g.
                  std::span<float> xs(x);

              * OpenMP target region access std::vector data via a pointer to vector.data(), e.g.
                  float *xd = x.data();
  Build:    Generate code that offloads to GPU using the OpenMP standard
              * COMPILER=<gcc|llvm|nvhpc|rocm> make jac
            Generate code that uses a CUDA kernel instead of OpenMP (or its HIP equivalent)
              * COMPILER=nvhpc make cujac
              * COMPILER=rocm make hipjac
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
# include <omp.h>
# ifdef USE_CUDA
# include "jacobi-update-cuda.h"
# endif


//
// ----- 2D index mapping
//         * idx_fuse: map 2D matrix indexing (iX,iY) to linear index i
//         * idx_split: maps linear index i to 2D matrix indexing (iX,iY)
// auto idx_fuse  = [NY](size_t iX, size_t iY) { return iY + NY * iX; };
// auto idx_split = [NY](size_t i) { return std::make_tuple(i / NY, i % NY); };


//
// ----- jacobi solution initialisation
//
void jac_init_cpu (const size_t NX, const size_t NY, std::vector<float> &u) {

  // some indices (declared before use to become visible to OpenMP pragmas)
  size_t iX,iY;

  // sanity check
  if (NX*NY != u.size()) {
    throw std::invalid_argument("jac_init: argument mismatch.");
  }

  // std::span to access matrix data
  std::span<float> u_span(u);

  // 2D index mapping
  auto idx_fuse = [NY](size_t iX, size_t iY) { return iY + NY * iX; };

  // initialise
  // * u[iX,iY] = min(iX,NX-iX-1) * min(iY,NY-iY-1) / (NX*NY)
  # pragma omp parallel default(none) shared(NX,NY, u_span, idx_fuse) private(iX,iY)
  # pragma omp for schedule (static) collapse(2)
  for (iX=0; iX<NX; iX++) {
    for (iY=0; iY<NY; iY++) {
      u_span[idx_fuse(iX,iY)] = float(std::min(iX,NX-iX-1))
                              * float(std::min(iY,NY-iY-1)) / float(NX*NY);
    }
  }

}

void jac_init_gpu (const size_t NX, const size_t NY, std::vector<float> &u) {

  // some indices (declared before use to become visible to OpenMP pragmas)
  size_t iX,iY;

  // sanity check
  if (NX*NY != u.size()) {
    throw std::invalid_argument("jac_init: argument mismatch.");
  }

  // pointers to access matrix data
  float *u_ptr = u.data();

  // 2D index mapping
  auto idx_fuse = [NY](size_t iX, size_t iY) { return iY + NY * iX; };

  // initialise
  // * u[iX,iY] = min(iX,NX-iX-1) * min(iY,NY-iY-1) / (NX*NY)
  # pragma omp target teams distribute parallel for collapse(2)	\
    map(tofrom: u_ptr[0:u.size()]) \
    default(none) shared(NX,NY, u_ptr, idx_fuse) private(iX,iY)
  for (iX=0; iX<NX; iX++) {
    for (iY=0; iY<NY; iY++) {
      u_ptr[idx_fuse(iX,iY)] = float(std::min(iX,NX-iX-1))
                             * float(std::min(iY,NY-iY-1)) / float(NX*NY);
    }
  }

}


//
// ----- jacobi solution iteration
//
void jac_iter_cpu (const size_t NX, const size_t NY, const size_t NT, std::vector<float> &u) {

  // some indices (declared before use to become visible to OpenMP pragmas)
  size_t iX,iY,iT;

  // sanity check
  if (NX*NY != u.size()) {
    throw std::invalid_argument("jac_init: argument mismatch.");
  }

  // extra vector for copies
  // * at each time step iT, u2 gets updated from u
  //   then the two are swapped
  std::vector<float> u2(NX*NY);

  // std::span to access matrix data
  std::span<float> u_span(u);
  std::span<float> u2_span(u2);

  // 2D index mapping
  auto idx_fuse = [NY](size_t iX, size_t iY) { return iY + NY * iX; };

  // Jacobi iterations
  # pragma omp parallel default(none) shared(NX,NY,NT, u_span,u2_span, idx_fuse) private(iX,iY,iT)
  for (iT=0; iT<NT; iT++) {
    // Jacobi update
    # pragma omp for schedule(static) collapse(2)
    for (iX=1; iX<NX-1; iX++) {
      for (iY=1; iY<NY-1; iY++) {
        u2_span[idx_fuse(iX,iY)] = 0.25f * ( u_span[idx_fuse(iX+1,iY)] + u_span[idx_fuse(iX-1,iY)]
                                           + u_span[idx_fuse(iX,iY+1)] + u_span[idx_fuse(iX,iY-1)] );
      }
    }
    // swap pointers
    # pragma omp master
    {
      std::swap (u_span, u2_span);
    }
  }

  // copy u2 to u, if u2 is latest updated
  if (NT%2) {
    std::copy(std::execution::par_unseq, u2.begin(), u2.end(), u.begin());
  }
}

void jac_iter_gpu (const size_t NX, const size_t NY, const size_t NT, std::vector<float> &u) {

  // some indices (declared before use to become visible to OpenMP pragmas)
  size_t iX,iY,iT;

  // sanity check
  if (NX*NY != u.size()) {
    throw std::invalid_argument("jac_init: argument mismatch.");
  }

  // extra vector for copies
  // * at each time step iT, u2 gets updated from u
  //   then the two are swapped
  std::vector<float> u2(NX*NY);

  // pointers to access matrix data
  float *u_ptr = u.data();
  float *u2_ptr = u2.data();

  // 2D index mapping
  auto idx_fuse = [NY](size_t iX, size_t iY) { return iY + NY * iX; };

  // Jacobi iterations
  // * copy u & u2 to device on entry
  # pragma omp target enter data \
    map(to: u_ptr[0:u.size()], u2_ptr[0:u2.size()])
  for (iT=0; iT<NT; iT++) {
    // Jacobi update
    # pragma omp target teams distribute				\
      default(none) shared(NX,NY,NT, u_ptr,u2_ptr, idx_fuse) private(iX,iY,iT)
    for (iX=1; iX<NX-1; iX++) {
      # pragma omp parallel for simd
      for (iY=1; iY<NY-1; iY++) {
        u2_ptr[idx_fuse(iX,iY)] = 0.25f * ( u_ptr[idx_fuse(iX+1,iY)] + u_ptr[idx_fuse(iX-1,iY)]
                                          + u_ptr[idx_fuse(iX,iY+1)] + u_ptr[idx_fuse(iX,iY-1)] );
      }
    }
    // swap pointers
    # pragma omp master
    {
      std::swap (u_ptr, u2_ptr);
    }
  }
  // * copy u & u2 back to host on exit
  # pragma omp target exit data map(from: u_ptr[0:u.size()], u2_ptr[0:u2.size()])

  // copy u2 to u, if u2 is latest updated
  if (NT%2) {
    std::copy(std::execution::par_unseq, u2.begin(), u2.end(), u.begin());
  }
}


//
// ----- jacobi solution output to disk
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
// ----- main
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
  // default option (CPU multithreaded)
  bool offload2gpu = 0;

  // optional input sizes
  if (argc > 1) NX = std::stoi(argv[1]);
  if (argc > 2) NY = std::stoi(argv[2]);
  if (argc > 3) NT = std::stoi(argv[3]);
  if (argc > 4) offload2gpu = std::stoi(argv[4])==1;
  if (argc > 5) dump2disk   = std::stoi(argv[5])==1;

  // vector declaration
  // * matrices are 1D allocated
  // * caution: std::vector<std::vector<type>> is not memory contiguous (performance hit)
  // * matrix u contains the actual data, u2 is a copy used during iterations
  // pointers to access matrix data
  std::vector<float> u(NX*NY);

  // report memory usage
  std::cout << " ... " << NX << "X" << NY << " matrix (" << float(sizeof(float) * u.size()) / float(1024*1024*1024) << "GB)" << std::endl;

  // report CPU / GPU execution and memory usage
  std::cout << " ... jacobi iterations:\t";
  if (offload2gpu) {
    if (! omp_get_num_devices()) throw std::runtime_error(" no GPUs present");
    std::cout << "GPU offloading" <<  std::endl;
  } else {
    std::cout << "CPU multithreading" <<  std::endl;
    uint num_threads;
    # pragma omp parallel
    {
      # pragma omp single
      num_threads = omp_get_num_threads();
    }
    std::cout << " ... number of CPU threads:\t" << num_threads << std::endl;
  }
  std::cout << " ... " << NX << "X" << NY << " matrix:\t" << float(sizeof(float) * u.size()) / float(1024*1024*1024) << "GB" << std::endl;


  // matrix initialisation
  auto t_start = std::chrono::high_resolution_clock::now();
  if (offload2gpu) {
    jac_init_gpu (NX, NY, u);
  } else {
    jac_init_cpu (NX, NY, u);
  }
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
  if (offload2gpu) {
    # ifdef USE_CUDA
    // special case: CUDA kernel
    jac_iter_cuda (NX, NY, NT, u.data());
    # else
    // default behaviour: OpenMP offloading
    jac_iter_gpu (NX, NY, NT, u);
    # endif
  } else {
    jac_iter_cpu (NX, NY, NT, u);
  }

  // stop time
  t_stop = std::chrono::high_resolution_clock::now();
  // report performance
  t_elap = std::chrono::duration_cast<std::chrono::duration<double>>(t_stop - t_start);
  std::cout << " ... iterations:\t" << t_elap.count() << " sec" << " @ " << 4.0f*NT*(NX-2)*(NY-2)*1.e-9/t_elap.count() << " Gflops" << std::endl;

  // write to disk for debug purposes
  if (dump2disk) jac_to_disk ("sol_iter.txt", u);

  return EXIT_SUCCESS;
}
