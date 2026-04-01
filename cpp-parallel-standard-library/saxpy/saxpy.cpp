/*
  File:     saxpy.cpp
  Synopsis: Illustrates the implementation of the BLAS SAXPY functionality using
            ISO C++ Standard Template Library (STL) parallelism
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
             * ./saxpy
 */

# include <chrono>
# include <iostream>
# include <random>
# include <functional>
# include <execution>
# include <algorithm>
# include <span>


int main(int argc, char** argv) {

  // sizes
  const uint N = 1 << 26; // size of arrays, e.g. 2**24 (size of 128MB in float64)
  const uint T = 1000;    // number of repeats

  // vectors
  std::vector<float> x(N); // vector container for input x
  std::vector<float> y(N); // vector container for input y and output z (overwritten)

  // scalar value
  const float a = -1.3;

  // iterators
  uint i,t;

  // initialise (C++ style using the vector containers directly)
  std::generate (x.begin(), x.end(), [i=0]    () mutable { return float(i++ + 1); });
  std::generate (y.begin(), y.end(), [i=0, N] () mutable { return float(i++ - N); });


  //
  // --- serial AXPY for reference
  //     * repeated only once (too slow to repeat)
  //     * vector data accessed via vector spans
  //
  std::span<float> xs(x); // xs is span of x
  std::span<float> ys(y); // ys is span of y

  auto t_start = std::chrono::high_resolution_clock::now();

  for (i=0; i<N; i++) {
    ys[i] = a*xs[i] + ys[i];
  }

  auto t_stop = std::chrono::high_resolution_clock::now();
  auto t_elap = std::chrono::duration_cast<std::chrono::duration<float>>(t_stop - t_start);

  std::cout << " serial:\t" << 2.0*N*1.e-9 / t_elap.count() << " gflops in " << t_elap.count() << " seconds" << std::endl;


  //
  // --- OpenMP CPU parallel AXPY
  //     * repeated T times
  //     * vector data accessed via vector spans
  //

  // define lambda to apply element-wise
  auto saxpy = [a](const float xi, const float yi){return a*xi + yi;};

  // ... daxpy (serial transform)
  t_start = std::chrono::high_resolution_clock::now();

  for (t=0; t<T; t++) {
    std::transform (std::execution::par_unseq, x.begin(), x.end(), y.begin(), y.begin(), saxpy);
  }

  t_stop = std::chrono::high_resolution_clock::now();
  t_elap = std::chrono::duration_cast<std::chrono::duration<float>>(t_stop - t_start);

  std::cout << " omp thread:\t" << 2.0*N*T*1.e-9 / t_elap.count() << " gflops in " << t_elap.count() << " seconds" << std::endl;

  // exit
  return EXIT_SUCCESS;

}
