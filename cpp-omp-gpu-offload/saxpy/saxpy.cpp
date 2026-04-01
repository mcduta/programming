/*
  File:     saxpy.cpp
  Synopsis: Illustrates the implementation of the BLAS SAXPY functionality using
             * OpenMP CPU multithreading and
             * OpenMP GPU offloading
  Details:
             * Implemented using std::vector, e.g.
                 std::vector<float> x(N)

             * OpenMP parallel regions access std::vector data via std::span, e.g.
                 std::span<float> xs(x);

             * OpenMP target region access std::vector data via a pointer to vector.data(), e.g.
                 float *xd = x.data();
  Build:
             * COMPILER=<gcc|llvm|nvhpc|rocm> make
  Run:
             * ./saxpy
 */

# include <iostream>
# include <cstdlib>
# include <vector>
# include <span>
# include <algorithm>
# ifdef _OPENMP
# include <omp.h>
# endif

int main (int argc, char **argv) {

  // sizes
  const uint N = 1 << 26; // size of arrays, e.g. 2**24 (size of 128MB in float64)
  const uint T = 1000;    // number of repeats

  // vectors
  std::vector<float> x(N); // vector container for input x
  std::vector<float> y(N); // vector container for input y and output z (overwritten)

  // scalar value
  const float a = -1.3;

  // timings
  double ts,tf,te; // time values: start, final, elapsed

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

  ts = omp_get_wtime ( );

  for (i=0; i<N; i++) {
    ys[i] = a*xs[i] + ys[i];
  }

  tf = omp_get_wtime ( );
  te = tf - ts;

  std::cout << " serial:\t" << 2.0*N*1.e-9/te << " gflops in " << te << " seconds" << std::endl;


  //
  // --- OpenMP CPU parallel AXPY
  //     * repeated T times
  //     * vector data accessed via vector spans
  //
  ts = omp_get_wtime ( );

  # pragma omp parallel default(none) shared (N,T, xs,ys,a) private (i,t)
  for (t=0; t<T; t++) {
    # pragma omp for schedule (static)
    for (i=0; i<N; i++) {
      ys[i] = a*xs[i] + ys[i];
    }
  }
  tf = omp_get_wtime ( );
  te = tf - ts;

  std::cout << " omp thread:\t" << 2.0*N*T*1.e-9/te << " gflops in " << te << " seconds" << std::endl;


  //
  // --- OpenMP GPU parallel AXPY
  //     * repeated T times
  //     * vector data accessed via pointers to vector.data()
  //
  float *xd = x.data(); // pointer to x contained data
  float *yd = y.data(); // pointer to y contained data

  ts = omp_get_wtime ( );

  # pragma omp target teams distribute         \
    map(to:xd[0:x.size()])                     \
    map(tofrom:yd[0:y.size()])
  for (t=0; t<T; t++) {
    # pragma omp parallel for
    for (i=0; i<N; i++) {
      yd[i] = a*xd[i] + yd[i];
    }
  }

  tf = omp_get_wtime ( );
  te = tf - ts;

  std::cout << " gpu teams:\t" << 2.0*N*T*1.e-9/te << " gflops in " << te << " seconds" << std::endl;

  // exit
  return EXIT_SUCCESS;
}
