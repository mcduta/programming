# include <chrono>
# include <iostream>
# include <random>
# include <functional>


//
// --- timer functions
//
// ... return restarted timer
std::chrono::high_resolution_clock::time_point get_time_start () {
  return std::chrono::high_resolution_clock::now();
}

// --- return seconds elapsed since timer restarted
double get_time_elapsed (std::chrono::high_resolution_clock::time_point timer) {
  std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - timer);
  return duration.count();
}


//
// --- main
//

int main() {

  // sizes
  const double alpha = -1.3;
  const uint size  = 128*1024*1024;  // 1GB in double precision
  const uint times = 40;

  // allocate
  std::vector<double> x(size);
  std::vector<double> y(size);
  std::vector<double> z(size, 0.0);

  // initialise
  for (uint k=0; k<size; k++) {
    x[k] = 1.0*(k+1);
    y[k] = 2.5*(k+1);
  }

  // amount of work
  auto gflops  = 1.e-9 * 2.0 * size * times;


  //
  // ... loop 1: straight loop
  //
  auto timer = get_time_start ();
  for (uint t=0; t<times; t++) {
    for (uint k=0; k<size; k++) {
      z[k] = alpha * x[k] + y[k];
    }
  }
  auto elapsed = get_time_elapsed (timer);
  std::cout << " standard loop: " << elapsed << " seconds, " << gflops / elapsed << " gflops" << std::endl;


  /*
  //
  // ... loop 2: manually unrolled
  //
  timer = get_time_start ();
  for (uint t=0; t<times; t++) {
    # pragma omp simd
    for (uint k=0; k<size; k+=4) {
      z[k]   = alpha * x[k]   + y[k];
      z[k+1] = alpha * x[k+1] + y[k+1];
      z[k+2] = alpha * x[k+2] + y[k+2];
      z[k+3] = alpha * x[k+3] + y[k+3];
    }
  }
  elapsed = get_time_elapsed (timer);
  std::cout << " unrolled loop: " << elapsed << " seconds, " << gflops / elapsed << " gflops" << std::endl;
  */


  //
  // ... loop 3: openmp simd
  //
  timer = get_time_start ();
  for (uint t=0; t<times; t++) {
    # pragma omp simd
    for (uint k=0; k<size; k++) {
      z[k] = alpha * x[k] + y[k];
    }
  }
  elapsed = get_time_elapsed (timer);
  std::cout << " simd loop: " << elapsed << " seconds, " << gflops / elapsed << " gflops" << std::endl;



  return EXIT_SUCCESS;
}
