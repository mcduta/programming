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

int main (int argc, char *argv[]) {

  // sizes
  double alpha = -1.3;     // scalar to multiply vector
  uint size    = 40000000; // ~0.3GB in double precision
  uint times   = 20;       // number of times to repeat test
  bool aligned = false;    // allocate aligned memory

  // arrays
  double* x, *y, *z;

  // options
  if ( argc > 1 ) size  = std::stoi(argv[1]);
  if ( argc > 2 ) times = std::stoi(argv[2]);
  if ( argc > 3 ) aligned = std::string(argv[3]) == "aligned";

  // report
  std::cout << " ... daxpy array test: size " << size << ", repeated " << times << " times." << std::endl;
  std::cout << " ... " << (aligned ? "aligned" : "not aligned") << " memory." << std::endl;

  // allocate
  if (aligned) {
    x = static_cast<double*>(std::aligned_alloc(16, size * sizeof(double)));
    y = static_cast<double*>(std::aligned_alloc(16, size * sizeof(double)));
    z = static_cast<double*>(std::aligned_alloc(16, size * sizeof(double)));
  } else {
    x = static_cast<double*>(std::malloc(size * sizeof(double)));
    y = static_cast<double*>(std::malloc(size * sizeof(double)));
    z = static_cast<double*>(std::malloc(size * sizeof(double)));
  }


  // initialise
  for (uint k=0; k<size; k++) {
    x[k] = 1.0*(k+1);
    y[k] = 2.5*(k+1);
    z[k] = 0.0;
  }

  // amount of work
  auto gflops  = 1.e-9 * 2.0 * size * times;

  //
  // ... loop 1: straight loop
  //
  auto timer = get_time_start ();
  for (uint t=0; t<times; t++) {
    alpha = 1.0 + ((double) t) / 100.;
    # pragma code_align 16
    for (uint k=0; k<size; k++) {
      z[k] = alpha * x[k] + y[k];
    }
  }
  auto elapsed = get_time_elapsed (timer);
  std::cout << " ... standard loop: " << elapsed << " seconds, " << gflops / elapsed << " gflops" << std::endl;


  //
  // ... loop 2: manually unrolled
  //
  timer = get_time_start ();
  for (uint t=0; t<times; t++) {
    alpha = 1.0 + ((double) t) / 1000.;
    # pragma code_align 16
    for (uint k=0; k<size; k+=4) {
      z[k]   = alpha * x[k]   + y[k];
      z[k+1] = alpha * x[k+1] + y[k+1];
      z[k+2] = alpha * x[k+2] + y[k+2];
      z[k+3] = alpha * x[k+3] + y[k+3];
    }
  }
  elapsed = get_time_elapsed (timer);
  std::cout << " ... unrolled loop: " << elapsed << " seconds, " << gflops / elapsed << " gflops" << std::endl;



  //
  // ... loop 3: openmp simd
  //
  timer = get_time_start ();
  for (uint t=0; t<times; t++) {
    alpha = 1.0 + ((double) t) / 1000.;
    # pragma omp simd
    for (uint k=0; k<size; k++) {
      z[k] = alpha * x[k] + y[k];
    }
  }
  elapsed = get_time_elapsed (timer);
  std::cout << " ... simd loop:     " << elapsed << " seconds, " << gflops / elapsed << " gflops" << std::endl;


  // deallocate
  if (x) std::free(x);
  if (y) std::free(y);
  if (z) std::free(z);

  return EXIT_SUCCESS;
}
