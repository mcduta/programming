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
  uint size    = 40000000; // ~0.3GB in double precision
  uint times   = 20;       // number of times to repeat test
  bool aligned = false;    // allocate aligned memory

  // structure (with scope for alignment)
  struct p {
    char  category;   // 1 byte
    char  name[4];    // 4 bytes
    float height;     // 4 bytes
  };

  // arrays
  struct p *x;

  // random number generation
  std::random_device rand_dev;
  std::mt19937 rand_gen (rand_dev());
  std::normal_distribution <float> rand_distr_sol (0, 1);   // normal distribution, mean=0, sigma=1


  // options
  if ( argc > 1 ) size  = std::stoi(argv[1]);
  if ( argc > 2 ) times = std::stoi(argv[2]);
  if ( argc > 3 ) aligned = std::string(argv[3]) == "aligned";

  // report
  std::cout << " ... daxpy array test: size " << size << ", repeated " << times << " times." << std::endl;
  std::cout << " ... " << (aligned ? "aligned" : "not aligned") << " memory: sizeof (struct) is " << sizeof(struct p) << std::endl;

  // allocate
  if (aligned) {
    x = static_cast<struct p *>(std::aligned_alloc(16, size * sizeof(struct p)));
  } else {
    x = static_cast<struct p *>(std::malloc(size * sizeof(struct p)));
  }


  // initialise
  for (uint k=0; k<size; k++) {
    x[k].height = 1.75 + 0.02 * rand_distr_sol (rand_gen);
  }

  //
  // ... loop 1: straight loop
  //
  auto timer = get_time_start ();
  for (uint t=0; t<times; t++) {
    float mean = 0.0;
    for (uint k=0; k<size; k++) {
      mean += x[k].height;
    }
    mean /= (float) size;
  }
  auto elapsed = get_time_elapsed (timer);
  std::cout << " ... standard loop: " << elapsed << " seconds" << std::endl;


  // //
  // // ... loop 2: manually unrolled
  // //
  // timer = get_time_start ();
  // for (uint t=0; t<times; t++) {
  //   alpha = 1.0 + ((double) t) / 1000.;
  //   # pragma code_align 16
  //   for (uint k=0; k<size; k+=4) {
  //     z[k]   = alpha * x[k]   + y[k];
  //     z[k+1] = alpha * x[k+1] + y[k+1];
  //     z[k+2] = alpha * x[k+2] + y[k+2];
  //     z[k+3] = alpha * x[k+3] + y[k+3];
  //   }
  // }
  // elapsed = get_time_elapsed (timer);
  // std::cout << " ... unrolled loop: " << elapsed << " seconds, " << gflops / elapsed << " gflops" << std::endl;



  // //
  // // ... loop 3: openmp simd
  // //
  // timer = get_time_start ();
  // for (uint t=0; t<times; t++) {
  //   alpha = 1.0 + ((double) t) / 1000.;
  //   # pragma omp simd
  //   for (uint k=0; k<size; k++) {
  //     z[k] = alpha * x[k] + y[k];
  //   }
  // }
  // elapsed = get_time_elapsed (timer);
  // std::cout << " ... simd loop:     " << elapsed << " seconds, " << gflops / elapsed << " gflops" << std::endl;


  // deallocate
  if (x) std::free(x);

  return EXIT_SUCCESS;
}
