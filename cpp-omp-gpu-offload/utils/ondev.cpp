# include <iostream>
# include <cstdlib>
# include <omp.h>

int main (int argc, char **argv) {
  int onGPU = 0;

  // test if GPU is available (OpenMP 4.5)
  # pragma omp target map(from:onGPU)
  {
    if (omp_is_initial_device() == 0) onGPU = 1;
  }

  // print result
  if (onGPU)
    std::cout << " === CPU and GPU available!"  << std::endl;
  else
    std::cout << " === CPU only!"  << std::endl;

  return EXIT_SUCCESS;
}
