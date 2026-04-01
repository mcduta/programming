# include <iostream>
# include <cstdlib>
# include <omp.h>

int main (int argc, char **argv) {
  std::cout << " devices: " << omp_get_num_devices () << std::endl;
  std::cout << " default: " << omp_get_default_device () << std::endl;
  std::cout << " initial: " << omp_get_initial_device () << std::endl;

  return EXIT_SUCCESS;
}
