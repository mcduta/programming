/*

  project: rd -- finite difference solution of a reaction-diffusion equation
  file:    time.hpp -- header file for timer functions

 */

# include <chrono>
std::chrono::high_resolution_clock::time_point get_time_start ();
double get_time_elapsed (std::chrono::high_resolution_clock::time_point);
