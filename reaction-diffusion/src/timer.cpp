/*

  project: rd -- finite difference solution of a reaction-diffusion equation
  file:    time.cpp -- timer functions (interface to chrono)

 */

# include "timer.hpp"

// ... return restarted timer
std::chrono::high_resolution_clock::time_point get_time_start () {
  return std::chrono::high_resolution_clock::now();
}

// ... return seconds elapsed since timer restarted
double get_time_elapsed (std::chrono::high_resolution_clock::time_point timer) {
  std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - timer);
  return duration.count();
}
