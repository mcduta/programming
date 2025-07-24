// ... basic includes
# include <stdio.h>
# include <getopt.h>

// ... sizes
# define MEGA (1<<20)
# define GIGA (1<<30)

// ... headers
void array_init (float* x, const size_t N);
__global__ void array_calcs (float* x, const size_t N, const uint T);
bool array_check (float* x, const size_t N, const float eps);
