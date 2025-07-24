
# include "gpu-mem-common.h"


//
// ----- random number initialisation
//
void array_init (float* x, const size_t N) {
  float f = 1.0f / ((float) RAND_MAX);
  for (size_t n=0; n<N; n++) {
    x[n] = f*rand();
  }
}


//
// ----- CUDA kernel
//
__global__ void array_calcs (float* x, const size_t N, const uint T) {
  // NB: type cast blockDim.x and blockIdx.x to size_t
  //     to handle evry large arrays (>32GB)
  size_t n = ((size_t) blockDim.x) * ((size_t) blockIdx.x) + threadIdx.x;
  if (n < N) {
    float v = x[n];
    for (size_t t=0; t<T; t++) {
      v = sqrtf (1.0f + v);
    }
    x[n] = v;
  }
}


//
// ----- verify calculations
//
bool array_check (float* x, const size_t N, const float eps) {
  float f = (1.0f + sqrt(5.0f)) / 2.0f;
  bool pass = true;
  for (size_t n=0; n<N; n++) {
    if (abs(x[n] - f) > eps) {
      pass = false;
      fprintf (stderr, " *** value mismatch: x[%lu] = %f\n", (unsigned long) n, x[n]);
      break;
    }
  }
  return pass;
}
