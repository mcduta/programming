/*

  CUDA kernel and utils shared by the test-*.cu examples

 */

//
// --- CUDA kernel (computationally expensive)
//     * computes the value 1.0 for all output array entries
//     * input x and output y could be aliased
//
template <class T> __global__ void cudaKernel (T *x, T *y, const size_t N, const size_t offset)
{
  size_t i = blockIdx.x*blockDim.x + threadIdx.x + offset;
  if (i < N) {
    T z =  fabs (x[i]);
    z = z / (1.0 + z);
    T s = sin (z);
    T c = cos (z);
    y[i] = sqrt (s*s+c*c);   // computed value is always 1.0
  }
}
