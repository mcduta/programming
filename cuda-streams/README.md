# Experimenting with CUDA Streams

## Background

In CUDA, kernel concurrency is achieved via CUDA streams, a mechanism in which one or more kernels can be assigned to different streams, with multiple streams able to run at the same time as long as there are sufficient resources. The main purpose of CUDA streams is to hide memory latency, which typically means that one kernel is loading or writing data, while another occupies the cores for computation. As a result, the GPU multi-processing cores can reach good utilization.

## Codes

The two streams examples in this repository are written in both CUDA and CuPY. They are similar, but not to the full extent, as there is no full equivalence between all CuPY and CUDA functions.

The main codes are `test-streams.cu` and `test-streams.py`. They both create an array of floating point numbers, which is processed in three different but computationally equivalent steps. (Equivalent means the same number of flops that lead to the same numerical results.) The three steps are:

1. The array is entirely transfered to GPU memory, then processed by a kernel, and then transfered back to host memory.
2. One quarter of the same arrays is dealt with by a different CUDA stream; each stream first transfers its quarter to GPU memory, then runs the kernel and then transfers the result back to host memory. This sequence of ops is carried out for each stream in a asynchronouns way, with a sync point at the end.
3. Same as 2 but the transfers and kernel launches are carried out in groups, for all streams at once.

Time is meeasured on kernel execution, without as well as without memory transfers. For sufficiently large arrays, it becomes clear the use of streams reduces overall execution time. Memory transfer latenciese are hidden by kernel execution by assigning work to more than one stream. (The choice of four streams is arbitrary.)

In addition to the main codes, the tests `test-kernel.cu` and `test-kernel.py` run a kernel without streams, with the purpose of checking whether the Python based execution is on a par with the CUDA one for the same kernel.


## Compile and Run

Run `make` to generate both the `test-kernel` and `test-streams` executable. Single and double precision floating point variants are generated for each, labelled `float32` and `float32`, respectively.

The executables can be run with something like
```
./test-kernel-float32 -g 4
```
while the equivalent Python code runs is
```
python ./test-streams.py -g 4
```


## References

Presentations of CUDA streams

  * https://leimao.github.io/blog/CUDA-Stream/
  * https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

The `test-stream` CUDA code is adapted from original source

  * https://github.com/NVIDIA-developer-blog/code-samples/tree/master/series/cuda-cpp/overlap-data-transfers

Notes for CuPY (including performance and timing tips):

  * https://carpentries-incubator.github.io/lesson-gpu-programming/instructor/streams.html
  * https://docs.cupy.dev/en/stable/user_guide/performance.html
