# GPU programming using C++ STL


## Background
These examples are meant to illustrate the potential the C++ STL has for writing single programs that target CPU or GPU execution. The examples are inspired by the developments in STL [[1]](#1) and the rewriting of Lulesh to make use of these [[2]](#2).


## Compiler support
The minimal C++ standard is 2017. GCC and Nvidia nvc++ support this.


## Source code
The source code consists of 1D and 3D explicit finite-difference solvers for the heat equation. The C++ code ``heat_*.cpp`` is complemented by equivalent OpenMP parallelised reference C code ``heat_*_omp.c``. Rather than using ``std::vector`` for the solution storage, the C++ codes follow the implementation of a Jacobi itaretion in [[3]](#3) and uses C arrays and a dedicated iterator definition in ``index.hpp``. All arrays are double precision. Note that, with GPU execution in mind, all values captured by the lambdas are by reference.


## CPU target
Compile the code with something like
```
g++ $CFLAGS --std=c++20 \
  -I $TBB_HOME/include -o heat_3d heat_3d.cpp -L $TBB_HOME/lib/intel64/gcc4.8 \
  -ltbb -Wl,-rpath=$TBB_HOME/lib/intel64/gcc4.8
```

and run with
```
./heat_1d 400000000 1000
./heat_3d 600 1000
```

The compiler optimisation flags could be something like ``-O3 -march=cascadelake``. ``$TBB_HOME`` is the installation path of the Threading Building Blocks. The first command line argument is the number of nodes in the computational grid. (In the 3D case, the total number of grid points is the cube of the input.) The second argument is the number of time steps (iterations). Both arguments are optional, for which default values are coded.

A third optional argument is a string (file name), which triggers the writing of the final computed solution to an ASCII file of the given name. This option is for debugging the numerics only and generates large files.

The reference OpenMP implementation is built with
```
gcc $CFLAGS -fopenmp -o heat_3d_omp heat_3d_omp.c -lm
```
and takes the same online arguments as the C++ code.


## GPU target
Compile the code with
```
nvc++ -std=c++20 -stdpar=gpu -o heat_3d_g heat_3d.cpp
```

Compilation automatically detects the compute capability of the GPU on the system which it is done. Recompilation is advised to target another GPU compute capability.


## Performance comparison
Performance measured for 1000 iterations on a 2 socket 40 core node (Intel Xeon Gold 6148) system with P100.

| | 1D<br>(400M grid points) | 3D<br>(200+M grid points) |
|------------ | :-----------: | :-----------: |
|OpenMP | 58.6s | 48.45s |
|C++ CPU | 80.2s | 69.1s |
|C++ GPU | 8.85s | 9.06s |


## Conclusion
OpenMP code is the fastest on CPU execution by a noticeable margin. However, C++ offers the convenience of a single source, which compiles to code that runs multithreaded on a multicore system and gives good performance on a GPU.


## References

<a id="1">[1]</a>
https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/

<a id="2">[2]</a>
https://asc.llnl.gov/codes/proxy-apps/lulesh

<a id="3">[3]</a>
https://on-demand.gputechconf.com/supercomputing/2019/pdf/sc1936-gpu-programming-with-standard-c++17.pdf
