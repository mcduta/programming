# Background

# Building the demo


module load gcc/14.2.0__cuda-12

g++ -I $TBBROOT/include -std=c++23 \
    -Wall -Wextra -pedantic-errors \
    -L $TBBROOT/lib/intel64/gcc4.8 -ltbb \
    -Wl,-rpath=$TBBROOT/lib/intel64/gcc4.8 \
    -o daxpy daxpy.cpp


 using std::vector
 ... alloc:	2.22168 s
 ... ini:	57.7817 s
 ... exe:	2.55164 s	[serial]
 ... exe:	0.0822274 s	[std::execution::par_unseq]
 ... exe:	0.0525372 s	[OpenMP]
 using C arrays
 ... alloc:	4.2536e-05 s
 ... ini:	55.3987 s
 ... exe:	0.548695 s



module load llvm/18_cuda12

clang++ -Wall -pedantic -std=c++23 -stdlib=libc++ -fexperimental-library \
        -o daxpy daxpy.cpp
clang++  -std=c++23 -stdlib=libc++ -fexperimental-library -Wall -pedantic -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o daxpy-llvm-gpu daxpy-vector.cpp



module load nvhpc/2025 gcc/14

nvc++ -pedantic --gcc-toolchain=$(which g++) -std=c++23 -stdpar=multicore -fopenmp -o daxpy-nvhpc daxpy.cpp








-fopenmp!!!

g++ -fopenmp -foffload=nvptx-none -o ondev ondev.cpp
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o ondev ondev.cpp


# Running the demo
