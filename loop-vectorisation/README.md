# Loop vectorisation demo

## Code
Main code is ``loops.cpp``, which uses heap allocated C arrays. It implements BLAS ``daxpy``-like operations on arrays of ``double``.

``loops.vectors.cpp`` is similar but uses ``std::vector``.

## Vectorisation options
 * compiler automatic vectorisation;
 * manual unrolling;
 * OpenMP ``pragma simd``.

## Memory allocation
 * aligned memory allocation (``std::aligned_alloc``);
 * non-aligned ``malloc`` allocation.

## Compilers
Tried: GCC 11.2, Intel C++ Compiler 2022, LLVM 10.0.

## Usage:
> loops <size> <times> aligned

where ``<size>`` and ``<times>`` are the vector size (number of elements) and the number of loop repeats. ``aligned`` is an optional argument to indicate aligned memory allocation (default is ``malloc``).
