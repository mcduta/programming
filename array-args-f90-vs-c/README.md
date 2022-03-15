# Array arguments passed to C and Fortran functions

## Background
Pointer aliasing is a hidden type of data dependency that can occur in C (and C++). Array data identified by pointers in C can overlap, because the C language itself puts very few restrictions on pointers. Therefore, two seemingly different pointers may point to storage locations in the same array (aliasing). As a result, data dependencies can arise when performing loop-based computations using pointers, as the pointers may potentially point to overlapping regions in memory. The compiler has to assume "the worst" in the absence of further information, and has to add extra tests to prevent any data dependency violation. This has two consequences for array-based floating point operations using **unaliased** pointers:

 * the resulting code is slower than should be and
 * loop vectorisation becomes hard or impossible.

Fortran 90/95 is not vulnerable to this issue because any pointer must be explicitly associated to a target variable of known size before it can be used.

## Example

This example works with both the GNU compilers and the Intel compilers. The coding consists of

 * `driver.c` -- the main C code that allocates the vectors and calls
   - the "naive" implementation of the C function that uses the vectors;
   - the unaliased implementation of the same C function;
   - the Fortran equivalent of the C functions.
 * `cfunc.c` -- the two C functions;
 * `ffunc.f90` -- the Fortran funtion.

To build with the GNU compilers, do
```
make clean && make
```

To build using the Intel compilers, do
```
make compiler=INTEL
```

Run with a large enough array length, _e.g._
```
./demo 500000000
```

## References
https://developers.redhat.com/blog/2020/06/02/the-joys-and-perils-of-c-and-c-aliasing-part-1
https://developers.redhat.com/blog/2020/06/03/the-joys-and-perils-of-aliasing-in-c-and-c-part-2
https://www.intel.com/content/www/us/en/developer/articles/technical/pointer-aliasing-and-vectorization.html
