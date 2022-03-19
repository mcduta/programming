# Comparison of intrinsic function cost

## Source code
The demo code computes the value of intrinsic functions (_e.g._ `exp()`) on the elements of a vector. Loop execution time is measured and reported. A baseline loop execution performance is also measured on simple multiply and adds floating point operations on the same vector.

## Build
The source code is compiled and linked
 * using the GNU `gfortran` compiler and the `libm` library
 > gfortran -O2 -o demo.lm demo.f90 -lm
 * using the Intel `ifort` compiler and the `libimf` library
 > ifort -O2 -o demo.limf demo.f90 -limf

## Comparison
Run the two resultnig executables with various vector sizes, _e.g._
```
 ./demo.limf 100000000
```

A size of `100000000` means a double precision vector size of 763MB.

It is expected that
 * the baseline loop executes in a time that does not depend on the choice of compiler;
 * the Intel `libimf` library outperforms the GNU `libm` library.
