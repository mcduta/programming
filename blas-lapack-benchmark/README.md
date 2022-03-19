# Node performance measurements using BLAS/LAPACK

## Background

Multithreaded node performance measurements

  * ``perf_dgemm.f90`` -- GFlops performance (larger is better) using BLAS ``dgemm`` matrix multiplication;
  * ``perf_dgesvd.f90`` -- walltime performance (shorter is better) using LAPACK ``dgesvd`` or ``dgesdd`` singular value solver.

The reference BLAS and LAPACK implementation of these functions is at [[1]](#1). The Fortran programs above can be built with modern implementations, such as Intel MKL [[2]](#2) and OpenBLAS [[3]](#3).

## Build and run

Edit ``makefile`` to include the appropriate ``.inc`` file to build with
  * ``make.openblas.inc`` for GNU ``gfortran`` and OpenBLAS;
  * ``make.gfortran-mkl.inc`` for GNU ``gfortran`` and Intel MKL;
  * ``make.ifort-mkl.inc`` for Intel ``ifort`` and Intel MKL.

Also, edit the included ``.inc`` file from the above list for the host specific paths.

Run the resulting DGEMM test with
```
perf_dgemm M N L
```
where ``M``, ``N`` and ``L`` are matrix dimensions.

Run the singular value solver providing matrix dimensions, a choice of method (SVD or SDD) and the option of whether the singular vectors are computed (more expensive) or not:
```
perf_dgesvd -m M -n N -method [svd|sdd] -vectors [yes|no]
```

## References

<a id="1">[1]</a>
http://www.netlib.org/lapack/explore-html/index.html

<a id="2">[2]</a>
https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html

<a id="3">[3]</a>
https://www.openblas.net/