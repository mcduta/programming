# FFTW Demos

## Background

The demo implements 1D, 2D and 3D Fast Fourier Transforms. The purpose of this is benchmarking, in particular benchmarking the FFTW library implementation [[1]](#1) against Intel OneAPI MKL library [[2]](#2).

There are two main implementation of the same operation:
  * ``fftw3.c`` -- uses interleaved input/output arrays of ``fftw_complex``;
  * ``fftw3s.c`` -- uses split input/output arrays of ``float`` or ``double``;

Both imlementations select the dimensionality (1D, 2D or 3D) and size of the transform using command line options. Also, the number of threads used by the transform is a command line option in the multithreaded executables. Single and double precision versions of the tests are generated as separate executables. So are single-threaded and multi-threaded versions.

Both implementations measure and report the walltime taken by the library transform, as a measure of library/node performance.


## Build

The code can be built by providing ``make`` the details of either the FFTW library or the MKL library. Thus, the following builds all the executable variants for FFTW
 (all executables are moved to a ``bin/`` directory):
```
export FFTWDIR=/dls_sw/apps/fftw/3.3.8/64/6-avx2
make FFTW_LIBRARY=fftw clean; make DBL=true  OMP=true  FFTW_LIBRARY=fftw; mv fftw3 bin/fftw3_dbl_omp_fftw;
make FFTW_LIBRARY=fftw clean; make DBL=false OMP=true  FFTW_LIBRARY=fftw; mv fftw3 bin/fftw3_sgl_omp_fftw;
make FFTW_LIBRARY=fftw clean; make DBL=true  OMP=false FFTW_LIBRARY=fftw; mv fftw3 bin/fftw3_dbl_fftw;
make FFTW_LIBRARY=fftw clean; make DBL=false OMP=false FFTW_LIBRARY=fftw; mv fftw3 bin/fftw3_sgl_fftw;

```

Similarly, the following builds all executables for MKL
```
export MKLROOT=/dls_sw/apps/intel-parallel-studio/2020/mkl
make FFTW_LIBRARY=mkl clean; make DBL=true  OMP=true  FFTW_LIBRARY=mkl; mv fftw3 bin/fftw3_dbl_omp_mkl;
make FFTW_LIBRARY=mkl clean; make DBL=false OMP=true  FFTW_LIBRARY=mkl; mv fftw3 bin/fftw3_sgl_omp_mkl;
make FFTW_LIBRARY=mkl clean; make DBL=true  OMP=false FFTW_LIBRARY=mkl; mv fftw3 bin/fftw3_dbl_mkl;
make FFTW_LIBRARY=mkl clean; make DBL=false OMP=false FFTW_LIBRARY=mkl; mv fftw3 bin/fftw3_sgl_mkl;
```


## Running tests
Tests can be run for a fixed transform size and varying number of threads. A comparison between FFTW and MKL highlights the superiority of the MKL implementation.

The folloing test template can be used to generate the walltime numbers:
```
for nt in 1 2 4 8 16; do
  tm=$(fftw3_dbl_omp_fftw -d 3 -n 512 512 512 -t $nt | tail -1 | awk '{print $NF}')
  echo $nt $tm
done
```


## References

<a id="1">[1]</a>
https://www.fftw.org/

<a id="2">[2]</a>
https://www.intel.com/content/www/us/en/developer/articles/technical/onemkl-ipp-choosing-an-fft.html
