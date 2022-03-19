# include make.blas-lapack.inc
# include make.openblas.inc
# include make.gfortran-mkl.inc
include make.ifort-mkl.inc


#
# ----- RULES
#


.SUFFIXES:
.SUFFIXES: .f90 .o

.f90.o:
		$(FC) $(FFLAGS) -c $<

default:	all

all:		perf_dgemm perf_dgesvd

perf_dgemm:	perf_dgemm.o
		$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

perf_dgesvd:	perf_dgesvd.o
		$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

clean:
		/bin/rm -f perf_dgemm perf_dgesvd *.o
realclean:	clean
		/bin/rm -f *.f90~ makefile~ make*.inc~
