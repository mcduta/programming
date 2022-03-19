CC = icc
FC = ifort

MKLROOT ?= /dls_sw/apps/intel/oneapi/2021/mkl/latest

CPPFLAGS = -I$(MKLROOT)/include
OMPFLAGS = -qopenmp
CFLAGS   = -Wall -O3 -mavx2 $(OMPFLAGS) -mkl=parallel
FFLAGS   = -warn all -O3 -mavx2 $(OMPFLAGS) -mkl=parallel

LIBS_STATIC  = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

LIBS_DYNAMIC = -liomp5 -lpthread -lm -ldl

LIBS = $(LIBS_DYNAMIC)
