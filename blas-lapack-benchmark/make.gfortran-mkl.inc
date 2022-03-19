CC = gcc
FC = gfortran

MKLROOT ?= /dls_sw/apps/intel/oneapi/2021/mkl/latest

CPPFLAGS = -I$(MKLROOT)/include
OMPFLAGS = -fopenmp
CFLAGS   = -Wall -O3 -mavx2 $(OMPFLAGS)
FFLAGS   = $(CFLAGS)

LIBS_STATIC  = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_gf_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

LIBS_DYNAMIC = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

LIBS = $(LIBS_DYNAMIC)
