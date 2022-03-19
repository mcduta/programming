CC = gcc
FC = gfortran

OBLASROOT = /dls_sw/apps/openblas/0.3.10

CPPFLAGS = -I $(OBLASROOT)/include
OMPFLAGS = -fopenmp
CFLAGS   = -Wall -g -O3 -mavx2 $(OMPFLAGS)
FFLAGS   = $(CFLAGS)

LIBS = -L $(OBLASROOT)/lib -Wl,-rpath=$(OBLASROOT)/lib -lopenblas
