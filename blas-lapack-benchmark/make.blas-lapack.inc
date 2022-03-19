CC = gcc
FC = gfortran

CPPFLAGS = 
OMPFLAGS = -fopenmp
CFLAGS   = -Wall -g -O3 -mavx2 $(OMPFLAGS)
FFLAGS   = $(CFLAGS)

LIBS = -lblas -llapack
