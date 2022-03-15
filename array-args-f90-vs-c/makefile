compiler ?= GNU

ifeq ($(compiler), GNU)
  CC = gcc
  FC = gfortran
  CFLAGS = -O2 -march=native -Wall -std=c99
  FFLAGS = -O2 -march=native -Wall
endif

ifeq ($(compiler), INTEL)
  CC = icc
  FC = ifort
  CFLAGS = -O2 -xHost -Wall -restrict
  FFLAGS = -O2 -xHost -warn all
endif


OBJ = driver.o cfunc.o ffunc.o

.SUFFIXES:
.SUFFIXES: .c .f90 .o

.c.o:
	$(CC) $(CFLAGS) -c $<

.f90.o:
	$(FC) $(FFLAGS) -c $<

.PHONY:	all
all:	demo

demo:	$(OBJ)
	$(CC) -o demo $(OBJ)

.PHONY:	clean
clean:
	/bin/rm -f *.o demo *__genmod.*
