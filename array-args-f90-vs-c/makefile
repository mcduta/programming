CC = icc
FC = ifort

CFLAGS = -O2 -xHost -Wall -restrict
FFLAGS = -O2 -xHost -warn all

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
