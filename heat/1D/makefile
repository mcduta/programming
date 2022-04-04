# compilers
COMPILER ?= gnu

ifeq ($(COMPILER), intel)
  CC       = icc
  CFLAGS   =
  OMPFLAGS = -openmp
  LDFLAGS  = 
endif

ifeq ($(COMPILER), gnu)
  CC       = gcc
  CFLAGS   =
  OMPFLAGS = -fopenmp
  LDFLAGS  = -lm
endif

MPICC = mpicc


.SUFFIXES:
.SUFFIXES: .c .h .o .oomp .ompi

# transformation rules
.c.o:
	$(CC) $(CFLAGS) $(COPTS) -c $<

.c.oomp:
	$(CC) $(CFLAGS) $(COPTS) $(OMPFLAGS) -c $<
	mv $*.o $*.oomp

.c.ompi:
	$(MPICC) $(CFLAGS) $(COPTS) -c $<
	mv $*.o $*.ompi

# dependencies
default:	all

all:		heat heat_omp heat_mpi

heat:		heat.o
		$(CC) $(CFLAGS) $(COPTS) -o heat heat.o $(LDFLAGS)
		/bin/rm heat.o

heat_omp:	heat_omp.oomp
		$(CC) $(CFLAGS) $(COPTS) $(OMPFLAGS) -o heat_omp heat_omp.oomp $(LDFLAGS)
		/bin/rm heat_omp.oomp

heat_mpi:	heat_mpi.ompi
		$(MPICC) $(CFLAGS) $(COPTS) -o heat_mpi heat_mpi.ompi $(LDFLAGS)
		/bin/rm heat_mpi.ompi

# doc:      heat.tex
# 	  latex heat.tex
# 	  dvips heat -o
# 	  ps2pdf14 heat.ps heat.pdf
# 	  /bin/rm heat.aux heat.dvi heat.log heat.ps

clean:
		/bin/rm -f try *.o *.ompi *.c~ *.h~ Makefile~ heat.pdf
