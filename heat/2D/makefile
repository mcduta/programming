
# INTEL compilers
CC = gcc
MPICC = mpicc

# compilation flag
COPTS = -O2 -mtune=native -g

CFLAGS  =
LDFLAGS = -lm


.SUFFIXES:
.SUFFIXES: .c .h .o .ompi

# transformation rules
.c.o:
	$(CC) $(CFLAGS) $(COPTS) -c $<

.c.ompi:
	$(MPICC) $(CFLAGS) $(COPTS) -c $<
	mv $*.o $*.ompi

# dependencies
default:  all

all:      heat heat_mpi

heat:     heat.o
	  $(CC) $(CFLAGS) $(COPTS) -o heat heat.o $(LDFLAGS)
	  /bin/rm heat.o

heat_mpi: heat_mpi.ompi
	  $(MPICC) $(CFLAGS) $(COPTS) -o heat_mpi heat_mpi.ompi $(LDFLAGS)
	  /bin/rm heat_mpi.ompi

heat_mpi_perf:	heat_mpi.ompi
		hpclink $(MPICC) $(CFLAGS) $(COPTS) -o heat_mpi_perf heat_mpi.ompi $(LDFLAGS)

doc:      heat.tex
	  latex heat.tex
	  dvips heat -o
	  ps2pdf14 heat.ps heat.pdf
	  /bin/rm heat.aux heat.dvi heat.log heat.ps

clean:
	/bin/rm -f try *.o *.ompi *.c~ *.h~ Makefile~ heat.pdf
