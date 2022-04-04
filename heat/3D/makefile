
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

# compilation flag
COPTS = -O2 -mavx2 -g

ifeq ($(DEBUG), yes)
  CFLAGS += -DDEBUG
endif


# info
$(info CC is ${CC})


# transformation rules
.SUFFIXES:
.SUFFIXES: .c .h .o .ompi .oomp .ohyb

.c.o:
	$(CC) $(CFLAGS) $(COPTS) -c $<

.c.ompi:
	$(MPICC) $(CFLAGS) $(COPTS) -c $<
	mv $*.o $*.ompi

.c.oomp:
	$(CC) $(CFLAGS) $(COPTS) $(OMPFLAGS) -c $<
	mv $*.o $*.oomp

.c.ohyb:
	$(MPICC) $(CFLAGS) $(COPTS) $(OMPFLAGS) -c $<
	mv $*.o $*.ohyb


# dependencies
default:	all

all:		heat heat_mpi

heat:		heat.o
		$(CC) $(CFLAGS) $(COPTS) -o heat heat.o $(LDFLAGS)
		/bin/rm heat.o

heat_mpi:	heat_mpi.ompi
		$(MPICC) $(CFLAGS) $(COPTS) -o heat_mpi heat_mpi.ompi $(LDFLAGS)
		/bin/rm heat_mpi.ompi

heat_omp:	 heat_omp.oomp
		$(CC) $(CFLAGS) $(COPTS) $(OMPFLAGS) -o heat_omp heat_omp.oomp $(LDFLAGS)
		/bin/rm heat_omp.oomp

heat_hyb:	heat_mpi-omp.ohyb
		$(MPICC) $(CFLAGS) $(COPTS) $(OMPFLAGS) -o heat_hyb heat_mpi-omp.ohyb $(LDFLAGS)
		/bin/rm heat_mpi-omp.ohyb

doc:		heat.tex
		latex heat.tex
		dvips heat -o
		ps2pdf14 heat.ps heat.pdf
		/bin/rm heat.aux heat.dvi heat.log heat.ps

merge:
		/bin/rm -f heat_mpi.out
		for file in `/bin/ls heat_mpi.out* | sort`; do echo "adding $$file ..."; cat $$file >> heat_mpi.out; done
		/bin/rm -f heat_mpi.out_p*

clean:
		/bin/rm -f heat heat_mpi *.o *.ompi *.c~ *.h~ makefile~ heat.pdf heat*out*
