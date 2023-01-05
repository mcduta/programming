#======================================================================#
#                                                                      #
#      makefile -- build tools to print out affinity info              #
#                    * single and multi-threaded (OpenMP)              #
#                    * single and multiple processes (MPI)             #
#               -- tested with gcc                                     #
#               -- intended use: test slurm srun affinity              #
#                  control setup                                       #
#                                                                      #
#======================================================================#


# compiler
CC    = gcc
MPICC = mpicc

# flags
CPPFLAGS = -Wpedantic -Wall -std=c99 -Wunused-but-set-variable
CFLAGS   = -O2
OMPFLAGS = -fopenmp

# libraries
LIBS = -lnuma

# target
all:		paff paff_omp paff_mpi paff_mpi_omp

# build
paff:		paff.c
		$(CC) $(CPPFLAGS) $(CFLAGS) -o paff paff.c $(LIBS)
paff_omp:	paff.c
		$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o paff_omp paff.c $(LIBS)
paff_mpi:	paff.c
		$(MPICC) -D_MPI $(CPPFLAGS) $(CFLAGS) -o paff_mpi paff.c $(LIBS)
paff_mpi_omp:	paff.c
		$(MPICC) -D_MPI $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o paff_mpi_omp paff.c $(LIBS)

# clean
clean:
		/usr/bin/rm -f paff paff_omp paff_mpi paff_mpi_omp
