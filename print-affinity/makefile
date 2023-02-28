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


# compiler$(OMPFLAGS)
CC        = gcc
NVCC      = nvcc
MPICC     = mpicc

# flags
CPPFLAGS  = -Wpedantic -Wall -std=c99 -Wunused-but-set-variable
CFLAGS    = -O2
OMPFLAGS  = -fopenmp

# libraries
LIBS      = -lnuma
NVLIBS    = -lcuda -L$(CUDA_PATH)/lib64 -lcudart


# target
all_cpu:	paff      paff_mpi
all_cuda:	paff_cuda paff_mpi_cuda
all:		all_cpu all_cuda


#
# --- build CPU only
paff:		paff.c
		$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o paff paff.c $(LIBS)
paff_mpi:	paff.c
		$(MPICC) -D_MPI $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) -o paff_mpi paff.c $(LIBS)


#
# --- build CPU and GPU
OBJ = paff.o paff_utils.o
paff_utils.o:	paff_utils.cu
		$(NVCC) -c paff_utils.cu

paff_cuda:	paff.c paff_utils.o
		$(CC) -D_CUDA $(OMPFLAGS) -c paff.c
		$(CC) $(OMPFLAGS) $(OBJ) -o paff_cuda $(LIBS) $(NVLIBS)
paff_mpi_cuda:	paff.c paff_utils.o
		$(MPICC) -D_CUDA -D_MPI $(OMPFLAGS) -c paff.c
		$(MPICC) $(OMPFLAGS) $(OBJ) -o paff_mpi_cuda $(LIBS) $(NVLIBS)

# clean
tidy:
		/usr/bin/rm -f *.o *.c~ *.cu~ *.h~
clean:		tidy
		/usr/bin/rm -f paff paff_mpi paff_cuda paff_mpi_cuda
