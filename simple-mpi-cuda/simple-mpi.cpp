/*

   simple.cpp -- simple example to demonstrate the combined use of MPI and CUDA

   tasks:
     * each MPI process allocates one array on CPU memory and initialises it with random numbers
     * each MPI process allocates two arrays on GPU and
        - makes a copy of the CPU array into one of the GPU arrays;
	- runs a GPU kernel to calculate the sqrt() of each entry of the first GPU array into the second GPU array;
        - makes a copy of the second GPU array back to the CPU memory array
	* the size of each GPU arrays is roughly an 8th of the available memory

*/

//
// --- includes
# include <mpi.h>
# include <iostream>
# include <cstdlib>
# include <getopt.h>
# include "simple-mpi-cuda.hpp"


//
// --- MPI error handling macro
# define MPI_SAFE_CALL(MPI_CALL) \
  if((MPI_CALL) != MPI_SUCCESS) { \
    std::cerr << "MPI error calling \""#MPI_CALL"\"" << std::endl; \
    simple_mpi_abort (-1); \
  }


//
// MPI abort wrapper function
void simple_mpi_abort (const int mpi_err) {
  MPI_Abort (MPI_COMM_WORLD, mpi_err);
}


//
// MPI command line help function
void simple_mpi_help () {
  std::cout << " *** Usage: simple-mpi-cuda -g -s" << std::endl;
  std::cout << "     -g -- report device for each MPI process (default: NO)" << std::endl;
  std::cout << "     -s -- automatically set device for each MPI process (default: NO)" << std::endl;
}


// ================================================================ //
// --- main code                                                    //
// ================================================================ //

int main (int argc, char *argv[]) {

  // dataset dimensions
  int block_size = 256,
      grid_size  = 10000;

  // initialize MPI
  MPI_SAFE_CALL ( MPI_Init (&argc, &argv) );

  // MPI rank and count
  int mpi_proc_num, mpi_proc_id;
  MPI_SAFE_CALL ( MPI_Comm_size (MPI_COMM_WORLD, &mpi_proc_num) );
  MPI_SAFE_CALL ( MPI_Comm_rank (MPI_COMM_WORLD, &mpi_proc_id) );

  // process args
  int option=0;
  int get_device = 0,
      set_device = 0;
  while ( (option = getopt(argc, argv, "sgh")) != -1 ) {
    switch (option) {
      case 'g' : get_device = 1; break;
      case 's' : set_device = 1; break;
      case 'h' : if (mpi_proc_id == 0) simple_mpi_help (); simple_mpi_abort (-1);
    }
  }

  // set device
  if (set_device) mpi_set_device (mpi_proc_num, mpi_proc_id);

  // get device
  if (get_device) mpi_get_device (mpi_proc_num, mpi_proc_id);

  // calculate
  mpi_data_process (mpi_proc_id, block_size,grid_size);

  // clean up
  MPI_SAFE_CALL ( MPI_Finalize() );

  std::exit (EXIT_SUCCESS);
}
