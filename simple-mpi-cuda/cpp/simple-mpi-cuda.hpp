# ifndef REAL
# define REAL float
# endif


// forward declarations
extern "C" {
  void simple_mpi_abort (int err);
  void mpi_get_device (const int mpi_proc_num, const int mpi_proc_id);
  void mpi_set_device (const int mpi_proc_num, const int mpi_proc_id);
  void data_cpu_init (const int size, REAL *data);
  void mpi_data_process (const int mpi_rank, const int block_size, const int grid_size);
}
