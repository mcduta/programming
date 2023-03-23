# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>

# define HOSTNAME_MAX_LEN 128
# define MPI_MSG_MAX_SIZE 1024


int main (int argc, char **argv) {

  int cnproc, ciproc;   /* child comm size and rank */
  int pnproc, piproc;   /* parent comm size and rank */
  char hostname [HOSTNAME_MAX_LEN] = "unknown";
  char message  [MPI_MSG_MAX_SIZE];
  MPI_Comm MPI_COMM_PARENT;

  /* initialize MPI */
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &ciproc);
  MPI_Comm_size (MPI_COMM_WORLD, &cnproc);

  /* get parent process info */
  /* -- MPI_Comm_get_parent returns the comm created by MPI_Comm_spawn */
  /* -- for a child procs, this comm is the same as MPI_COMM_WORLD */
  MPI_Comm_get_parent (&MPI_COMM_PARENT);
  if (MPI_COMM_PARENT == MPI_COMM_NULL) perror ("MPI_Comm_get_parent: no process parent.");
  MPI_Comm_remote_size(MPI_COMM_PARENT, &pnproc);

  /* get hostname */
  if (gethostname (hostname, HOSTNAME_MAX_LEN) == -1) {
    perror("gethostname");
  }

  /* form message */
  sprintf (message, "child %d/%d @ %s", ciproc, cnproc, hostname);

  /* send message to parent */
  MPI_Send (message, MPI_MSG_MAX_SIZE, MPI_CHARACTER, 0, 99, MPI_COMM_PARENT);

  /* finalize MPI */
  MPI_Barrier (MPI_COMM_WORLD);
  MPI_Finalize ();

  return EXIT_SUCCESS;
}
