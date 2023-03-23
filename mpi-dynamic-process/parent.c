# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <string.h>


# define HOSTNAME_MAX_LEN 128
# define CHILDEXE_MAX_LEN 128
# define MPI_MSG_MAX_SIZE 1024

int main (int argc, char **argv) {

  int pnproc, piproc;                             /* parent comm size and rank */
  int cnproc, ciproc;                             /* child comm size and rank */
  char childexe [CHILDEXE_MAX_LEN];               /* child executable name */
  char hostname [HOSTNAME_MAX_LEN] = "unknown";   /* hostname */
  char message  [MPI_MSG_MAX_SIZE];
  MPI_Comm MPI_COMM_CHILD;
  MPI_Status status;

  /* initialize MPI */
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &piproc);
  MPI_Comm_size (MPI_COMM_WORLD, &pnproc);

  /* command line */
  if (argc < 3) {
    perror (" *** error: not enough arguments.");
  }
  strcpy (childexe, argv[1]);   /* name of child executable */
  cnproc = atoi(argv[2]);       /* number of child processes */

  /* spawn child processes */
  /* -- MPI_Comm_spawn is a collective  */
  /* -- first 4 args interpreted by root=0 */
  MPI_Comm_spawn (childexe, MPI_ARGV_NULL, cnproc, MPI_INFO_NULL, 0,
                  MPI_COMM_SELF, &MPI_COMM_CHILD, MPI_ERRCODES_IGNORE);

  /* get parent process hostname (basename, no domain) */
  if (gethostname (hostname, HOSTNAME_MAX_LEN) == -1) {
    perror("gethostname");
  }

  /* parent report */
  printf (" parent %d/%d @ %s\n", piproc, pnproc, hostname);

  /* child report */
  for (ciproc=0; ciproc<cnproc; ciproc++) {
    MPI_Recv (message, MPI_MSG_MAX_SIZE, MPI_CHARACTER, ciproc, 99, MPI_COMM_CHILD, &status);
    printf (" parent %d: %s\n", piproc, message);
  }

  /* finalize MPI */
  MPI_Barrier (MPI_COMM_WORLD);
  MPI_Finalize ();

  return EXIT_SUCCESS;
}
