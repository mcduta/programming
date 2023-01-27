/* filename: shm_read.c */
#include "shm_read_write.h"


/*
  ================================================================
  main -- fills up the shared memory buffer
          in a cycle
  ================================================================
*/
int main(int argc, char *argv[]) {

  // shared memory variables
  int shmid;
  struct shm_segment *shmp;

  // timing variable
  double secs;

  // other
  int repeats;
  char *buffer;


  printf (" reading from shmem: start...\n");


  // heap allocated buffer
  buffer = (char *) malloc( sizeof(char) * SHM_RW_BUF_SIZE );
  if (buffer == NULL) {
    perror ("malloc: buffer");
    return EXIT_FAILURE;
  }

  // create a shared memory segment
  shmid = shmget (SHM_RW_KEY, sizeof(struct shm_segment), 0644 | IPC_CREAT);
  if (shmid == -1) {
    perror ("shmget: create shared memory segment");
    return EXIT_FAILURE;
  }

  // attach shared memory segment to process and get a pointer to it
  shmp = shmat (shmid, NULL, 0);
  if (shmp == (void *) -1) {
    perror("shmat: attach to shared memory segment");
    return EXIT_FAILURE;
  }

  // handle CTRL+C
  signal (SIGINT, sigint_handler);

  // transfer blocks of data from shared memory to stdout*/
  while (shmp->complete != 1 && keep_running) {

    // timeit
    secs = wall_clock ();

    // copy shared memory segment to buffer
    memcpy (buffer, shmp->buffer, SHM_RW_BUF_SIZE);

    // timeit
    secs = wall_clock () - secs;

    // report
    printf(" read %f MB in %fs\n", shmp->mbytes, secs);
  }

  // detach process from shared memory segment
  if (shmdt(shmp) == -1) {
    perror("shmdt");
    return EXIT_FAILURE;
  }

  // finalize
  free (buffer);
  printf (" reading from shmem: ...complete.\n");
  return EXIT_SUCCESS;
}
