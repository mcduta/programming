/* filename: shm_write.c */
# include "shm_read_write.h"


/*
  ================================================================
  fill_char_buffer -- function to fill up buffer
                      with characters using memset
  ================================================================
*/
void fill_char_buffer (char *bufptr, const int bufsize) {

  // character ch, init on A, persistene between calls
  static char ch = 'A';

  // fill buffer up with character ch
  memset (bufptr, ch, bufsize - 1);
  bufptr [bufsize-1] = '\0';

  // to make things more interesting, character ch cycles
  // between calls from A=65 to Z=90
  ch = (ch < 90) ? ch+1 : 65;

}


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


  printf (" writing to shmem: start...\n");

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

  // transfer blocks of data from buffer to shared memory
  for (repeats = 0; repeats < SHM_RW_REPEATS && keep_running; repeats++) {

    // fill heap allocated buffer
    fill_char_buffer (buffer, SHM_RW_BUF_SIZE);

    // timeit
    secs = wall_clock ();

    // copy buffer to shared memory segment
    memcpy (shmp->buffer, buffer, SHM_RW_BUF_SIZE);

    // timeit
    secs = wall_clock () - secs;

    // set struct values
    shmp->mbytes = ((float) sizeof(char)*SHM_RW_BUF_SIZE) / ((float) MEGA);
    shmp->complete = 0;

    // report
    printf(" wrote %f MB in %fs\n", shmp->mbytes, secs);

  }

  shmp->complete = 1;

  // detach process from shared memory segment
  if (shmdt(shmp) == -1) {
    perror ("shmdt");
    return EXIT_FAILURE;
  }

  // remove shared memoruy segment
  if (shmctl (shmid, IPC_RMID, 0) == -1) {
    perror ("shmctl");
    return EXIT_FAILURE;
  }


  // finalize
  free (buffer);
  printf (" writing to shmem: ...complete.\n");
  return EXIT_SUCCESS;
}
