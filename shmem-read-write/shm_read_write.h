/*
  adapted from

    https://www.tutorialspoint.com/inter_process_communication/inter_process_communication_shared_memory.htm

 */
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <errno.h>
# include <string.h>
# include <signal.h>
# include <unistd.h>
# include <sys/ipc.h>
# include <sys/shm.h>
# include <sys/types.h>
# include <sys/time.h>


/* read / write parameters */
# define KILO 1024
# define MEGA 1024*KILO
# define GIGA 1024*MEGA

# define SHM_RW_BUF_SIZE GIGA
# define SHM_RW_REPEATS 128
# define SHM_RW_KEY 0x1234


/*
  ================================================================
  shm_segment -- read / write structure
  ================================================================
*/
struct shm_segment {
  float mbytes;
  int complete;
  char buffer[SHM_RW_BUF_SIZE];
};


/*
  ================================================================
  sigint_handler -- function to handle CTRL+C
  ================================================================
*/
static volatile int keep_running = 1;

void sigint_handler (int dummy) {
  keep_running = 0;
}


/*
  ================================================================
  wall_clock -- timing function
  ================================================================
*/

double wall_clock (void){

  # define MILLION 1000000.0

  double secs;
  struct timeval tp;

  gettimeofday (&tp,NULL);
  secs = (MILLION * (double) tp.tv_sec + (double) tp.tv_usec) / MILLION;
  return secs;

}
