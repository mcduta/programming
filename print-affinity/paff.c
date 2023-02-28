# define _GNU_SOURCE /* defines CPU_SETSIZE */
# include <stdio.h>
# include <stdlib.h>
# include <sched.h>
# include <numa.h>
# include <unistd.h>
# ifdef _MPI
# include <mpi.h>
# endif
# ifdef _OPENMP
# include <omp.h>
# endif

// utils: cuda_get_device_id
# include "paff_utils.h"

# define HOSTNAME_MAX_LEN     128   /* max hostname length before truncation */
# define AFFINITY_MAX_LEN     256   /* max length for affinity info */
# define REPORT_MAX_LEN      1024   /* max length for report info */
# define REPORT_PART_MAX_LEN  256   /* max length for part report info */


/*
  -----------------------------------------
  cpuset2str -- cpu set to printable string
  -----------------------------------------
  */
void cpuset2str (cpu_set_t *mask, char *str) {
  char *ptr = str;                                        /* pointer to string sprintf start */
  int i=0, j=0, go=1;                                     /* counters, etc. */
  while (go) {                                            /* go through all cpuset entries */
    while ((i < CPU_SETSIZE) & !CPU_ISSET(i, mask)) i++;  /* skip the 0s */
    j = i+1;
    while ((j < CPU_SETSIZE) &  CPU_ISSET(j, mask)) j++;  /* go past all 1s */
    if ((i >= CPU_SETSIZE) || (j >= CPU_SETSIZE)) {       /* if out of cpuset range */
      go = 0;                                             /* finished */
    } else {  
      if (i==j-1) {                                       /* write cpu ids */
        sprintf (ptr, "%d,", i);                          /* to return string */
      } else {                                            /* in human readable */
        sprintf (ptr, "%d-%d,", i,j-1);                   /* format */
      }
      i = j;                                              /* start with the next group of 0s in next round */
      while (*ptr != 0) ptr++;                            /* adjust string position for next sprintf */
    }   
  }
  *(--ptr) = 0;                                           /* remove last comma */
}

/*
  -----------------------------------------
  name2base -- get fully qualified hostname and return base
  -----------------------------------------
  */
void str2base (char *str) {
  while (*str != '.') str++;                              /* find first stop */
  *str = 0;                                               /* basename is evrything before that stop */
}

/*
  ----
  main
  ----
   */
int main (int argc, char **argv) {

  /*
    info strings
  */
  char affinity    [AFFINITY_MAX_LEN] = "\0";          /* affinity info */
  char hostname    [HOSTNAME_MAX_LEN] = "unknown";     /* name of host machine */
  char report      [REPORT_MAX_LEN] = "\0";            /* report string printed to stdout */
  char report_part [REPORT_PART_MAX_LEN] = "\0";       /* string to be strcat to report */
  int cpu, node, gpu;                                  /* CPU, NUMA node and GPU IDs */
  cpu_set_t mask;                                      /* taskset info */
  int nproc, iproc;                                    /* MPI proc info */
  int err;                                             /* error code */

 
  /*
    MPI
  */
# ifdef _MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &iproc);
  MPI_Comm_size (MPI_COMM_WORLD, &nproc);
# endif


  /*
    get hostname
  */
  if (gethostname (hostname, HOSTNAME_MAX_LEN) == -1) {
    perror("gethostname");
  }
  str2base (hostname);


 /*
    OpenMP
  */
# ifdef _OPENMP
  int nthread, ithread;
  # pragma omp parallel default(none) \
    shared (iproc, hostname) \
    private (nthread,ithread, cpu,node,gpu, mask,affinity, err)	\
    firstprivate (report,report_part)
    {
# endif


  /*
    OpenMP info
  */
# ifdef _OPENMP
  nthread = omp_get_num_threads ();
  ithread = omp_get_thread_num ();
# endif


  /*
    get cpu and node
    note: use the glibc sched_getcpu rather than the kernel getcpu,
          see https://man7.org/linux/man-pages/man3/sched_getcpu.3.html
  */
  cpu = sched_getcpu();
  if (cpu == -1) perror ("sched_getcpu");
  node = numa_node_of_cpu(cpu);
  if (node == -1) perror ("numa_node_of_cpu");


  /*
    get cpu affinity
  */
  if (sched_getaffinity (0, sizeof(mask), &mask) == -1) {
    perror ("sched_getaffinity");
  }
  cpuset2str (&mask, affinity);


  /*
    set gpu affinity
  */
# ifdef _CUDA
  err = 0;
# ifdef _MPI
  err = cuda_set_device_id (iproc);
# else
# ifdef _OPENMP
  err = cuda_set_device_id (ithread);
# else
  err = cuda_set_device_id (0);
# endif
# endif
  if (err == -1) perror("cuda_set_device_id");
# endif


  /*
    get gpu affinity (per thread, per process)
  */
# ifdef _CUDA
  gpu = cuda_get_device_id ();
  if (gpu == -1) perror ("cuda_get_device");
# endif


  /*
    report
  */
  sprintf (report_part, "Host=%s", hostname); strcat (report, report_part);
# ifdef _MPI
  sprintf (report_part, " MPIrank=%d", iproc); strcat (report, report_part);
# endif
# ifdef _OPENMP
  if (nthread > 1) sprintf (report_part, " OMPthread=%d", ithread); strcat (report, report_part);
# endif
  sprintf (report_part, " CPU=%d", cpu); strcat (report, report_part);
# ifdef _CUDA
  sprintf (report_part, " GPU=%d", gpu); strcat (report, report_part);
# endif
  sprintf (report_part, " NUMAnode=%d Affinity=%s", node, affinity); strcat (report, report_part);

   puts(report);


# ifdef _OPENMP
    } /* OpenMP parallel region*/
# endif


# ifdef _MPI
  MPI_Finalize ();
# endif

  return EXIT_SUCCESS;
}
