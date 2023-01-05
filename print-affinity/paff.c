#define _GNU_SOURCE
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

# define HOSTNAME_MAX_LEN  64   /* max hostname length before truncation */
# define MPI_INFO_MAX_LEN  32   /* max length for OpenMP info */
# define OMP_INFO_MAX_LEN  32   /* max length for OpenMP info */
# define AFFINITY_MAX_LEN 256   /* max length for affinity info */


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
        sprintf(ptr, "%d,", i);                           /* to return string */
      } else {                                            /* in human readable */
        sprintf(ptr, "%d-%d,", i,j-1);                    /* format */
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
  char mpi_info [MPI_INFO_MAX_LEN] = "";               /* MPI info */
  char omp_info [OMP_INFO_MAX_LEN] = "";               /* OpenMP info */
  char affinity [AFFINITY_MAX_LEN] = "";               /* affinity info */
  char hostname [HOSTNAME_MAX_LEN] = "unknown_host";   /* host */
  int cpu, node;                                       /* CPU and NUMA node */
  cpu_set_t mask;                                      /* taskset info */

 
  /*
    MPI
  */
# ifdef _MPI
  int nproc, iproc;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &iproc);
  MPI_Comm_size (MPI_COMM_WORLD, &nproc);
  sprintf (mpi_info, "MPIrank=%d ", iproc);
# endif


 /*
    OpenMP
  */
# ifdef _OPENMP
  int nthread, ithread;
  # pragma omp parallel default(none) \
    shared (mpi_info) \
    private (nthread,ithread, omp_info, hostname,cpu,node, mask, affinity)
    {
# endif


  /*
    OpenMP info
  */
# ifdef _OPENMP
  nthread = omp_get_num_threads ();
  ithread = omp_get_thread_num ();
  sprintf (omp_info, "OMPthread=%d ", ithread);
# endif


  /*
    get hostname
  */
  if (gethostname (hostname, HOSTNAME_MAX_LEN) == -1) {
    perror("gethostname");
  }
  str2base (hostname);

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
    get affinity
  */
  if (sched_getaffinity (0, sizeof(mask), &mask) == -1) {
    perror("sched_getaffinity");
  }
  cpuset2str (&mask, affinity);


  /*
    report
  */
  printf("hostname=%s %s%sCPU=%d NUMAnode=%d affinity=%s\n", hostname, mpi_info, omp_info, cpu, node, affinity);


# ifdef _OPENMP
    } /* OpenMP parallel region*/
 # endif


# ifdef _MPI
  MPI_Finalize();
# endif

  return EXIT_SUCCESS;
}
