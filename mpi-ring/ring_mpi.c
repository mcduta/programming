
/*

  ring.c -- non-blocked communication in a ring.

 */


# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <getopt.h>
# include <mpi.h>

# define NPROC_MAX 64

// ------------------------------------------------------------------- //
//                                                                     //
//                              H E L P                                //
//                                                                     //
// ------------------------------------------------------------------- //
void print_ring_help () {
  printf(" *** usage: ring_mpi -s msg_size [-l num_loops -f out_freq]\n");
}

// ------------------------------------------------------------------- //
//                                                                     //
//                             M  A  I  N                              //
//                                                                     //
// ------------------------------------------------------------------- //

int main(int narg, char** varg) {

  // MPI variables
  int    iproc,                 // number of this process
         nproc;                 // total number of processes

  // message info
  int	   buffsize=0,            // buffer size
         iproc_prev,            // process from which iproc receives
         iproc_next;            // process to which iproc sends

  // data buffers
  double *sendbuff, *recvbuff,  // sending, receiving data buffers
         sendbuffMean,          // mean of sent data
         recvbuffMean,          // mean of received data
         sendbuffMeanArray[NPROC_MAX], // gathered means (before communication)
         recvbuffMeanArray[NPROC_MAX]; // gathered means (after communication)

  // loops
  int    iloop,nloop=1,ofreq=1;

  // MPI variables
  MPI_Status  status;
  MPI_Request send_request, recv_request;

  // timing variables
  double time_start, time_end, time_tot;

  // extra vars
  int    ibuf, ip;
  double checkDiff;
  int    option=0;


  //
  // ----- init MPI starts parallel independent processes
  //
  MPI_Init (&narg, &varg);


  //
  // ----- this process obtains total number of processes (nproc) and own number (proc)
  //
  MPI_Comm_size (MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank (MPI_COMM_WORLD, &iproc);

  if (nproc > NPROC_MAX) {
    if (iproc == 0) fprintf(stderr, " *** error: too many processes required\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }


  //
  // ----- process arguments
  //
  while ( (option = getopt(narg, varg, "s:l:f:")) != -1 ) {
    switch (option) {
      case 's' : buffsize = atoi(optarg);
        break;
      case 'l' : nloop = atoi(optarg);
        break;
      case 'f' : ofreq = atoi(optarg);
        break;
      default: if (iproc == 0) print_ring_help();
        MPI_Abort(MPI_COMM_WORLD, -2);
    }
  }

  if (buffsize==0 || nloop<0 || ofreq<0) {
    if (iproc == 0) print_ring_help();
    MPI_Abort(MPI_COMM_WORLD, -3);
  }

  //
  // ----- allocate memory
  //
  sendbuff = (double *) malloc(buffsize * sizeof(double));
  recvbuff = (double *) malloc(buffsize * sizeof(double));


  //
  // ----- neighbouring process (messages go round in a circle)
  //
  iproc_prev = iproc - 1;
  iproc_next = iproc + 1;

  if (iproc_prev == -1)    iproc_prev = nproc-1;
  if (iproc_next == nproc) iproc_next = 0;


  //
  // ----- average of communication time
  //
  time_tot = 0.0;


  //
  // ----- loop a number of times
  //
  for (iloop=0; iloop<nloop; iloop++) {

    //
    // ----- output minimal info
    //
    if (iproc == 0 && (iloop%ofreq) == 0 && nloop > 1) {
      fprintf(stdout, " iteration %d...\n", iloop);
    }


    //
    // ----- initialise
    //
    srand (time(NULL) + iproc);

    for(ibuf=0; ibuf<buffsize; ibuf++) {
      sendbuff[ibuf] = (double)rand() / RAND_MAX;
    }


    //
    // ----- print out mean values before communication
    //

    // compute mean
    sendbuffMean = 0.0;
    for (ibuf=0; ibuf<buffsize; ibuf++) {
      sendbuffMean += sendbuff[ibuf];
    }
    sendbuffMean /= ((double) buffsize);


    // root process gather mean values
    MPI_Gather (&sendbuffMean, 1, MPI_DOUBLE,
                sendbuffMeanArray, 1, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // print
    if (nloop == 1 && iproc == 0) {
      fprintf(stdout, "\n ----- before communication\n");
      for (ip=0; ip<nproc; ip++) {
        fprintf(stdout, " process %d: mean of data sent to %d = %e\n",
                ip, (ip+1)%nproc, sendbuffMeanArray[ip]);
      }
    }


    //
    // ----- communication
    //
    time_start = MPI_Wtime();

    MPI_Isend (sendbuff, buffsize, MPI_DOUBLE,
               iproc_next, 0, MPI_COMM_WORLD, &send_request);

    MPI_Irecv (recvbuff, buffsize, MPI_DOUBLE,
               iproc_prev, 0, MPI_COMM_WORLD, &recv_request);

    MPI_Wait(&send_request, &status);
    MPI_Wait(&recv_request, &status);

    time_end = MPI_Wtime();

    time_tot += time_end - time_start;


    //
    // ----- print out mean values after communication
    //

    // compute mean
    recvbuffMean = 0.0;
    for (ibuf=0; ibuf<buffsize; ibuf++) {
      recvbuffMean += recvbuff[ibuf];
    }
    recvbuffMean /= ((double) buffsize);


    // root process gather mean values
    MPI_Gather (&recvbuffMean, 1, MPI_DOUBLE,
                recvbuffMeanArray, 1, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // print
    if (nloop == 1 && iproc == 0) {
      fprintf(stdout, "\n ----- after communication\n");

      // print mean values
      for (ip=0; ip<nproc; ip++) {
        fprintf(stdout, " process %d: mean of data received = %e\n",
                ip, recvbuffMeanArray[ip]);
      }
    }

    // check results
    if (iproc == 0) {
      checkDiff = 0.0;
      for (ip=0; ip<nproc; ip++) {
        checkDiff += abs( sendbuffMeanArray[ip]
                          - recvbuffMeanArray[(ip+1)%nproc] );
      }

      if (nloop == 1) {
        if (checkDiff < 1.e-14) {
          fprintf(stdout, "\n correct communication\n");
        } else {
          fprintf(stdout, "\n correct communication\n");
        }
      } else {
        if (checkDiff > 1.e-14) {
          fprintf(stderr, " *** warning: possible communication problem, checkDiff = %e\n", checkDiff);
        }
      }

    }

  }


  //
  // ----- average communication time
  //
  if (iproc == 0) {
    fprintf(stdout, " communication time = %e\n", time_tot / ((double) nloop));
  }


  //
  // ----- finalise MPI
  //
  free(sendbuff);
  free(recvbuff);


  //
  // ----- finalise MPI
  //
  MPI_Finalize ( );


  return 0;
}


/*
  end
 */
