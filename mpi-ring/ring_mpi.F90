
!
!      ring.f90 -- non-blocked communication in a ring.
!


! --------------------------------------------------------------------- !
!                                                                       !
!                              M  A  I  N                               !
!                                                                       !
! --------------------------------------------------------------------- !

program ring

  implicit none

  include 'mpif.h'

# define NPROC_MAX 64


  ! MPI variables
  integer :: iproc,             & ! number of this process
             nproc                ! total number of processes

  ! message info
  integer :: buffsize,          & ! buffer size
             iproc_prev,        & ! process from which iproc receives
             iproc_next           ! process to which iproc sends

  ! sending, receiving data buffers
  double precision, allocatable :: sendbuff(:), recvbuff(:)
  double precision ::           &
             sendbuffMean,      & ! mean of sent data
             recvbuffMean         ! mean of received data
  double precision ::           &
             sendbuffMeanArray(NPROC_MAX), & ! gathered means (before communication)
             recvbuffMeanArray(NPROC_MAX)    ! gathered means (after communication)

  ! command line arguments
  integer :: iargc
  character(len=16) :: arg

  ! loops
  integer :: iloop,nloop,outFreq

  ! MPI variables
  integer :: ierr
  integer :: status(MPI_STATUS_SIZE,2)
  integer :: send_request(1), recv_request(1)

  ! timing variables
  double precision :: time_start, time_end, time_tot

  ! random numbers
  integer :: nseed
  integer, allocatable :: seed(:)
  double precision :: rnum

  ! extra vars
  integer :: ibuf, ip
  double precision :: checkDiff

  !
  ! ----- init MPI starts parallel independent processes
  !
  call MPI_Init ( ierr );


  !
  ! ----- this process obtains total number of processes (nproc) and own number (proc)
  !
  call MPI_Comm_size ( MPI_COMM_WORLD, nproc, ierr )
  call MPI_Comm_rank ( MPI_COMM_WORLD, iproc, ierr )

  if (nproc > NPROC_MAX) then
    if (iproc == 0) then
      print *, ' *** error: too many processes required'
    end if
    call MPI_Abort ( MPI_COMM_WORLD, -1, ierr );
  end if


  !
  ! ----- process arguments
  !
  if (iargc() == 1 .or. iargc() == 2 .or. iargc() == 3) then

    ! buffer size
    call getarg(1, arg)
    read (arg, '(I16)') buffsize

    if (iargc() >= 2) then
      call getarg(2, arg)
      read (arg, '(I16)') nloop
    else
      nloop = 1
    end if

    if (iargc() == 3) then
      call getarg(3, arg)
      read (arg, '(I16)') outFreq
    else
      outFreq = 10
    end if

  else
    print *, ' *** error: wrong number of arguments'
    call MPI_Abort ( MPI_COMM_WORLD, -2, ierr )
  end if


  !
  ! ----- allocate memory
  !
  allocate (sendbuff(buffsize))
  allocate (recvbuff(buffsize))


  !
  ! ----- neighbouring process (messages go round in a circle)
  !
  iproc_prev = iproc - 1
  iproc_next = iproc + 1

  if (iproc_prev == -1)    iproc_prev = nproc-1
  if (iproc_next == nproc) iproc_next = 0


  !
  ! ----- average of communication time
  !
  time_tot = 0.0


  !
  ! ----- loop a number of times
  !
  do iloop = 1, nloop

    !
    ! ----- output minimal info
    !
    if (iproc == 0 .and. &
        modulo(iloop, outFreq) == 0 .and. &
        nloop > 1) then
      print *, ' iteration ', iloop, '...'
    end if


    !
    ! ----- initialise
    !
    nseed = 1
    allocate(seed(nseed))
    seed(1) = iproc
    call random_seed (size = nseed)
    call random_seed (put  = seed(1:nseed))


    do ibuf = 1, buffsize
      call random_number(rnum)
      sendbuff(ibuf) = rnum
    end do


    !
    ! ----- print out mean values before communication
    !

    ! compute mean
    sendbuffMean = 0.0
    do ibuf = 1, buffsize
      sendbuffMean = sendbuffMean + sendbuff(ibuf)
    end do

    sendbuffMean = sendbuffMean / dble(buffsize)


    ! root process gather mean values
    call MPI_Gather (sendbuffMean,      1, MPI_DOUBLE_PRECISION,       &
                     sendbuffMeanArray, 1, MPI_DOUBLE_PRECISION,       &
                     0, MPI_COMM_WORLD, ierr)

    ! print
    if (nloop == 1 .and. iproc == 0) then
      print *, ' ----- before communication'
      do ip = 0, nproc-1
        write(*,"(A9,I2,A23,I2,A3,E12.6)")  ' process ', ip,           &
                                            ': mean of data sent to ', &
                                            mod(ip+1, nproc), ' = ',   &
                                            sendbuffMeanArray(ip+1)
      end do
    end if


    !
    ! ----- communication
    !
    time_start = MPI_Wtime()

    call MPI_Isend (sendbuff, buffsize, MPI_DOUBLE_PRECISION,          &
                    iproc_next, 0, MPI_COMM_WORLD,                     &
                    send_request, ierr)

    call MPI_Irecv (recvbuff, buffsize, MPI_DOUBLE_PRECISION,          &
                    iproc_prev, 0, MPI_COMM_WORLD,                     &
                    recv_request, ierr)

    call MPI_Wait(send_request, status, ierr)
    call MPI_Wait(recv_request, status, ierr)

    time_end = MPI_Wtime()

    time_tot = time_tot + ( time_end - time_start )


    !
    ! ----- print out mean values after communication
    !

    ! compute mean
    recvbuffMean = 0.0
    do ibuf = 1, buffsize
      recvbuffMean = recvbuffMean + recvbuff(ibuf)
    end do
    recvbuffMean = recvbuffMean / dble(buffsize)


    ! root process gather mean values
    call MPI_Gather (recvbuffMean,      1, MPI_DOUBLE_PRECISION,       &
                     recvbuffMeanArray, 1, MPI_DOUBLE_PRECISION,       &
                     0, MPI_COMM_WORLD, ierr)

    ! print
    if (nloop == 1 .and. iproc == 0) then
      print *, ' ----- after communication'

      ! print mean values
      do ip = 0, nproc-1
        write(*,"(A9,I2,A26,E12.6)")  ' process ', ip,                 &
                                      ': mean of data received = ',    &
                                       recvbuffMeanArray(ip+1)
      end do
    end if

    ! check results
    if (iproc == 0) then
      checkDiff = 0.0
      do ip = 0, nproc-1
        checkDiff = checkDiff                                          &
                  + dabs( sendbuffMeanArray(ip+1)                      &
                        - recvbuffMeanArray(modulo(ip+1, nproc)+1) );
      end do

      if (nloop == 1) then
        if (checkDiff < 1.e-14) then
          print *, ' correct communication'
        else
          print *, ' correct communication'
        end if
      else
        if (checkDiff > 1.e-14) then
          print *, ' *** warning: possible communication problem, checkDiff = ', checkDiff
        end if
      end if

    end if


    deallocate(seed)

  end do


  !
  ! ----- average communication time
  !
  if (iproc == 0) then
    print *, ' communication time = ', time_tot / dble(nloop)
  end if


  !
  ! ----- finalise MPI
  !
  deallocate(sendbuff);
  deallocate(recvbuff);



  !
  ! ----- finalise MPI
  !
  call MPI_Finalize ( ierr )



  stop
end program ring

!
!      end
!
