program demo

  ! nothing implicit
  implicit none

  ! double precision
  integer, parameter :: dp = selected_real_kind(15, 307)

  ! arguments
  character(len=32) :: arg

  ! vectors to work on
  integer :: n, i
  real (kind = dp), dimension(:), allocatable :: x,y

  ! timing
  integer :: time_start, time_stop, time_rate
  real (kind = dp) :: time_elap


  ! dimension n is read from command line
  if (command_argument_count() < 1) then
     n = 1024
  else if (command_argument_count() == 1) then
     call get_command_argument (1, arg)
     read (arg, *) n
  else
     write(*,*) ' *** error: only one argument expected'
     stop
  end if

  if ( n <= 0 ) then
     write(*,*) ' *** error: zero or negative arguments found'
     stop
  end if


  ! allocate and initialise vectors
  allocate (x(1:n), y(1:n))
  call random_number( x(1:n) )


  ! baseline timing (adds & mults)
  call system_clock( time_start, time_rate )
  do i = 1, n
     y(i) = 1.0_dp + x(i) * (1.0_dp + 0.5_dp * x(i))
  end do
  call system_clock( time_stop, time_rate )
  time_elap = real( time_stop - time_start, 8 ) / time_rate
  write (*, *) " baseline performance (adds and mults) : ", time_elap


  ! intrinsic 1: exp
  call system_clock( time_start, time_rate )
  do i = 1, n
     y(i) = exp(x(i))
  end do
  call system_clock( time_stop, time_rate )
  time_elap = real( time_stop - time_start, 8 ) / time_rate
  write (*, *) " intrinsic 1 (exp) : ", time_elap


  ! intrinsic 2: sin
  call system_clock( time_start, time_rate )
  do i = 1, n
     y(i) = sin(x(i))
  end do
  call system_clock( time_stop, time_rate )
  time_elap = real( time_stop - time_start, 8 ) / time_rate
  write (*, *) " intrinsic 2 (sin) : ", time_elap


  ! free memory
  deallocate (x, y)

end program demo
