!======================================================================
!
!   name:        perf_dgemm.f90
!
!   synopsis:    demonstrate and time the use of BLASS DGEMM
!
!   purpose:     provide a way to test the performance of
!                DGEMM implementations
!
!   description: This example initialises matrices A, B and C
!                and computes alpha*A*B+beta*C. The example is
!                in double precision.
!                DGEMM performs C = alpha*A*B + beta*C,
!                with alpha and beta scalars and A,B,C
!                matrices of size
!
!                  A MxK
!                  B KxN
!                  C MxN
!
!                Schematically, the algorithm is
!
!                  for j=1,N
!                    for i=1,M
!                      C(i,j) = beta*C(i,j);
!                    end
!                    for l=1,K
!                      temp = alpha*B(l,j);
!                      for i = 1,M
!                        C(i,j) = C(i,j) + temp*A(i,l);
!                      end
!                    end
!                  end
!
!                Total number of flops is ((2*M+1)*K+M)*N
!
!   compile:     ...
!
!   run:         ...
!
!======================================================================

program perf_dgemm

  ! nothing implicit
  implicit none

  ! double precision
  integer, parameter :: dp = selected_real_kind(15, 307)

  ! main variables
  integer :: m, k, n
  real (kind=dp), allocatable ::  a(:,:), b(:,:), c(:,:)
  real (kind=dp) ::  alpha, beta

  ! secondary variables
  integer :: i, j
  real (kind=dp) :: si
  character (32) :: arg

  ! timing variables
  integer :: time_start, time_finish, time_rate

  ! performance
  real (kind=dp) :: perf_gflops, perf_time


  ! dimensions m, k, n are r ead from command line
  if (command_argument_count().ne.3) then
     write(*,*) ' *** error: less than three arguments found'
     stop
  end if

  call get_command_argument (1, arg)
  read (arg, *) m
  call get_command_argument (2, arg)
  read (arg, *) k
  call get_command_argument (3, arg)
  read (arg, *) n

  if ( m<=0 .or. k<=0 .or. n<=0 ) then
     write(*,*) ' *** error: zero or negative arguments found'
     stop
  end if

  ! allocate memory
  allocate ( a (m,k) )
  allocate ( b (k,n) )
  allocate ( c (m,n) )

  ! initialise
  alpha =  1.2_dp
  beta  = -0.3_dp

  !$omp parallel shared(a,b,c, m,k,n) private(i,j,si)
  si = 1.0_dp / real ( m*k, dp )
  !$omp do
  do j = 1, k
     do i = 1, m
        a(i,j) = si * real ( +  (i-1) * k + j, dp )
     end do
  end do
  !$omp end do nowait

  si = 1.0_dp / real ( k*n, dp )
  !$omp do
  do j = 1, n
     do i = 1, k
        b(i,j) = si * real ( - ((i-1) * n + j), dp )
     end do
  end do
  !$omp end do nowait

  si = 1.0_dp / real ( m*n, dp )
  !$omp do
  do j = 1, n
     do i = 1, m
        c(i,j) = si * real ( 1 + (m-i)*(n-j), dp )
     end do
  end do
  !$omp end do
  !$omp end parallel

  ! write (*,*) 'A'
  ! call print_matrix (a,m,k)
  ! write (*,*) 'B'
  ! call print_matrix (b,k,n)
  ! write (*,*) 'C'
  ! call print_matrix (c,m,n)


  ! compute alpha*A*B+beta*C
  call system_clock( time_start,  time_rate )

  call dgemm ('n','n', m,n,k, alpha,a,m, b,k, beta,c,m)

  call system_clock( time_finish, time_rate )

  ! write (*,*) 'alpha*A*B+beta*C'
  ! call print_matrix (c,m,n)

  ! report performance
  perf_time   = real ( time_finish - time_start, dp ) &
              / real ( time_rate, dp )
  perf_gflops = ( ( 2.0_dp*real(m,dp)+1.0_dp)*real(k,dp) + real(m,dp) ) &
              * real(n,dp) / perf_time
  write( *,*) " time [s] = ", perf_time
  write( *,*) " gflops   = ", perf_gflops * 1.d-9

  ! deallocate memory
  deallocate (a)
  deallocate (b)
  deallocate (c)

end program perf_dgemm

! ----------------------------------------
! subroutine print_matrix (a,m,k)
!   implicit none
!   double precision, intent(in), dimension(m,k) :: a
!   integer, intent(in) :: m,k
!   integer :: i,j
!   do i = 1, m
!      write (*, *) (a(i,j), j = 1, k)
!   end do
! end subroutine print_matrix
