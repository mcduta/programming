!======================================================================
!
!   name:        perf_dgesvd.f90
!
!   synopsis:    demonstrate and time the use of LAPACK DGESVD and DGESDD
!
!   purpose:     provide a way to test the performance of DGESVD and
!                DGESDD implementations
!
!   description: This example initialises a matrix A and computes its
!                singular value decomposition SVD decomposition in
!                double precision.
!
!                Both routines DGESVD and DGESDD compute the SVD of a
!                rectangular real matrix A, optionally the left and/or
!                right singular vectors.
!
!                The SVD is
!
!                  A = U*S*VT
!
!                where
!
!                  A  = original real-valued m-by-n matrix
!                  S  = m-by-n matrix, which is zero except for its
!                       min(m,n) diagonal elements
!                  U  = m-by-m orthogonal matrix (left singular vectors)
!                  VT = (V transposed) n-by-n orthogonal matrix (right singular vectors)
!
!                DGESVD computes the SVD of a general rectangular matrix.
!                DGESDD computes the SVD of a general rectangular matrix
!                using a divide-and-conquer method, which is usually
!                substantially faster if singular vectors are required.
!                Computing the singular vectors is the slow part for both
!                routines in the case of large matrices.
!
!   compile:     ...
!
!   run:         ./perf_dgesvd -m 1200 -n 1000
!                ./perf_dgesvd -m 1200 -n 1000 -method sdd -vectors no
!
!======================================================================

program perf_dgesvd

  !
  ! === nothing implicit
  !
  implicit none

  !
  ! === double precision
  !
  integer, parameter :: dp = selected_real_kind(15, 307)

  !
  ! === main variables
  !
  ! problem dimensions
  integer :: m=0, n=0
  ! the original mxn matrix
  real (kind=dp), allocatable :: a(:,:)
  ! storage for SVD decomposition
  real (kind=dp), allocatable :: s(:), u(:,:), vt(:,:)

  !
  ! === command line options
  !
  ! command line args
  character (32) :: arg, arg2
  ! method option
  character (32) :: method  = "svd"
  ! compute option: "yes" - all singular vectors are computed, "no" - none
  character (32) :: vectors = "yes"

  !
  ! === workspaces
  !
  ! IWORK integer workspace (DGESDD only)
  integer, allocatable :: iw(:)
  ! WORK main workspace
  real (kind=dp), allocatable :: ww(:)

  !
  ! === ancillary variables
  !
  ! dummy WORK (for optimal LWORK calculation)
  real (kind=dp) :: dw(1,1)
  ! iteration indices
  integer :: i
  ! length of WORK and DGESVD/DGESDD info
  integer :: lw, info, stat
  ! option (compute vectors)
  character (32) :: opt

  !
  ! === functions
  !
  intrinsic :: max, min, nint, mod
  real (kind=dp), external :: dlange

  !
  ! === performance
  !
  ! timing
  integer :: time_start, time_finish, time_rate
  ! perf
  real (kind=dp) :: perf_time
  ! error
  real (kind=dp) :: perf_err


  !
  ! === process command line
  !
  info = command_argument_count ()

  if ( info == 0 ) then
     write (*,*) " === help: perf_dgesvd -help"
     stop
  end if

  i = 1
  do while ( i <= info )
     call get_command_argument(i, arg)
     i = i + 1

     if ( arg == "-help" ) then
        call print_help ()
        stop
     else
        call get_command_argument(i, value=arg2, status=stat)
        i = i + 1

        if ( stat /= 0 ) then
           write (*,*) " *** error: unspecified value for option ", arg
           stop
        end if

        select case (arg)
        case ("-m")
           read (arg2, *) m
        case ("-n")
           read (arg2, *) n
        case ("-method")
           read (arg2, *) method
        case ("-vectors")
           read (arg2, *) vectors
        case default
           write (*,*) " === help: perf_dgesvd -help"
           stop
        end select
     end if
  end do


  if ( m<=0 .or. n<=0 ) then
     write(*,*) " *** error: matrix dimenasions are zero, negative or unspecified: ", m, n
     stop
  end if

  if ( method /= "svd" .and. method /= "sdd" ) then
     write(*,*) " *** error: wrong method option: ", method
     stop
  end if

  if ( vectors /= "yes" .and. vectors /= "no" ) then
     write(*,*) " *** error: wrong vector option: ", vectors
     stop
  end if

  if ( vectors == "yes" ) then
     opt = "all"
  else
     opt = "none"
  endif


  !
  ! === allocate memory
  !
  allocate ( a (m,n) )
  allocate ( s (m)   )
  if ( vectors == "yes" ) then
     allocate ( u (m,m) )
     allocate ( vt(n,n) )
  end if
  if ( method == "sdd" ) then
     allocate ( iw (8*min(m,n)) )
  end if


  !
  ! === initialise
  !
  call init_matrix (a,m,n)


  !
  ! === query the optimal workspace size and allocate
  !
  lw = -1

  if ( method == "svd" ) then
     call dgesvd ( opt, opt, m,n,a, m,s, u,m, vt,n, dw,lw,     info )
  else
     call dgesdd ( opt     , m,n,a, m,s, u,m, vt,n, dw,lw, iw, info )
  end if

  if (info /= 0) then
     write(*,*) ' *** error: failure in DGESVD/DGESDD: info =', info
     stop
  end if

  if ( method == "svd" ) then
     lw = max(1,3*min(m,n)+max(m,n),5*min(m,n))
  else
     if ( vectors == "yes") then
        lw = 3*min(m,n)*min(m,n) + max(max(m,n),4*min(m,n)*min(m,n)+4*min(m,n))
     else
        lw = 3*min(m,n) + max(max(m,n),7*min(m,n))
     end if
  end if

  lw = max (lw, nint(dw(1,1)))
  allocate ( ww (lw) )


  !
  ! ===  compute singular values and vectors
  !
  call system_clock( time_start,  time_rate )

  if ( method == "svd" ) then
     call dgesvd ( opt, opt, m,n,a, m,s, u,m, vt,n, ww,lw,     info )
  else
     call dgesdd ( opt,      m,n,a, m,s, u,m, vt,n, ww,lw, iw, info )
  end if

  call system_clock( time_finish, time_rate )

  if (info /= 0) then
     write(*,*) ' *** error: failure in DGESVD/DGESDD: info =', info
     stop
  end if


  !
  ! === verify accuracy
  !
  if (vectors=="yes")  then
     perf_err = frob_norm_diff (a,m,n, u,s,vt)
     perf_err = perf_err / sqrt ( real(m*n, dp) )
     write( *,*) " || A-U*S*VT ||_F = ", perf_err
  end if


  !
  ! === report performance
  !
  perf_time   = real ( time_finish - time_start, dp ) &
              / real ( time_rate, dp )
  write( *,*) " time [s] = ", perf_time


  !
  ! === deallocate memory
  !
  deallocate ( a )
  deallocate ( s )
  if ( vectors == "yes" ) then
     deallocate ( u )
     deallocate ( vt )
  end if
  if ( method == "sdd" ) then
     deallocate ( iw )
  end if
  deallocate ( ww )

contains

  ! ----------------------------------------------------------------
  !
  ! initialise matrix A
  !
  ! ----------------------------------------------------------------
  subroutine init_matrix (a,m,n)
    implicit none
    integer, intent(in) :: m,n
    real (kind=dp), intent(inout) :: a(m,n)
    integer :: i, j
    real (kind=dp) :: tmp

    !$omp parallel shared(a, m,n) private(i,j,tmp)
    tmp = 1.0_dp / real ( m*n, dp )
    !$omp do
    do j = 1, n
       do i = 1, m
          a(i,j) = tmp * real ( (i+1) * (j+1) + j*j, dp )
       end do
    end do
    !$omp end do nowait
    !$omp do
    do i = 1, min(m,n)
       a(i,i) = a(i,i) + 1.0_dp
    end do
    !$omp end do nowait
    !$omp end parallel

  end subroutine init_matrix


  ! ----------------------------------------------------------------
  !
  ! check SVD of matrix A by calculating the Frobenius norm of
  ! A - U*S*VT
  !
  ! ----------------------------------------------------------------
  function frob_norm_diff (a,m,n, u,s,vt)
    implicit none
    real (kind=dp) :: frob_norm_diff
    integer, intent(in) :: m,n
    real (kind=dp), intent(inout) :: a(m,n), s(m), u(m,m), vt(n,n)

    integer :: i,j

    ! initial values of A are lost
    call init_matrix (a,m,n)

    if (m < n) then
       ! VT := S*VT
       !$omp parallel shared(vt,s, m,n) private(i)
       !$omp do
       do i = 1, m
          vt(i,:) = s(i) * vt(i,:)
       end do
       !$omp end do nowait
       !$omp do
       do i = m+1, n
          vt(i,:) = 0.0_dp
       end do
       !$omp end do
       !$omp end parallel
    else
       ! U := U*S
       !$omp parallel shared(u,s, m,n) private(j)
       !$omp do
       do j = 1, n
          u(:,j) = s(j) * u(:,j)
       end do
       !$omp end do nowait
       !$omp do
       do j = n+1, m
          u(:,j) = 0.0_dp
       end do
       !$omp end do
       !$omp end parallel
    end if

    ! A := 1.0*U*VT - 1.0*A
    call dgemm ("no","no", m,n,n, -1.0_dp,u,m, vt,n, +1.0_dp,a,m)

    ! Frobenius norm of difference
    ! Note: last argument is work array, not referenced for Frobenius norm
    frob_norm_diff = dlange ("Frobenius", m,n,a, m,s)

  end function frob_norm_diff


  ! ----------------------------------------------------------------
  !
  ! print help
  !
  ! ----------------------------------------------------------------
  subroutine print_help ()
    implicit none
    write (*,*) " === usage: perf_dgesvd -m M -n N -method [svd|sdd] -vectors [yes|no]"
    write (*,*) "            M, N    -- matrix dimensions"
    write (*,*) "            svd|sdd -- use DGESVD or DGESDD     (default=svd)"
    write (*,*) "            yes|no  -- compute singular vectors (default=yes)"
  end subroutine print_help

  ! ----------------------------------------------------------------
  !
  ! print matrix
  !
  ! ----------------------------------------------------------------
  subroutine print_matrix (a,m,n)
    implicit none
    double precision, intent(in), dimension(m,n) :: a
    integer, intent(in) :: m,n
    integer :: i,j
    do i = 1, min(m,20)
       write (*, *) (a(i,j), j = 1, min(n,20))
    end do
  end subroutine print_matrix

end program perf_dgesvd
