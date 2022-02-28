subroutine fmult (n, x,y,z)
  implicit none
  integer :: n
  real (kind=8) :: x(n),y(n),z(n)

  z = x + 2.d0*y + x*y

end subroutine fmult
