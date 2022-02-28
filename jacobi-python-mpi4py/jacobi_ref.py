# python jacobi.py  --nrows 600 --ncols 400 -i 1000 -t 1.e-6

def jacobi (nrows, ncols, itermax=100, itertol=1.e-6, iterupdate=1):
    """
    Jacobi iterator
      call:
        jacobi (comm, nrows, ncols)
      args:
        comm  = MPI communicator
        nrows = grid size (number of rows)
        ncols = grid size (number of cols)
       * notional "global" matrix is sliced along rows, so that
         process iproc owns u(i1:i2,ncols) from "global" u(nrows,ncols)
       * the extra nr2 rows "above" and "below" are for halo communication
    """

    # initialise matrix for jacobi calculation
    u = numpy.zeros((nrows, ncols))
    u[ :, 0] = 1.0
    u[ :,-1] = 1.0
    u[ 0, :] = 1.0
    u[-1, :] = 1.0

    # iteration count
    iternum  = 0

    # iteration error (global)
    errtot   = itertol + 1.0

    # iterate
    print (F" *** iteration history")

    while iternum < itermax and errtot > itertol:
        iternum += 1

        # the actual Jacobi update
        v = u.copy()
        u[1:-1, 1:-1] = 0.25 * ( u[ 0:-2, 1:-1] + u[ 2:,  1:-1] +
                                 u[ 1:-1, 0:-2] + u[ 1:-1,  2:] )
        v = u - v

        # step process error
        err = numpy.sqrt ( numpy.mean (v.flat) )

        # global error
        if iternum % iterupdate == 0:
            print (F"     {iternum} {errtot}")

    # return solution (for plotting)
    return u


# ======================================================================
#
# ----- main
#
# ======================================================================
#

if __name__ == "__main__":
  # process arguments
  import argparse
  parser = argparse.ArgumentParser (description="Jacobi iteration demo in mpi4py")

  parser.add_argument ("--nrows", dest="nrows", type=int, default=100, help="number of grid rows")
  parser.add_argument ("--ncols", dest="ncols", type=int, default=100, help="number of grid cols")

  parser.add_argument ("-i", "--itermax", dest="itermax", type=int,   default=100,   help="maximum number of iterations")
  parser.add_argument ("-t", "--itertol", dest="itertol", type=float, default=1.e-6, help="iterations tolerance")

  parser.add_argument ("-p", "--plot", action="store_true", help="plot solution option")
  args = parser.parse_args ()


  # import libraries
  import numpy
  import math
  import time

  # summary of processing
  print (F" *** Jacobi iterations on ({args.nrows},{args.ncols}) grid points")
  print (F"     {args.itermax} iterations max, {args.itertol} tolerance")

  # Jacobi iterator
  tstart = time.time ()
  u = jacobi(nrows=args.nrows, ncols=args.ncols, itermax=args.itermax, itertol=args.itertol, iterupdate=max(1,args.itermax//10))
  tstop  = time.time ()

  # report time
  print(F" *** runtime = {tstop - tstart} sec")

  # plot solution
  if args.plot:
      import matplotlib.pyplot as plt
      plt.imshow(u, cmap="jet", interpolation="nearest")
      plt.show()
