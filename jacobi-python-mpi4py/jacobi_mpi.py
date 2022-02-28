# mpirun -np 4 python jacobi_mpi4py.py  --nrows 600 --ncols 600 -i 4000 -t 1.e-6

def jacobi (comm, nrows, ncols, itermax=100, itertol=1.e-6, iterupdate=1):
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

    # process
    nproc = comm.size  # total number of processes mpirun/mpiexec launched
    iproc = comm.rank  # process ID (integer from 0 to nproc-1)

    # process neighbours
    iproc_hi = iproc - 1 if iproc > 0         else MPI.PROC_NULL
    iproc_lo = iproc + 1 if iproc < nproc - 1 else MPI.PROC_NULL

    # sizes and limits (per process)
    i1 = (iproc    * nrows) // nproc
    i2 = (iproc+1) * nrows  // nproc - 1
    nr = i2 - i1 + 1
    ### print (F"iproc={iproc}, (i1,i2)=({i1},{i2}), nr={nr}")

    # initialise matrix for jacobi calculation (with extra 2 rows for halo comms)
    nr2 = 2 if iproc > 0 and iproc < nproc-1 else 1
    u = numpy.zeros((nr+nr2, ncols))
    u[:, 0] = 1.0
    u[:,-1] = 1.0
    if iproc == 0:       u[ 0, :] = 1.0
    if iproc == nproc-1: u[-1, :] = 1.0

    # iteration count
    iternum  = 0

    # iteration error (global)
    errtot   = itertol + 1.0
    errtot2  = numpy.empty(1, dtype=numpy.float64)

    # iterate
    if iproc == 0:
        print (F" *** iteration history")

    while iternum < itermax and errtot > itertol:
        iternum += 1

        # halo exchange
        if iproc % 2 == 0:
            # even numbered processes
            comm.Recv ( [ u[ 0,:], ncols, MPI.DOUBLE], source=iproc_hi, tag=10)
            comm.Send ( [ u[ 1,:], ncols, MPI.DOUBLE],   dest=iproc_hi, tag=20)
            comm.Recv ( [ u[-1,:], ncols, MPI.DOUBLE], source=iproc_lo, tag=30)
            comm.Send ( [ u[-2,:], ncols, MPI.DOUBLE],   dest=iproc_lo, tag=40)
        else:
            # odd numbered processes
            comm.Send ( [ u[-2,:], ncols, MPI.DOUBLE],   dest=iproc_lo, tag=10)
            comm.Recv ( [ u[-1,:], ncols, MPI.DOUBLE], source=iproc_lo, tag=20)
            comm.Send ( [ u[ 1,:], ncols, MPI.DOUBLE],   dest=iproc_hi, tag=30)
            comm.Recv ( [ u[ 0,:], ncols, MPI.DOUBLE], source=iproc_hi, tag=40)

        # the actual Jacobi update
        v = u.copy()
        u[1:-1, 1:-1] = 0.25 * ( u[ 0:-2, 1:-1] + u[ 2:,  1:-1] +
                                 u[ 1:-1, 0:-2] + u[ 1:-1,  2:] )
        v = u - v

        # step process error
        err = numpy.sqrt ( numpy.mean (v[1:-1,:].flat) )

        # global error
        if iternum % iterupdate == 0:
            comm.Allreduce ([numpy.array(err), 1, MPI.DOUBLE], [errtot2, 1, MPI.DOUBLE], op = MPI.SUM)
            errtot = errtot2[0] / nproc
            if iproc == 0:
                print (F"     {iternum} {errtot}")


    # return solution (for plotting)
    i1 = +1 if iproc > 0         else 0
    i2 = -1 if iproc < nproc - 1 else None
    return u[i1:i2, :]


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

  # import MPI
  from mpi4py import MPI
  ### assert MPI.COMM_WORLD.Get_size() > 1, "just 1"

  # MPI process
  nproc = MPI.COMM_WORLD.size
  iproc = MPI.COMM_WORLD.rank

  # summary of processing
  if iproc == 0:
      print (F" *** Jacobi iterations on ({args.nrows},{args.ncols}) grid points")
      print (F"     {nproc} MPI processes")
      print (F"     {args.itermax} iterations max, {args.itertol} tolerance")

  # Jacobi iterator
  tstart = MPI.Wtime ()
  u = jacobi(comm=MPI.COMM_WORLD, nrows=args.nrows, ncols=args.ncols, itermax=args.itermax, itertol=args.itertol, iterupdate=max(1,args.itermax//10))
  tstop  = MPI.Wtime ()

  # report time
  if iproc == 0:
      print(F" *** runtime = {tstop - tstart} sec")

  # plot solution
  if args.plot:
      import matplotlib.pyplot as plt
      utot = numpy.empty ((args.nrows, args.ncols))
      MPI.COMM_WORLD.Gatherv (u, [utot, MPI.DOUBLE], root=0)

      if iproc == 0:
          plt.imshow(utot, cmap="jet", interpolation="nearest")
          plt.show()
