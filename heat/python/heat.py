#!/usr/bin/env python


"""Advanced Research Computing -- Python for High Performance Computing

This script compares different ways of implementing an iterative
procedure to solve the 2D heat equation. 

The Software
------------
The python code provides a general guideline to using Python for High
Performance Computing and provides a simple means to compare the
computational time taken by the different approaches.

The script compares functions implemented in

     * pure Python
     * NumPy
     * weave.blitz
     * weave.inline
     * numba
     * Fortran (OpenMP)
     * C (OpenMP)
     * Cython

The modules are built (using gcc and gfortran) with the command

     make all



The Maths
---------
The heat equation is the Initial Value Boundary Problem (IVBP)
Partial Differential Equation

      2      2      2
     d u    d u    d u
     ---2 = ---2 + ---2
     d t    d x    d y

whose solution u(x, y, t) is sought.

For this exercise, the problem is defined on the unit square

     0 <= x, y, z <= 1

The boundary conditions are

     u(0, y, t) = 0
     u(x, 0, t) = 0

and the initial value is

     u(x, y, 0) = sin(pi*y) * sin(pi*z)

With the above, the analytic solution is

     u(x, y, t) = sin(pi*y) * sin(pi*z) * exp(-2*pi**2 * t)


History
-------
Sep 2004: original version by Prabhu Ramachandran
          http://scipy-cookbook.readthedocs.io/items/PerformancePython.html
Sep 2016: modifications by Mihai Duta


License
-------
BSD
Creative Commons(?)

Last modified: Feb 2017

"""


# ======================================================================
#
# ----- imports
#
# ======================================================================
#
import numpy
import sys
import time
import h5py


# ======================================================================
#
# ----- grid class
#
# ======================================================================
#
class grid:
    """
    grid -- class to store the data of the computational grid:
            number of grid points, grid spacing and solution
    """
    def __init__ (self,
                  nx=64, ny=64,
                  xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                  h5name=None):

        #
        # ... if no solution file, initialise from input
        if h5name is None:

            # store grid parameters
            self.nx,   self.ny   = nx, ny
            self.xmin, self.xmax = xmin, xmax
            self.ymin, self.ymax = ymin, ymax
            # grid x, y data
            self.x, self.y = self.coords ()
            # initial compute time
            self.t = 0.0
            # initialise solution numerically (and store in uo)
            self.uo = numpy.sin (numpy.pi * (self.x)) * numpy.sin (numpy.pi * (self.y))
            # apply Dirichlet boundary values
            self.uo[0 , :] = 0
            self.uo[-1, :] = 0
            self.uo[: , 0] = 0
            self.uo[: ,-1] = 0

        #
        # ... if solution file, initialise from file
        else:

            import os.path
            if os.path.exists (h5name):
                # open HDF5 file
                h5f = h5py.File(h5name, "r")
                # store grid parameters
                self.nx,   self.ny   = h5f ["grid/numnodes"][...]
                self.xmin, self.xmax = h5f ["grid/limits/x"][...]
                self.ymin, self.ymax = h5f ["grid/limits/y"][...]
                # check if file dimensions match
                if (self.nx,self.ny) is not (nx,ny):
                    raise RuntimeError (" grid size mismatch: input size is " + repr((nx,ny))
                                        + ", file size is " + repr((self.nx,self.ny)))
                # check if problem limists match
                if (self.xmin,self.xmax,self.ymin,self.ymax) is not (xmin,xmax,ymin,ymax):
                    raise RuntimeError (" grid geometric limits mismatch: input is " + repr((xmin,xmax,ymin,ymax))
                                        + ", file values " + repr((self.xmin,self.xmax,self.ymin,self.ymax)))
                # grid x, y data
                self.x = h5f ["grid/coords/x"][...]
                self.y = h5f ["grid/coords/y"][...]
                # initial compute time
                self.t = h5f ["solution/time"][...]
                # read from file
                self.uo = h5f ["solution/solution"][...]
                # close HDF5 file
                h5f.close()
            else:
                raise RuntimeError (" file " + h5name + " does not exist")

        # u stores current-step solution
        self.u = self.uo.copy ()


    def coords (self):
        """
        computes the coordinates x, y in numpy.meshgrid format starting from self
        """
        nx, ny = self.nx, self.ny
        xx = numpy.linspace (self.xmin, self.xmax, nx)
        yy = numpy.linspace (self.ymin, self.ymax, ny)
        x, y = numpy.meshgrid (xx, yy)
        return x, y


    def error (self):
        """
        computes L2-norm absolute error for the solution
        (this requires that self.u and self.uo must be appropriately setup)
        """
        # pi
        pi = numpy.pi
        # analytic solution at time value
        self.uo = numpy.sin (pi*self.x) * numpy.sin (pi*self.y) * numpy.exp (-2.0*pi*pi*self.t)
        # get difference between analytic and computed solution
        v = ( (self.u - self.uo) / (self.uo + 1.e-14) ).flat
        return numpy.sqrt (numpy.dot (v,v))


    def save (self, h5name=None):
        if h5name is not None:
            # open HDF5 file
            h5f = h5py.File(h5name, "a")
            # write grid (first time only)
            if "grid" not in h5f.keys():
                h5f.create_dataset("grid/numnodes", data=numpy.array((self.nx,  self.ny  )), dtype="i4")
                h5f.create_dataset("grid/limits/x", data=numpy.array((self.xmin,self.xmax)), dtype="f8")
                h5f.create_dataset("grid/limits/y", data=numpy.array((self.ymin,self.ymax)), dtype="f8")
                h5f.create_dataset("grid/coords/x", data=self.x, dtype="f8")
                h5f.create_dataset("grid/coords/y", data=self.y, dtype="f8")

            # write solution
            if "solution" in h5f.keys():
                # update
                h5f ["solution/time"][...] = self.t
                h5f ["solution/solution"][...] = self.u
            else:
                # create
                h5f.create_dataset("solution/time",data=self.t, dtype="f8")
                h5f.create_dataset("solution/solution",data=self.u, dtype="f8")

            # close HDF5 file
            h5f.close()


    def plot (self):
        """
        produces a 3d plot of the solution
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        x, y = self.coords ()
        fig = plt.figure ()
        ax = fig.gca (projection='3d')
        surf = ax.plot_surface (x, y, self.u)
        plt.show ()
        plt.draw ()


# ======================================================================
#
# ----- solver class
#
# ======================================================================
#
class solution:
    """
    solution -- class to implement an explicit time-stepping solution for
                the time-dependent 2D heat equation
    """
    def __init__ (self, grid, stepper = "numpy", nu=0.25):
        # initialise grid data with argument
        self.grid = grid
        # initialise stepper with argument
        self.setStepper (stepper)
        # scheme parameter (<=0.25 for stability)
        self.nu = nu
        # Fortran trickery (cast solution and copy to Fortran storage)
        if (stepper == "fortran"):
            self.grid.uo =  numpy.array (self.grid.uo, order="Fortran")
            self.grid.u  =  numpy.ndarray (shape=self.grid.uo.shape,
                                           dtype=self.grid.uo.dtype,
                                           order="Fortran")


    def setStepper (self, stepper="numpy"):
        """
        Set stepper used by timeStep()
        """
        if   stepper == "python":
            self.stepper = self.pythonStepper
        elif stepper == "numpy":
            self.stepper = self.numpyStepper
        elif stepper == "fortran":
            self.stepper = self.fortranStepper
        elif stepper == "ctypes":
            self.stepper = self.cStepper
        elif stepper == "numba":
            self.stepper = self.numbaStepper
        elif stepper == "blitz":
            self.stepper = self.blitzStepper
        elif stepper == "inline":
            self.stepper = self.inlineStepper
        elif stepper == "cython":
            self.stepper = self.cythonStepper
        elif stepper == "pycuda":
            self.stepper = None
        else:
            self.stepper = self.numpyStepper


    def timeStep (self, numIters=0):
        """
        Advances the solution numIters timesteps using stepper set by setStepper()
        """
        #
        # ... solution parameters
        # number of grid points in x, y
        nx, ny = self.grid.u.shape
        # solution vectors
        u  = self.grid.u
        uo = self.grid.uo
        # scheme parameter (<=0.25 for stability)
        nu = self.nu
        # grid spacing
        dx = float (self.grid.xmax - self.grid.xmin) / (self.grid.nx - 1)

        #
        # ... time-step solution numIters times
        for t in range (1, numIters):
            # apply numerical scheme
            self.stepper (nx,ny, u,uo, nu)
            # swap pointers
            u, uo = uo, u

        # swap pointers
        # (at the end of the iterations, uo points to the updated solution)
        u, uo = uo, u
        #
        # ... update current time
        self.grid.t += numIters*nu*dx*dx


    def timeStepCuda (self, numIters=0):
        """
        Advances the solution numIters timesteps using a pycuda stepper
        """
        # pycuda
        import pycuda.autoinit
        import pycuda.driver
        import pycuda.compiler
        # kernel
        srcmod = pycuda.compiler.SourceModule ("""
        __global__ void cuda_stepper_naive (int I, int J, double *u, double *uo, double nu)
        {
          int i,j, ks,ksn,kss,ksw,kse;

          i   = blockDim.x * blockIdx.x + threadIdx.x;
          j   = blockDim.y * blockIdx.y + threadIdx.y;

          if (i>0 && i<(I-1) && j>0 && j<(J-1)) {
            ks  = i*J + j;
            ksn = ks - J;
            kss = ks + J;
            kse = ks + 1;
            ksw = ks - 1;

            u[ks] = uo[ks]
                  + nu * ( uo[ksn] + uo[kss] + uo[ksw] + uo[kse]
                         - 4.0*uo[ks] );
          }
        }

        __global__ void cuda_stepper (const int I, const int J, double * __restrict__ u, const double * __restrict__ uo, const double nu)
        {
          # define DIM_BLOCK_I 16
          # define DIM_BLOCK_J  8

          int Im1,Jm1, Is,Js, is,js,ks, ksn,kss,ksw,kse, i,j,k;
          int active;

          __shared__ double us[(DIM_BLOCK_I+2)*(DIM_BLOCK_J+2)];

          // extra limits
          Im1 = I - 1;
          Jm1 = J - 1;

          // shared memory size
          Is = DIM_BLOCK_I + 2;
          Js = DIM_BLOCK_J + 2;

          // global i,j indices
          i  = blockDim.x * blockIdx.x + threadIdx.x;
          j  = blockDim.y * blockIdx.y + threadIdx.y;

          // shared memory indices
          is = threadIdx.x + 1;
          js = threadIdx.y + 1;

          // active thread?
          active = i<I && j<J;

          // active thread
          if (active) {

            // global indices
            k = i*J + j;

            // local (shared memory) indices
            ks  = is*Js + js;
            ksn = ks - Js;
            kss = ks + Js;
            kse = ks + 1;
            ksw = ks - 1;

            // copy central node
            us[ks] = uo[k];

            //
            // ----- halo treatment: i index changes fastest, so coallesced mem transfer along i
            //
            if (threadIdx.x==0 && i>0) {
              us[ksn] = uo[k-J];
            }

            if (threadIdx.x==(DIM_BLOCK_I-1) && i<Im1) {
              us[kss] = uo[k+J];
            }

            if (threadIdx.y==0 && j>0) {
              us[ksw] = uo[k-1];
            }

            if (threadIdx.y==(DIM_BLOCK_J-1) && j<Jm1) {
              us[kse] = uo[k+1];
            }
          }

          __syncthreads();

          //
          // ----- update central node using shared memory
          //
          if (i>0 && i<I-1 && j>0 && j<J-1) {
             u[k] = us[ks]
                  + nu * ( us[ksn] + us[kss] + us[ksw] + us[kse]
                         - 4.0*us[ks] );
          }

          __syncthreads();

        }

        """)

        #
        # ... pycuda events (timing & synchronization)
        eventStart = pycuda.driver.Event()
        eventEnd   = pycuda.driver.Event()
        #
        # ... record start event
        eventStart.record ()
        #
        # ... solution parameters
        # number of grid points in x, y
        nx, ny = self.grid.u.shape
        # solution vectors
        u  = self.grid.u
        uo = self.grid.uo
        # scheme parameter (<=0.25 for stability)
        nu = self.nu
        # grid spacing
        dx = float (self.grid.xmax - self.grid.xmin) / (self.grid.nx - 1)
        # device memory
        ud  = pycuda.driver.mem_alloc (u.nbytes)
        udo = pycuda.driver.mem_alloc (u.nbytes)
        # copy memory (host to device)
        pycuda.driver.memcpy_htod (udo, u)
        pycuda.driver.memcpy_htod (ud,  u)
        # device block and grid
        cuda_block = (16, 8, 1)
        cuda_grid  = (nx//cuda_block[0] + (nx%cuda_block[0]>0),
                      ny//cuda_block[1] + (ny%cuda_block[1]>0),
                      1)
        # device kernel
        cuda_stepper = srcmod.get_function ("cuda_stepper")
        #
        # ... time-step solution numIters times
        for t in range (1, numIters):
            # apply numerical scheme
            cuda_stepper (numpy.int32(nx), numpy.int32(ny),
                          ud,udo, numpy.float64(nu),
                          block=cuda_block, grid=cuda_grid)
            # swap pointers
            ud, udo = udo, ud

        # swap pointers
        # (at the end of the iterations, udo points to the updated solution)
        ud, udo = udo, ud
        # copy memory (device to host)
        pycuda.driver.memcpy_dtoh (u, ud)
        #
        # ... update current time
        self.grid.t += numIters*nu*dx*dx
        #
        # ... pycuda sync and execution time
        eventEnd.record ()
        eventEnd.synchronize ()
#       eventTime = eventStart.time_till (eventEnd) * 1e-3


    def pythonStepper (self, nx,ny, u,uo, nu):
        """ time-steps implemented using straight python array indexing"""
        # apply numerical scheme (one time-step)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u[i,j] = uo[i,j] + ( nu * ( uo [i-1, j] + uo [i+1, j] +
                                            uo [i, j-1] + uo [i, j+1]
                                            - 4.0 * uo [i,j] ) )


    def numpyStepper (self, nx,ny, u,uo, nu):
        """ time-steps implemented using numpy array indexing"""
        # apply numerical scheme (one time-step)
        u[1:-1, 1:-1] = uo[1:-1, 1:-1] + ( nu * ( uo [0:-2, 1:-1] + uo [2:, 1:-1] +
                                                  uo [1:-1, 0:-2] + uo [1:-1, 2:]
                                                  - 4.0 * uo [1:-1, 1:-1] ) )


    def numbaStepper (self, nx,ny, u,uo, nu):
        """ time-steps implemented using straight python array indexing dispatched via JIT compiling"""
        # apply numerical scheme (one time-step)
        numbaStepperJIT (nx,ny, u,uo, nu)


    def blitzStepper (self, nx,ny, u,uo, nu):
        """ time-steps implemented using numpy expression dispatched via blitz"""
        from scipy import weave
        # define expression (same one as for numpyStepper)
        expr = "u[1:-1, 1:-1] = uo[1:-1, 1:-1] + ( nu * ( uo [0:-2, 1:-1] + uo [2:, 1:-1] + " \
               "                                   uo [1:-1, 0:-2] + uo [1:-1, 2:]" \
               "                                   - 4.0 * uo [1:-1, 1:-1] ) )"
        weave.blitz (expr, check_size=0)


    def inlineStepper (self, nx,ny, u,uo, nu):
        """ time-steps implemented using C code dispatched via weave"""
        from scipy import weave
        from scipy.weave import converters
        # define code (same one as for C code cStepper
        #  * cannot use u[i][j]
        #  * instead use u[k], with k = i*ny + j
        code = """
               int i,j;
               int k,kn,ks,kw,ke;
               for (i=1; i<nx-1; i++) {
                 for (j=1; j<ny-1; j++) {
                   k    = i*ny + j;
                   kn   = k + nx;
                   ks   = k - nx;
                   ke   = k + 1;
                   kw   = k - 1;
                   u[k] = uo[k]
                        + nu * ( uo[kn] + uo[ks]
                               + uo[ke] + uo[kw]
                               - 4.0 * uo[k]);
                 }
               }
               """
        # compiler keyword only needed on windows with MSVC installed
        err = weave.inline (code,
                            ["nx","ny","u","uo","nu"])


    def fortranStepper (self, nx,ny, u,uo, nu):
        """ time-steps implemented using Fortran code"""
        import sys
        sys.path.append("./lib/python2.7/site-packages")

        import fortran_stepper
        # apply numerical scheme (one time-step)
        fortran_stepper.timestep (nu, uo,u)


    def cStepper (self, nx,ny, u,uo, nu):
        """ time-steps implemented using C code"""
        import ctypes
        from numpy.ctypeslib import ndpointer

        c_stepper = ctypes.cdll.LoadLibrary("./lib/python2.7/site-packages/c_stepper.so")
        c_stepper.timestep.restype = None
        c_stepper.timestep.argtypes = [ctypes.c_double,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

        # apply numerical scheme (one time-step)
        c_stepper.timestep (nu, nx,ny, uo,u)


    def cythonStepper (self, nx,ny, u,uo, nu):
        """ time-steps implemented using cython"""

        import sys
        sys.path.append("./lib/python2.7/site-packages")

        import cython_stepper
        # apply numerical scheme (one time-step)
        cython_stepper.timestep (uo,u, nu)


# ======================================================================
#
# ----- numba stepper (defined separately)
#
# ======================================================================
#
from numba import jit
# numba / JIT compiler decorator
@jit ("void(int32,int32,float64[:,:],float64[:,:],float64)")
def numbaStepperJIT (nx,ny, u,uo, nu):
    # same code as the straight python stepper
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u[i,j] = uo[i,j] + ( nu * ( uo [i-1, j] + uo [i+1, j] +
                                        uo [i, j-1] + uo [i, j+1]
                                        - 4.0 * uo [i,j] ) )


# ======================================================================
#
# ----- runAllStepperOptions :: run all steppers available
#                               (except pycuda)
#
# ======================================================================
#
def runAllStepperOptions (numNodes=1000, numIters=100):

    # stepper implementations to try
    stepperTypeList = [
#       "python",
        "numpy",
        "numba",
#       "blitz",
#       "inline",
#       "fortran",
        "ctypes",
        "cython"
    ]

    # try all steppers
    for stepperType in stepperTypeList:
        # initialise grid
        g = grid (numNodes, numNodes)
        # initialise solution
        s = solution (g, stepper=stepperType)
        # solve
        t = time.time ()
        s.timeStep (numIters=numIters)
        t = time.time () - t
        # compute error
        err = g.error ()
        # number of floating point ops: 8 flops/grid node * (N-2)**2 grid nodes * I iterations
        flops = 8.0 * (numNodes-2)**2 * numIters / (t * 1000.**3)
        # report
        print " stepper %s, %d iterations, %f seconds, %f Gflops, %f rel error" % (stepperType, numIters, t, flops, err)


# ======================================================================
#
# ----- runStepper :: run a single stepper
#
# ======================================================================
#
def runStepper (numNodes=1000, numIters=100, h5name=None, stepper="numpy"):

    # initialise grid
    g = grid (nx=numNodes, ny=numNodes, h5name=h5name)
    # initialise solution
    s = solution (g, stepper=stepper)
    # solve
    t = time.time ()
    s.timeStep (numIters=numIters)
    t = time.time () - t
    # save solution
    g.save (h5name)
    # compute error
    err = g.error ()
    # number of floating point ops: 8 flops/grid node * (N-2)**2 grid nodes * I iterations
    flops = 8.0 * (numNodes-2)**2 * numIters / (t * 1000.**3)
    # report
    print " stepper %s, %d iterations, %f seconds, %f Gflops, %f rel error" % (stepper, numIters, t, flops, err)


# ======================================================================
#
# ----- runStepperCuda :: run the pycuda stepper
#                         (specialised)
#
# ======================================================================
#
def runStepperCuda (numNodes=1000, numIters=100, h5name=None):

    import pycuda.driver

    # initialise grid
    g = grid (nx=numNodes, ny=numNodes, h5name=h5name)
    # initialise solution
    s = solution (g, stepper="pycuda")
    # solve
    t = time.time ()
    s.timeStepCuda (numIters=numIters)
    t = time.time () - t
    # save solution
    g.save (h5name)
    # compute error
    err = g.error ()
    # number of floating point ops: 8 flops/grid node * (N-2)**2 grid nodes * I iterations
    flops = 8.0 * (numNodes-2)**2 * numIters / (t * 1000.**3)
    # report
    print " stepper %s, %d iterations, %f seconds, %f Gflops, %f rel error" % ("pycuda", numIters, t, flops, err)


# ======================================================================
#
# ----- main
#
# ======================================================================
#
if __name__ == "__main__":
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser (description="demonstrator of python performance using compiled languages")
    parser.add_argument ("-n", dest="numNodes", metavar="n", type=int, default=400,     help="grid size")
    parser.add_argument ("-i", dest="numIters", metavar="i", type=int, default=100,     help="number of iterations")
    parser.add_argument ("-f", dest="h5name",   metavar="f", type=str, default=None,    help="solution HDF5 file")
    parser.add_argument ("-s", dest="stepper",  metavar="s", type=str, default="numpy", help="stepper implementation")
    args = parser.parse_args ()

    #
    # ... opening message
    print " computing %d iterations on a %dx%d grid...\n" % (args.numIters, args.numNodes,args.numNodes)


#   # run all options and compare execution time
    runAllStepperOptions (numNodes=args.numNodes, numIters=args.numIters)

    # run the checkpointing sequeuce
#    runStepper (numNodes=args.numNodes, numIters=args.numIters, h5name=args.h5name, stepper="ctypes")
#    runStepper (numNodes=args.numNodes, numIters=args.numIters, h5name=args.h5name, stepper="cython")
    runStepperCuda (numNodes=args.numNodes, numIters=args.numIters, h5name=args.h5name)


    #
    # ... closing message
    print "\n ...finished.\n"

    # ... end
    #
