# ====================================================================== #
#                                                                        #
#                                                                        #
#                                                                        #
#                                                                        #
# ====================================================================== #

import nvtx

class gs:
    """
    gs -- class for the 2D solution of Gray--Scott reaction-diffusion equation on a rectangular domain with periodic boundary conditions
    """

    #
    # --- initialise class
    @nvtx.annotate ("__in-t__()")
    def __init__ (self, params=None):
        """
        initialise parameters and solution storage
        """
        # parameters
        if params is not None:
            self.params = params;
        else:
            raise Exception (" *** error: no equation parameters provided")

        # unpacking parameters
        self.nx    = params["nx"]
        self.ny    = params["ny"]
        self.Du    = params["Du"]
        self.Dv    = params["Dv"]
        self.alpha = params["alpha"]
        self.beta  = params["beta"]
        self.niter = params["niter"]
        self.us    = params["us"]
        self.vs    = params["vs"]
        self.ns    = params["ns"]

        # solution and iteration storage and init (vecpy is numpy or cupy)
        self.u1 = 1.0 + 0.02 * vecpy.random.randn(self.nx, self.ny)
        self.v1 = 0.0 + 0.02 * vecpy.random.randn(self.nx, self.ny)

        # solution init spots
        randx = vecpy.random.rand(self.ns)
        randy = vecpy.random.rand(self.ns)
        rands = vecpy.random.rand(self.ns)

        for i in range (self.ns):
            # spot centre
            cx = 1 + round((self.nx-2)*random.uniform(0,1)); cx = min(self.nx-2, cx)
            cy = 1 + round((self.ny-2)*random.uniform(0,1)); cy = min(self.ny-2, cy)
            # spot size
            ss = 1 + round((min(self.nx,self.ny)/20.)*random.uniform(0,1))
            # spots
            ss2 = 1 + round(ss*ss/4);
            for iy in range(max(0,cy-ss), min(self.ny-1,cy+ss)):
                for ix in range(max(0,cx-ss), min(self.nx-1,cx+ss)):
                    if((ix-cx)*(ix-cx)+(iy-cy)*(iy-cy)) < ss2: 
                        self.u1[ix,iy] = self.us
                        self.v1[ix,iy] = self.vs

        self.u2 = vecpy.zeros_like(self.u1)
        self.v2 = vecpy.zeros_like(self.v1)

        # pointers (shallow copies) to solution storage
        self.u  = self.u1
        self.v  = self.v1
        self.un = self.u2
        self.vn = self.v2


    #
    # --- iterator class
    @nvtx.annotate ("iterate()")
    def iterate (self):
        """
        iterator class for an explicit time integration solution
        """

        # ... laplacian and nonlinear term
        Lu  = vecpy.zeros ((self.nx-2, self.ny-2))
        Lv  = vecpy.zeros ((self.nx-2, self.ny-2))
        uvv = vecpy.zeros ((self.nx-2, self.ny-2))

        # pointers (shallow copies) to solution storage
        u  = self.u
        v  = self.v
        un = self.un
        vn = self.vn

        # ... time iterations
        for iter in range(1, self.niter):
            with nvtx.annotate ("loop"):

               # laplacians
               Lu = u[0:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 0:-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
               Lv = v[0:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, 0:-2] + v[1:-1, 2:] - 4.0 * v[1:-1, 1:-1]

               # u*v*v nonlinear term
               uvv = u[1:-1, 1:-1] * v[1:-1, 1:-1] * v[1:-1, 1:-1]

               # update
               un[1:-1, 1:-1] = u[1:-1, 1:-1] + self.Du*Lu - uvv + self.alpha * (1.0 - u[1:-1, 1:-1])
               vn[1:-1, 1:-1] = v[1:-1, 1:-1] + self.Dv*Lv + uvv - (self.alpha + self.beta) * v[1:-1, 1:-1]

               # periodic boundaries
               un[0, :]  = un[-2, :]; vn[0, :]  = vn[-2, :]
               un[-1, :] = un[1, :];  vn[-1, :] = vn[1, :]
               un[:, 0]  = un[:, -2]; vn[:, 0]  = vn[:, -2]
               un[:, -1] = un[:, 1];  vn[:, -1] = vn[:, 1]

               # swap
               un, u = u, un
               vn, v = v, vn


        # u, v point to the last iterate
        self.u  = u
        self.v  = v
        self.un = un
        self.vn = vn



#
# --- main
#
if __name__ == "__main__":

    print (" reaction-diffusion demo")

    # process arguments
    import argparse
    parser = argparse.ArgumentParser (description="solver demo for the Gray--Scott reaction-diffusion equation")
    parser.add_argument ("-i", "--input",  dest="config_file", default=None,    help="configuration file")
    parser.add_argument ("-a", "--array",  dest="array_lib",   default="numpy", help="n-dim array library (numpy[default], cupy)")
    parser.add_argument ("-o", "--output", dest="output_file", default=None,    help="output plot file (file name, no extension)")
    args = parser.parse_args ()

    print (f" ... importing {args.array_lib}")
    if args.array_lib == "numpy":
        import numpy as vecpy
    elif args.array_lib == "cupy":
        import cupy as vecpy
    else:
        raise Exception (f" *** error: {args.array_lib}: wrong n-dim array library required")

    import random
    import time

    #
    # ... read in parameters (avoind using numpy)
    with open (args.config_file, "r") as conf_file:
        config_data = conf_file.readlines()
        config_params = {
            "nx":      int(config_data[0]),
            "ny":      int(config_data[1]),
            "Du":    float(config_data[2]),
            "Dv":    float(config_data[3]),
            "alpha": float(config_data[4]),
            "beta":  float(config_data[5]),
            "niter":   int(config_data[6]),
            "us":    float(config_data[7]),
            "vs":    float(config_data[8]),
            "ns":      int(config_data[9]),
        }

    #
    # .... initialise
    t = time.time()
    demo = gs (params=config_params)
    t = time.time() - t
    print (f" ... init time: {t} sec")

    #
    # .... iterate
    t = time.time()
    demo.iterate ()
    t = time.time() - t
    print (f" ... iter time: {t} sec")

    #
    # .... plot
    if args.output_file is not None:
        from matplotlib import pyplot

        pyplot.figure(1);
        if args.array_lib == "cupy":
            pyplot.imshow (demo.u.get(), interpolation="bilinear", cmap="hot");
        else:
            pyplot.imshow (demo.u,       interpolation="bilinear", cmap="hot");
        pyplot.axis("off")

        pyplot.savefig (args.output_file + ".png")
