import sys
import numpy

if len(sys.argv) == 4:
    NX = int(sys.argv[1])
    NY = int(sys.argv[2])
    NT = int(sys.argv[3])
else:
    print(F" *** usage: {sys.argv[0]} NX NY NT")
    exit(1)

u0 = numpy.loadtxt("sol_init.txt").reshape(NX, NY)
u  = numpy.loadtxt("sol_iter.txt").reshape(NX, NY)

ua = u0.copy();
for i in range(NT):
    ua[1:NX-1, 1:NY-1] = 0.25 * ( ua[2:NX, 1:NY-1] + ua[0:NX-2, 1:NY-1]
                                + ua[1:NX-1, 2:NY] + ua[1:NX-1, 0:NY-2] )

err_max = numpy.max(numpy.abs(ua - u))
err_fro = numpy.linalg.norm(ua - u, "fro") / numpy.sqrt(NX*NY)
print(F" (NX,NY)=({NX},{NY}), {NT} iters, max error={err_max}, fro error={err_fro}")
