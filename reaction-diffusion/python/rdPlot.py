import numpy

with open("sol.dat", "rb") as solfile:
    header = numpy.fromfile (solfile, dtype=numpy.uint64, count=2)
    data   = numpy.fromfile (solfile, dtype=numpy.float64)

nx, ny = header[0], header[1];
u = data[0:nx*ny].reshape(ny,nx).T
v = data[nx*ny:].reshape(ny,nx).T

from matplotlib import pyplot

pyplot.figure(1); pyplot.imshow (u);
pyplot.figure(2); pyplot.imshow (v); pyplot.show ();
