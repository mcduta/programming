#
# --- sol_read
#       * read solution from file given as argument
def sol_read (filename):

    extension = filename.split(".")[-1]
    if extension == "dat":
        [nx, ny, u, v] = sol_read_dat (filename)
    elif extension == "h5":
        [nx, ny, u, v] = sol_read_h5  (filename)
    else:
        raise ValueError (F" *** unrecognised file extension, must be dat or h5.")

    return nx, ny, u, v


#
# --- sol_read_dta
#       * read solution from generic binary file
def sol_read_dat (filename):

    import numpy

    nx, ny = 0, 0
    u = numpy.empty(0)
    v = numpy.empty(0)

    with open(sol_filename, "rb") as solfile:
        sizereal = numpy.fromfile (solfile, dtype=numpy.uint64, count=1)
        header   = numpy.fromfile (solfile, dtype=numpy.uint64, count=2)
        if sizereal[0] == 4:
            typereal = numpy.float32
        elif sizereal[0] == 8:
            typereal = numpy.float64
        else:
            raise ValueError (F" *** sizeof(REAL): expected 4 or 8 bytes, got {sizereal[0]}")

        data = numpy.fromfile (solfile, dtype=typereal)

    nx, ny = header[0], header[1];
    u = data[0:nx*ny].reshape(ny,nx).T
    v = data[nx*ny:].reshape(ny,nx).T

    return nx, ny, u, v


#
# --- sol_read_dta
#       * read solution from generic binary file
def sol_read_h5 (filename):

    import h5py
    
    with h5py.File(filename, "r") as file:
        u = file["sol/u"]
        v = file["sol/v"]
        nx, ny = u.shape

        return ny, nx, u[:,:].T, v[:,:].T


#
# --- sol_plot
#       * read solution from file given as argument
def sol_plot (nx, ny, u, v):

    from matplotlib import pyplot

    pyplot.figure(1);
    pyplot.imshow (u, interpolation="bilinear", cmap="hot");
    pyplot.axis("off")

    pyplot.figure(2);
    pyplot.imshow (v, interpolation="bilinear", cmap="hot");
    pyplot.axis("off")

    pyplot.show ();


#
# --- main:
#       * takes solution file as argument
#
if __name__ == "__main__":
    import sys
    filename = "sol.dat"
    if len(sys.argv) > 1:
        filename = str(sys.argv[1])
    try:
        [nx, ny, u, v] = sol_read (filename)
    except:
        print (F" *** error opening file {filename}")
    else:
        sol_plot (nx, ny, u, v)
