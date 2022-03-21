# Using Random Number Generators in distributed computing

This demo runs a MPI-distributed Monte Carlo simulation employing RNG from the Intel mkl/vsl library. Specifically, it uses MT2203 as the basic generator and a uniform distribution between 0 and 1.

Minimal setup using ``makefile``. With MPI binaries in the path and ``MKLROOT`` defined, ``make`` will generate the executable. If something breaks, edit to update for site specifics.

Adapted from ``${MKLROOT}/examples/vslc/makefile``.
