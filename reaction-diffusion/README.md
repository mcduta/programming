# A Finite Difference Solver for the Gray--Scott Reaction--Diffusion System

## Background

This C++ code implements an explicit finite-difference solver for a set of reaction-diffusion equations [[1]](#1) on a rectangular 2D domain with periodic boundary conditions. The equations model the time-variation of the concentration of two chemical species in a reaction-diffusion chemical reaction. Chemical reactions are localised, hence the numerical model is a good candidate for a parallel implementation.

The emphasis of the implementation is on exposing enough floating point operations to make the code interesting for data loop parallelisation (OpenMP) and instruction level parallelisation (vectorisation). Solution accuracy is not important, and the explicit scheme is adequate enough. 


## Source tree

The source has a number of directories

  * ``python/`` -- directory for a Python visualiser for the solution file;
  * ``config/`` -- directory with run configuration examples;
  * ``src/`` -- the C++ source;
  * ``cupy/`` -- a Python ``cupy`` implementation of the finite-difference solver (with a separate README).


## Build

The minimal build sequence of operations from the source root
```
mkdir build && cd build
cmake ../src -D CMAKE_INSTALL_PREFIX=../
make install
```

The above will create an extra ``bin`` directory for the executables alongside the other directories listed above and copy the resulting executables in it.

The executables are
  * ``rd`` -- the main solver;
  * ``rdGL`` -- an OpenGL based solver that renders the solution as it is being iterated on.

``cmake`` attempts to build the executables with
  * OpenMP support for multithreaded execution;
  * OpenGL support for the rendering ``rdGL`` executable;
  * HDF5 support for solution file storage.

None of the above features is ``REQUIRED``, and ``cmake`` still builds a single threaded ``rd`` executable in the absence of all three above.

Tweaks:
  * use something like ``CXX=$(type -p g++) cmake`` to pick up the C++ compiler from the environment (in the case ``CXX`` is not set on an adequate compiler);
  * specify the ``Debug`` build option with ``cmake -D CMAKE_BUILD_TYPE="Debug"`` to generate executables with debug symbols (``Release`` is the default), useful when debugging or profiling;
  * provide the path to a serial HDF5 library installation with ``-D HDF5_ROOT=/path/to/hdf5`` to output the solution file in HDF5 format;
  * while the default is to build with single precision solution, the executables can be generated in double precision by specifying the build option ``-D USE_DOUBLE_PRECISION=ON``.


## Run

Run the executable ``rd`` in a directoruy in which it picks up the default configuration file ``config.in``. The structure of this file is like this

  * ``300`` -- the number of finite difference points in x;
  * ``400`` -- the number of finite difference points in y;
  * ``0.1000`` -- diffusion coefficient ``Du`` for first chemical species;
  * ``0.0500`` -- diffusion coefficient ``Dv`` for second chemical species;
  * ``0.0545`` -- ``alpha`` coefficient ("feed rate");
  * ``0.0620`` -- ``beta`` coefficient ("kill rate");
  * ``100`` -- number of time iterations
  * ``0.50`` -- value of spot initialisation for first chemical species;
  * ``0.25`` -- value of spot initialisation for second chemical species;
  * ``7`` -- number or random spots to initialise.

The solution is random in both species initially, with the exception of a number of spots (random position and size), where both species are initialised to the respective values specified in the config file. Note: a ``json`` configuration file would be a friendlier option but that would bring library dependencies that would break the simplicity of this code.

An alternative configuratrion file can be passed on to the executable as an argument. A minimal run sequence is
```
./bin/rd ./config/config.in
python ./python/rdPlot.py
```

The ``rd`` executable outputs the computed solution to the file ``sol.dat`` (bespoke binary file format) or ``sol.h5`` (HDF5 format), depending on a viable HDF5 installation having been found by ``cmake``. ``sol.dat`` is the default solution file that ``rdPlot.py`` reads but ``sol.h5`` (or any other name) can replace that if supplied as a command line argument; ``python ./python/rdPlot.py sol.h5``.

Multithreading is via OpenMP, hence tunable via the environment variable ``OMP_NUM_THREADS``. To run with 4 threads of execution, for example, do
```
OMP_NUM_THREADS=4 ./bin/rd ./config/config.in
```

New configuration files can be added using model paramaters found in similar demonstrators: [[2]](#2), [[3]](#3), [[4]](#4).

Another executable built is ``rdGL``, which renders the first species solution (while it is being computed) using OpenGL. The configuration setup and the solution solver is shared with ``rd``. Users can interact with the rendering window via the following keys:
  * ``h`` -- print help;
  * ``k`` -- recalibrate colours;
  * ``s`` -- save current solution to file (to visualise using the Python tool);
  * ``q`` -- terminate and quit.

One ``rdGL`` caveat is the rendering "idle" function keeps iterating and rendering ad infinitum, which means the number of iterations in the config file is the number after which OpenGL renders rather than the total number of iterations.



## References
<a id="1">[1]</a>
https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system

<a id="2">[2]</a>
https://mrob.com/pub/comp/xmorphia/

<a id="3">[3]</a>
https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/

<a id="4">[4]</a>
http://pmneila.github.io/jsexp/grayscott/
