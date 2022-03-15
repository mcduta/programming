# A Finite Difference Solver for the Gray--Scott Reaction--Diffusion System

## Background

This C++ code implements an explicit finite-difference solver for a set of reaction-diffusion equations [[1]](#1) on a rectangular 2D domain with periodic boundary conditions. The equations model the time-variation of the concentration of two chemical species in a reaction-diffusion chemical reaction. Chemical reactions are localised, hence the numerical model is a good candidate for a parallel implementation.

The emphasis of the implementation is on exposing enough floating point operations to make the code interesting for data loop parallelisation (OpenMP) and instruction level parallelisation (vectorisation). Solution accuracy is not important, and the explicit scheme is adequate enough. 


## Source tree

The source has a number of directories

  * ``python/`` -- directory for a Python visualiser for the solution file;
  * ``config/`` -- directory with run configuration examples;
  * ``src/`` -- the C++ source.


## Build

The minimal build sequence of operations from the source root
```
mkdir build && cd build
cmake ../src -D CMAKE_INSTALL_PREFIX=../
make install
```

The above will create an extra ``bin`` directory for the executables alongside the other directories listed above and copy the resulting executables in it.

Tweaks:
  * use something like ``CXX=$(type -p g++) cmake`` to pick up the C++ compiler from the environment (in the case ``CXX`` is not set on the adequate compiler);
  * specify the ``Debug`` build option with ``cmake -D CMAKE_BUILD_TYPE="Debug"`` to generate executables with debug symbols (``Release`` is the default)

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

The solution is zero in both species initially, with the exception of a number of spots (random position and size), where both species are initialised to the respective values specified in the config file. Note: a ``json`` configuration file would be a friendlier option but that would bring library dependencies that would break the simplicity of this code.

An alternative configuratrion file can be passed on to the executable as an argument. A minimal run sequence is
```
./bin/rd ./config/config.in
python ./python/rdPlot.py
```

Another executable built is ``rdGL``, which renders the first species solution (while it is being computed) using OpenGL. The configuration setup and the solution solver is shared with ``rd``. Users can interact with the rendering window via the following keys:
  * ``h`` -- print help;
  * ``k`` -- recalibrate colours;
  * ``s`` -- save current solution to file (to visualise using the Python tool);
  * ``q`` -- terminate and quit.

``rdGL`` limitations:
  * rendering works only if the number of finited difference points in x and y are the same (this is a bug!);
  * the rendering "idle" function keeps iterating and rendering ad infinitum, which means the number of iterations in the config file is the number after which OpenGL renders rather than the total number of iterations.



## References
<a id="1">[1]</a>
https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system

<a id="2">[2]</a>
https://mrob.com/pub/comp/xmorphia/

<a id="3">[3]</a>
https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/
