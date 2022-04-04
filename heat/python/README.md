# Control which iterator kernel is being used
Edit the list in function ``runAllStepperOptions`` in ``heat.py``.

# Typical output using all options

> computing 400 iterations on a 600x600 grid
> stepper python, 400 iterations, 158.138310 seconds, 0.007236 Gflops, 0.295148 rel error
> stepper numpy, 400 iterations, 1.572167 seconds, 0.727870 Gflops, 0.295148 rel error
> stepper numba, 400 iterations, 1.520735 seconds, 0.752487 Gflops, 0.295148 rel error
> stepper fortran, 400 iterations, 0.284791 seconds, 4.018150 Gflops, 0.295148 rel error
> stepper ctypes, 400 iterations, 0.076332 seconds, 14.991503 Gflops, 0.295148 rel error
> stepper cython, 400 iterations, 0.146718 seconds, 7.799538 Gflops, 0.295491 rel error
> computing 400 iterations on a 600x600 grid
> stepper pycuda, 400 iterations, 4.396053 seconds, 0.260309 Gflops, 0.295148 rel error
