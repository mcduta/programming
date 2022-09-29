# A Finite Difference Solver for the Gray--Scott Reaction--Diffusion System

## Background

This Python code implements the same reaction-diffusion explicit finite-difference solver as the C++ source.

The emphasis of the implementation is on ``numpy`` and ``cupy``, first illustrating the ease of use and second, comparing performance.

There is no package installation (the Python source is just a demo) but need it needs ``cupy`` installed.


## Help

```
python rd.py --help
```

The demo uses the same configuration input files as the C++ code, specifies the use of either ``numpy`` (the default) or ``cupy`` as well as the name of a PNG output file.


## Run with ``numpy``
```
python rd.py -i flower.in -a numpy -o flower_numpy

```


## Run with ``cupy``
```
python rd.py -i flower.in -a cupy -o flower_cupy
```


## Performance notes

  * loading ``cupy`` is slow;
  * ``cupy`` random number initialisation is slow (not the RNG as such but the ``rd`` initialiser as a whole);
  * ``cupy`` iterations are fast.
