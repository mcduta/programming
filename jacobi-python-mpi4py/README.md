# An mpi4py implementation of a 2D Jacobi update

## Material

* `jacobi_ref.py` -- serial reference Python implementation
* `jacobi_mpi.py` -- MPI distributed computing Python implementation (via `mpi4py` support)

## Run examples

```
python jacobi_ref.py  --nrows 600 --ncols 400 -i 1000 -t 1.e-6
mpirun -np 4 python jacobi_mpi.py  --nrows 600 --ncols 600 -i 4000 -t 1.e-6
```

## References

Inspired by [github.com/whdlgp/MPI_jacobi_iteration_example/blob/master/jacobi_mpi.py](https://github.com/whdlgp/MPI_jacobi_iteration_example/blob/master/jacobi_mpi.py).