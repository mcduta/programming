# C memory allocation wrappers for multi-dimensional arrays

## Background
Using heap allocatable multi-dimensional arrays in C presents the programmer with a challenge when using corresponding multi-dimensional indexing. This easily works with multi-dimensional arrays of fixed dimensions at compile time using the familiar indexing (_e.g._ ``[i][j]``) but not with allocatable arrays.

One solution is to allocate a linear memory space of appropriate dimension and use a map of multi-dimensional indexing to a 1D linear indexing. This has the advantage that row- and column-major ordering are easily dealt with [[1]](#1). For example
```
int M=200, N=200; // matrix is of size MxN
float *A = (float *) malloc(M*N*sizeof(float));
int m=15, n=30;   // we want A[m][n]
int k = n*M + m;  // linear memory indexing (m,n) -> k (column-major!)
printf("%f\n", A[k]);
```

Another solution is to present the allocated array as a array of pointers to arrays. With this solution, the multi-dimensional indexing works "naturally". However, the way in which this method is explained very often is first allocate an array of pointers along the first dimension, then for each one of those allocate another array of pointers and so on to the last dimension, where the allocation is for the desired data type. For example, for a 2D matrix, this would be
```
int M=200, N=200; // matrix is of size MxN
float **A = (float **) malloc(M*sizeof(float *));
for(m=0; m<M; m++) A[m] = (float *) malloc(N*sizeof(float));
printf("%f\n", A[m][n]);
```

The resulting array allocation is not memory-contiguous and this solution is therefore not adequate for matrix based computations for a couple of reasons:
  * the element addressing is not cache friendly and
  * the allocation is incompatible with existing numerical libraries.

## A solution

A good solution a combination of the two above:
  * allocate the entire memory with a single ``malloc`` and
  * use arrays of arrays pointing to already allocated memory.

This avoids the disadvantages of cache-unfriendliness and library incompatibility. The file ``array_alloc.c`` contains simple wrapper functions for array allocation that implements the above. Supported are
  * 1, 2, 3 dimensions;
  * ``char``, ``int``, ``float``, ``double`` data types.

The naming convention for routines is
```
alloc_<dimensionality>_<type>
```

where ``<dimensionality>`` is of the form ``nd``, where ``n`` is the number of dimensions, and ``<type>`` is the data type. For example, ``alloc_2d_float`` creates a 2D matrix of ``float``.

Each function returns a pointer with the appropriate degree of indirection for the dimensionality required. ``NULL`` is returned on error.

The arguments to the functions are the dimensions in natural order.


## References

<a id="1">[1]</a>
https://en.wikipedia.org/wiki/Row-_and_column-major_order
