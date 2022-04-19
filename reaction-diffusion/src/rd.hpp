/*

  project: rd -- finite difference solution of a reaction-diffusion equation
  file:    rd.h -- general header file

 */


// ----- common includes
# include <iostream>
# include <fstream>
# include <cstdlib>
# include <random>
# ifdef _OPENMP
  # include <omp.h>
# endif
# ifdef _HDF5
  # include "H5Cpp.h"
# endif


// ----- definitions
# ifndef MAX
  # define MAX(a,b) (((a) > (b)) ? (a) : (b))
# endif
# ifndef MIN
  # define MIN(a,b) (((a) < (b)) ? (a) : (b))
# endif

# ifndef COL_MAJOR_INDEX_2D
# define COL_MAJOR_INDEX_2D(M,N,m,n) (n)*(M) + (m)
# endif
# ifndef ROW_MAJOR_INDEX_2D
# define ROW_MAJOR_INDEX_2D(M,N,m,n) (n) + (m)*(N)
# endif

# ifndef COL_MAJOR_INDEX_3D
# define COL_MAJOR_INDEX_3D(M,N,L,m,n,l) ((l)*(N) + (n))*(M) + (m)
# endif
# ifndef ROW_MAJOR_INDEX_3D
# define ROW_MAJOR_INDEX_3D(M,N,L,m,n,l) (l) + ((n) + (m)*(N))*(L)
# endif


// ----- determine real number precision
# ifdef _DOUBLE_PRECISION
  # define REAL double
# else
  # define REAL float
# endif

# if (REAL != float) && (REAL != double)
  # error wrong REAL definition.
# endif


/* end */
