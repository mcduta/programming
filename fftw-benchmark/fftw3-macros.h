
//
// ----- FFTW standard include files
//
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <fftw3.h>
# include <sys/time.h>
# ifdef FFTW3_OMP
  # include <omp.h>
# endif


//
// ----- check macros
//
# if ( defined(FFTW3_DOUBLE) && defined(FFTW3_FLOAT) )
  # error " *** either FFTW3_DOUBLE or FFTW3_FLOAT must be defined"
# endif

# if ( !defined(FFTW3_DOUBLE) && !defined(FFTW3_FLOAT) )
  # warning " *** FFTW3_DOUBLE defined as default"
  # define FFTW3_DOUBLE
# endif


//
// ----- define FFTW library function name macros
//
# if ( defined(FFTW3_DOUBLE) )
  # define __FFTW3_REAL                    double
  # define __FFTW3_COMPLEX                 fftw_complex
  # define __FFTW3_PLAN                    fftw_plan
  # define __FFTW3_MALLOC                  fftw_malloc
  # define __FFTW3_INIT_THREADS            fftw_init_threads
  # define __FFTW3_PLAN_WITH_NTHREADS      fftw_plan_with_nthreads
  # define __FFTW3_PLAN_DFT_1D             fftw_plan_dft_1d
  # define __FFTW3_PLAN_DFT_2D             fftw_plan_dft_2d
  # define __FFTW3_PLAN_DFT_3D             fftw_plan_dft_3d
  # define __FFTW3_PLAN_DFT_R2C_1D         fftw_plan_dft_r2c_1d
  # define __FFTW3_PLAN_DFT_R2C_2D         fftw_plan_dft_r2c_2d
  # define __FFTW3_PLAN_DFT_R2C_3D         fftw_plan_dft_r2c_3d
  # define __FFTW3_PLAN_DFT_C2R_1D         fftw_plan_dft_c2r_1d
  # define __FFTW3_PLAN_DFT_C2R_2D         fftw_plan_dft_c2r_2d
  # define __FFTW3_PLAN_DFT_C2R_3D         fftw_plan_dft_c2r_3d
  # define __FFTW3_PLAN_GURU_DFT           fftw_plan_guru_dft
  # define __FFTW3_PLAN_GURU_SPLIT_DFT     fftw_plan_guru_split_dft
  # define __FFTW3_PLAN_GURU_SPLIT_DFT_R2C fftw_plan_guru_split_dft_r2c
  # define __FFTW3_PLAN_GURU_SPLIT_DFT_C2R fftw_plan_guru_split_dft_c2r
  # define __FFTW3_EXECUTE                 fftw_execute
  # define __FFTW3_EXECUTE_SPLIT_DFT       fftw_execute_split_dft
  # define __FFTW3_EXECUTE_SPLIT_DFT_R2C   fftw_execute_split_dft_r2c
  # define __FFTW3_EXECUTE_SPLIT_DFT_C2R   fftw_execute_split_dft_c2r
  # define __FFTW3_DESTROY_PLAN            fftw_destroy_plan
  # define __FFTW3_CLEANUP_THREADS         fftw_cleanup_threads
  # define __FFTW3_FREE                    fftw_free
  # define __FFTW3_EXPORT_WISDOM_TO_FILE   fftw_export_wisdom_to_file
  # define __FFTW3_IMPORT_WISDOM_FROM_FILE fftw_import_wisdom_from_file
# else
  # define __FFTW3_REAL                    float
  # define __FFTW3_COMPLEX                 fftwf_complex
  # define __FFTW3_PLAN                    fftwf_plan
  # define __FFTW3_INIT_THREADS            fftwf_init_threads
  # define __FFTW3_PLAN_WITH_NTHREADS      fftwf_plan_with_nthreads
  # define __FFTW3_MALLOC                  fftwf_malloc
  # define __FFTW3_PLAN_DFT_1D             fftwf_plan_dft_1d
  # define __FFTW3_PLAN_DFT_2D             fftwf_plan_dft_2d
  # define __FFTW3_PLAN_DFT_3D             fftwf_plan_dft_3d
  # define __FFTW3_PLAN_DFT_R2C_1D         fftwf_plan_dft_r2c_1d
  # define __FFTW3_PLAN_DFT_R2C_2D         fftwf_plan_dft_r2c_2d
  # define __FFTW3_PLAN_DFT_R2C_3D         fftwf_plan_dft_r2c_3d
  # define __FFTW3_PLAN_DFT_C2R_1D         fftwf_plan_dft_c2r_1d
  # define __FFTW3_PLAN_DFT_C2R_2D         fftwf_plan_dft_c2r_2d
  # define __FFTW3_PLAN_DFT_C2R_3D         fftwf_plan_dft_c2r_3d
  # define __FFTW3_PLAN_GURU_DFT           fftwf_plan_guru_dft
  # define __FFTW3_PLAN_GURU_SPLIT_DFT     fftwf_plan_guru_split_dft
  # define __FFTW3_PLAN_GURU_SPLIT_DFT_R2C fftwf_plan_guru_split_dft_r2c
  # define __FFTW3_PLAN_GURU_SPLIT_DFT_C2R fftwf_plan_guru_split_dft_c2r
  # define __FFTW3_EXECUTE                 fftwf_execute
  # define __FFTW3_EXECUTE_SPLIT_DFT       fftwf_execute_split_dft
  # define __FFTW3_EXECUTE_SPLIT_DFT_R2C   fftwf_execute_split_dft_r2c
  # define __FFTW3_EXECUTE_SPLIT_DFT_C2R   fftwf_execute_split_dft_c2r
  # define __FFTW3_DESTROY_PLAN            fftwf_destroy_plan
  # define __FFTW3_CLEANUP_THREADS         fftwf_cleanup_threads
  # define __FFTW3_FREE                    fftwf_free
  # define __FFTW3_EXPORT_WISDOM_TO_FILE   fftwf_export_wisdom_to_file
  # define __FFTW3_IMPORT_WISDOM_FROM_FILE fftwf_import_wisdom_from_file
# endif


//
// ----- define FFTW library function name macros
//
# if ( defined(FFTW3_DOUBLE) )
  # define __MX_REAL_CLASS mxDOUBLE_CLASS
# else
  # define __MX_REAL_CLASS mxSINGLE_CLASS
# endif


//
// ----- FFTW wisdom options
//
enum FFTW3_WISDOM_OPT {
  FFTW3_WISDOM_UNDEF,
  FFTW3_WISDOM_NONE,
  FFTW3_WISDOM_CREATE,
  FFTW3_WISDOM_READ
};


//
// ----- row major indexing
//
# define ROW_MAJOR_INDEX_1D(M,m)                   m
# define ROW_MAJOR_INDEX_2D(M,N,m,n)           n + m*N
# define ROW_MAJOR_INDEX_3D(M,N,L,m,n,l)  l + (n + m*N)*L


//
// ----- column major indexing
//
# define COL_MAJOR_INDEX_1D(M,m)                   m
# define COL_MAJOR_INDEX_2D(M,N,m,n)             n *M + m
# define COL_MAJOR_INDEX_3D(M,N,L,m,n,l)  (l*N + n)*M + m



/*                              E  N  D                               */
