ls
vi makefile 
make perf_dgemm
ll
ls -a
OMP_NUM_THREADS=4 ./perf_dgemm_openblas 4000 4000 4000
gfortran -v
ldd ./perf_dgemm_openblas
OMP_NUM_THREADS=4 ./perf_dgemm 4000 4000 4000
OMP_NUM_THREADS=4 ./perf_dgemm 4000 4000 4000
OMP_NUM_THREADS=4 ./perf_dgemm 4000 4000 4000
exit
ls
ldd ./perf_dgemm_mkl
ldd ./perf_dgemm_openblas 
exit
