cd basics/
nvidia-smi 
cd basics/
g++ -fopenmp -o lsdevices lsdevices.cpp 
./lsdevices 
cd ../
cd daxpy/daxpy.cpp 
cd daxpy/
ls
g++ -O3 -fopenmp -o daxpy daxpy.cpp 
vim daxpy.cpp 
vi daxpy.cpp 
g++ -O3 -fopenmp -o daxpy daxpy.cpp 
./daxpy 
vi daxpy.cpp 
g++ -O3 -fopenmp -o daxpy daxpy.cpp 
./daxpy 
vi daxpy.cpp 
g++ -O3 -fopenmp -o daxpy daxpy.cpp 
./daxpy 
nvprof ./daxpy 
ldd basics/lsdevices 
gcc -v
cd basics/
g++ -O3 -fopenmp -o lsdevices lsdevices.cpp --static
g++ -O3 -fopenmp -o lsdevices lsdevices.cpp 
./lsdevices 
