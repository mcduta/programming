# compilers
COMPILER ?= nvhpc

ifeq ($(COMPILER), gcc)
  ifeq ($(TBBROOT), )
    $(error TBBROOT not set)
  endif

  CXX         = g++
  CXXFLAGS    = -O3 -mavx2 --std=c++23 -I $(TBBROOT)/include
  OMPFLAGS    = -fopenmp
  OMPFLAGSGPU = -foffload=nvptx-none -fno-lto
  LFLAGS      = -L $(TBBROOT)/lib/intel64/gcc4.8 -ltbb -Wl,-rpath=$(TBBROOT)/lib/intel64/gcc4.8

else ifeq ($(COMPILER), llvm)
  CXX         = clang++
  CXXFLAGS    = -O3 -mtune=emeraldrapids --std=c++23 -stdlib=libc++ -fexperimental-library
  OMPFLAGS    = -fopenmp
  OMPFLAGSGPU = -fopenmp-targets=nvptx64-nvidia-cuda -foffload-lto
  LFLAGS      =

else ifeq ($(COMPILER), nvhpc)
  CXX         = nvc++
  CXXFLAGS    = -O3 -mavx2 --std=c++23 --gcc-toolchain=$(shell which g++)
  OMPFLAGS    = -fopenmp
  OMPFLAGSGPU = -mp=gpu -ltbb
  LFLAGS      = -lcuda
  LFLAGSEXTRA = -L ${CUDA_HOME}/lib64 -lcudart

else ifeq ($(COMPILER), rocm)
  ROCM_GPU   ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

  CXX         = amdclang++
  CXXFLAGS    = -O3 -mavx2 --std=c++23
  OMPFLAGS    = -fopenmp
  OMPFLAGSGPU = --offload-arch=${ROCM_GPU} -ltbb
  LFLAGSEXTRA = --hip-link -L${ROCM_PATH}/hip/lib -lroctx64

else
  $(error COMPILER not set: gcc, llvm, nvhpc, rocm)
endif

default:		jac

#                       jac -- CPU multithreading and GPU offloading using OpenMP
jac:			jacobi.cpp
			$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(OMPFLAGSGPU) $(LFLAGS) -o $@ $^

#                       cujac -- CPU multithreading using OpenMP and GPU acceleration using CUDA
ifeq ($(COMPILER), nvhpc)
jacobi-update-cuda.o:	jacobi-update-cuda.cu
			nvcc -O3 -I ${CUDA_HOME}/include -c $^
cujac:			jacobi.cpp jacobi-update-cuda.o
			$(CXX) -D USE_CUDA $(CXXFLAGS) $(OMPFLAGS) $(OMPFLAGSGPU) $(LFLAGS) $(LFLAGSEXTRA) -o $@ $^
endif

#                       hipjac -- CPU multithreading using OpenMP and GPU acceleration using HIP
ifeq ($(COMPILER), rocm)
jacobi-update-hip.o:	jacobi-update-cuda.cu
			hipify-perl $^ > jacobi-update-hip.cpp
			hipcc -O3 -c jacobi-update-hip.cpp
hipjac:			jacobi.cpp jacobi-update-hip.o
			$(CXX) -D USE_CUDA $(CXXFLAGS) $(OMPFLAGS) $(OMPFLAGSGPU) $(LFLAGS) $(LFLAGSEXTRA) -o $@ $^
endif

clean:
			/bin/rm -f *.o jac cujac hipjac
