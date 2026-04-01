# compilers
COMPILER ?= nvhpc

ifeq ($(COMPILER), gcc)
  CXX         = g++
  CXXFLAGS    = -O3 -mavx2 --std=c++23 -Wall -Wextra -pedantic-errors
  OMPFLAGS    = -fopenmp
  OMPFLAGSGPU = -fopenmp -foffload=nvptx-none -fno-lto
else ifeq ($(COMPILER), llvm)
  CXX         = clang++
  CXXFLAGS    = -O3 -mtune=core-avx2 --std=c++23 -stdlib=libc++ -fexperimental-library
  OMPFLAGS    = -fopenmp
  OMPFLAGSGPU = -fopenmp-targets=nvptx64-nvidia-cuda -foffload-lto
else ifeq ($(COMPILER), nvhpc)
  CXX         = nvc++
  CXXFLAGS    = -O3 -mavx2 --std=c++23 --gcc-toolchain=$(shell which g++)
  OMPFLAGS    = -fopenmp
  OMPFLAGSGPU = -mp=gpu
else ifeq ($(COMPILER), amdclang)
  CXX         = amdclang++
  CXXFLAGS    = -O3 -mavx2 --std=c++23
  OMPFLAGS    = -fopenmp
  ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
  OMPFLAGSGPU = --offload-arch=${ROCM_GPU}
else
  $(error *** COMPILER not set: gcc, llvm, nvhpc, amdclang)
endif

.PHONY:	default
default:saxpy
saxpy:	saxpy.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(OMPFLAGSGPU) -o $@ $^
clean:
	/bin/rm -f saxpy
