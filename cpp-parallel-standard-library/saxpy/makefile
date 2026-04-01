# compilers
COMPILER ?= nvhpc

ifeq ($(COMPILER), nvhpc)
#module load nvhpc/2025 gcc/14
  CXX         = nvc++
  CXXFLAGS    = -O3 -march=native -pedantic -stdpar=gpu --std=c++23 --gcc-toolchain=$(shell which g++)
  LDFLAGS     =

else ifeq ($(COMPILER), rocm)
  ifeq ($(STDPAR_PATH), )
    $(error STDPAR_PATH not set)
  endif
  ROCM_GPU   ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

  CXX         = amdclang++
  CXXFLAGS    = -O3 -march=native -fstrict-aliasing -std=c++23 --hipstdpar --hipstdpar-path=$(STDPAR_PATH) --offload-arch=$(ROCM_GPU)
  LDFLAGS     = -fno-lto -lm
# --offload-arch=gfx942

else
  $(error *** COMPILER not set: nvhpc, rocm)
endif

.PHONY:	default
default:saxpy
saxpy:	saxpy.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^
clean:
	/bin/rm -f saxpy


# Possible extra compiler options
#   * work by targeting CPU multithreading;
# ---
#
# ifeq ($(COMPILER), gcc)
#   # module load gcc/14_cuda-12
#   ifeq ($(TBBROOT), )
#     $(error TBBROOT not set)
#   endif

#   CXX         = g++
#   CXXFLAGS    = -O3 -mavx2 -Wall -Wextra -pedantic-errors -I $(TBBROOT)/include --std=c++23
#   LDFLAGS     = -L $(TBBROOT)/lib/intel64/gcc4.8 -ltbb -Wl,-rpath=$(TBBROOT)/lib/intel64/gcc4.8

# else ifeq ($(COMPILER), llvm)
#   # module load llvm/18_cuda12
#   CXX         = clang++
#   CXXFLAGS    = -O3 -mtune=core-avx2 -Wall -pedantic --std=c++23 -stdlib=libc++ -fexperimental-library
#   LDFLAGS     =
# endif
