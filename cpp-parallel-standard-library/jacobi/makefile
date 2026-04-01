# compilers
COMPILER ?= nvhpc

ifeq ($(COMPILER), nvhpc)
#module load nvhpc/2025 gcc/14
  CXX         = nvc++
  CXXFLAGS    = -O3 -march=native -pedantic -stdpar=gpu --std=c++23 --gcc-toolchain=$(shell which g++)
  LDFLAGS     =

else ifeq ($(COMPILER), rocm)
#export HSA_XNACK=1
#export HIP_PLATFORM=amd
#module load rocm/therock-23.1.0 amdclang/23.0.0-23.1.0
  ifeq ($(STDPAR_PATH), )
    $(error STDPAR_PATH not set)
  endif

  CXX         = amdclang++
  ROCM_GPU   ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
  CXXFLAGS    = -O3 -march=native -fstrict-aliasing -std=c++23 --hipstdpar --hipstdpar-path=$(STDPAR_PATH) --offload-arch=$(ROCM_GPU)
  LDFLAGS     = -fno-lto -lm
# --offload-arch=gfx942

else
  $(error *** COMPILER not set: nvhpc, rocm)
endif

.PHONY:	default
default:jac
jac:	jacobi.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^
clean:
	/bin/rm -f jac
