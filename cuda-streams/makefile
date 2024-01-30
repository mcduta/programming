# tools
CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# options
CCFLAGS       ?= -g -O0 -m64
LIBS          ?=

# rules
.PHONY:			clean test-kernel test-streams

all:			test-kernel test-streams
test-kernel:		test-kernel-float32 test-kernel-float64
test-streams:		test-streams-float32 test-streams-float64
test-kernel-float32:	test-kernel.cu test-utils.cpp test-utils-cuda.cu
			$(NVCC) $(CCFLAGS) -D REAL=float  -o $@ test-kernel.cu $(LIBS)
test-kernel-float64:	test-kernel.cu test-utils.cpp test-utils-cuda.cu
			$(NVCC) $(CCFLAGS) -D REAL=double -o $@ test-kernel.cu$(LIBS)
test-streams-float32:	test-streams.cu test-utils.cpp test-utils-cuda.cu
			$(NVCC) $(CCFLAGS) -D REAL=float  -o $@ test-streams.cu $(LIBS)
test-streams-float64:	test-streams.cu test-utils.cpp test-utils-cuda.cu
			$(NVCC) $(CCFLAGS) -D REAL=double -o $@ test-streams.cu$(LIBS)
clean:
			/usr/bin/rm -f *.o \
			test-kernel-float32 test-kernel-float64 \
			test-streams-float32 test-streams-float64
clobber:		clean
