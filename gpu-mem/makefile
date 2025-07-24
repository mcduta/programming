# tools
NVCC = nvcc
NVFLAGS = -O3

# rules
.PHONY:			clean gpu-mem-separate gpu-mem-unified gpu-mem-heterogeneous

all:			gpu-mem-separate gpu-mem-unified gpu-mem-heterogeneous
gpu-mem-common.o:	gpu-mem-common.cu
			$(NVCC) $(NVFLAGS) -c $<
gpu-mem-separate:	gpu-mem-separate.cu gpu-mem-common.o
			$(NVCC) $(NVFLAGS) -o $@ $^
gpu-mem-unified:	gpu-mem-unified.cu gpu-mem-common.o
			$(NVCC) $(NVFLAGS) -o $@ $^
gpu-mem-heterogeneous:	gpu-mem-heterogeneous.cu gpu-mem-common.o
			$(NVCC) $(NVFLAGS) -o $@ $^
clean:
			/usr/bin/rm -f *.o \
			gpu-mem-separate gpu-mem-unified gpu-mem-heterogeneous
