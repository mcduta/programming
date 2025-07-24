# Memory management in CUDA

## Background

CUDA (Compute Unified Device Architecture) provides two main memory management models to optimize performance across different GPU programming use cases: the __separate memory model__ and the __unified memory model__.

In the __separate memory model__, the host (CPU) and device (GPU) maintain separate memory spaces, and the programmer is responsible for explicitly managing data movement between host and device using CUDA API functions. Thus, the host memory is allocated with ``malloc`` and the device memory with ``cudaMalloc``, while the data transfers are managed using ``cudaMemcpy``.

In the __unified memory model__ (UM) [[1]](#1), the host and device use a single shared memory space, and the programmer uses a simplified memory allocation with ``cudaMallocManaged``, without the need of explicit data copies; the operating system performs on-demand migration of data between host and device.

While separate memory programming gives best performance control, with predictable and reproducible behaviour, UM simplifies programming, has a reduced memory management complexity, and is ideal for rapid prototyping. A further attraction of UM is the ability to allocate more memory thank is physically available on the GPU.

__Heterogeneous Memory Management___ (HMM) [[3]](#3) is an enhancement of the __unified memory model__, which allows the GPU to access system memory directly, even without requiring data migration, using standard C/C++ pointers allocated with ``malloc`` and ``new``. Unlike UM, which requires little in terms of system and hardware support, HMM relies on close coordination between the GPU hardware, CUDA driver/runtime, operating system, and system architecture. HMM tightly integrates the GPU into the system's memory management, letting both CPU and GPU share and access the same virtual address space.

## The CUDA code examples

This project illustrates the programming for the __separate memory model__, the __UM model__, and the __HMM model__ using a very simple example application, in which an array of floats is initialised on the host with random values, then processed on the device and finally verified on the host. The CUDA device kernel applies the sequence ``x:=sqrt(1+x)`` for a number of iterations to each entry of the array in turn. The sequence converges to the golden ratio ``(1+sqrt(5)/2, so the correctness of the results can be easily checked.

There are three implementations to this application, and they differ only slightly in how the host/device memory is managed for the array:
* ``gpu-mem-separate.cu``;
* ``gpu-mem-unified.cu``;
* ``gpu-mem-heterogeneous.cu``.

In ``gpu-mem-separate.cu``, there are two separate memory allocation, one on the CPU side using ``new`` for the array ``xCPU`` and one on the GPU using ``cudaMalloc`` for the array ``xGPU``. ``xCPU`` is then initialised by the function ``array_init`` on the CPU and copied into ``xGPU`` using ``cudaMemcpy``. The kernel ``array_calcs`` is then applied to ``xGPU``, which is then copied back to host into ``xCPU`` using a second ``cudaMemcpy``. The checker function ``array_check`` is then run on ``xCPU``. This represents the traditional programming with the system (CPU) and device memories separated.

In ``gpu-mem-unified.cu``, there is a single array, ``xGPU``, allocated using ``cudaMallocManaged``. All the functions and steps described above (``array_init``, ``array_calcs`` and ``array_check``) are then applied to ``xGPU`` directly, without any explicit data transfers between host and device. On any system supporting CUDA 6.0 and above, any memory page migration is managed automatically by the operating system.

In ``gpu-mem-heterogeneous.cu``, the simplified programming of the unified memory is taken even further, replacing the allocation using ``cudaMallocManaged`` with a simple ``new`` and the deallocation using ``cudaFree`` with a ``delete``.

## Build and run

Use the command ``make`` to build the three corresponding executables. The CUDA compiler ``nvcc`` has to be in the path.

All versions of the application has the same command line arguments that change the number of GB allocated (default is 1GB) and the number of times each array entry is iterated on to converge to the golden ratio (default is 1000). For example,
```
./gpu-mem-unified -g 50
```
allocates an array of 50GB, with the CUDA kernel iterating 1000 times on each element. The programming uses 64 bit array indexing to allow for testing very large arrays.

## Memory management

By design, the __UM model__ programming and, as an extension, the __HMM model__, can both allocate and use more memory than is available on the device. While both __UM__ and __HMM__ support GPU memory oversubscription, they do so in fundamentally different ways, especially in how data migration and memory access are handled.

Thus, when __UM__ uses ``cudaMallocManaged`` to allocate memory that exceeds the GPU’s physical memory, CUDA automatically migrates memory pages between host and device, with pages brought in on demand, as the GPU accesses them. Only the data actively used on the GPU is resident on the GPU at any time, with the rest remainingh in host system memory, and migrated in or out of the device memory based on usage patterns. This is managed in software by the CUDA runtime and the GPU cannot access host pages directly; data must be copied to GPU memory first. Expectedly, the advantages of simplified programming in __UM__ come with a performance hit from page migration overheads, possibly poor memory access locality, limited control over placement, etc. Performance pitfalls can be avoided via careful programming [[2]](#2) but that negates to a point the initial easy programming proposition.

__HMM__ on the other hand manages oversubscription naturally, with the GPU treating host memory as its own. This is supported in hardware, with page fault handling managed at the OS level and using the Input-Output Memory Management Unit (IOMMU), thus creating a true shared virtual memory between CPU and GPU. __HMM__ alleviates the overheads of the __UM__ but requires running on adequate hardware, such as the Grace Hopper GH200 superchip. Verifying whether the hardware supports __HMM__ is easily done with the ``nvaccelinfo`` utility, which has to display the field
```
...
Memory Models Flags:           -gpu=mem:separate, -gpu=mem:managed, -gpu=mem:unified
...

```


## References

<a id="1">[1]</a>
https://developer.nvidia.com/blog/unified-memory-cuda-beginners/

<a id="2">[2]</a>
https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/

<a id="3">[3]</a>
https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/