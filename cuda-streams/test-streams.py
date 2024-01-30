"""

  Demo for CUDA using streams

  Demo adapted from original source
    https://github.com/NVIDIA-developer-blog/code-samples/tree/master/series/cuda-cpp/overlap-data-transfers

"""


import cupy
import numpy
import sys
import math


#
# --- argument parser
#
def parseArgv () -> int:
    import argparse
    parser = argparse.ArgumentParser (description="demonstrator of CUDA kernel execution in CuPy")
    parser.add_argument ("-m", dest="totalMega", metavar="MB", type=int, help="GPU array size in MB")
    parser.add_argument ("-g", dest="totalGiga", metavar="GB", type=int, help="GPU array size in GB")
    args = parser.parse_args ()

    if args.totalMega is None and args.totalGiga is None:
        # 1GB is default
        totalBytes = 2**30
    else:
        if args.totalMega is not None:
            totalBytes = args.totalMega * 2**20
        if args.totalGiga is not None:
            totalBytes = args.totalGiga * 2**30

    return totalBytes


#
# --- report size
#
def printSize (totalBytes: int, sizeofREAL: int):
    if totalBytes < 2**30:
        print (F" used total GPU memory = {totalBytes/(2**20)} MB")
    else:
        print (F" used total GPU memory = {totalBytes/(2**30)} GB")

    print (F" size of REAL = {sizeofREAL} bytes")


#
# --- the MAIN
#
if __name__ == '__main__':

    # process args
    totalBytes = parseArgv ()

    # set type (4 or 8 bytes)
    sizeofREAL = 4
    dataType = "float32" if sizeofREAL == 4 else "float64"

    # fixed parameters
    blockSize = 256
    streamNum = 4

    # sizes
    totalSize = totalBytes // sizeofREAL

    blockNum = totalSize // (streamNum * blockSize );
    streamSize  = blockNum * blockSize;
    streamBytes = streamSize * sizeofREAL;

    assert streamNum * streamBytes == totalBytes

    # report size
    printSize (totalBytes, sizeofREAL)

    # device ID
    devId = 0;

    # cudaDeviceProp struct reference
    devProp = cupy.cuda.runtime.getDeviceProperties(devId)
    assert totalBytes < devProp["totalGlobalMem"]
    print (F" found device {devProp['name']} with {devProp['totalGlobalMem']/(2**30):6.2f} GB\n")


    #
    # ..... create events and streams
    #
    startEvent = cupy.cuda.Event()
    stopEvent  = cupy.cuda.Event()
    stream = []
    for i in range(streamNum):
        stream.append(cupy.cuda.stream.Stream())


    #
    # ===== element-wise kernels
    #
    cudaKernel = cupy.ElementwiseKernel (
        'T x', 'T y',
        '''
        T z = abs(x);
        z = z / (1.0 + z);
        T s = sin(z);
        T c = cos(z);
        y = sqrt (s*s + c*c)
        ''',
        name='cudaKernel')

    cudaRawKernelFloat32 = cupy.RawKernel (
        r'''
        extern "C" __global__ void cudaRawKernelFloat32 (float *x, const int offset)
        {
        int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
        float z = fabsf (x[i]);
        z = z / (1.0 + z);
        float s = sinf (z);
        float c = cosf (z);
        x[i] = sqrtf (s*s+c*c);
        }
        ''', 'cudaRawKernelFloat32')



    #
    # ===== baseline case - sequential transfer and execute
    #
    #
    xCPU = numpy.random.randn(totalSize).astype(dataType)
    xGPU = cupy.asarray(xCPU)
    startEvent.record()
    # cudaKernel(xGPU, xGPU)
    cudaRawKernelFloat32 ( (int(math.ceil(totalSize/blockSize)), 1, 1), (blockSize, 1, 1), (xGPU, 0))
    stopEvent.record()
    xCPU = cupy.asnumpy(xGPU)
    elapsedTime = cupy.cuda.get_elapsed_time(startEvent, stopEvent)
    print (" sequential transfer and execute");
    print (F"    ... time  = {elapsedTime} ms");
    print (F"    ... error = {numpy.max(xCPU - 1.0)}\n");


    #
    # ===== asynchronous version 1: loop over {copy, kernel, copy}
    #
    startEvent.record()

    # allocate pinned host memory and device memory
    xCPU = numpy.random.randn(totalSize).astype(dataType)

    for i in range(streamNum):
        i1 = (i    * totalSize) // streamNum
        i2 = (i+1) * totalSize  // streamNum

        with stream[i]:
            xGPUstream = cupy.asarray(xCPU[i1:i2])
            # cudaKernel(xGPUstream, xGPUstream)
            cudaRawKernelFloat32 ( (blockNum, 1, 1), (blockSize, 1, 1), (xGPUstream, 0))
            xCPU[i1:i2] = cupy.asnumpy(xGPUstream)

    stopEvent.record()
    cupy.cuda.Stream.null.synchronize()   ## should sync on stopEvent!!!!
    elapsedTime = cupy.cuda.get_elapsed_time(startEvent, stopEvent)

    print (" asynchronous V1 transfer and execute");
    print (F"    ... time  = {elapsedTime} ms");
    print (F"    ... error = {numpy.max(xCPU - 1.0)}\n");


    #
    # ===== asynchronous version 2:
    #       loop over copy, loop over kernel, loop over copy
    #
    startEvent.record()

    # allocate pinned host memory and device memory
    xCPU = numpy.random.randn(totalSize).astype(dataType)

    for i in range(streamNum):
        i1 = (i    * totalSize) // streamNum
        i2 = (i+1) * totalSize  // streamNum

        with stream[i]:
            xGPUstream = cupy.asarray(xCPU[i1:i2])

    for i in range(streamNum):
        # cudaKernel(xGPUstream, xGPUstream)
        cudaRawKernelFloat32 ( (blockNum, 1, 1), (blockSize, 1, 1), (xGPUstream, 0))

    for i in range(streamNum):
        i1 = (i    * totalSize) // streamNum
        i2 = (i+1) * totalSize  // streamNum

        with stream[i]:
            xCPU[i1:i2] = cupy.asnumpy(xGPUstream)

    stopEvent.record()
    cupy.cuda.Stream.null.synchronize()   ## should sync on stopEvent!!!!
    elapsedTime = cupy.cuda.get_elapsed_time(startEvent, stopEvent)

    print (" asynchronous V2 transfer and execute");
    print (F"    ... time  = {elapsedTime} ms");
    print (F"    ... error = {numpy.max(xCPU - 1.0)}\n");

    # cleanup
