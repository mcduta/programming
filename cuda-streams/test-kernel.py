"""

  Simple CUDA kernel test that times one kernel execution:
  * overall timing, inc. memory transfers and
  * kernel timing only.

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

    # sizes
    totalSize = totalBytes // sizeofREAL

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
    startEvent       = cupy.cuda.Event()
    startEventAllOps = cupy.cuda.Event()
    stopEvent        = cupy.cuda.Event()
    stopEventAllOps  = cupy.cuda.Event()


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

    cudaKernelFloat32 = cupy.RawKernel (
        r'''
        extern "C" __global__ void cudaKernelFloat32 (float *x, float *y, const size_t N)
        {
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < N) {
          float z =  fabs (x[i]);
          z = z / (1.0 + z);
          float s = sin (z);
          float c = cos (z);
          y[i] = sqrt (s*s+c*c);
        }
        }
        ''', 'cudaKernelFloat32')



    #
    # ===== baseline case - sequential transfer and execute
    #
    #
    xCPU = numpy.random.randn(totalSize).astype(dataType)
    xGPU = cupy.asarray(xCPU)

    """
    # warm-up
    yGPU = cupy.asarray(xCPU)
#   cudaKernel (xGPU, yGPU)
    cudaKernelFloat32 ( (int(math.ceil(totalSize/1024)), 1, 1), (1024, 1, 1), (xGPU, yGPU))
    xGPU = cupy.asarray(xCPU)
    """


    startEventAllOps.record()
    yGPU = cupy.asarray(xCPU)
    startEvent.record()
#   cudaKernel (xGPU, yGPU)
    cudaKernelFloat32 ( (int(math.ceil(totalSize/1024)), 1, 1), (1024, 1, 1), (xGPU, yGPU, totalSize))
    stopEvent.record()
    stopEvent.synchronize()
    yCPU = cupy.asnumpy(yGPU)
    stopEventAllOps.record()
    stopEventAllOps.synchronize()
    elapsedTime       = cupy.cuda.get_elapsed_time(startEvent,       stopEvent)
    elapsedTimeAllOps = cupy.cuda.get_elapsed_time(startEventAllOps, stopEventAllOps)
    print (" timing transfer and execute");
    print (F"    ... time (overall) = {elapsedTimeAllOps} ms");
    print (F"    ... time (kernel)  = {elapsedTime} ms");
    print (F"    ... error          = {numpy.max(yCPU - 1.0)}\n");

