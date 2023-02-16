# ======================================================================
#
# ----- GPU side processing
#
# ======================================================================

def mpi_data_process (mpi_proc_num, mpi_proc_id, size_data, dev_get=False, dev_set=False):

    # imports
    import cupy

    # device numbers
    if dev_set:
        dev_num = cupy.cuda.runtime.getDeviceCount ()
        cupy.cuda.runtime.setDevice (mpi_proc_id % dev_num)

    dev_id   = cupy.cuda.runtime.getDevice ()
    dev_prop = cupy.cuda.runtime.getDeviceProperties (dev_id)

    if dev_get:
        import platform
        hostname = platform.node().split(".")[0]
        print (F" {hostname}: rank {mpi_proc_id} : device {dev_id}")

    # total GPU memory available
    size_avail = dev_prop["totalGlobalMem"]
    # accommodate everything in memory (1 array per MPI process, 8 byte reals)
    size_avail = size_avail // (8 * mpi_proc_num)
    # how many data_size in available size
    size_data = (size_avail // size_data) * size_data

    # allocate data on GPU memory and initialise
    data_gpu = cupy.arange (size_data).astype(cupy.float32) % 10.0 + 1.0
    # data_gpu = cupy.random.rand (data_size)

    # run device kernel
    data_gpu = cupy.sqrt (data_gpu)



# ======================================================================
#
# ----- main (CPU side processing)
#
# ======================================================================
#

if __name__ == "__main__":

    # imports
    from mpi4py import MPI

    # MPI process
    mpi_proc_num = MPI.COMM_WORLD.size
    mpi_proc_id  = MPI.COMM_WORLD.rank

    # process arguments
    import argparse
    parser = argparse.ArgumentParser (description="GPU device use from mpi4py and cupy")
    parser.add_argument ("-g", "--get_device", action="store_true", help="report device for each MPI process (default: NO)")
    parser.add_argument ("-s", "--set_device", action="store_true", help="automatically set device for each MPI process (default: NO)")
    args = parser.parse_args ()

    # data set dimensions
    size_block = 256
    size_grid  = 10000

    # call GPU-side processing function
    mpi_data_process (mpi_proc_num, mpi_proc_id, size_block * size_grid, args.get_device, args.set_device)
