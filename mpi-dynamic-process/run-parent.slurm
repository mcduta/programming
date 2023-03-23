#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:05:00
#SBATCH --job-name=test-mpi-spawn
#SBATCH --partition=cs05r


# modules
module purge
module load intel/mpi/2022

# run
hostlist=$(scontrol show hostnames | tr "\n" "," | sed "s/.$//")
I_MPI_SPAWN=on mpirun -np 8 --ppn 4 -hosts ${hostlist} ./parent_intel2022 ./child_intel2022 2
