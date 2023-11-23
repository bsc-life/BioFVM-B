#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=1
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --x11=batch
#SBATCH --exclusive

# set application and parameters
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
#ddt mpiexec --map-by socket --bind-to socket ./examples/tutorial1
ddt --connect srun ./examples/tutorial1
