#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=112
#SBATCH --qos=gp_debug
#SBATCH -t 02:00:00
#SBATCH --account=bsc08
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive

export OMP_DISPLAY_ENV=true
#export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

module purge
module load gcc/13.2.0 openmpi/4.1.5-gcc ddt

srun --nodes=4  --ntasks-per-node=1 --cpus-per-task=112 ./examples/tutorial1 $voxels 1> ./dirichlet/${voxels}_112_th.log 2>  ./dirichlet/${voxels}_112_th.log
