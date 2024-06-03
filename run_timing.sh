#!/bin/bash
#SBATCH --nodes=1
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


for voxels in  1000 1500 2000 2500 3000 4000 5000
do
	srun --cpus-per-task=56 ./dirichlet_test $voxels 1> ./dirichlet/${voxels}_56_th.log 2>  ./dirichlet/${voxels}_56_th.log
	srun --cpus-per-task=112 ./dirichlet_test $voxels 1> ./dirichlet/${voxels}_112_th.log 2>  ./dirichlet/${voxels}_112_th.log
done
