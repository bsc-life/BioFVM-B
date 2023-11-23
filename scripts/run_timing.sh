#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=48
#SBATCH --qos=debug
#SBATCH -t 02:00:00
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive


export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

module purge
module load openmp gcc/7.2.0 openmpi/3.1.1 #openmpi/3.1.4 #openmpi/2.1.3
#openmpi/4.0.1
#module load DDT

for voxels in  1000 1500 2000 2500 3000
do
	for threads in 48 32 16 8 4 2
	do
		srun --cpus-per-task=$threads ./capVoxels $voxels 1> ./timing/kunpeng/2_nodes/${voxels}_${threads}_th 2>  ./timing/kunpeng/2_nodes/${voxels}_${threads}_th
	done
done
