#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
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
module load openmp gcc/8.1.0 openmpi/3.1.1  #openmpi/2.1.3  #openmpi/3.1.4 #openmpi/2.1.3
#openmpi/4.0.1
module load DDT

for voxels in 1000 1500 2000 2500 3000 3500 4000 
#10000 12500 15000 17500 20000 22500 25000 30000 32500 35000 40000 42500 45000 47500 50000
do
	for steps in 4 8 16 32 64 128 256 512 1024
	do
		# srun ./perf_voxels_substrates 4000 1 >> 1_node_4000_hf.log
		srun ./capVoxels $voxels $steps 1> ./timing/steps/nodes_4/steps_${steps}/${voxels}_complete 2> ./timing/steps/nodes_4/steps_${steps}/${voxels}_complete
	done
done
