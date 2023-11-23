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
module load openmp gcc/8.1.0 openmpi/4.0.1

for voxels in 500 1000 1500 2000 2500 3000 
do
    for substrates in 1 4 8 12 16 24 32 48 64 96 128
    do
    	srun ./perf_voxels_substrates $voxels $substrates 1> timing/vs_parallel/nodes_4/voxels_${voxels}_subs_${substrates} 2> timing/vs_parallel/nodes_4/voxels_${voxels}_subs_${substrates}
    done	
done
