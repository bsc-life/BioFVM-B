#!/bin/bash
#SBATCH --nodes=1
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

for voxels in 3000 5000 7500 10000 15000 20000
do
    srun ./perfVoxels $voxels >> voxcap/nodes_1/step_$voxels
done
