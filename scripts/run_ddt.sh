#!/bin/bash
#SBATCH --nodes=2
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
module load openmp gcc/8.1.0 openmpi/3.1.1
#openmpi/2.1.3
#openmpi/4.0.1
module load DDT

for voxels in 500
do
    for substrates in  4 
    do
	ddt --connect mpiexec -n 2 ./perf_voxels_substrates 500 4
    done	
done
