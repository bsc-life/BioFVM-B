#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --qos=debug
#SBATCH -t 02:00:00
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive
#SBATCH --constraint=perfparanoid 

export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

module purge
module load openmp gcc/8.1.0 openmpi/4.0.1 #openmpi/3.1.4 #openmpi/2.1.3
#openmpi/4.0.1
module load DDT

for voxels in 8000
do
    for substrates in  4 
    do
	#srun gprof ./capVoxels 2000 
	srun perf record -e cycles mpirun -n 4  ./capVoxels 2000 
    done	
done
