#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=normal
#SBATCH -t 02:00:00
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive


export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

module purge

#openmpi/4.0.1
module load gcc/11.2.0 openmpi/4.1.1
#module load DDT

for voxels in 8000
do
    for substrates in  4 
    do
	# srun ./perf_voxels_substrates 4000 1 >> 1_node_4000_hf.log
	#ddt --connect  mpirun -n 4 ./capVoxels 4000 1> 4_node_4000_hf.log 2> 4_node_4000_hf.log
        srun ./capVoxels 1000 1> test.log 2> test.log
#	ddt --connect mpirun -n 2 ./capVoxels 500
    done	
done
