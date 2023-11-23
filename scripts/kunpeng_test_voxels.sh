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

for voxels in 1000 2000 3000 4000 5000 6000 7000 8000
do
	# srun ./perf_voxels_substrates 4000 1 >> 1_node_4000_hf.log
	#ddt --connect  mpirun -n 4 ./capVoxels 4000 1> 4_node_4000_hf.log 2> 4_node_4000_hf.log
        srun ./capVoxels $voxels 1> ./voxcap/kunpeng/nodes_1/${voxels}.log 2> ./voxcap/kunpeng/nodes_1/${voxels}.log
#	ddt --connect mpirun -n 2 ./capVoxels 500
done
