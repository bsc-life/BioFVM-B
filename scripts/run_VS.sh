#!/bin/bash
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --qos=class_a
#SBATCH -t 02:00:00
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive


export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

module purge
module load openmp gcc/7.2.0 openmpi/3.1.1 # gcc/8.1.0 openmpi/3.1.1  #openmpi/2.1.3  #openmpi/3.1.4 #openmpi/2.1.3
#openmpi/4.0.1
module load DDT/18.2

for voxels in 8000
do
    for substrates in  4 
    do
	# srun ./perf_voxels_substrates 4000 1 >> 1_node_4000_hf.log
	#ddt --connect  mpirun -n 4 ./capVoxels 4000 1> 4_node_4000_hf.log 2> 4_node_4000_hf.log
	#mpirun -n 4  ./capVoxels 2000 20 # 1> test.log 2> test.log
	#ddt --connect mpirun -n 4 ./capVoxels 100 33
	#ddt --connect mpirun -n 4  ./capVoxels	100 33
	#ddt --connect mpirun -n 4 ./capVoxels 150 3
	#ddt --connect mpirun -n 4 ./capVoxels 500 33
	#mpirun -n 4 ./capVoxels 100 33
	srun ./test_VS 3000 128 2 ./test.csv
    done	
done
