#!/bin/bash
#SBATCH --nodes=160
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --qos=class_a
#SBATCH -t 3-00:00:00
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive


export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=threads

module purge
module load openmp gcc/7.2.0 openmpi/3.1.1 # gcc/8.1.0 openmpi/3.1.1  #openmpi/2.1.3  #openmpi/3.1.4 #openmpi/2.1.3
#openmpi/4.0.1
module load DDT/18.2

export nodes=160

#overlap/voxels_{15000,20000,25000,30000,40000,100000}/factor_{1,2,4,8,16,32,64,128}

for voxels in 60000 50000 40000 30000 25000 20000 15000 10000
do
	for substrates in  1 2 4 8
	do
		for factor in 1 2 4 8 16 32 64 128
		do
			overlap=$((nodes * factor))
			echo "Executing HF: ${voxels} | Substrates: ${substrates} | Steps: ${overlap}"
			srun ./test_VS $voxels  $overlap  $substrates ./overlap/voxels_${voxels}/substrates_${substrates}/factor_${factor}/${nodes}_node.csv 1> ./overlap/voxels_${voxels}/substrates_${substrates}/factor_${factor}/${nodes}_node.log  2> ./overlap/voxels_${voxels}/substrates_${substrates}/factor_${factor}/${nodes}_node.log 
		done
	done	
done
