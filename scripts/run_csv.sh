#!/bin/bash
#SBATCH --nodes=32
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

export nodes=32

for voxels in 1000 1500 2000 2500 3000 4000 5000 6000 7000 8000 9000 10000
do
	for substrates in  1 2 4 8 12 16 24 32
	do
		for factor in 1 2 4 6 8 12 16 32 64
		do
			overlap=$((nodes * factor))
			echo "Executing HF: ${voxels} | Substrates: ${substrates} | Steps: ${overlap}"
			srun ./test_VS $voxels  $overlap  $substrates ./overlap2/voxels_${voxels}/substrates_${substrates}/factor_${factor}/${nodes}_node.csv 1> /dev/null 2> /dev/null
		done
	done	
done
