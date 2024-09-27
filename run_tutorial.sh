#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --qos=gp_debug
#SBATCH -t 02:00:00
#SBATCH --account=bsc08
#SBATCH -o resize_big.log
#SBATCH -e resize_big.log
#SBATCH --exclusive

export OMP_DISPLAY_ENV=false
#export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

module purge
module load gcc/13.2.0 openmpi/4.1.5-gcc ddt

for i in 1
do
	echo "Nodes is $i"
	#mpirun -bootstrap slurm  valgrind --tool=massif --pages-as-heap=yes ./examples/tutorial1 
	srun --nodes=${i} --ntasks-per-node=1 --cpus-per-task=112 valgrind ./examples/tutorial1 # valgrind --tool=massif --pages-as-heap=yes ./examples/tutorial1 
done
#ddt --connect mpirun -n 8 ./examples/tutorial1 

