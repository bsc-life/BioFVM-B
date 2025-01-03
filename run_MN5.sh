#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=112
#SBATCH --qos=gp_debug
#SBATCH -t 02:00:00
#SBATCH --account=bsc08
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive


export OMP_DISPLAY_ENV=true
#export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores

module purge
module load gcc/13.2.0 openmpi/4.1.5-gcc ddt


#test_VS usage:  <program_name> <domain_half_side> <number_of_blocks> <num_substrates> [csv_name] 
srun --nodes=4 --ntasks-per-node=1 --cpus-per-task=112 ./test_VS 2000 4 2 vector.csv

