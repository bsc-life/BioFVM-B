#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=1
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --x11=batch
#SBATCH --exclusive

# set application and parameters
#module load octave
octave --force-gui
