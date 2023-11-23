#!/bin/bash
#SBATCH -n 48
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=1
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --x11=batch

# set application and parameters
emacs
