#!/bin/bash
#SBATCH -n 48
#SBATCH -t 00:15:00
#SBATCH --cpus-per-task=1
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --x11=batch

# set application and parameters

paraprof
#jumpshot tau.slog2
