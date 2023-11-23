#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH -t 00:60:00
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive


export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

 mpiexec --map-by ppr:1:socket:pe=24  --report-bindings ./examples/perfVoxels 100
