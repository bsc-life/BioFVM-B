#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH -t 00:60:00
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive

#export OMP_SCHEDULE=STATIC
export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=spread
#export OMP_PLACES="{0:1}:48:1"
#export OMP_PLACES='cores(48)'
export OMP_PLACES=threads
mpiexec --map-by ppr:1:socket:pe=24  --report-bindings ./examples/tutorial1
#mpiexec --report-bindings --map-by node --bind-to none ./examples/tutorial1	
#mpiexec --map-by socket --bind-to core  --report-bindings ./examples/tutorial1
#mpiexec --map-by node --bind-to none --report-bindings ./examples/tutorial1
