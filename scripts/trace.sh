#!/bin/bash

source /apps/BSCTOOLS/extrae/4.0.4/openmpi_3_1_1/etc/extrae.sh
#export OMP_NUM_THREADS=4
export EXTRAE_CONFIG_FILE=extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitrace.so # For C apps
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitracef.so # For Fortran apps

## Run the desired program
$*

