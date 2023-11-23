#include "../BioFVM_microenvironment.h"

using namespace BioFVM; 

void create_point_sources(double cell_radius, double dt, int num_sources, Microenvironment & microenvironment, int *mpi_Dims, int mpi_Rank, int mpi_Size, MPI_Comm mpi_Cart_comm);
void create_point_sinks(double cell_radius, double dt, int num_sinks, Microenvironment &microenvironment, int *mpi_Dims, int mpi_Rank, int mpi_Size, MPI_Comm mpi_Cart_comm);
