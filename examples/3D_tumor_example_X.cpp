/*
#############################################################################
# If you use BioFVM in your project, please cite BioFVM and the version     #
# number, such as below:                                                    #
#                                                                           #
# We solved the diffusion equations using BioFVM (Version 1.1.6) [1]        #
#                                                                           #
# [1] A. Ghaffarizadeh, S.H. Friedman, and P. Macklin, BioFVM: an efficient #
#    parallelized diffusive transport solver for 3-D biological simulations,#
#    Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730 #
#                                                                           #
#############################################################################
#                                                                           #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)   #
#                                                                           #
# Copyright (c) 2015-2017, Paul Macklin and the BioFVM Project              #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions are    #
# met:                                                                      #
#                                                                           #
# 1. Redistributions of source code must retain the above copyright notice, #
# this list of conditions and the following disclaimer.                     #
#                                                                           #
# 2. Redistributions in binary form must reproduce the above copyright      #
# notice, this list of conditions and the following disclaimer in the       #
# documentation and/or other materials provided with the distribution.      #
#                                                                           #
# 3. Neither the name of the copyright holder nor the names of its          #
# contributors may be used to endorse or promote products derived from this #
# software without specific prior written permission.                       #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED #
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           #
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER #
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       #
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#                                                                           #
#############################################################################
*/

#include "../BioFVM.h"
#include <omp.h>

/*----------------Include MPI Header------------*/
#include <mpi.h>
/*----------------------------------------------*/

using namespace BioFVM;

int live_cells  = 0;
int blood_vessels = 1;
int oxygen    = 2;

// some globals
double prolif_rate = 1.0 /24.0;
double death_rate = 1.0 / 6; //
double cell_motility = 50.0 / 365.25 / 24.0 ;
// 50 mm^2 / year --> mm^2 / hour
double o2_uptake_rate = 3.673 * 60.0; // 165 micron length scale
double vessel_degradation_rate = 1.0 / 2.0 / 24.0 ;
// 2 days to disrupt tissue

double max_cell_density = 1.0;

double o2_supply_rate = 10.0;
double o2_normoxic  = 1.0;
double o2_hypoxic   = 0.2;

void supply_function( Microenvironment* microenvironment, int voxel_index, std::vector<double>* write_here )
{
	// use this syntax to access the jth substrate write_here
	// (*write_here)[j]
	// use this syntax to access the jth substrate in voxel voxel_index of microenvironment:
	// microenvironment->density_vector(voxel_index)[j]

		static double temp1 = prolif_rate / (o2_normoxic - o2_hypoxic);

     (*write_here)[live_cells] =
          microenvironment->density_vector(voxel_index)[oxygen];
     (*write_here)[live_cells] -= o2_hypoxic;

     if( (*write_here)[live_cells] < 0.0 )
     {
          (*write_here)[live_cells] = 0.0;
     }
     else
     {
          (*write_here)[live_cells] = temp1;
          (*write_here)[live_cells] *=
               microenvironment->density_vector(voxel_index)[live_cells];
     }

     (*write_here)[blood_vessels] = 0.0;
     (*write_here)[oxygen] = o2_supply_rate;
     (*write_here)[oxygen] *=
          microenvironment->density_vector(voxel_index)[blood_vessels];

	return;
}

void supply_target_function( Microenvironment* microenvironment, int voxel_index, std::vector<double>* write_here )
{
	// use this syntax to access the jth substrate write_here
	// (*write_here)[j]
	// use this syntax to access the jth substrate in voxel voxel_index of microenvironment:
	// microenvironment->density_vector(voxel_index)[j]

	(*write_here)[live_cells] = max_cell_density;
  (*write_here)[blood_vessels] =  1.0;
  (*write_here)[oxygen] = o2_normoxic;

	return;
}

void uptake_function( Microenvironment* microenvironment, int voxel_index, std::vector<double>* write_here )
{
	// use this syntax to access the jth substrate write_here
	// (*write_here)[j]
	// use this syntax to access the jth substrate in voxel voxel_index of microenvironment:
	// microenvironment->density_vector(voxel_index)[j]

	(*write_here)[live_cells] = o2_hypoxic;
     (*write_here)[live_cells] -=
          microenvironment->density_vector(voxel_index)[oxygen];
     if( (*write_here)[live_cells] < 0.0 )
     {
          (*write_here)[live_cells] = 0.0;
     }
     else
     {
          (*write_here)[live_cells] *= death_rate;
     }

     (*write_here)[oxygen] = o2_uptake_rate ;
     (*write_here)[oxygen] *=
          microenvironment->density_vector(voxel_index)[live_cells];

     (*write_here)[blood_vessels] = vessel_degradation_rate ;
     (*write_here)[blood_vessels] *=
          microenvironment->density_vector(voxel_index)[live_cells];

	return;
}

int main( int argc, char* argv[] )
{

  /*--------------------------------------------*/
	/*		Initialize MPI		                      */
	/*--------------------------------------------*/

	unsigned int mpi_Error;
	int mpi_Requested_level = MPI_THREAD_FUNNELED;
	int mpi_Provided_level;
	MPI_Comm mpi_Comm = MPI_COMM_WORLD;
	MPI_Comm mpi_Cart_comm;
	int mpi_Size, mpi_Rank;
	int mpi_Dims[3], mpi_Is_periodic[3], mpi_Coords[3];
	int mpi_Reorder;


  mpi_Error = MPI_Init_thread(NULL, NULL, mpi_Requested_level, &mpi_Provided_level);
	MPI_Barrier(mpi_Comm);


	if(mpi_Error != MPI_SUCCESS)
	  MPI_Abort(mpi_Comm,-1);

	MPI_Comm_size(mpi_Comm, &mpi_Size);
	MPI_Comm_rank(mpi_Comm, &mpi_Rank);

	if(mpi_Provided_level != mpi_Requested_level && mpi_Rank == 0)
	{
	  std::cout<<"The MPI Requested Level is not available, please lower the level and try again"<<std::endl;
	  MPI_Abort(mpi_Comm, -1);
	}

  /*------------------------------------------------------------*/
	/*			MPI Comm Size and Rank                                */
  /*      ONLY 1-D X decomposition supported                    */
	/*------------------------------------------------------------*/

	mpi_Dims[0] = 1;
	mpi_Dims[1] = mpi_Size;                                          //Number of Y processes, since Y-processes divide X-axis, we have X-decomposition
	mpi_Dims[2] = 1;

	mpi_Is_periodic[0]=0;
	mpi_Is_periodic[1]=0;
	mpi_Is_periodic[2]=0;
	mpi_Reorder = 0;

  /*------------------------------------------------------------*/
	/*			Create MPI topology, find coords                      */
	/*------------------------------------------------------------*/

	MPI_Cart_create(mpi_Comm, 3, mpi_Dims, mpi_Is_periodic, mpi_Reorder, &mpi_Cart_comm);
	MPI_Cart_coords(mpi_Cart_comm, mpi_Rank, 3, mpi_Coords);

	/*--------------------------------------------*/
	/*		OpenMP threads set	                    */
	/*--------------------------------------------*/

	//omp_set_num_threads( 1 );                          //To test just one thread

	// create a microenvironment, and set units

	Microenvironment M;
	M.name = "Tumor microenvironment";
	M.time_units = "hr";
	M.spatial_units = "mm";
	M.mesh.units = M.spatial_units;

	// set up and add all the densities you plan

	M.set_density( 0 , "live cells" , "cells" );
	M.add_density( "blood vessels" , "vessels/mm^2" );
	M.add_density( "oxygen" , "cells" );

	// set the properties of the diffusing substrates

  M.diffusion_coefficients[live_cells] = cell_motility;
  M.diffusion_coefficients[blood_vessels] = 0;
  M.diffusion_coefficients[oxygen] = 6.0;

  //1e5 microns^2/min in units mm^2 / hr

  M.decay_rates[live_cells] = 0;
  M.decay_rates[blood_vessels] = 0;
  M.decay_rates[oxygen] = 0.01 * o2_uptake_rate;

	//set the mesh size

	double dx = 0.05;    //50 microns
	M.resize_space_uniform( 0, 4.0, 0, 4.0, 0.0, 4.0, dx, mpi_Dims, mpi_Coords );

	// display summary information
  if(mpi_Rank == 0)
	 M.display_information( std::cout );

	// set up metadata

	BioFVM_metadata.program.user.surname = "Kirk";
	BioFVM_metadata.program.user.given_names = "James T.";
	BioFVM_metadata.program.user.email = "Jimmy.Kirk@starfleet.mil";
	BioFVM_metadata.program.user.organization = "Starfleet";
	BioFVM_metadata.program.user.department = "U.S.S. Enterprise (NCC 1701)";

	BioFVM_metadata.program.creator.surname = "Roykirk";
	BioFVM_metadata.program.creator.given_names = "Jackson";
	BioFVM_metadata.program.creator.organization = "Yoyodyne Corporation";

	BioFVM_metadata.program.program_name = "Nomad";
	BioFVM_metadata.program.program_version = "MK-15c";
	BioFVM_metadata.program.program_URL = "";

	// set initial conditions

	std::vector<double> center(3);
  center[0] = (M.mesh.bounding_box[0]+M.mesh.bounding_box[3])/2;
	center[1] = (M.mesh.bounding_box[1]+M.mesh.bounding_box[4])/2;
	center[2] = (M.mesh.bounding_box[2]+M.mesh.bounding_box[5])/2;

	double radius = 1.0;
	std::vector<double> one( M.density_vector(0).size() , 1.0 );

	double pi = 2.0 * asin( 1.0 );

	// use this syntax to create a zero vector of length 3
	// std::vector<double> zero(3,0.0);

	// use this syntax for a parallelized loop over all the
	// voxels in your mesh:
	#pragma omp parallel for
	for( int i=0 ; i < M.number_of_voxels() ; i++ )
	{

		std::vector<double> displacement = M.voxels(i).center - center;
     double distance = norm( displacement );

     if( distance < radius )
     {
          M.density_vector(i)[live_cells] = 0.1;
     }
     M.density_vector(i)[blood_vessels]= 0.5
                                        + 0.5*cos(0.4* pi * M.voxels(i).center[0])*cos(0.3*pi *M.voxels(i).center[1])*cos(0.2*pi *M.voxels(i).center[2]);
     M.density_vector(i)[oxygen] = o2_normoxic;

		// use this syntax to access the coordinates (as a vector) of
		// the ith voxel;
		// M.mesh.voxels[i].center

		// use this access the jth substrate at the ith voxel
		// M.density_vector(i)[j]

	}

	// save the initial profile

	M.write_to_matlab( "initial.mat", mpi_Rank, mpi_Size, mpi_Cart_comm ); // barebones
	//save_BioFVM_to_MultiCellDS_xml_pugi( "initial" , M , 0.0 ); // MultiCellDS digital snapshot

	// set up the diffusion solver, sources and sinks

	M.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D;

	M.bulk_supply_rate_function = supply_function;
	M.bulk_supply_target_densities_function = supply_target_function;
	M.bulk_uptake_rate_function = uptake_function;

	double t     = 0.0;
	double t_max = 10.0 * 24.0;
	double dt    = 0.1;

	double output_interval  = 12.0;  // how often you save data
	double next_output_time = t;     // next time you save data

	while( t < t_max )
	{
		// if it's time, save the simulation
		if( fabs( t - next_output_time ) < dt/2.0 )
		{
      if(mpi_Rank == 0)
        std::cout << "simulation time: " << t << " " << M.time_units << " (" << t_max << " " << M.time_units << " max)" << std::endl;

			char* filename;
			filename = new char [1024];

			sprintf( filename, "output_%6f.mat" , next_output_time );
			M.write_to_matlab( filename, mpi_Rank, mpi_Size, mpi_Cart_comm );

			//sprintf( filename, "output_%6f" , next_output_time );
			//save_BioFVM_to_MultiCellDS_xml_pugi( filename , M , 0.0 ); // MultiCellDS digital snapshot

			delete [] filename;
			next_output_time += output_interval;
		}

		M.simulate_bulk_sources_and_sinks( dt );
		M.simulate_diffusion_decay( dt, mpi_Size, mpi_Rank, mpi_Coords, mpi_Dims, mpi_Cart_comm );
		M.simulate_cell_sources_and_sinks( dt );

		t += dt;
	}

	M.write_to_matlab( "final.mat", mpi_Rank, mpi_Size, mpi_Cart_comm); // barebones
	//save_BioFVM_to_MultiCellDS_xml_pugi( "final" , M , 0.0 ); // MultiCellDS digital snapshot

  if(mpi_Rank == 0)
	std::cout << std::endl << "Done!" << std::endl;

  MPI_Finalize();
	return 0;
}
