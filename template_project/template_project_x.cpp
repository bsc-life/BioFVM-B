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

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* This program can be taken as a starting template for building your own */
/* parallel i.e. MPI+OpenMP BioFVM program 																*/
/* This shows the parts which will remain the same for ALL the programs 	*/
/* and also an example of what can be done in the non-fixed parts 				*/
/* We STRONGLY suggest that users follow the tutorial in the README file  */
/* and ALSO check the parallel/serial examples BEFORE they start creating */
/* their own programs.																										*/
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include "../BioFVM.h"
#include <omp.h>

/*----------------Include MPI Header------------*/
#include <mpi.h>
/*----------------------------------------------*/

using namespace BioFVM;

/*==============================================================================*/
/* Users can define global variables here (though if possible, global variables */
/* should be avoided)																														*/
/*==============================================================================*/

void supply_function( Microenvironment* microenvironment, int voxel_index, std::vector<double>* write_here )
{

/*==============================================================================*/
/* Users can see other examples to see the use of supply_function(...)					*/
/* and to understand the syntax for access of variables.												*/
/*==============================================================================*/

}

void supply_target_function( Microenvironment* microenvironment, int voxel_index, std::vector<double>* write_here )
{

/*==============================================================================*/
/* Users can see other examples to see the use of supply_target_function(...)		*/
/* and to understand the syntax for access of variables.												*/
/*==============================================================================*/

}

void uptake_function( Microenvironment* microenvironment, int voxel_index, std::vector<double>* write_here )
{

/*==============================================================================*/
/* Users can see other examples to see the use of uptake_function(...)					*/
/* and to understand the syntax for access of variables.												*/
/*==============================================================================*/

}

int main( int argc, char* argv[] )
{

/*===================================================================*/
/* The following declarations, initializations, function invocations */
/* will remain common to ALL parallel i.e. MPI+OpenMP programs			 */
/* Users can copy paste these into new programs or use this template */
/*===================================================================*/

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

/*----------------------------------------------------------*/
/* Set the number of OpenMP threads	                    		*/
/* As an example if you have 2 sockets (or processors)			*/
/* and you are running 1 MPI process per socket 						*/
/* then you should set the number of threads = maximum cores*/
/* per-socket																								*/
/*----------------------------------------------------------*/

	//omp_set_num_threads( 1 );  <---- uncomment to set threads
	
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/	
/* FIXED PART HAS ENDED i.e. the part above this in main(...) will remain the same in ALL the programs */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*========================================================*/
/* Create a microenvironment, and set units (see examples)*/
/*========================================================*/



/*==========================================================*/
/* Set up and add all the densities you plan (see examples) */
/*==========================================================*/


/*===============================================================*/
/* Set the properties of the diffusing substrates (see examples) */
/*===============================================================*/



/*================================================*/
/* Set 	the mesh size 														*/
/* functions ending with mpi_Dims, mpi_Coords are */
/* parallel equivalent of serial functions 				*/
/*================================================*/

	//double dx = 0.05;    //50 microns
	//M.resize_space_uniform( 0, 4.0, 0, 4.0, 0.0, 4.0, dx, mpi_Dims, mpi_Coords ); 

/*==============================*/
/*	Display summary information	*/
/*==============================*/

 
  //if(mpi_Rank == 0) <--- Makes sure that only a single process displays the information
	// 	M.display_information( std::cout );

/*================================*/
/* 	Set up metadata (see examples)*/
/*================================*/


/*=====================================================================*/
/* set initial conditions (this is just an example of what you can do) */
/* This will change from program to program and is certainly NOT fixed */
/* (see examples)																											 */
/*=====================================================================*/

	std::vector<double> center(3);
  center[0] = (M.mesh.bounding_box[0]+M.mesh.bounding_box[3])/2;
	center[1] = (M.mesh.bounding_box[1]+M.mesh.bounding_box[4])/2;
	center[2] = (M.mesh.bounding_box[2]+M.mesh.bounding_box[5])/2;

	double radius = 1.0;
	std::vector<double> one( M.density_vector(0).size() , 1.0 );

	double pi = 2.0 * asin( 1.0 );

/*========================================================*/
/* An example of how to loop over ALL voxels using threads*/
/*========================================================*/

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

	}

/*==================================================*/
/* Save the initial profile in a MATLAB file 				*/
/* write_to_matlab(...) below is a parallel function*/
/*==================================================*/

	M.write_to_matlab( "initial.mat", mpi_Rank, mpi_Size, mpi_Cart_comm ); 
	
/*================================================*/
/* set up the diffusion solver, sources and sinks */
/*================================================*/


	M.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D;

	M.bulk_supply_rate_function = supply_function;
	M.bulk_supply_target_densities_function = supply_target_function;
	M.bulk_uptake_rate_function = uptake_function;

/*============================================================================*/
/* Example of how to consider times - this can change from program to program */
/*============================================================================*/

	double t     = 0.0;
	double t_max = 10.0 * 24.0;
	double dt    = 0.1;

	double output_interval  = 12.0;  // how often you save data
	double next_output_time = t;     // next time you save data

/*=================================================================*/
/* The main simulation loop will most of the times looks like this */
/*=================================================================*/


	while( t < t_max )
	{
		if( fabs( t - next_output_time ) < dt/2.0 )
		{
      if(mpi_Rank == 0)
        std::cout << "simulation time: " << t << " " << M.time_units << " (" << t_max << " " << M.time_units << " max)" << std::endl;

			char* filename;
			filename = new char [1024];

			sprintf( filename, "output_%6f.mat" , next_output_time );
			M.write_to_matlab( filename, mpi_Rank, mpi_Size, mpi_Cart_comm );

			delete [] filename;
			next_output_time += output_interval;
		}

		M.simulate_bulk_sources_and_sinks( dt );
		M.simulate_diffusion_decay( dt, mpi_Size, mpi_Rank, mpi_Coords, mpi_Dims, mpi_Cart_comm );
		M.simulate_cell_sources_and_sinks( dt );

		t += dt;
	}

	M.write_to_matlab( "final.mat", mpi_Rank, mpi_Size, mpi_Cart_comm); 

/*==================================================================*/
/* The next 2 statements are a must in EVERY program 								*/
/* i.e. (1) correctly close MPI (2) return 0 to calling environment */
/*==================================================================*/

  MPI_Finalize();
	return 0;
}
