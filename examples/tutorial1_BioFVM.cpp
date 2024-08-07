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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <random>


/*----------------Include MPI Header-------------*/
#include <mpi.h>
/*----------------------------------------------*/

#include "../BioFVM.h"
#include "tutorial1_BioFVM.h"

using namespace BioFVM;

//int omp_num_threads = 48; // set number of threads for parallel computing, set this to # of CPU cores x 2 (for hyperthreading)


#define N 5000
double pi= 3.1415926535897932384626433832795;

double UniformRandom()
{
	return ((double) rand() / (RAND_MAX));
}


void create_point_sources(double cell_radius, double dt, int num_sources, Microenvironment &microenvironment, int *mpi_Dims, int mpi_Rank, int mpi_Size, MPI_Comm mpi_Cart_comm)
{
        std::vector<double> tempPoint(3,0.0);                       //Create three random points i.e. x,y and z

        std::vector<std::vector<double>> vector_coords(mpi_Size);   //vector_coords[r] contains list of coordinates to be sent to rank 'r'
        std::vector<std::vector<int>> vector_IDs(mpi_Size);    		//vector_IDs[r] contains IDs of Basic_Agent IDs to be sent to rank 'r' (IDs are unique over domain)
        std::vector<int> sources_per_proc_at_root(mpi_Size);

        int proc_x_coord, proc_y_coord, proc_z_coord;			    //X,Y,Z coordinates of processes
        int proc_index_rank;
        int unique_ID = 0;                                          //The Basic_Agent will have position tempPoint[] and a global ID

        std::vector<MPI_Request> send_req1(mpi_Size);
        std::vector<MPI_Request> send_req2(mpi_Size);
        MPI_Request recv_req1, recv_req2;

        int sources_per_proc;								       //Processes receives no. of IDs for itself in this variable
        std::vector<double> list_of_coords;                        //3 * sources_per_proc for each process
        std::vector<int> list_of_IDs;  			                   //1 * sources_per_proc for each process

        /*----------------------------------------------------------------------------------------*/
        /* First find total voxels in each direction: globally and locally                        */
        /*----------------------------------------------------------------------------------------*/

        int x_voxels = (microenvironment.mesh.bounding_box[3] - microenvironment.mesh.bounding_box[0])/microenvironment.mesh.dx;
        int y_voxels = (microenvironment.mesh.bounding_box[4] - microenvironment.mesh.bounding_box[1])/microenvironment.mesh.dy;
        int z_voxels = (microenvironment.mesh.bounding_box[5] - microenvironment.mesh.bounding_box[2])/microenvironment.mesh.dz;

        int local_x_voxels = x_voxels/mpi_Dims[1];                 //Remember X and Y are different for Domain and MPI Topology
        int local_y_voxels = y_voxels/mpi_Dims[0];
        int local_z_voxels = z_voxels/mpi_Dims[2];

        int dx = microenvironment.mesh.dx;
        int dy = microenvironment.mesh.dy;
        int dz = microenvironment.mesh.dz;

        if(mpi_Rank == 0)
        {

            for(int i=0; i<num_sources; i++)
            {

                /*------------------------------------------------------------------------------------------------*/
                /* Generate 3 random numbers between [0,1] and multiply by Range to get random position in domain */
                /* i.e. Multiply by x_max - x_min OR y_max - y_min OR z_max-z_min => in general the Range         */
                /*------------------------------------------------------------------------------------------------*/

                for(int j=0; j < 3 ; j++ )
                    tempPoint[j] = UniformRandom()*N;

                /*------------------------------------------------------------------------------------------------*/
                /* Find the MPI process coordinate using local_x/y/z_voxels                                       */
                /* Remember BioFVM X direction is left to right                                                   */
                /* MPI Process X direction is top to bottom                                                       */
                /* Use the floor() function, as type promotion was creating a problem in proc_x_coord expression  */
                /*------------------------------------------------------------------------------------------------*/

                /*------------------------------------------------------------------------------------------------------------------------------------------------------*/
                /* When the starting point of domain is not zero i.e. minX != 0, minY != 0 and minZ != 0, then we should subtract these starting points from tempPoints */
                /* otherwise the strategy below fails                                                                                                                   */
                /*------------------------------------------------------------------------------------------------------------------------------------------------------*/

                tempPoint[0] = tempPoint[0] - microenvironment.mesh.bounding_box[0];
                tempPoint[1] = tempPoint[1] - microenvironment.mesh.bounding_box[1];
                tempPoint[2] = tempPoint[2] - microenvironment.mesh.bounding_box[2];

                proc_y_coord = tempPoint[0]/(local_x_voxels * dx);                             //Can use floor() here but no need
                proc_x_coord = (mpi_Dims[0] - 1) - floor(tempPoint[1]/(local_y_voxels * dy));  //Automatic promotion to double was creating problems
                proc_z_coord = tempPoint[2]/(local_z_voxels * dz);                             //Can use floor() here but no need
                proc_index_rank = proc_x_coord * mpi_Dims[1] * mpi_Dims[2] + proc_y_coord * mpi_Dims[2] + proc_z_coord;

                vector_coords[proc_index_rank].push_back(tempPoint[0]);          //Create a list of coordinates for a particular rank
                vector_coords[proc_index_rank].push_back(tempPoint[1]);          //i.e. for some rank 'r', vector_coords[r] = tempPoint[0]-->temPoint[1]-->temPoint[2]
                vector_coords[proc_index_rank].push_back(tempPoint[2]);
                vector_IDs[proc_index_rank].push_back(unique_ID);                //Similarly list of unique IDs for that particular rank

                unique_ID++;
            }

            for(int k=0 ; k<vector_IDs.size(); k++)                                   //Need to know if there is a rank which has got no points
                sources_per_proc_at_root[k] = vector_IDs[k].size();                   //sources_per_proc[k] contains no. of points going to kth rank

            for(int k=1 ; k<mpi_Size; k++)
            {
                MPI_Isend(&sources_per_proc_at_root[k], 1, MPI_INT, k, 1111, mpi_Cart_comm, &send_req2[k]); //Every process needs to know how many points it will receive
                MPI_Wait(&send_req2[k], MPI_STATUS_IGNORE);

                if(sources_per_proc_at_root[k] != 0)                                                        //Send to only those processes which have got at least one point
                {
                   /*--------------------------------------------------------------------------------------------*/
                   /* Remember vector_coords[k] is the base address of the vectors of coordinates for rank k     */
                   /* sending &vector_coords[k] is incorrect                                                     */
                   /*--------------------------------------------------------------------------------------------*/


                   MPI_Isend(&vector_coords[k][0], 3 * sources_per_proc_at_root[k], MPI_DOUBLE, k, 2222, mpi_Cart_comm, &send_req1[k]);
                   MPI_Wait(&send_req1[k], MPI_STATUS_IGNORE);

                   MPI_Isend(&vector_IDs[k][0], sources_per_proc_at_root[k], MPI_INT, k, 3333, mpi_Cart_comm, &send_req2[k]);
                   MPI_Wait(&send_req2[k], MPI_STATUS_IGNORE);

                }
            }

                /*------------------------------------------------------------------------------------------------*/
                /* Since the send above was not for the root, we need to copy parameters at root                  */
                /*------------------------------------------------------------------------------------------------*/

            if(sources_per_proc_at_root[0] != 0)
            {
               sources_per_proc = sources_per_proc_at_root[0];

               list_of_coords.resize(3 * sources_per_proc, 0.0);
               list_of_IDs.resize(sources_per_proc, 0);

               for(int k=0; k<vector_coords[0].size(); k++)
                   list_of_coords[k] = vector_coords[0][k];

               for(int k=0; k<sources_per_proc; k++)
                   list_of_IDs[k] = vector_IDs[0][k];

            }
        }
        else    //Non-root processes code
        {

                /*------------------------------------------------------------------------------------------------*/
                /* For non-root processes, first receive no. of points that process gets (>=0)                    */
                /* if num points = 0 then nothing to be done                                                      */
                /* If num points > 0 then store all (1) coordinates and (2) IDs                                           */
                /*------------------------------------------------------------------------------------------------*/

          MPI_Irecv(&sources_per_proc, 1, MPI_INT, 0, 1111, mpi_Cart_comm, &recv_req2);
          MPI_Wait(&recv_req2, MPI_STATUS_IGNORE);

          if(sources_per_proc != 0)
          {
              list_of_coords.resize(3 * sources_per_proc,0.0);
              list_of_IDs.resize(sources_per_proc,0);


              MPI_Irecv(&list_of_coords[0], 3 * sources_per_proc, MPI_DOUBLE, 0, 2222, mpi_Cart_comm, &recv_req1);
              MPI_Wait(&recv_req1, MPI_STATUS_IGNORE);

              MPI_Irecv(&list_of_IDs[0], sources_per_proc, MPI_INT, 0, 3333, mpi_Cart_comm, &recv_req2);
              MPI_Wait(&recv_req2, MPI_STATUS_IGNORE);
          }

        }
                /*------------------------------------------------------------------------------------------------*/
                /* All processes now allocate Basic_Agents separately                                             */
                /* Remember ID is now being sent by the root so it must be written here as the constructor of     */
                /* this class is also generating an ID, and we do not want the constructor ID                     */
                /*------------------------------------------------------------------------------------------------*/


                for(int k=0 ; k<sources_per_proc; k++)
                {
                    Basic_Agent *temp_point_source = create_basic_agent();

                    temp_point_source->register_microenvironment(&microenvironment);

                    for(int i=0 ; i<3 ; i++)
                        tempPoint[i] = list_of_coords[3*k + i];
                    temp_point_source->assign_position(tempPoint, mpi_Rank, mpi_Dims);
                    temp_point_source->ID = list_of_IDs[k];         //Added this statement to overwrite Constructor generated ID.

                    temp_point_source->set_total_volume( (4.0/3.0)*pi*pow(cell_radius,3.0) );
                    (*temp_point_source->secretion_rates).resize(1,10); //[0]=10;
                    (*temp_point_source->saturation_densities).resize(1,1); //[0]=1;
                    temp_point_source->set_internal_uptake_constants(dt);
                }


}

void create_point_sinks(double cell_radius, double dt, int num_sinks, Microenvironment &microenvironment, int *mpi_Dims, int mpi_Rank, int mpi_Size, MPI_Comm mpi_Cart_comm)
{

        /*----------------------------------------------------------------------------------------*/
        /* First find total voxels in each direction: globally and locally                        */
        /*----------------------------------------------------------------------------------------*/

        int x_voxels = (microenvironment.mesh.bounding_box[3] - microenvironment.mesh.bounding_box[0])/microenvironment.mesh.dx;
        int y_voxels = (microenvironment.mesh.bounding_box[4] - microenvironment.mesh.bounding_box[1])/microenvironment.mesh.dy;
        int z_voxels = (microenvironment.mesh.bounding_box[5] - microenvironment.mesh.bounding_box[2])/microenvironment.mesh.dz;

        int local_x_voxels = x_voxels/mpi_Dims[1];
        int local_y_voxels = y_voxels/mpi_Dims[0];
        int local_z_voxels = z_voxels/mpi_Dims[2];

        int dx = microenvironment.mesh.dx;
        int dy = microenvironment.mesh.dy;
        int dz = microenvironment.mesh.dz;

        int proc_x_coord, proc_y_coord, proc_z_coord;
        int proc_index_rank;
        int unique_ID = 0;

        std::vector<double> tempPoint(3,0.0);
        std::vector<std::vector<double>> vector_coords(mpi_Size); //vector_coords[r] contains list of coordinates to be sent to rank 'r'
        std::vector<std::vector<int>> vector_IDs(mpi_Size);       //vector_IDs[r] contains list of IDs to be sent to rank 'r', IDs of Basic_Agents are unique in domain
        std::vector<int> sinks_per_proc_at_root(mpi_Size);
        std::vector<MPI_Request> send_req(mpi_Size);
        std::vector<double> list_of_coords;
        std::vector<int> list_of_IDs;

        MPI_Request recv_req;
        int sinks_per_proc;


        if(mpi_Rank == 0)
        {

            for(int i=0; i<num_sinks; i++)
            {

                /*------------------------------------------------------------------------------------------------*/
                /* Generate 3 random numbers between [0,1] and multiply by Range to get random position in domain */
                /* i.e. Multiply by x_max - x_min OR y_max - y_min OR z_max-z_min => in general the Range         */
                /*------------------------------------------------------------------------------------------------*/

                for(int j=0; j < 3 ; j++ )
                    tempPoint[j] = UniformRandom()*N;

                /*------------------------------------------------------------------------------------------------*/
                /* Find the MPI process coordinate using local_x/y/z_voxels                                       */
                /* Remember BioFVM X direction is left to right                                                   */
                /* MPI Process X direction is top to bottom                                                       */
                /*------------------------------------------------------------------------------------------------*/

                /*------------------------------------------------------------------------------------------------------------------------------------------------------*/
                /* When the starting point of domain is not zero i.e. minX != 0, minY != 0 and minZ != 0, then we should subtract these starting points from tempPoints */
                /* otherwise the strategy below fails                                                                                                                   */
                /*------------------------------------------------------------------------------------------------------------------------------------------------------*/

                tempPoint[0] = tempPoint[0] - microenvironment.mesh.bounding_box[0];
                tempPoint[1] = tempPoint[1] - microenvironment.mesh.bounding_box[1];
                tempPoint[2] = tempPoint[2] - microenvironment.mesh.bounding_box[2];

                proc_y_coord = tempPoint[0]/(local_x_voxels * dx);
                proc_x_coord = (mpi_Dims[0] - 1) - floor(tempPoint[1]/(local_y_voxels * dy));
                proc_z_coord = tempPoint[2]/(local_z_voxels * dz);

                proc_index_rank = proc_x_coord * mpi_Dims[1] * mpi_Dims[2] + proc_y_coord * mpi_Dims[2] + proc_z_coord;

                vector_coords[proc_index_rank].push_back(tempPoint[0]);          //Create a list of coordinates for a particular rank
                vector_coords[proc_index_rank].push_back(tempPoint[1]);          //i.e. for some rank 'r', vector_coords[r] = tempPoint[0]-->temPoint[1]-->temPoint[2]
                vector_coords[proc_index_rank].push_back(tempPoint[2]);
                vector_IDs[proc_index_rank].push_back(unique_ID);               //Similarly list of unique IDs for that particular rank

                unique_ID++;
            }

            for(int k=0 ; k<vector_IDs.size(); k++)                             //Need to know if there is a rank which has got no points
                sinks_per_proc_at_root[k] = vector_IDs[k].size();               //sinks_per_proc[k] contains no. of points going to kth rank


            for(int k=1 ; k<mpi_Size; k++)
            {
                MPI_Isend(&sinks_per_proc_at_root[k], 1, MPI_INT, k, 0, mpi_Cart_comm, &send_req[k]); //Every process needs to know how many points it will receive
                MPI_Wait(&send_req[k], MPI_STATUS_IGNORE);

                if(sinks_per_proc_at_root[k] != 0)                                                    //Send to only those processes which have got at least one point
                {
                   MPI_Isend(&vector_coords[k][0], 3 * sinks_per_proc_at_root[k], MPI_DOUBLE, k, 0, mpi_Cart_comm, &send_req[k]);
                   MPI_Wait(&send_req[k], MPI_STATUS_IGNORE);

                   MPI_Isend(&vector_IDs[k][0], sinks_per_proc_at_root[k], MPI_INT, k, 0, mpi_Cart_comm, &send_req[k]);
                   MPI_Wait(&send_req[k], MPI_STATUS_IGNORE);

                }
            }

                /*------------------------------------------------------------------------------------------------*/
                /* Since the send above was not for the root, we need to copy parameters at root                  */
                /*------------------------------------------------------------------------------------------------*/

            if(sinks_per_proc_at_root[0] != 0)
            {
               sinks_per_proc = sinks_per_proc_at_root[0];

               list_of_coords.resize(3 * sinks_per_proc);
               list_of_IDs.resize(sinks_per_proc);

               for(int k=0; k<vector_coords[0].size(); k++)
                   list_of_coords[k] = vector_coords[0][k];

               for(int k=0; k<sinks_per_proc; k++)
                   list_of_IDs[k] = vector_IDs[0][k];

            }
        }
        else
        {

                /*------------------------------------------------------------------------------------------------*/
                /* For non-root processes, first receive no. of points that process gets (>=0)                    */
                /* if num points = 0 then nothing to be done                                                      */
                /* If num points > 0 then store all coordinates and IDs                                           */
                /*------------------------------------------------------------------------------------------------*/

          MPI_Irecv(&sinks_per_proc, 1, MPI_INT, 0, 0, mpi_Cart_comm, &recv_req);
          MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

          if(sinks_per_proc != 0)
          {
              list_of_coords.resize(3 * sinks_per_proc);
              list_of_IDs.resize(sinks_per_proc);

              MPI_Irecv(&list_of_coords[0], 3 * sinks_per_proc, MPI_DOUBLE, 0, 0, mpi_Cart_comm, &recv_req);
              MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

              MPI_Irecv(&list_of_IDs[0], sinks_per_proc, MPI_INT, 0, 0, mpi_Cart_comm, &recv_req);
              MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

          }

        }
                /*------------------------------------------------------------------------------------------------*/
                /* All processes now allocate Basic_Agents separately                                             */
                /* Remember ID is now being sent by the root so it must be written here as the constructor of     */
                /* this class is also generating an ID, and we do not want the constructor ID                     */
                /*------------------------------------------------------------------------------------------------*/


            for(int k=0 ; k<sinks_per_proc; k++)
            {
                Basic_Agent *temp_point_sink = create_basic_agent();

                temp_point_sink->register_microenvironment(&microenvironment);

                for(int i=0 ; i<3 ; i++)
                    tempPoint[i] = list_of_coords[3*k + i];
                temp_point_sink->assign_position(tempPoint, mpi_Rank, mpi_Dims);

                temp_point_sink->ID = list_of_IDs[k];                                         //Added this statement to overwrite Constructor generated ID.
                temp_point_sink->set_total_volume( (4.0/3.0)*pi*pow(cell_radius,3.0) );
                (*temp_point_sink->uptake_rates)[0]=0.8;
                temp_point_sink->set_internal_uptake_constants(dt);
            }


}

int main( int argc, char* argv[] )
{
	double minX=0, minY=0, minZ=0, maxX=N, maxY=N, maxZ=N, mesh_resolution=10;

	/*--------------------------------------------*/
	/*		Initialize MPI		      */
	/*--------------------------------------------*/

    double t_start, t_end, t_total_start, t_total_end;

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
    t_total_start = MPI_Wtime();

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
	/*			MPI Comm Size and Rank                            		*/
  /*      Right now it should be 1-D X decomposition            */
	/*------------------------------------------------------------*/



	mpi_Dims[0] = 1;
	mpi_Dims[1] = mpi_Size;                                        //Number of Y processes, since Y-processes divide X-axis, we have X-decomposition
	mpi_Dims[2] = 1;


	mpi_Is_periodic[0]=0;
	mpi_Is_periodic[1]=0;
	mpi_Is_periodic[2]=0;
	mpi_Reorder = 0;



  /*------------------------------------------------------------*/
	/*			Create MPI topology, find coords                  		*/
	/*------------------------------------------------------------*/

	MPI_Cart_create(mpi_Comm, 3, mpi_Dims, mpi_Is_periodic, mpi_Reorder, &mpi_Cart_comm);
	MPI_Cart_coords(mpi_Cart_comm, mpi_Rank, 3, mpi_Coords);

	/*--------------------------------------------*/
	/*		OpenMP threads set	      */
	/*--------------------------------------------*/

	//omp_set_num_threads(omp_num_threads);

	/*--------------------------------------------*/
	/*		Create Microenvironment	      */
	/*--------------------------------------------*/


	Microenvironment microenvironment;

	microenvironment.name="substrate scale";
	microenvironment.set_density(0, "substrate1" , "dimensionless" );
	microenvironment.spatial_units = "microns";
	microenvironment.mesh.units = "microns";
	microenvironment.time_units = "minutes";



	/*------------------------------------------------------------*/
	/* Resize local Microenvironment on each process 	      */
	/*------------------------------------------------------------*/

    t_start = MPI_Wtime();
	microenvironment.resize_space_uniform( minX,maxX,minY,maxY,minZ,maxZ, mesh_resolution, mpi_Dims, mpi_Coords);
    t_end = MPI_Wtime();
    if(mpi_Rank==0)
        std::cout<<"TIME FOR RESIZING MICROENVIRONMENT = "<< (t_end-t_start)<< std::endl;



	std::vector<double> center(3);
	center[0] = (microenvironment.mesh.bounding_box[0]+microenvironment.mesh.bounding_box[3])/2;
	center[1] = (microenvironment.mesh.bounding_box[1]+microenvironment.mesh.bounding_box[4])/2;
	center[2] = (microenvironment.mesh.bounding_box[2]+microenvironment.mesh.bounding_box[5])/2;
	double stddev_squared = -100.0 * 100.0;
	std::vector<double> one( microenvironment.number_of_densities() , 1.0 );



    t_start = MPI_Wtime();
	#pragma omp parallel for
	for( int i=0; i < microenvironment.number_of_voxels() ; i++ )
	{
        std::vector<double> displacement = microenvironment.voxels(i).center - center;
		double distance_squared = norm_squared( displacement );
		double coeff = distance_squared;
		coeff /=  stddev_squared;
		microenvironment.density_vector(i)[0]= exp( coeff );
	}
	t_end = MPI_Wtime();
    if(mpi_Rank==0)
        std::cout<<"TIME FOR GENERATING GAUSSIAN PROFILE = "<< (t_end-t_start)<< std::endl;

	t_start = MPI_Wtime();
    microenvironment.write_to_matlab( "/home/bsc/bsc008383/CI/biofvm-b/output/initial_concentration.mat", mpi_Rank, mpi_Size, mpi_Cart_comm );
    t_end = MPI_Wtime();
    if(mpi_Rank==0)
        std::cout<<"TIME FOR WRITING INITIAL CONCENTRATION FILE = "<< (t_end-t_start)<< std::endl;

	// register the diffusion solver
	microenvironment.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D_vectorized;

	// register substrates properties
	microenvironment.diffusion_coefficients[0] = 1000; // microns^2 / min
	microenvironment.decay_rates[0] = 0.01;


    microenvironment.granurality = mpi_Size;

	double dt          = 0.01;
	double cell_radius = 5;
    int num_sources    = 500;
    int num_sinks      = 500;

    t_start = MPI_Wtime();
    create_point_sources(cell_radius, dt, num_sources, microenvironment, mpi_Dims, mpi_Rank, mpi_Size, mpi_Cart_comm);
    create_point_sinks(cell_radius, dt, num_sinks, microenvironment, mpi_Dims, mpi_Rank, mpi_Size, mpi_Cart_comm);
    t_end = MPI_Wtime();
    if(mpi_Rank==0)
        std::cout<<"TIME FOR CREATING ALL BASIC AGENTS = "<< (t_end-t_start)<< std::endl;


	double t = 0.0;
	double t_max=5;

    t_start = MPI_Wtime();
    while( t < t_max )
	{
		microenvironment.simulate_cell_sources_and_sinks( dt );
		microenvironment.simulate_diffusion_decay( dt, mpi_Size, mpi_Rank, mpi_Coords, mpi_Dims, mpi_Cart_comm );
		t += dt;
	}
	t_end = MPI_Wtime();
	if(mpi_Rank==0)
        std::cout<<"TIME FOR SIMULATING (SOURCES+SINKS+DIFFUSION) = "<< (t_end-t_start)<< std::endl;

	t_start = MPI_Wtime();
    microenvironment.write_to_matlab( "/home/bsc/bsc008383/CI/biofvm-b/output/final.mat", mpi_Rank, mpi_Size, mpi_Cart_comm );            //Remember to use Parallel Version !
    t_end = MPI_Wtime();
    if(mpi_Rank==0)
        std::cout<<"TIME FOR WRITING FINAL FILE = "<< (t_end-t_start)<< std::endl;


    MPI_Barrier(mpi_Comm);
    t_total_end = MPI_Wtime();
    if(mpi_Rank == 0)
        std::cout<<"TOTAL PROGRAM EXECUTION TIME = " <<(t_total_end-t_total_start)<<std::endl;

	/*-------------------------------------------------------------------------*/
	/*				MPI Finalize                                               */
	/*-------------------------------------------------------------------------*/

	MPI_Finalize();

	return 0;
}
