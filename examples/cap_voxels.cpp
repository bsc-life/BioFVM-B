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
#include <unistd.h>

#include "../BioFVM.h"

/*----------------Include MPI Header-------------*/
#include <mpi.h>
/*----------------------------------------------*/

using namespace BioFVM;
using namespace std;

/*------------------*/
/* Global variables */
/*------------------*/

int center_voxel_index = 0;



void process_mem_usage(double& vm_usage, double& resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

void supply_function(Microenvironment *microenvironment, int voxel_index, double *write_here)
{
	if (voxel_index == center_voxel_index)
	{
		write_here[0] = 100;
	}

	return;
}

void supply_target_function(Microenvironment *m, int voxel_index, double *write_here)
{
	if (voxel_index == center_voxel_index)
	{
		write_here[0] = 100;
	}
	return;
}

int write_report(int num_voxels, double time)
{
	std::ofstream report("scaling_test_results_voxels.txt", std::ofstream::app);
	report << num_voxels << "\t" << time << std::endl;
	report.close();
	return 0;
}

void process_output(double t, double dt, double mesh_resolution)
{

	std::cout << "current simulated time: " << t << " minutes " << std::endl;
	std::cout << "interval wall time: ";
	BioFVM::TOC();
	BioFVM::display_stopwatch_value(std::cout, BioFVM::stopwatch_value());
	std::cout << std::endl;
	std::cout << "total wall time: ";
	BioFVM::RUNTIME_TOC();
	BioFVM::display_stopwatch_value(std::cout, BioFVM::runtime_stopwatch_value());
	std::cout << std::endl;

	std::cout << "time: " << t << std::endl;
	BioFVM::TIC();
}

int main(int argc, char *argv[])
{
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

	if (mpi_Error != MPI_SUCCESS)
		MPI_Abort(mpi_Comm, -1);

	MPI_Comm_size(mpi_Comm, &mpi_Size);
	MPI_Comm_rank(mpi_Comm, &mpi_Rank);

	if (mpi_Provided_level != mpi_Requested_level && mpi_Rank == 0)
	{
		std::cout << "The MPI Requested Level is not available, please lower the level and try again" << std::endl;
		MPI_Abort(mpi_Comm, -1);
	}

	/*------------------------------------------------------------*/
	/*			MPI Comm Size and Rank                            		*/
	/*      Right now it should be 1-D X decomposition            */
	/*------------------------------------------------------------*/

	mpi_Dims[0] = 1;
	mpi_Dims[1] = mpi_Size; // Number of Y processes, since Y-processes divide X-axis, we have X-decomposition
	mpi_Dims[2] = 1;

	mpi_Is_periodic[0] = 0;
	mpi_Is_periodic[1] = 0;
	mpi_Is_periodic[2] = 0;
	mpi_Reorder = 0;

	/*------------------------------------------------------------*/
	/*			Create MPI topology, find coords                  		*/
	/*------------------------------------------------------------*/

	MPI_Cart_create(mpi_Comm, 3, mpi_Dims, mpi_Is_periodic, mpi_Reorder, &mpi_Cart_comm);
	MPI_Cart_coords(mpi_Cart_comm, mpi_Rank, 3, mpi_Coords);

	/*----------------------*/
	/* For time measurement */
	/*----------------------*/

	double t = 0.0;
	double t_output_interval = 1.0;
	double t_next_output_time = 0;
	int next_output_index = 0;

	double dt = 0.01;
	double t_max = 0.1; // t_max = iter * dt = 30 * 0.01
	double mesh_resolution = 10.0;

	/*--------------------------------------------------------------------*/
	/* One input is expected i.e. the HALF length of the CUBIC domain 		*/
	/* Suppose we give 100 then domain in all directions is -100 to +100 	*/
	/*--------------------------------------------------------------------*/

	double domain_half_side = strtod(argv[1], NULL);
	int num_substrates = atoi(argv[2]);


	/*------------------------------------------------------------------------------------*/
	/* For the parallel version, we are going to update the central voxel of EACH process */
	/*------------------------------------------------------------------------------------*/

	std::vector<double> center(3);

	/*------------------------------------------------------------------*/
	/* We control the threads in the SLURM submission script, hence no 	*/
	/* need for statement below UNLESS you want to reduce the number 		*/
	/* of threads from the maximum per process (i.e. tpp which is 24)		*/
	/*------------------------------------------------------------------*/

	// omp_set_num_threads(omp_num_threads);
	
	// create a microenvironment;
	double vm, rss;
    process_mem_usage(vm, rss);
	double vm_array[mpi_Size];
	double rss_array[mpi_Size];
	double vm_total_ini = 0.0;
	double rss_total_ini = 0.0;
	double vm_total_end = 0.0;
	double rss_total_end = 0.0;

	MPI_Gather(&vm, 1, MPI_DOUBLE, vm_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Gather rss values from all processes to the root
    MPI_Gather(&rss, 1, MPI_DOUBLE, rss_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if (mpi_Rank == 0) {
		
		for (int i = 0; i < mpi_Size; ++i) {
			vm_total_ini += vm_array[i];
			rss_total_ini += rss_array[i];
			//cout <<"rss[" << i << "]: " << rss_array[i] << endl;
		}
		#ifdef PAPER
			//cout << domain_half_side*2 << "," << num_substrates << "," << rss_total/1000000 << ",";
		#else
			cout << "Before microenvironment (" << domain_half_side*2 << ","<< num_substrates << ") VM: " << vm_total << " kB; RSS: " << rss_total/1000000 << " GB" << endl;
		#endif
	}
   

	Microenvironment microenvironment;
	microenvironment.set_density(0, "substrate0", "dimensionless");
	microenvironment.diffusion_coefficients[0] = 100000;
	microenvironment.decay_rates[0] = 0.1;

	for (int i = 1; i < num_substrates; i++)
	{
		std::string substrate_name;
		substrate_name.resize(25, '\0');
		microenvironment.add_density(substrate_name, "dimensionless");
		microenvironment.diffusion_coefficients[i] = 75000.0 + (i*1000.0);
		microenvironment.decay_rates[i] = 0.05 + (i*0.005);
	}



	
	double minX = -domain_half_side, minY = -domain_half_side, minZ = -domain_half_side, maxX = domain_half_side, maxY = domain_half_side, maxZ = domain_half_side; //, mesh_resolution=10;
	microenvironment.resize_space_uniform(minX, maxX, minY, maxY, minZ, maxZ, mesh_resolution, mpi_Dims, mpi_Coords);

	
	// register the diffusion solver
	microenvironment.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D;
	microenvironment.print_voxels_densities = print_voxels_densities;

	microenvironment.bulk_supply_rate_function = supply_function;
	microenvironment.bulk_supply_target_densities_function = supply_target_function;

	vector<double> subs(num_substrates, 1);
	#pragma omp parallel for collapse(3)
	for (int i = 0; i < microenvironment.mesh.x_size; i++)
	{
		for (int j = 0; j < microenvironment.mesh.y_size; j++)
		{
			for (int k = 0; k < microenvironment.mesh.z_size; k++)
			{
				int index =  i*microenvironment.mesh.y_size*microenvironment.mesh.z_size*num_substrates + 
							 j*microenvironment.mesh.z_size*num_substrates+
							 k*num_substrates;
				copy(subs.begin(), subs.end(), (*microenvironment.p_density_vectors).begin() + index);
			}
		}
	}
	process_mem_usage(vm, rss);

	MPI_Gather(&vm, 1, MPI_DOUBLE, vm_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Gather rss values from all processes to the root
    MPI_Gather(&rss, 1, MPI_DOUBLE, rss_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if (mpi_Rank == 0) {
		for (int i = 0; i < mpi_Size; ++i) {
			vm_total_end += vm_array[i];
			rss_total_end += rss_array[i];
		}
		#ifdef PAPER
			//cout << domain_half_side*2 << "," << num_substrates << "," << rss_total_ini/1000000 << "," << rss_total_end/1000000 << endl;
			cout << domain_half_side*2 << "," << num_substrates << ","  << rss_total_ini/1000000 << "," << rss_total_end/1000000 << endl;
		#else
    		cout << "After microenvironment  RSS: " << rss_total /1000000<< " GB" << endl;
		#endif
	}

	
	
	MPI_Finalize();
	return 0;
}
