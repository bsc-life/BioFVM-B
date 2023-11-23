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

#include "BioFVM_solvers.h" 
#include "BioFVM_vector.h" 

#include <iostream>
#include <omp.h>

namespace BioFVM{

// do I even need this? 
void diffusion_decay_solver__constant_coefficients_explicit( Microenvironment& M, double dt )
{
	static bool precomputations_and_constants_done = false; 
	if( !precomputations_and_constants_done )
	{
		std::cout	<< std::endl << "Using solver: " << __FUNCTION__ << std::endl 
					<< "     (constant diffusion coefficient with explicit stepping, implicit decay) ... " << std::endl << std::endl;  

		if( M.mesh.uniform_mesh == true )
		{
			std::cout << "Uniform mesh detected! Consider switching to a more efficient method, such as " << std::endl  
			<< "     diffusion_decay_solver__constant_coefficients_explicit_uniform_mesh" << std::endl  
			<< std::endl; 
		}

		precomputations_and_constants_done = true; 
	}

	return; 
}

void diffusion_decay_solver__constant_coefficients_explicit_uniform_mesh( Microenvironment& M, double dt )
{
	static bool precomputations_and_constants_done = false; 
	if( !precomputations_and_constants_done )
	{
		std::cout	<< std::endl << "Using solver: " << __FUNCTION__ << std::endl 
					<< "     (constant diffusion coefficient with explicit stepping, implicit decay, uniform mesh) ... " << std::endl << std::endl;  

		if( M.mesh.uniform_mesh == false )
		{ std::cout << "Error. This code is only supported for uniform meshes." << std::endl; }

		precomputations_and_constants_done = true; 
	}

	return; 
}

/*----------------------------------------------------------*/
/* WRITING PARALLEL VERSION OF THIS FUNCTION IN             */
/* BioFVM_parallel.cpp                                      */
/* DO NOT CHANGE THIS VERSION                               */
/*----------------------------------------------------------*/
// void diffusion_decay_solver__constant_coefficients_LOD_3D( Microenvironment& M, double dt )
// {
// 	if( M.mesh.uniform_mesh == false || M.mesh.Cartesian_mesh == false )
// 	{
// 		std::cout << "Error: This algorithm is written for uniform Cartesian meshes. Try: other solvers!" << std::endl << std::endl; 
// 	return; 
// 	}
// 
// 	// define constants and pre-computed quantities 
// 	
// 	if( !M.diffusion_solver_setup_done )
// 	{
// 		std::cout << std::endl << "Using method " << __FUNCTION__ << " (implicit 3-D LOD with Thomas Algorithm) ... " 
// 		<< std::endl << std::endl;  
// 		/*-------------------------------------------------------------*/
//         /* x_coordinates                                               */
//         /* are of size local_x_nodes (see function resize()            */
//         /* of class Cartesian Mesh in BioFVM_parallel.cpp.             */ 
//         /* Each line of Voxels going                                   */
//         /* from left to right forms a tridiagonal system of Equations  */
//         /*-------------------------------------------------------------*/
// 		
//         M.thomas_denomx.resize( M.mesh.x_coordinates.size() , M.zero );           //sizeof(x_coordinates) = local_x_nodes, denomx is the main diagonal elements
// 		M.thomas_cx.resize( M.mesh.x_coordinates.size() , M.zero );               //Both b and c of tridiagonal matrix are equal, hence just one array needed
// 		
// 		/*-------------------------------------------------------------*/
//         /* y_coordinates are of size of local_y_nodes.                 */
//         /* Each line of Voxels going                                   */
//         /* from bottom to top forms a tridiagonal system of Equations  */
//         /*-------------------------------------------------------------*/
// 
// 		M.thomas_denomy.resize( M.mesh.y_coordinates.size() , M.zero );           
// 		M.thomas_cy.resize( M.mesh.y_coordinates.size() , M.zero );
//         
//         /*-------------------------------------------------------------*/
//         /* z_coordinates are of size of local_z_nodes.                 */
//         /* Each line of Voxels going                                   */
//         /* from front to back forms a tridiagonal system of Equations  */
//         /* In 1-D Z decomposition, these lines                         */
//         /* are going to be split over multiple processes.              */
//         /*-------------------------------------------------------------*/
// 		
// 		M.thomas_denomz.resize( M.mesh.z_coordinates.size() , M.zero );
// 		M.thomas_cz.resize( M.mesh.z_coordinates.size() , M.zero );
// 
//         /*-------------------------------------------------------------*/
//         /* For x, y direction, AND 1-D decomposition in the Z-dir      */
//         /* the thomas_i_jump and thomas_j_jump will always be within   */
//         /* the process but for thomas_k_jump, this will be within      */
//         /* process for non-boundary voxels and in neighbouring         */
//         /* processes for boundary voxels.                              */
//         /*-------------------------------------------------------------*/
// 
// 		M.thomas_i_jump = 1; 
// 		M.thomas_j_jump = M.mesh.x_coordinates.size(); 
// 		M.thomas_k_jump = M.thomas_j_jump * M.mesh.y_coordinates.size(); 
//         
//         /*-------------------------------------------------------------*/
//         /* This part below of defining constants SHOULD typically      */
//         /* not change during parallelization.                          */
//         /*-------------------------------------------------------------*/
// 
// 		M.thomas_constant1  = M.diffusion_coefficients;      // dt*D/dx^2 
// 		M.thomas_constant1a = M.zero;                        // -dt*D/dx^2; 
// 		M.thomas_constant2  = M.decay_rates;                 // (1/3)* dt*lambda 
// 		M.thomas_constant3  = M.one;                         // 1 + 2*constant1 + constant2; 
// 		M.thomas_constant3a = M.one;                         // 1 + constant1 + constant2; 		
// 			
// 		M.thomas_constant1 *= dt; 
// 		M.thomas_constant1 /= M.mesh.dx; 
// 		M.thomas_constant1 /= M.mesh.dx; 
// 
// 		M.thomas_constant1a = M.thomas_constant1; 
// 		M.thomas_constant1a *= -1.0; 
// 
// 		M.thomas_constant2 *= dt; 
// 		M.thomas_constant2 /= 3.0;                            // for the LOD splitting of the source, division by 3 is for 3-D 
// 
// 		M.thomas_constant3 += M.thomas_constant1; 
// 		M.thomas_constant3 += M.thomas_constant1; 
// 		M.thomas_constant3 += M.thomas_constant2; 
// 
// 		M.thomas_constant3a += M.thomas_constant1; 
// 		M.thomas_constant3a += M.thomas_constant2; 
// 
// 		// Thomas solver coefficients 
//         
//         /*--------------------------------------------------------------------*/
//         /* In 1-D Z decomposition, x and y-lines are contiguous adn typically */
//         /* the assignments below for x,y should not be changed                */
//         /* Both the first voxel i.e. index 0 and last voxel i.e. index=       */
//         /* x_coordinates.size()-1 are on the same process                     */
//         /*--------------------------------------------------------------------*/
// 
// 		M.thomas_cx.assign( M.mesh.x_coordinates.size() , M.thomas_constant1a );                  //Fill b and c elements with -D * dt/dx^2 
// 		M.thomas_denomx.assign( M.mesh.x_coordinates.size()  , M.thomas_constant3 );              //Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dx^2)
// 		M.thomas_denomx[0] = M.thomas_constant3a;                                                 //First diagonal element is   (1 + 1/3 * lambda * dt + 1*D*dt/dx^2)
// 		M.thomas_denomx[ M.mesh.x_coordinates.size()-1 ] = M.thomas_constant3a;                   //Last diagonal element  is   (1 + 1/3 * lambda * dt + 1*D*dt/dx^2) 
// 		
// 		if( M.mesh.x_coordinates.size() == 1 )                                                    //This is an extreme case, won't exist
// 		{ 
//             M.thomas_denomx[0] = M.one; 
//             M.thomas_denomx[0] += M.thomas_constant2; 
//         } 
// 
// 		M.thomas_cx[0] /= M.thomas_denomx[0];                                                     //The first c element of tridiagonal matrix is div by first diagonal el. 
// 		for( int i=1 ; i <= M.mesh.x_coordinates.size()-1 ; i++ )                                 
// 		{ 
// 			axpy( &M.thomas_denomx[i] , M.thomas_constant1 , M.thomas_cx[i-1] );                 //axpy(1st, 2nd, 3rd) => 1st = 1st + 2nd * 3rd
// 			M.thomas_cx[i] /= M.thomas_denomx[i];                                                //the value at  size-1 is not actually used  
// 		}                                                                                        //Since value of size-1 is not used, it means it is the value after the last 
//                                                                                                  //Diagonal element
// 
// 		/*--------------------------------------------------------------------*/
//         /* In 1-D Z decomposition, x and y-lines are contiguous adn typically */
//         /* the assignments below for x,y should not be changed                */
//         /* Both the first voxel i.e. index 0 and last voxel i.e. index=       */
//         /* y_coordinates.size()-1 are on the same process                     */
//         /*--------------------------------------------------------------------*/
// 		
// 		M.thomas_cy.assign( M.mesh.y_coordinates.size() , M.thomas_constant1a ); 
// 		M.thomas_denomy.assign( M.mesh.y_coordinates.size()  , M.thomas_constant3 ); 
// 		M.thomas_denomy[0] = M.thomas_constant3a; 
// 		M.thomas_denomy[ M.mesh.y_coordinates.size()-1 ] = M.thomas_constant3a; 
// 		if( M.mesh.y_coordinates.size() == 1 )
// 		{ M.thomas_denomy[0] = M.one; M.thomas_denomy[0] += M.thomas_constant2; } 
// 
// 		M.thomas_cy[0] /= M.thomas_denomy[0]; 
// 		for( int i=1 ; i <= M.mesh.y_coordinates.size()-1 ; i++ )
// 		{ 
// 			axpy( &M.thomas_denomy[i] , M.thomas_constant1 , M.thomas_cy[i-1] ); 
// 			M.thomas_cy[i] /= M.thomas_denomy[i];                                                 // the value at  size-1 is not actually used  
// 		}
// 
// 		/*--------------------------------------------------------------------*/
//         /* for nProcs > 1 and 1-D decomposition in Z-direction, index 0 and   */
//         /* index z_coordinates.size()-1, the latter being the ACTUAL boundary */
//         /* point of the domain, are on different processes                    */
//         /*--------------------------------------------------------------------*/
// 		
// 		M.thomas_cz.assign( M.mesh.z_coordinates.size() , M.thomas_constant1a ); 
// 		M.thomas_denomz.assign( M.mesh.z_coordinates.size()  , M.thomas_constant3 ); 
//         
//         /*--------------------------------------------------------------------*/
//         /* This is tricky:                                                    */ 
//         /* Now processes in Z direction will ALL have their individual        */
//         /* thomas_denomz[] and thomas_cz[].                                   */
//         /* Only processes which contain the front face of the domain          */
//         /* will set thomas_denomz[0]=thomas_constant3a                        */
//         /* These processes have mpi_Coord[2]=0                                */
//         /* Also ONLY processes containing the back boundary of domain will    */
//         /* set thomas_denomz[M.mesh.z_coordinates.size()-1]=thomas_constant3a */
//         /* Such processes will have mpi_Coord[2]=dims[2]-1                    */
//         /*--------------------------------------------------------------------*/
//         
// 		M.thomas_denomz[0] = M.thomas_constant3a; 
//         
// 		M.thomas_denomz[ M.mesh.z_coordinates.size()-1 ] = M.thomas_constant3a; 
//         
// 		if( M.mesh.z_coordinates.size() == 1 )                                                    //Extreme case - unlikely to occur
// 		{ 
//             M.thomas_denomz[0] = M.one; 
//             M.thomas_denomz[0] += M.thomas_constant2; 
//         } 
// 
// 		/*-------------------------------------------------------------------*/
//         /* This again is to be done on the processes containing the front    */
//         /* boundary i.e. ONLY for mpi_Coord[2]=0;                            */
//         /*-------------------------------------------------------------------*/
// 		
//         M.thomas_cz[0] /= M.thomas_denomz[0]; 
//         
// 		/*-------------------------------------------------------------------*/
//         /* This will be serialized because thomas_cz[i-1] is needed which    */
//         /* modified in every iteration of for loop.                          */
//         /* For rank 0 process in 1-D Z decomposition, calculation will       */
//         /* begin at i=1 but for other processes, calculation will begin at   */
//         /* i=0 AND we need to pass the value of last thomas_cz[i-1] in       */
//         /* in every process to next process.                                 */
//         /* I think this will just be a single double value - NO              */
//         /* thomas_cz[i] is a vector in itself, so we will need to send       */
//         /* thomas_cz[i][0] to thomas_cz[i].size()-1 to receiver              */
//         /*-------------------------------------------------------------------*/
//         
//         for( int i=1 ; i <= M.mesh.z_coordinates.size()-1 ; i++ )
// 		{ 
// 			axpy( &M.thomas_denomz[i] , M.thomas_constant1 , M.thomas_cz[i-1] ); 
// 			M.thomas_cz[i] /= M.thomas_denomz[i];                                                  // the value at  size-1 is not actually used  
// 		}	
// 
// 		M.diffusion_solver_setup_done = true; 
// 	}
// 
// 	// x-diffusion 
// 	
// 	M.apply_dirichlet_conditions();
//     
//     /*-----------------------------------------------------------------------*/
//     /* For X/Y directions, I don't think any change is needed                */
//     /*-----------------------------------------------------------------------*/
//     
// 	#pragma omp parallel for 
// 	for( int k=0; k < M.mesh.z_coordinates.size() ; k++ )
// 	{
// 		for( int j=0; j < M.mesh.y_coordinates.size() ; j++ )
// 		{
// 			// Thomas solver, x-direction
// 
// 			// remaining part of forward elimination, using pre-computed quantities 
// 			int n = M.voxel_index(0,j,k);
// 			(*M.p_density_vectors)[n] /= M.thomas_denomx[0]; 
// 
// 			for( int i=1; i < M.mesh.x_coordinates.size() ; i++ )
// 			{
// 				n = M.voxel_index(i,j,k); 
// 				axpy( &(*M.p_density_vectors)[n] , M.thomas_constant1 , (*M.p_density_vectors)[n-M.thomas_i_jump] );
// 				(*M.p_density_vectors)[n] /= M.thomas_denomx[i]; 
// 			}
// 
// 			for( int i = M.mesh.x_coordinates.size()-2 ; i >= 0 ; i-- )
// 			{
// 				n = M.voxel_index(i,j,k); 
// 				naxpy( &(*M.p_density_vectors)[n] , M.thomas_cx[i] , (*M.p_density_vectors)[n+M.thomas_i_jump] ); 
// 			}
// 
// 		}
// 	}
// 
// 	// y-diffusion 
// 
// 	M.apply_dirichlet_conditions();
// 	#pragma omp parallel for 
// 	for( int k=0; k < M.mesh.z_coordinates.size() ; k++ )
// 	{
// 		for( int i=0; i < M.mesh.x_coordinates.size() ; i++ )
// 		{
//    // Thomas solver, y-direction
// 
// 	// remaining part of forward elimination, using pre-computed quantities 
// 
// 	int n = M.voxel_index(i,0,k);
// 	(*M.p_density_vectors)[n] /= M.thomas_denomy[0]; 
// 
// 	for( int j=1; j < M.mesh.y_coordinates.size() ; j++ )
// 	{
// 		n = M.voxel_index(i,j,k); 
// 		axpy( &(*M.p_density_vectors)[n] , M.thomas_constant1 , (*M.p_density_vectors)[n-M.thomas_j_jump] ); 
// 		(*M.p_density_vectors)[n] /= M.thomas_denomy[j]; 
// 	}
// 
// 	// back substitution 
// 	// n = voxel_index( mesh.x_coordinates.size()-2 ,j,k); 
// 
// 	for( int j = M.mesh.y_coordinates.size()-2 ; j >= 0 ; j-- )
// 	{
// 		n = M.voxel_index(i,j,k); 
// 		naxpy( &(*M.p_density_vectors)[n] , M.thomas_cy[j] , (*M.p_density_vectors)[n+M.thomas_j_jump] ); 
// 	}
// 
//   }
//  }
// 
//  // z-diffusion
//  /*--------------------------------------------------------------------------------*/
//  /* This will change. Why is the Dirichlet condition being applied again and again?*/
//  /*--------------------------------------------------------------------------------*/
// 
// 	M.apply_dirichlet_conditions();
//  #pragma omp parallel for 
//  for( int j=0; j < M.mesh.y_coordinates.size() ; j++ )
//  {
// 	 
//   for( int i=0; i < M.mesh.x_coordinates.size() ; i++ )
//   {
//    // Thomas solver, z-direction
// 
// 	// remaining part of forward elimination, using pre-computed quantities 
// 
// 	/*------------------------------------------------------------------------------*/
//     /* This will be done only on rank = 0, or basically the processes containing    */
//     /* the front boundary of the process.                                           */
//     /* MPI divides domain into thick slices in Z-direction                          */
//     /* OpenMP divides the thick slices in vertical direction                        */
//     /*------------------------------------------------------------------------------*/
//     int n = M.voxel_index(i,j,0);
// 	(*M.p_density_vectors)[n] /= M.thomas_denomz[0]; 
// 
// 	
//     /*------------------------------------------------------------------------------*/
//     /* Let all the processing finish at rank 0, now we need to send a the 'back'    */
//     /* square of p_density_vectors[] to rank 1. Do the same at rank 2 so on...      */
//     /* Possibly the thomas_k_jump will also change                                  */
//     /* Instead of using thomas_k_jump, we can use prev=voxel_index(i,j,k-1)         */
//     /* and then use it in (*M.p_density_vectors)[prev]                              */
//     /* Also for rank > 0, it needs to start at k=0                                  */
//     /* The values of p_density_vectors sent from the previous neighbour will be     */
//     /* stored in temp_p_density_vectors -- only used for k=1 else just use          */
//     /* p_density_vectors. Vector of vectors is not stored contiguously              */ 
//     /* So we need to pack them into a single array and then send it to other end    */
//     /*------------------------------------------------------------------------------*/
//     
//     // should be an empty loop if mesh.z_coordinates.size() < 2  
// 	for( int k=1; k < M.mesh.z_coordinates.size() ; k++ )
// 	{
// 		n = M.voxel_index(i,j,k); 
// 		axpy( &(*M.p_density_vectors)[n] , M.thomas_constant1 , (*M.p_density_vectors)[n-M.thomas_k_jump] ); 
// 		(*M.p_density_vectors)[n] /= M.thomas_denomz[k]; 
// 	}
// 
// 	// back substitution 
// 
// 	// should be an empty loop if mesh.z_coordinates.size() < 2 
// 	
// 	/*---------------------------------------------------------------------------------*/
//     /* This starts from the last process i.e. rank = size-1 then proceeds backwards    */
//     /* i.e. from back boundary to front boundary. On the last process k starts from    */
//     /* z_coordinates.size()-2 but on other processes, it starts from                   */
//     /* z_coordinates.size()-1. 
//     /*---------------------------------------------------------------------------------*/
// 	for( int k = M.mesh.z_coordinates.size()-2 ; k >= 0 ; k-- )
// 	{
// 		n = M.voxel_index(i,j,k); 
// 		naxpy( &(*M.p_density_vectors)[n] , M.thomas_cz[k] , (*M.p_density_vectors)[n+M.thomas_k_jump] ); 
// 		// n -= i_jump; 
// 	}
//   }
//  }
//  
// 	M.apply_dirichlet_conditions();
// 	
// 	// reset gradient vectors 
// //	M.reset_all_gradient_vectors(); 
// 
// 	return; 
// }

// void diffusion_decay_solver__constant_coefficients_LOD_2D( Microenvironment& M, double dt, int mpi_Size, int mpi_Rank, int *mpi_Coords, int *mpi_Dims, MPI_Comm mpi_Cart_comm )
// {   //-->Gaurav Saxena changed this function signature (although the function body is not parallel) ---> just avoiding compilation problems.
// 	if( M.mesh.uniform_mesh == false )
// 	{
// 		std::cout << "Error: This algorithm is written for uniform Cartesian meshes. Try: something else." << std::endl << std::endl; 
// 		return; 
// 	}
// 	
// 	// constants for the linear solver (Thomas algorithm) 
// 	
// 	if( !M.diffusion_solver_setup_done )
// 	{
// 		std::cout << std::endl << "Using method " << __FUNCTION__ << " (2D LOD with Thomas Algorithm) ... " << std::endl << std::endl;  
// 		
// 		M.thomas_denomx.resize( M.mesh.x_coordinates.size() , M.zero );
// 		M.thomas_cx.resize( M.mesh.x_coordinates.size() , M.zero );
// 
// 		M.thomas_denomy.resize( M.mesh.y_coordinates.size() , M.zero );
// 		M.thomas_cy.resize( M.mesh.y_coordinates.size() , M.zero );
// 		
// 		// define constants and pre-computed quantities 
// 
// 		M.thomas_i_jump = 1; 
// 		M.thomas_j_jump = M.mesh.x_coordinates.size(); 
// 
// 		M.thomas_constant1 =  M.diffusion_coefficients; //   dt*D/dx^2 
// 		M.thomas_constant1a = M.zero; // -dt*D/dx^2; 
// 		M.thomas_constant2 =  M.decay_rates; // (1/2)*dt*lambda 
// 		M.thomas_constant3 = M.one; // 1 + 2*constant1 + constant2; 
// 		M.thomas_constant3a = M.one; // 1 + constant1 + constant2; 
// 		
// 		M.thomas_constant1 *= dt; 
// 		M.thomas_constant1 /= M.mesh.dx; 
// 		M.thomas_constant1 /= M.mesh.dx; 
// 
// 		M.thomas_constant1a = M.thomas_constant1; 
// 		M.thomas_constant1a *= -1.0; 
// 
// 		M.thomas_constant2 *= dt; 
// 		M.thomas_constant2 *= 0.5; // for splitting via LOD
// 
// 		M.thomas_constant3 += M.thomas_constant1; 
// 		M.thomas_constant3 += M.thomas_constant1; 
// 		M.thomas_constant3 += M.thomas_constant2; 
// 
// 		M.thomas_constant3a += M.thomas_constant1; 
// 		M.thomas_constant3a += M.thomas_constant2; 
// 		
// 		// Thomas solver coefficients 
// 
// 		M.thomas_cx.assign( M.mesh.x_coordinates.size() , M.thomas_constant1a ); 
// 		M.thomas_denomx.assign( M.mesh.x_coordinates.size()  , M.thomas_constant3 ); 
// 		M.thomas_denomx[0] = M.thomas_constant3a; 
// 		M.thomas_denomx[ M.mesh.x_coordinates.size()-1 ] = M.thomas_constant3a; 
// 		if( M.mesh.x_coordinates.size() == 1 )
// 		{ M.thomas_denomx[0] = M.one; M.thomas_denomx[0] += M.thomas_constant2; } 
// 
// 		M.thomas_cx[0] /= M.thomas_denomx[0]; 
// 		for( int i=1 ; i <= M.mesh.x_coordinates.size()-1 ; i++ )
// 		{ 
// 			axpy( &M.thomas_denomx[i] , M.thomas_constant1 , M.thomas_cx[i-1] ); 
// 			M.thomas_cx[i] /= M.thomas_denomx[i]; // the value at  size-1 is not actually used  
// 		}
// 
// 		M.thomas_cy.assign( M.mesh.y_coordinates.size() , M.thomas_constant1a ); 
// 		M.thomas_denomy.assign( M.mesh.y_coordinates.size()  , M.thomas_constant3 ); 
// 		M.thomas_denomy[0] = M.thomas_constant3a; 
// 		M.thomas_denomy[ M.mesh.y_coordinates.size()-1 ] = M.thomas_constant3a; 
// 		if( M.mesh.y_coordinates.size() == 1 )
// 		{ M.thomas_denomy[0] = M.one; M.thomas_denomy[0] += M.thomas_constant2; } 
// 
// 
// 		M.thomas_cy[0] /= M.thomas_denomy[0]; 
// 		for( int i=1 ; i <= M.mesh.y_coordinates.size()-1 ; i++ )
// 		{ 
// 			axpy( &M.thomas_denomy[i] , M.thomas_constant1 , M.thomas_cy[i-1] ); 
// 			M.thomas_cy[i] /= M.thomas_denomy[i]; // the value at  size-1 is not actually used  
// 		}
// 
// 		M.diffusion_solver_setup_done = true; 
// 	}
// 
// 	// set the pointers
// 
// 	M.apply_dirichlet_conditions();
// 	// x-diffusion 
// 	#pragma omp parallel for 
// 	for( int j=0; j < M.mesh.y_coordinates.size() ; j++ )
// 	{
// 		// Thomas solver, x-direction
// 
// 		// remaining part of forward elimination, using pre-computed quantities 
// 		int n = M.voxel_index(0,j,0);
// 		(*M.p_density_vectors)[n] /= M.thomas_denomx[0]; 
// 
// 		n += M.thomas_i_jump; 
// 		for( int i=1; i < M.mesh.x_coordinates.size() ; i++ )
// 		{
// 			axpy( &(*M.p_density_vectors)[n] , M.thomas_constant1 , (*M.p_density_vectors)[n-M.thomas_i_jump] ); 
// 			(*M.p_density_vectors)[n] /= M.thomas_denomx[i]; 
// 			n += M.thomas_i_jump; 
// 		}
// 
// 		// back substitution 
// 		n = M.voxel_index( M.mesh.x_coordinates.size()-2 ,j,0); 
// 
// 		for( int i = M.mesh.x_coordinates.size()-2 ; i >= 0 ; i-- )
// 		{
// 			naxpy( &(*M.p_density_vectors)[n] , M.thomas_cx[i] , (*M.p_density_vectors)[n+M.thomas_i_jump] ); 
// 			n -= M.thomas_i_jump; 
// 		}
// 	}
// 
// 	// y-diffusion 
// 
// 	M.apply_dirichlet_conditions();
// 	#pragma omp parallel for 
// 	for( int i=0; i < M.mesh.x_coordinates.size() ; i++ )
// 	{
// 		// Thomas solver, y-direction
// 
// 		// remaining part of forward elimination, using pre-computed quantities 
// 
// 		int n = M.voxel_index(i,0,0);
// 		(*M.p_density_vectors)[n] /= M.thomas_denomy[0]; 
// 
// 		n += M.thomas_j_jump; 
// 		for( int j=1; j < M.mesh.y_coordinates.size() ; j++ )
// 		{
// 			axpy( &(*M.p_density_vectors)[n] , M.thomas_constant1 , (*M.p_density_vectors)[n-M.thomas_j_jump] ); 
// 			(*M.p_density_vectors)[n] /= M.thomas_denomy[j]; 
// 			n += M.thomas_j_jump; 
// 		}
// 
// 		// back substitution 
// 		n = M.voxel_index( i,M.mesh.y_coordinates.size()-2, 0); 
// 
// 		for( int j = M.mesh.y_coordinates.size()-2 ; j >= 0 ; j-- )
// 		{
// 			naxpy( &(*M.p_density_vectors)[n] , M.thomas_cy[j] , (*M.p_density_vectors)[n+M.thomas_j_jump] ); 
// 			n -= M.thomas_j_jump; 
// 		}
// 	}
// 
// 	M.apply_dirichlet_conditions();
// 	
// 	// reset gradient vectors 
// //	M.reset_all_gradient_vectors(); 
// 	
// 	return; 
// }

void diffusion_decay_explicit_uniform_rates( Microenvironment& M, double dt )
{


	using std::vector; 
	using std::cout; 
	using std::endl; 
	cout << "Diffusion decay explicit uniform rates desactivated!" << endl;
	/*
	static int n_jump_i = 1; 
	static int n_jump_j = M.mesh.x_coordinates.size(); 
	static int n_jump_k = M.mesh.x_coordinates.size() * M.mesh.y_coordinates.size(); 

	if( !M.diffusion_solver_setup_done )
	{	
		M.thomas_i_jump = 1; 
		M.thomas_j_jump = M.mesh.x_coordinates.size(); 
		M.thomas_k_jump = M.thomas_j_jump * M.mesh.y_coordinates.size(); 
	
		M.diffusion_solver_setup_done = true; 
	}
	
	if( M.mesh.uniform_mesh == false )
	{
		cout << "Error: This algorithm is written for uniform Cartesian meshes. Try: something else" << endl << endl; 
		return; 
	}

	// double buffering to reduce memory copy / allocation overhead 

	static vector< vector<double> >* pNew = &(M.temporary_density_vectors1);
	static vector< vector<double> >* pOld = &(M.temporary_density_vectors2);

	// swap the buffers 

	vector< vector<double> >* pTemp = pNew; 
	pNew = pOld; 
	pOld = pTemp; 
	M.p_density_vectors = pNew; 

	static bool reaction_diffusion_shortcuts_are_set = false; 

	static vector<double> constant1 = (1.0 / ( M.mesh.dx * M.mesh.dx )) * M.diffusion_coefficients; 
	static vector<double> constant2 = dt * constant1; 
	static vector<double> constant3 = M.one + dt * M.decay_rates;

	static vector<double> constant4 = M.one - dt * M.decay_rates;

	#pragma omp parallel for
	for( int i=0; i < (*(M.p_density_vectors)).size() ; i++ )
	{
		int number_of_neighbors = M.mesh.connected_voxel_indices[i].size(); 

		double d1 = -1.0 * number_of_neighbors; 

		(*pNew)[i] = (*pOld)[i];  
		(*pNew)[i] *= constant4; 

		for( int j=0; j < number_of_neighbors ; j++ )
		{
			axpy( &(*pNew)[i], constant2, (*pOld)[  M.mesh.connected_voxel_indices[i][j] ] ); 
		}
		vector<double> temp = constant2; 
		temp *= d1; 
		axpy( &(*pNew)[i] , temp , (*pOld)[i] ); 
	}
	
	// reset gradient vectors 
//	M.reset_all_gradient_vectors(); 
*/
	return; 
}

};
