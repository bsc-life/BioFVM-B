#include "BioFVM_microenvironment.h"
#include "BioFVM_mesh.h"
#include "BioFVM_basic_agent.h"
#include <cmath>
#include <immintrin.h>

namespace BioFVM
{
    #define DENSITY(X,Y,Z) (*(M.p_density_vectors[(X)]))[(Y)*M.mesh.z_size+(Z)]
    void diffusion_decay_solver__constant_coefficients_LOD_3D(Microenvironment &M, double dt, int size, int rank, int *coords, int *dims, MPI_Comm mpi_Cart_comm)
    {
        
        MPI_Request send_req, recv_req;
        double t_strt_set, t_end_set;
        double t_strt_x, t_end_x;
        double t_strt_y, t_end_y;
        double t_strt_z, t_end_z;

        if (M.mesh.uniform_mesh == false || M.mesh.Cartesian_mesh == false)
        {
            std::cout << "Error: This algorithm is written for uniform Cartesian meshes. Try: other solvers!" << std::endl
                      << std::endl;
            return;
        }

        // define constants and pre-computed quantities

        if (!M.diffusion_solver_setup_done)
        {
            // t_strt_set = MPI_Wtime();
            // std::cout << std::endl << "Using method " << __FUNCTION__ << " (implicit 3-D LOD with Thomas Algorithm) ... "
            //<< std::endl << std::endl;

            /*-------------------------------------------------------------*/
            /* x_coordinates are of size local_x_nodes                     */
            /* (see function resize() of class Cartesian Mesh in           */
            /* BioFVM_parallel.cpp.                                        */
            /* Each line of Voxels going from left to right forms          */
            /* a tridiagonal system of Equations                           */
            /* Now these lines are going to split in the X decomposition   */
            /*-------------------------------------------------------------*/

            M.thomas_denomx.resize(M.mesh.x_coordinates.size(), M.zero); // sizeof(x_coordinates) = local_x_nodes, denomx is the main diagonal elements
            M.thomas_cx.resize(M.mesh.x_coordinates.size(), M.zero);     // Both b and c of tridiagonal matrix are equal, hence just one array needed

            /*-------------------------------------------------------------*/
            /* y_coordinates are of size of local_y_nodes.                 */
            /* Each line of Voxels going                                   */
            /* from bottom to top forms a tridiagonal system of Equations  */
            /*-------------------------------------------------------------*/

            M.thomas_denomy.resize(M.mesh.y_coordinates.size(), M.zero);
            M.thomas_cy.resize(M.mesh.y_coordinates.size(), M.zero);

            /*-------------------------------------------------------------*/
            /* z_coordinates are of size of local_z_nodes.                 */
            /* Each line of Voxels going                                   */
            /* from front to back forms a tridiagonal system of Equations  */
            /*-------------------------------------------------------------*/

            M.thomas_denomz.resize(M.mesh.z_coordinates.size(), M.zero);
            M.thomas_cz.resize(M.mesh.z_coordinates.size(), M.zero);

            /*-------------------------------------------------------------*/
            /* For X-decomposition thomas_i_jump - 1 can be in the previous*/
            /* process and thomas_i_jump+1 can be in the next processs     */
            /* hence we can use thomas_j_jump and thomas_k_jump safely     */
            /* but we CANNOT use thomas_i_jump safely                      */
            /*-------------------------------------------------------------*/

            M.thomas_i_jump = 1;
            M.thomas_j_jump = M.mesh.x_coordinates.size();
            M.thomas_k_jump = M.thomas_j_jump * M.mesh.y_coordinates.size();

            /*-------------------------------------------------------------*/
            /* This part below of defining constants SHOULD typically      */
            /* not change during parallelization.                          */
            /*-------------------------------------------------------------*/

            M.thomas_constant1 = M.diffusion_coefficients; // dt*D/dx^2
            M.thomas_constant1a = M.zero;                  // -dt*D/dx^2;
            M.thomas_constant2 = M.decay_rates;            // (1/3)* dt*lambda
            M.thomas_constant3 = M.one;                    // 1 + 2*constant1 + constant2;
            M.thomas_constant3a = M.one;                   // 1 + constant1 + constant2;

            M.thomas_constant1 *= dt;
            M.thomas_constant1 /= M.mesh.dx;
            M.thomas_constant1 /= M.mesh.dx;

            M.thomas_constant1a = M.thomas_constant1;
            M.thomas_constant1a *= -1.0;

            M.thomas_constant2 *= dt;
            M.thomas_constant2 /= 3.0; // for the LOD splitting of the source, division by 3 is for 3-D

            M.thomas_constant3 += M.thomas_constant1;
            M.thomas_constant3 += M.thomas_constant1;
            M.thomas_constant3 += M.thomas_constant2;

            M.thomas_constant3a += M.thomas_constant1;
            M.thomas_constant3a += M.thomas_constant2;

            // Thomas solver coefficients

            /*--------------------------------------------------------------------*/
            /* In 1-D X decomposition, y and z-lines are contiguous and typically */
            /* the assignments below for y,z should not be changed                */
            /*--------------------------------------------------------------------*/

            M.thomas_cx.assign(M.mesh.x_coordinates.size(), M.thomas_constant1a);    // Fill b and c elements with -D * dt/dx^2
            M.thomas_denomx.assign(M.mesh.x_coordinates.size(), M.thomas_constant3); // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dx^2)

            if (rank == 0)
                M.thomas_denomx[0] = M.thomas_constant3a; // First diagonal element is   (1 + 1/3 * lambda * dt + 1*D*dt/dx^2)

            if (rank == (size - 1))
                M.thomas_denomx[M.mesh.x_coordinates.size() - 1] = M.thomas_constant3a; // Last diagonal element  is   (1 + 1/3 * lambda * dt + 1*D*dt/dx^2)

            if (rank == 0)
                if (M.mesh.x_coordinates.size() == 1) // This is an extreme case, won't exist, still if it does
                {                                     // then this must be at rank 0
                    M.thomas_denomx[0] = M.one;
                    M.thomas_denomx[0] += M.thomas_constant2;
                }
            if (rank == 0)
                M.thomas_cx[0] /= M.thomas_denomx[0]; // The first c element of tridiagonal matrix is div by first diagonal el.

            // axpy(1st, 2nd, 3rd) => 1st = 1st + 2nd * 3rd
            // the value at  size-1 is not actually used
            // Since value of size-1 is not used, it means it is the value after the last Diagonal element
            //cout << "Rank " << rank << endl;
            for (int ser_ctr = 0; ser_ctr <= size - 1; ser_ctr++)
            {
                if (rank == ser_ctr)
                {
                    if (rank == 0 && rank <= size - 1) // If size=1, then this process does not send data
                    {

                        for (int i = 1; i <= M.mesh.x_coordinates.size() - 1; i++)
                        {
                            axpy(&M.thomas_denomx[i], M.thomas_constant1, M.thomas_cx[i - 1]);
                            M.thomas_cx[i] /= M.thomas_denomx[i]; // the value at  size-1 is not actually used
                        }
                    }
                    else
                    {
                        for (int i = 1; i <= M.mesh.x_coordinates.size() - 1; i++)
                        {
                            axpy(&M.thomas_denomx[i], M.thomas_constant1, M.thomas_cx[i - 1]);
                            M.thomas_cx[i] /= M.thomas_denomx[i]; // the value at  size-1 is not actually used
                        }
                    }

                    if (rank < (size - 1))
                    {
                        MPI_Isend(&(M.thomas_cx[M.mesh.x_coordinates.size() - 1][0]), M.thomas_cx[M.mesh.x_coordinates.size() - 1].size(), MPI_DOUBLE, ser_ctr + 1, 1111, mpi_Cart_comm, &send_req);
                        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
                    }
                }

                if (rank == (ser_ctr + 1) && (ser_ctr + 1) <= (size - 1))
                {

                    std::vector<double> temp_cx(M.thomas_cx[0].size());

                    MPI_Irecv(&temp_cx[0], temp_cx.size(), MPI_DOUBLE, ser_ctr, 1111, mpi_Cart_comm, &recv_req);
                    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

                    axpy(&M.thomas_denomx[0], M.thomas_constant1, temp_cx); // CHECK IF &temp_cz[0] is OK, axpy() in BioFVM_vector.cpp
                    M.thomas_cx[0] /= M.thomas_denomx[0];                   // the value at  size-1 is not actually used
                }

                MPI_Barrier(mpi_Cart_comm);
            }
            cout << "Diffusion set up is done" << endl;

            /*--------------------------------------------------------------------*/
            /* In 1-D X decomposition, z and y-lines are contiguous adn typically */
            /* the assignments below for z,y should not be changed                */
            /* Both the first voxel i.e. index 0 and last voxel i.e. index=       */
            /* y_coordinates.size()-1 are on the same process                     */
            /*--------------------------------------------------------------------*/

            M.thomas_cy.assign(M.mesh.y_coordinates.size(), M.thomas_constant1a);
            M.thomas_denomy.assign(M.mesh.y_coordinates.size(), M.thomas_constant3);
            M.thomas_denomy[0] = M.thomas_constant3a;
            M.thomas_denomy[M.mesh.y_coordinates.size() - 1] = M.thomas_constant3a;
            if (M.mesh.y_coordinates.size() == 1)
            {
                M.thomas_denomy[0] = M.one;
                M.thomas_denomy[0] += M.thomas_constant2;
            }
            M.thomas_cy[0] /= M.thomas_denomy[0];
            for (int i = 1; i <= M.mesh.y_coordinates.size() - 1; i++)
            {
                axpy(&M.thomas_denomy[i], M.thomas_constant1, M.thomas_cy[i - 1]);
                M.thomas_cy[i] /= M.thomas_denomy[i]; // the value at  size-1 is not actually used
            }

            M.thomas_cz.assign(M.mesh.z_coordinates.size(), M.thomas_constant1a);
            M.thomas_denomz.assign(M.mesh.z_coordinates.size(), M.thomas_constant3);
            M.thomas_denomz[0] = M.thomas_constant3a;
            M.thomas_denomz[M.mesh.z_coordinates.size() - 1] = M.thomas_constant3a;
            if (M.mesh.z_coordinates.size() == 1)
            {
                M.thomas_denomz[0] = M.one;
                M.thomas_denomz[0] += M.thomas_constant2;
            }
            M.thomas_cz[0] /= M.thomas_denomz[0];
            for (int i = 1; i <= M.mesh.z_coordinates.size() - 1; i++)
            {
                axpy(&M.thomas_denomz[i], M.thomas_constant1, M.thomas_cz[i - 1]);
                M.thomas_cz[i] /= M.thomas_denomz[i]; // the value at  size-1 is not actually used
            }

            M.diffusion_solver_setup_done = true;
            // t_end_set = MPI_Wtime();
            // std::cout<<"Set-up time = "<<(t_end_set-t_strt_set)<<std::endl;
        }

        // x-diffusion
        //cout << "Rank " << rank << " starting x-diffusion" << endl;
        M.apply_dirichlet_conditions();
        //cout << "Rank " << rank << " apply dirichlet condition done" << endl;
        /*-----------------------------------------------------------------------------------*/
        /*                        FORWARD ELIMINATION - x DIRECTION/DECOMPOSITION            */
        /*-----------------------------------------------------------------------------------*/

        /* For data packing...                                                                                 */
        /* My direction of traversing is go up up up i.e. y direction points then go in i.e. Z-direction       */
        /* Remember to visualize 3-D as 2-D plates kept after one another. Hence Z-direction data is farther   */
        /* apart than X/Y direction                                                                            */

        int y_size = M.mesh.y_coordinates.size();
        int z_size = M.mesh.z_coordinates.size();
        int p_size = M.number_of_densities(); // Jose (*M.p_density_vectors)[0].size(); // All p_density_vectors elements have same size, use anyone

        int snd_data_size = z_size * y_size * p_size; // Number of data elements to be sent
        int rcv_data_size = z_size * y_size * p_size; // All p_density_vectors elements have same size, use anyone
        //cout << "Rank " << rank << " snd_data_size: " << snd_data_size << " rcv_data_size: " << rcv_data_size << endl;
        //std::vector<double> snd_data(snd_data_size);
        //std::vector<double> rcv_data(rcv_data_size);

        /* So row is along Z axis, column of each row is along Y-axis and each element has p_density_vector*/

        std::vector<std::vector<double>> block3d(z_size*y_size, std::vector<double>(p_size));

        // t_strt_x = MPI_Wtime();
        //cout << "Rank " << rank << " starting forward" << endl;
        for (int ser_ctr = 0; ser_ctr <= size - 1; ser_ctr++)
        {
            if (rank == ser_ctr)
            {
                if (rank == 0)
                {
                    //cout << "Rank " << rank << "is computing" << endl; 
                    #pragma omp parallel for
                    for (int k = 0; k <= M.mesh.z_coordinates.size() - 1; k++)
                    {
                        for (int j = 0; j <= M.mesh.y_coordinates.size() - 1; j++)
                        {
                            // int n = M.voxel_index(0, j, k);
                            //(*M.p_density_vectors)[n] /= M.thomas_denomx[0];
                            DENSITY(0,j,k) /= M.thomas_denomx[0];
                            for (int i = 1; i < M.mesh.x_coordinates.size(); i++)
                            {
                                axpy(&(DENSITY(i,j,k)), M.thomas_constant1,DENSITY(i-1,j,k), M.number_of_densities());
                                DENSITY(i,j,k) /= M.thomas_denomx[i];
                            }
                        }
                    }
                }
                else
                {
                    //cout << "Rank " << rank << "is computing" << endl; 
                    #pragma omp parallel for
                    for (int k = 0; k <= M.mesh.z_coordinates.size() - 1; k++)
                    {
                        for (int j = 0; j <= M.mesh.y_coordinates.size() - 1; j++)
                        {
                            /*
                            int n = M.voxel_index(0, j, k); // Need to consider case separately for k=0, as k-1 would be -1 !
                            axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, block3d[k][j]);
                            (*M.p_density_vectors)[n] /= M.thomas_denomx[0];*/
            
                            axpy(&(DENSITY(0,j,k)), M.thomas_constant1,
                                 block3d[j*z_size+k], M.number_of_densities());

                                DENSITY(0,j,k)/= M.thomas_denomx[0];
                            

                            for (int i = 1; i < M.mesh.x_coordinates.size(); i++)
                            {
                                // int n = M.voxel_index(i, j, k);
                                //  int n1 = M.voxel_index(i-1,j,k);
                                //  axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                                //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                                
                                axpy(&(DENSITY(i,j,k)), M.thomas_constant1,
                                     DENSITY(i-1,j,k), M.number_of_densities());
                                DENSITY(i,j,k)/= M.thomas_denomx[i];
                            }
                        }
                    }
                }

                if (rank < (size - 1))
                {
                    int x_end = M.mesh.x_coordinates.size() - 1;
                    /*int ctr = 0;

                    for (int k = 0; k <= M.mesh.z_coordinates.size() - 1; k++)
                    {
                        for (int j = 0; j <= M.mesh.y_coordinates.size() - 1; j++)
                        {
                            int n = M.voxel_index(x_end, j, k);
                            for (int ele = 0; ele <= (*M.p_density_vectors)[n].size() - 1; ele++)
                                snd_data[ctr++] = (*M.p_density_vectors)[n][ele];
                        }
                    }
                    MPI_Isend(&snd_data[0], snd_data_size, MPI_DOUBLE, ser_ctr + 1, 1111, mpi_Cart_comm, &send_req);*/
                    //cout << "Rank " << rank << " is sending direction " << M.p_density_vectors[x_end] << endl;
                    //cout << "Rank " << rank << " the direction contain " << (*(M.p_density_vectors[x_end])).size() << " vector " << endl;
                    //cout << "Rank " << rank << " the second vector " << ((*(M.p_density_vectors[x_end]))[0]).size() << " vector " << endl;
                    //cout << "Rank " << rank << " the direction contain " << &(*(M.p_density_vectors[x_end])) << " vector " << endl;
                    //int error = MPI_Isend( M.p_density_vectors[x_end], snd_data_size, MPI_DOUBLE, ser_ctr + 1, 1111, mpi_Cart_comm, &send_req);
                    //int error = MPI_Isend( &aux, 1, MPI_DOUBLE, ser_ctr + 1, 1111, mpi_Cart_comm, &send_req);
                    /*
                    double *buffer = new double[y_size*z_size*p_size];
                    int index = 0; 
                    for (int i = 0; i < y_size*z_size; ++i) {
                        for (int j = 0; j < p_size; ++j) {
                            buffer[index] = (*(M.p_density_vectors[x_end]))[i][j];
                            ++index;
                        }
                    }*/
                    double *buffer = &((*(M.p_density_vectors[x_end]))[0][0]);
                    MPI_Isend( buffer, snd_data_size, MPI_DOUBLE, ser_ctr + 1, 1111, mpi_Cart_comm, &send_req);
                    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
                    //cout << "Rank " << rank << " have send" << endl;
                }
            }
            if (rank == (ser_ctr + 1) && rank <= (size - 1))
            {
                // Receive the data here and try to put in same format as vector of vectors in block3d
                //cout << "Rank " << rank << "is receiving" << endl;
                double *buffer = &(block3d[0][0]); //new double[y_size*z_size*p_size]; //&block3d;              
                MPI_Irecv(buffer, rcv_data_size, MPI_DOUBLE, ser_ctr, 1111, mpi_Cart_comm, &recv_req);

                //MPI_Irecv(&aux, rcv_data_size, MPI_DOUBLE, ser_ctr, 1111, mpi_Cart_comm, &recv_req);
                //MPI_Irecv(&aux, 1, MPI_DOUBLE, ser_ctr, 1111, mpi_Cart_comm, &recv_req);
                MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
                //cout << "Rank " << rank << "has received" << endl; 
                /*
                int index = 0; 
                for (int i = 0; i < y_size*z_size; ++i) {
                    for (int j = 0; j < p_size; ++j) {
                            block3d[i][j] = buffer[index];
                            ++index;
                    }
                }*/
                
                /*
                int ctr = 0;
                for (int m = 0; m < z_size; m++)
                {
                    for (int n = 0; n < y_size; n++)
                    {
                        for (int p = 0; p < p_size; p++)
                        {
                            block3d[m][n][p] = rcv_data[ctr++];
                        }
                    }
                }*/
            }
            //cout << "Rank " << rank << "in the barrier" << endl; 
            MPI_Barrier(mpi_Cart_comm);
        }

        /*-----------------------------------------------------------------------------------*/
        /*                         CODE FOR BACK SUBSITUTION                                 */
        /*-----------------------------------------------------------------------------------*/
        //cout << "Rank " << rank << " starting backward substitution" << endl;
        for (int ser_ctr = size - 1; ser_ctr >= 0; ser_ctr--)
        {
            if (rank == ser_ctr)
            {
                if (rank == (size - 1))
                {
                    #pragma omp parallel for
                    for (int k = M.mesh.z_coordinates.size() - 1; k >= 0; k--)
                    {
                        for (int j = M.mesh.y_coordinates.size() - 1; j >= 0; j--)
                        {
                            //int index = j * z_size * p_size + k * p_size;
                            for (int i = M.mesh.x_coordinates.size() - 2; i >= 0; i--)
                            {
                                // int n = M.voxel_index(i, j, k);
                                //  int n2 = M.voxel_index(i+1,j,k);                                    //can remove overhead of finding index here
                                // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[i], (*M.p_density_vectors)[n + M.thomas_i_jump])
                                naxpy(&(DENSITY(i,j,k)), M.thomas_cx[i],
                                      DENSITY(i+1,j,k), M.number_of_densities());
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for
                    for (int k = M.mesh.z_coordinates.size() - 1; k >= 0; k--)
                    {
                        for (int j = M.mesh.y_coordinates.size() - 1; j >= 0; j--)
                        {
                            //int index = j * z_size * p_size + k * p_size;
                            // int n = M.voxel_index(M.mesh.x_coordinates.size() - 1, j, k);
                            // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[M.mesh.x_coordinates.size() - 1], block3d[k][j]);
                            naxpy(&(DENSITY(M.mesh.x_coordinates.size()-1,j,k)), M.thomas_cx[M.mesh.x_coordinates.size() - 1],
                                  block3d[j*z_size+k], M.number_of_densities());
                            int index = j * z_size * p_size + k * p_size;
                            for (int i = M.mesh.x_coordinates.size() - 2; i >= 0; i--)
                            {
                                // int n = M.voxel_index(i, j, k);
                                //  int n2 = M.voxel_index(i+1,j,k);
                                // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[i], (*M.p_density_vectors)[n + M.thomas_i_jump]);
                                naxpy(&(DENSITY(i,j,k)), M.thomas_cx[i],
                                      DENSITY(i+1,j,k), M.number_of_densities());
                            }
                        }
                    }
                }
                if (rank > 0)
                {
                    int x_start = 0;
                    int ctr = 0;
                    /*
                    for (int k = 0; k <= M.mesh.z_coordinates.size() - 1; k++)
                    {
                        for (int j = 0; j <= M.mesh.y_coordinates.size() - 1; j++)
                        {
                            int n = M.voxel_index(x_start, j, k);
                            for (int ele = 0; ele <= (*M.p_density_vectors)[n].size() - 1; ele++)
                                snd_data[ctr++] = (*M.p_density_vectors)[n][ele];
                        }
                    }

                    MPI_Isend(&snd_data[0], snd_data_size, MPI_DOUBLE, ser_ctr - 1, 1111, mpi_Cart_comm, &send_req);*/
                    /*
                    double *buffer = new double[y_size*z_size*p_size];
                    int index = 0; 
                    for (int i = 0; i < y_size*z_size; ++i) {
                        for (int j = 0; j < p_size; ++j) {
                            buffer[index] = (*(M.p_density_vectors[x_start]))[i][j];
                            ++index;
                        }
                    }*/
                    //cout << "Rank " << rank << " is sending" << endl;
                    double *buffer = &((*(M.p_density_vectors[x_start]))[0][0]);
                    MPI_Isend(buffer, snd_data_size, MPI_DOUBLE, ser_ctr - 1, 1111, mpi_Cart_comm, &send_req);
                    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
                    cout << "Rank " << rank << " has send" << endl;
                }
            }
            if (rank == (ser_ctr - 1) && rank >= 0)
            {
                // Receive the data here and try to put in same format as vector of vectors
                // MPI_Irecv(&rcv_data[0], rcv_data_size, MPI_DOUBLE, ser_ctr, 1111, mpi_Cart_comm, &recv_req);
                //double *buffer = new double[y_size*z_size*p_size]; //&block3d; 
                //cout << "Rank " << rank << " is receiving" << endl;
                double *buffer = &(block3d[0][0]);
                MPI_Irecv(buffer, rcv_data_size, MPI_DOUBLE, ser_ctr, 1111, mpi_Cart_comm, &recv_req);
                MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
                //cout << "Rank " << rank << " has received" << endl;
                /*
                int index = 0; 
                for (int i = 0; i < y_size*z_size; ++i) {
                    for (int j = 0; j < p_size; ++j) {
                            block3d[i][j] = buffer[index];
                            ++index;
                    }
                }*/
                /*
                int ctr = 0;
                for (int m = 0; m < z_size; m++)
                {
                    for (int n = 0; n < y_size; n++)
                    {
                        for (int p = 0; p < p_size; p++)
                        {
                            block3d[m][n][p] = rcv_data[ctr++];
                        }
                    }
                }*/
            }
            MPI_Barrier(mpi_Cart_comm);
        }
        //cout << "Rank " << rank << " finished X diffusion" << endl;
        // t_end_x = MPI_Wtime();
        // std::cout<<"X solve time = "<<(t_end_x-t_strt_x)<<std::endl;

        // y-diffusion
        //cout << "Rank " << rank << " apply dirichlet" << endl;
        M.apply_dirichlet_conditions();

        //cout << "Rank " << rank << " Y diffusion" << endl;
// t_strt_y = MPI_Wtime();
#pragma omp parallel for
        for (int k = 0; k < M.mesh.z_coordinates.size(); k++)
        {
            for (int i = 0; i < M.mesh.x_coordinates.size(); i++)
            {
                // Thomas solver, y-direction

                // remaining part of forward elimination, using pre-computed quantities

                // int n = M.voxel_index(i, 0, k);
                //(*M.p_density_vectors)[n] /= M.thomas_denomy[0];
                
                DENSITY(i,0,k) /= M.thomas_denomy[0];
                

                for (int j = 1; j < M.mesh.y_coordinates.size(); j++)
                {
                    // n = M.voxel_index(i, j, k);
                    //index = j * z_size * p_size + k * p_size;
                    // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_j_jump]);
                    axpy(&(DENSITY(i,j,k)),  M.thomas_constant1,
                                      DENSITY(i,j-1,k), M.number_of_densities());
                    //(*M.p_density_vectors)[n] /= M.thomas_denomy[j];
                    DENSITY(i,j,k) /= M.thomas_denomy[j];
                }

                // back substitution
                // n = voxel_index( mesh.x_coordinates.size()-2 ,j,k);

                for (int j = M.mesh.y_coordinates.size() - 2; j >= 0; j--)
                {
                    // n = M.voxel_index(i, j, k);
                    // naxpy(&(*M.p_density_vectors)[n], M.thomas_cy[j], (*M.p_density_vectors)[n + M.thomas_j_jump]);
                    //index = j * z_size * p_size + k * p_size;
                   naxpy(&(DENSITY(i,j,k)),   M.thomas_cy[j],
                                      DENSITY(i,j+1,k), M.number_of_densities());
                }
            }
        }
        // t_end_y = MPI_Wtime();
        // std::cout<<"Y solve time = "<<(t_end_y-t_strt_y)<<std::endl;

        // z-diffusion
        /*--------------------------------------------------------------------------------*/
        /* This will change. Why is the Dirichlet condition being applied again and again?*/
        /* This is where serialization will begin, I think after the apply_dirichlet()    */
        /*                                                                                */
        /*--------------------------------------------------------------------------------*/

        //cout << "Rank " << rank << " apply dirichlet" << endl;
        M.apply_dirichlet_conditions();

        /*------------------------------------------------------------------------------*/
        /* PROCESSING OF 0TH ELEMENT IN ARRAY                                           */
        /* The processing on the 0th element of array will be done                      */
        /* only on rank = 0, or basically the processes containing                      */
        /* the front boundary of the domain.                                            */
        /* MPI divides domain into thick slices in Z-direction                          */
        /* OpenMP divides the thick slices in vertical direction i.e. horizontal lines. */
        /*------------------------------------------------------------------------------*/

        /*------------------------------------------------------------------------------*/
        /* FORWARD SUBSITUTION AND DATA SENDING OF BACK PLANE                           */
        /* Let all the processing finish at rank 0, now we need to send  the 'back'     */
        /* square of p_density_vectors[] to rank 1. Do the same at rank 2 so on...      */
        /* Possibly the thomas_k_jump will also change                                  */
        /* Instead of using thomas_k_jump, we can use prev=voxel_index(i,j,k-1)         */
        /* and then use it in (*M.p_density_vectors)[prev]                              */
        /* Also for rank > 0, it needs to start at k=0                                  */
        /* The values of p_density_vectors sent from the previous neighbour will be     */
        /* stored in temp_p_density_vectors -- only used for k=1 else just use          */
        /* p_density_vectors. Vector of vectors is not stored contiguously              */
        /* So we need to pack them into a single array and then send it to other end    */
        /*------------------------------------------------------------------------------*/

        /*---------------------------------------------------------------------------------*/
        /* BACK SUBSTITUTION AND DATA SENDING OF FRONT PLANE                               */
        /* This starts from the last process i.e. rank = size-1 then proceeds backwards    */
        /* i.e. from back boundary to front boundary. On the last process k starts from    */
        /* z_coordinates.size()-2 but on other processes, it starts from                   */
        /* z_coordinates.size()-1.                                                         */
        /*---------------------------------------------------------------------------------*/

        /*------------------COMMENT OUT FROM HERE-----------------------------------------*/
        /* So the code is like M.apply_dirichlet_conditions() then code for Z-diffusion   */
        /* Then M.apply_dirichlet_conditions();                                           */
        /*--------------------------------------------------------------------------------*/

        /*----------------------------TILL HERE----------------------------------------------*/

        /*----------------------------------------------------------------------------------------------------------*/
        /*Declaring a matrix with z_size rows, y_size columns and each element of matrix is a vector of size p_size.*/
        /*First will store received data from rcv_data into this 3-d block because we need to pass base address     */
        /*of vectors in the routine axpy()                                                                          */
        /*----------------------------------------------------------------------------------------------------------*/

        // t_strt_z = MPI_Wtime();

        //cout << "Rank " << rank << " Z diffusion" << endl;
#pragma omp parallel for
        for (int j = 0; j < M.mesh.y_coordinates.size(); j++)
        {

            for (int i = 0; i < M.mesh.x_coordinates.size(); i++)
            {

                // remaining part of forward elimination, using pre-computed quantities
                // int n = M.voxel_index(i, j, 0);
                //(*M.p_density_vectors)[n] /= M.thomas_denomz[0];
                //int index = j * z_size * p_size;
                DENSITY(i,j,0) /= M.thomas_denomz[0];
                

                // should be an empty loop if mesh.z_coordinates.size() < 2
                for (int k = 1; k < M.mesh.z_coordinates.size(); k++)
                {
                   // n = M.voxel_index(i, j, k);
                   // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_k_jump]);
                   //(*M.p_density_vectors)[n] /= M.thomas_denomz[k];
                   //index = j * z_size * p_size + k * p_size;
                    axpy(&(DENSITY(i,j,k)),  M.thomas_constant1,
                                      DENSITY(i,j,k-1), M.number_of_densities());
                    DENSITY(i,j,k) /= M.thomas_denomz[k];
                }

                // for parallelization need to break forward elimination and back substitution into
                // should be an empty loop if mesh.z_coordinates.size() < 2

                for (int k = M.mesh.z_coordinates.size() - 2; k >= 0; k--)
                {
                    // n = M.voxel_index(i, j, k);
                    // naxpy(&(*M.p_density_vectors)[n], M.thomas_cz[k], (*M.p_density_vectors)[n + M.thomas_k_jump]);
                    //index = j * z_size * p_size + k * p_size;
                    naxpy(&(DENSITY(i,j,k)),   M.thomas_cz[k],
                                      DENSITY(i,j,k+1), M.number_of_densities());
                }
            }
        }

        // t_end_z = MPI_Wtime();
        // std::cout<<"Z solve time = "<<(t_end_z-t_strt_z)<<std::endl;
        //cout << "Rank " << rank << " apply dirichlet" << endl;
        M.apply_dirichlet_conditions();

        // reset gradient vectors
        //	M.reset_all_gradient_vectors();
        //cout << "Rank " << rank << " have finished the diffusion" << endl;
        return;
    }

    void microenvironment::diffusion_decay() {
    
        MPI_Request send_req[granurality], recv_req[granurality];
        double block3d[i_jump]; //Aux structure of the size: Y*Z*Substrates
        /*
        string path  = "./timing/voxels_" + std::to_string((int)cube_side/2.0) + "/substrates_" + std::to_string(number_of_densities) 
        + "/factor_" + std::to_string(factor) +  "/" + std::to_string(mpi_size) + "_node.csv";*/
        std::ofstream file(timing_csv, std::ios::app);
        int vl = 4;
        auto start_time = std::chrono::high_resolution_clock::now();
        apply_dirichlet_conditions();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto apply_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        //cout << "Peto aqui!" << endl;
        start_time = std::chrono::high_resolution_clock::now();
        //X-diffusion vector_x_v2
        //start_time = std::chrono::high_resolution_clock::now();
        if (mpi_rank == 0)
            {
                for (int step = 0; step < granurality; ++step)
                {
                    int initial_index = step * snd_data_size;
                    int limit = (initial_index + snd_data_size);
                    int limit_vec = limit -(snd_data_size%vl);
                    #pragma omp parallel for
                    for (int index = initial_index; index < limit_vec; index += vl)
                    {
                        int index_dec = index;
                        int gd = index%gvec_size;

                        __m256d denomx1 = _mm256_loadu_pd(&gthomas_denomx[0][gd]);
                        __m256d density1 = _mm256_loadu_pd(&densities[index]);
                        __m256d aux1 = _mm256_div_pd(density1, denomx1);

                        _mm256_storeu_pd(&densities[index], aux1);

                        for (int i = 1; i < x_size; i++)
                        {
                            int index_inc = index_dec + i_jump;
                            __m256d constant1 = _mm256_loadu_pd(&gthomas_constant1[gd]);
                            __m256d density_curr1 = _mm256_loadu_pd(&densities[index_dec ]);
                            __m256d density_inc1 = _mm256_loadu_pd(&densities[index_inc]);
                            __m256d denomy1 = _mm256_loadu_pd(&gthomas_denomx[i][gd]);
                        
                            density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
                    
                            density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                            _mm256_storeu_pd(&densities[index_inc], density_curr1);
                            
                            index_dec = index_inc;
                        }
                    }

                    
                    //Epilogo vectorization
                    //int ep =limit - limit_vec;
                    //int ep_it = ep / number_of_densities + (ep > 0);
                    for (int index = limit_vec; index < limit; ++index)
                    {
                        int index_dec = index;
                        int d = index % number_of_densities; 

                        densities[index] /= thomas_denomx[0][d];

                        for (int i = 1; i < x_size; i++)
                        {
                            int index_inc = index_dec + i_jump;
                            // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
                            densities[index_inc] += thomas_constant1[d] * densities[index_dec];
                            
                            //(*M.microenvironment)[n] /= M.thomas_denomx[i];
                            
                            densities[index_inc] /= thomas_denomx[i][d];
                            
                            index_dec = index_inc;
                        }
                    }

                    if (mpi_size > 1) {
                        int x_end = x_size - 1;
                        int offset = step * snd_data_size;
                        MPI_Status status;
                        MPI_Isend(&(densities[x_end * i_jump + offset]), snd_data_size, MPI_DOUBLE, mpi_rank + 1, step, MPI_COMM_WORLD, &send_req[step]);
                    }
                }
                //Last iteration
                if (snd_data_size_last != 0) {
                    int initial_index = granurality * snd_data_size;
                    #pragma omp parallel for
                    for (int index = initial_index; index < initial_index + snd_data_size_last; index += number_of_densities)
                    {
                        int index_dec = index; 
                        for (int d = 0; d < number_of_densities; d++)
                        {
                            densities[index + d] /= thomas_denomx[0][d];
                        }

                        for (int i = 1; i < x_size; i++)
                        {
                            int index_inc = index_dec + i_jump;
                            // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
                            for (int d = 0; d < number_of_densities; d++)
                            {
                                densities[index_inc + d] += thomas_constant1[d] * densities[index_dec + d];
                            }

                            //(*M.microenvironment)[n] /= M.thomas_denomx[i];
                            for (int d = 0; d < number_of_densities; d++)
                            {
                                densities[index_inc + d] /= thomas_denomx[i][d];
                            }
                            index_dec = index_inc;
                        }
                    }

                    if (mpi_size > 1) {
                        int x_end = x_size - 1;
                        int offset = granurality * snd_data_size;
                        MPI_Status status;
                        MPI_Isend(&(densities[x_end * i_jump + offset]), snd_data_size_last, MPI_DOUBLE, mpi_rank + 1, granurality, MPI_COMM_WORLD, &send_req[granurality]);
                        
                    }
                }
            }
            else
            {
                if (mpi_rank >= 1 && mpi_rank <= (mpi_size - 1))
                {
                    for (int step = 0; step < granurality; ++step)
                    {
                        int initial_index = step * snd_data_size;
                        MPI_Irecv(&(block3d[initial_index]), rcv_data_size, MPI_DOUBLE, mpi_rank-1, step, MPI_COMM_WORLD, &(recv_req[step]));
                    }
                    if (snd_data_size_last != 0)
                        MPI_Irecv(&(block3d[granurality*snd_data_size]), rcv_data_size_last, MPI_DOUBLE, mpi_rank-1, granurality, MPI_COMM_WORLD, &(recv_req[granurality]));
                    for (int step = 0; step < granurality; ++step)
                    {
                        int initial_index = step * snd_data_size;
                        int limit = (initial_index + snd_data_size);
                        int limit_vec = limit - (snd_data_size%vl);
                        MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                        #pragma omp parallel for
                        for (int index = initial_index; index < limit; index += vl)
                        {
                            // axpy(&(*M.microenvironment)[n], M.thomas_constant1, block3d[k][j]);
                            int index_dec = index;
                            int gd = index%gvec_size;
                            __m256d constant1 = _mm256_loadu_pd(&gthomas_constant1[gd]);
                            __m256d density_curr1 = _mm256_loadu_pd(&densities[index]);
                            __m256d density_inc1 = _mm256_loadu_pd(&block3d[index]);
                            __m256d denomy1 = _mm256_loadu_pd(&gthomas_denomx[0][gd]);
                    

                            density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
                
                            //_mm512_storeu_pd(&microenvironment[index_inc + zd], density_curr);
                
                            //(*M.p_density_vectors)[n] /= M.thomas_denomy[j];
                            //Fer unrolling aqui
                    
                            //__m512d density = _mm512_loadu_pd(&microenvironment[index_inc + zd]);
                            density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                            _mm256_storeu_pd(&densities[index], density_curr1);

                            for (int i = 1; i < x_size; i++)
                            {

                                int index_inc = index_dec + i_jump;
                                __m256d constant1 = _mm256_loadu_pd(&gthomas_constant1[gd]);
                                __m256d density_curr1 = _mm256_loadu_pd(&densities[index_dec ]);
                                __m256d density_inc1 = _mm256_loadu_pd(&densities[index_inc]);
                                __m256d denomy1 = _mm256_loadu_pd(&gthomas_denomx[i][gd]);
                        

                                density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
                    
                                //_mm512_storeu_pd(&microenvironment[index_inc + zd], density_curr);
                    
                                //(*M.p_density_vectors)[n] /= M.thomas_denomy[j];
                                //Fer unrolling aqui
                    
                                //__m512d density = _mm512_loadu_pd(&microenvironment[index_inc + zd]);
                                density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                                _mm256_storeu_pd(&densities[index_inc], density_curr1);

                                index_dec = index_inc;
                            }
                            
                        }
                        //Epilogo vectorizacion
                        /*
                        int ep = snd_data_size - limit;
                        int index = limit;
                        int d_ini = (ep%number_of_densities);
                        int ep_it = ep / number_of_densities + (ep > 0);
                        */
                        for (int index = limit_vec; index < limit; ++index)
                        {
                            int index_dec = index;
                            int d = index % number_of_densities; 
                            
                            densities[index] += thomas_constant1[d] * block3d[index];
                            densities[index] /= thomas_denomx[0][d];
                            
                            for (int i = 1; i < x_size; i++)
                            {
                                int index_inc = index_dec + i_jump;
                                // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
                            
                                densities[index_inc] += thomas_constant1[d] * densities[index_dec];
                            
                                //(*M.microenvironment)[n] /= M.thomas_denomx[i];
                                densities[index_inc] /= thomas_denomx[i][d];
                                index_dec = index_inc;
                            }
                        }


                        if (mpi_rank < (mpi_size - 1))
                        {
                            int x_end = x_size - 1;
                            MPI_Isend(&(densities[x_end * i_jump + initial_index]), snd_data_size, MPI_DOUBLE, mpi_rank + 1, step, MPI_COMM_WORLD, &send_req[step]);
                        }
                    }
                    if (snd_data_size_last != 0)
                    {
                        int initial_index = granurality * snd_data_size;
                        MPI_Wait(&recv_req[granurality], MPI_STATUS_IGNORE); //Need to change
                        #pragma omp parallel for
                        for (int index = initial_index; index < initial_index + snd_data_size_last; index += number_of_densities)
                        {
                            // axpy(&(*M.microenvironment)[n], M.thomas_constant1, block3d[k][j]);
                            int index_dec = index;
                            for (int d = 0; d < k_jump; d++)
                            {
                                densities[index + d] += thomas_constant1[d] * block3d[index + d];
                            }

                            //(*M.microenvironment)[n] /= M.thomas_denomx[0];
                            for (int d = 0; d < number_of_densities; d++)
                            {
                                densities[index + d] /= thomas_denomx[0][d];
                            }

                            for (int i = 1; i < x_size; i++)
                            {

                                int index_inc = index_dec + i_jump;
                                // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
                                for (int d = 0; d < number_of_densities; d++)
                                {
                                    densities[index_inc + d] += thomas_constant1[d] * densities[index_dec + d];
                                }
                                //(*M.microenvironment)[n] /= M.thomas_denomx[i];
                                for (int d = 0; d < number_of_densities; d++)
                                {
                                    densities[index_inc + d] /= thomas_denomx[i][d];
                                }

                                index_dec = index_inc;
                            }
                            
                        }
                        // End of computation region
                        if (mpi_rank < (mpi_size - 1))
                        {
                            int x_end = x_size - 1;
                            MPI_Request aux;
                            MPI_Isend(&(densities[x_end * i_jump + initial_index]), snd_data_size_last, MPI_DOUBLE, mpi_rank + 1, granurality, MPI_COMM_WORLD, &send_req[granurality]);
                        
                        }
                    }
                }
            }
            
            /*-----------------------------------------------------------------------------------*/
            /*                         CODE FOR BACK SUBSITUTION                                 */
            /*-----------------------------------------------------------------------------------*/
            
            //cout << "Rank " << rank << " starting backward substitution" << endl;

            if (mpi_rank == (mpi_size - 1))
            {
                for (int step = 0; step < granurality; ++step)
                {
                    int last_xplane = ((x_size - 1)*i_jump);
                    int initial_index = last_xplane + (step * snd_data_size);
                    int limit = initial_index + snd_data_size;
                    int limit_vec = limit - (snd_data_size%vl);
                    #pragma omp parallel for 
                    for (int index = initial_index; index < limit_vec; index += vl)
                    {
                        int index_aux = index;
                        int gd = (index - last_xplane)%gvec_size;
                        for (int i = x_size - 2; i >= 0; i--)
                        {

                            int index_dec = index_aux - i_jump;
                            __m256d cy1 = _mm256_loadu_pd(&gthomas_cx[i][gd]);
                            __m256d density_curr1 = _mm256_loadu_pd(&densities[index_aux]);
                            __m256d density_dec1 = _mm256_loadu_pd(&densities[index_dec]);

                            density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                            _mm256_storeu_pd(&densities[index_dec], density_curr1);
                            index_aux = index_dec;
                        }
                    }

                    //Epilogo Vectorizacion Back Last rank
                    
                    for (int index = limit_vec; index < limit; ++index){
                        int index_aux = index;
                        int d = (index - last_xplane) % number_of_densities;
                        for (int i = x_size - 2; i >= 0; i--)
                        {
                            int index_dec = index_aux - i_jump;
                            // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
                            densities[index_dec] -= thomas_cx[i][d] * densities[index_aux];
                            
                            index_aux = index_dec;
                        }
                    }

                    if (mpi_size > 1) {
                        MPI_Request aux;
                        MPI_Isend(&(densities[step * snd_data_size]), snd_data_size, MPI_DOUBLE, mpi_rank - 1, step, MPI_COMM_WORLD, &send_req[step]);
                    }
                }

                //Last iteration
                if (snd_data_size_last != 0) {
                    int initial_index = ((x_size - 1)*i_jump) + (granurality * snd_data_size);
                    #pragma omp parallel for
                    for (int index = initial_index; index < initial_index + snd_data_size_last; index += number_of_densities)
                    {
                        int index_aux = index;
                        for (int i = x_size - 2; i >= 0; i--)
                        {

                            int index_dec = index_aux - i_jump;
                            // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
                            for (int d = 0; d < k_jump; d++)
                            {
                                densities[index_dec + d] -= thomas_cx[i][d] * densities[index_aux + d];
                            }
                            index_aux = index_dec;
                        }
                    }
                    if (mpi_size > 1) {
                        MPI_Request aux;
                        MPI_Isend(&(densities[granurality * snd_data_size]), snd_data_size_last, MPI_DOUBLE, mpi_rank - 1, granurality, MPI_COMM_WORLD, &send_req[granurality]);
                        //cout << "Rank " << rank << " has send" << endl;
                    }
                
                }
            }
            else
            {
                for (int step = 0; step < granurality; ++step)
                {
                    MPI_Irecv(&(block3d[step*snd_data_size]), rcv_data_size, MPI_DOUBLE, mpi_rank+1, step, MPI_COMM_WORLD, &recv_req[step]);
                }
                if (snd_data_size_last != 0)
                    MPI_Irecv(&(block3d[granurality*snd_data_size]), rcv_data_size_last, MPI_DOUBLE, mpi_rank+1, granurality, MPI_COMM_WORLD, &recv_req[granurality]);

                
                for (int step = 0; step < granurality; ++step)
                {
                    int last_xplane = ((x_size - 1)*i_jump);
                    int initial_index = last_xplane + (step * snd_data_size);
                    int limit = initial_index + snd_data_size;
                    int limit_vec = limit - (snd_data_size%vl);
                    MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                    #pragma omp parallel for
                    for (int index = initial_index; index < limit_vec; index += vl)
                    {
                        int index_aux = index;
                        int index_3d = index - last_xplane;
                        int gd = index_3d%gvec_size;
                        __m256d cy1 = _mm256_loadu_pd(&gthomas_cx[x_size-1][gd]);
                        __m256d density_curr1 = _mm256_loadu_pd(&densities[index_aux]);
                        __m256d density_dec1 = _mm256_loadu_pd(&block3d[index_3d]);

                        density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                        _mm256_storeu_pd(&densities[index_aux], density_curr1);

                        for (int i = x_size - 2; i >= 0; i--)
                        {

                            int index_dec = index_aux - i_jump;
                            __m256d cy1 = _mm256_loadu_pd(&gthomas_cx[i][gd]);
                            __m256d density_curr1 = _mm256_loadu_pd(&densities[index_aux]);
                            __m256d density_dec1 = _mm256_loadu_pd(&densities[index_dec]);

                            density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                            _mm256_storeu_pd(&densities[index_dec], density_curr1);
                            index_aux = index_dec;
                        }
                    }
                    
                    //Epilogo Vectorizacion
                    for (int index = limit_vec; index < limit; ++index){
                        int index_aux = index;
                        int index_3d = index - last_xplane;
                        int d = (index - last_xplane) % number_of_densities;

                        densities[index_aux] -= thomas_cx[x_size - 1][d] * block3d[index_3d];


                        for (int i = x_size - 2; i >= 0; i--)
                        {
                            int index_dec = index_aux - i_jump;
                            // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
                            densities[index_dec] -= thomas_cx[i][d] * densities[index_aux];
                            
                            index_aux = index_dec;
                        }
                    }

                    if (mpi_rank > 0)
                    {
                        MPI_Request aux;
                        MPI_Isend(&(densities[step * snd_data_size]), snd_data_size, MPI_DOUBLE, mpi_rank - 1, step, MPI_COMM_WORLD, &send_req[step]);
                        // cout << "Rank " << rank << " has send" << endl;
                    }
                }
                if (snd_data_size_last != 0)
                {
                    int initial_index = ((x_size - 1)*i_jump) + (granurality * snd_data_size);
                    int index_3d_initial = (granurality * snd_data_size);
                    MPI_Wait(&recv_req[granurality], MPI_STATUS_IGNORE);
                    #pragma omp parallel for
                    for (int offset = 0; offset < snd_data_size_last; offset += number_of_densities)
                    {
                        int index_aux = initial_index + offset;
                        //int index = j * M.j_jump + k * M.k_jump + (M.mesh.x_coordinates.size() - 1) * M.i_jump;
                        int index_3d = index_3d_initial + offset;
                        // naxpy(&(*M.microenvironment)[n], M.thomas_cx[M.mesh.x_coordinates.size() - 1], block3d[k][j]);
                        for (int d = 0; d < k_jump; d++)
                        {
                            densities[index_aux + d] -= thomas_cx[x_size - 1][d] * block3d[index_3d + d];
                        }

                        for (int i = x_size - 2; i >= 0; i--)
                        {

                            int index_dec = index_aux - i_jump;
                            // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
                            for (int d = 0; d < k_jump; d++)
                            {
                                densities[index_dec + d] -= thomas_cx[i][d] * densities[index_aux + d];
                            }
                            index_aux = index_dec;
                            
                        }
                    }
                    if (mpi_rank > 0)
                    {
                        MPI_Request aux;
                        MPI_Isend(&(densities[granurality * snd_data_size]), snd_data_size_last, MPI_DOUBLE, mpi_rank - 1, granurality, MPI_COMM_WORLD, &send_req[granurality]);
                    }
                }
            }
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        //cout << "Peto aqui2!" << endl;

        if (mpi_rank == 0)
            file << duration_us << ",";
        start_time = std::chrono::high_resolution_clock::now();
        apply_dirichlet_conditions();
        end_time = std::chrono::high_resolution_clock::now();
        apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        //Y-diffusion
        start_time = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < x_size; i++)
        {
            //Forward Elimination
            //J = 0
            int gd = 0; //Density pointer
            int index = i * i_jump;
            int zd;
            for (zd = 0; zd + vl < j_jump; zd+=vl)
            {
                __m256d denomy1 = _mm256_loadu_pd(&gthomas_denomy[0][gd]);
                __m256d density1 = _mm256_loadu_pd(&densities[index + zd]);
                gd+=vl;
                if (gd == gvec_size) gd = 0;
                __m256d aux1 = _mm256_div_pd(density1, denomy1);
                _mm256_storeu_pd(&densities[index + zd], aux1);
            }
            //Epilogo
            int ep = j_jump - zd;
            int z_ini = zd / number_of_densities;
            int d_ini = zd % number_of_densities;
            index = index + z_ini * k_jump;
            for (int k = z_ini; k < z_size; ++k)
            {
                for (int d = d_ini; d < number_of_densities; ++d){
                    d_ini = 0;
                    densities[index + d] /= thomas_denomy[0][d];
                }
                index+=k_jump;
            }
            //J = 1..(y_size-1)
            for (int j = 1; j < y_size; j++)
            {
                int index_base = i * i_jump +  (j-1)*j_jump;
                int index_inc =  index_base + j_jump;
                int zd;
                gd = 0;
                for (zd = 0; zd + vl < j_jump; zd+=vl)
                {
                    __m256d constant1 = _mm256_loadu_pd(&gthomas_constant1[gd]);
                    __m256d density_curr1 = _mm256_loadu_pd(&densities[index_base + zd]);
                    __m256d density_inc1 = _mm256_loadu_pd(&densities[index_inc + zd]);
                    __m256d denomy1 = _mm256_loadu_pd(&gthomas_denomy[j][gd]);
                    gd+=vl;
                    if (gd == gvec_size) gd = 0;
                    density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
                    density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                    _mm256_storeu_pd(&densities[index_inc + zd], density_curr1);
                }
                //Epilogo
                int ep = j_jump - zd;
                int z_ini = zd / number_of_densities;
                int d_ini = zd % number_of_densities;
                index_base = index_base + z_ini * k_jump;
                index_inc = index_inc + z_ini * k_jump;
                for (int k = z_ini; k < z_size; ++k)
                {
                    for (int d = d_ini; d < number_of_densities; ++d){
                        d_ini = 0;
                        densities[index_inc + d] += thomas_constant1[d] * densities[index_base + d];
                        densities[index_inc + d] /= thomas_denomy[j][d];
                    }
                    index_base+=k_jump;
                    index_inc+=k_jump;
                }
            }
            // Back substitution
            for (int j = y_size - 2; j >= 0; j--)
            {
                int index_base = i * i_jump + (j+1) * j_jump;
                int index_dec = index_base - j_jump;
                int zd;
                gd = 0;
                for ( zd = 0; zd + vl < j_jump; zd+=vl)
                {
                    __m256d cy1 = _mm256_loadu_pd(&gthomas_cy[j][gd]);
                    __m256d density_curr1 = _mm256_loadu_pd(&densities[index_base + zd]);
                    __m256d density_dec1 = _mm256_loadu_pd(&densities[index_dec+ zd]);
                    gd+=vl;
                    if (gd == gvec_size) gd = 0;

                    density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                    _mm256_storeu_pd(&densities[index_dec + zd], density_curr1);
                    
                }

                //Epilogo
                int ep = j_jump - zd;
                int z_ini = zd / number_of_densities;
                int d_ini = zd % number_of_densities;
                index_base = index_base + z_ini * k_jump;
                index_dec = index_dec + z_ini * k_jump;
                for (int k = z_ini; k < z_size; ++k)
                {
                    for (int d = d_ini; d < number_of_densities; ++d){
                        d_ini = 0;
                        densities[index_dec + d] -= thomas_cy[j][d] * densities[index_base + d];
                    }
                    index_base+=k_jump;
                    index_dec+=k_jump;
                }

            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (mpi_rank == 0)
            file << duration_us << ",";
        apply_dirichlet_conditions();
        end_time = std::chrono::high_resolution_clock::now();
        apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        //Z-diffusion
        //cout << "Peto aqui 3!" << endl;

        start_time = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < x_size; i++)
            {
            for (int j = 0; j < y_size; j++)
                {
                int index = i * i_jump + j * j_jump;
                //(*M.p_density_vectors)[n] /= M.thomas_denomz[0];
                for (int d = 0; d < number_of_densities; d++)
                {
                    densities[index + d] /= thomas_denomz[0][d];
                }

                // should be an empty loop if mesh.z_coordinates.size() < 2
                for (int k = 1; k < z_size; k++)
                {

                    int index_inc = index + k_jump;
                    // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_k_jump]);
                    for (int d = 0; d < number_of_densities; d++)
                    {
                        densities[index_inc + d] += densities[d] * densities[index + d];
                    }
                    //(*M.p_density_vectors)[n] /= M.thomas_denomz[k];
                    for (int d = 0; d < number_of_densities; d++)
                    {
                        densities[index_inc + d] /= thomas_denomz[k][d];
                    }

                    index = index_inc;
                }

                index = i * i_jump + j * j_jump + (k_jump * (z_size - 1));
                for (int k = z_size - 2; k >= 0; k--)
                {

                    int index_dec = index - k_jump;
                    // naxpy(&(*M.p_density_vectors)[n], M.thomas_cz[k], (*M.p_density_vectors)[n + M.thomas_k_jump]);
                    for (int d = 0; d < number_of_densities; d++)
                    {
                        densities[index_dec + d] -= thomas_cz[k][d] * densities[index + d];
                    }
                    index = index_dec;
                }
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (mpi_rank == 0)
            file << duration_us << ",";
        
        start_time = std::chrono::high_resolution_clock::now();
        apply_dirichlet_conditions();
        end_time = std::chrono::high_resolution_clock::now();
        apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (mpi_rank == 0)
                file << apply_us/4.0 << std::endl;
    }
}