#include "BioFVM_microenvironment.h"
#include "BioFVM_mesh.h"
#include "BioFVM_basic_agent.h"
#include <cmath>
#include <chrono>
#include <thread>
#include <immintrin.h>

namespace BioFVM
{

    /*-------------------------------------------------------------------------------------------*/
    /* This is a dummy function which calls a version of resize_space(...) that calls a function */
    /* which divides the x-voxels among processes. No division of y and z voxels occurs. 				 */
    /*-------------------------------------------------------------------------------------------*/

    void Microenvironment::resize_space_uniform(double x_start, double x_end, double y_start, double y_end, double z_start, double z_end, double dx_new, int *dims, int *coords)
    {
        return resize_space(x_start, x_end, y_start, y_end, z_start, z_end, dx_new, dx_new, dx_new, dims, coords);
    }

    /*------------------------------------------------------------------------------------------------*/
    /* This function calls the resize(...) function of the Mesh class which is responsible for domain */
    /* partitioning in the x-direction. In addition, it initializes several other useful vectors. 		*/
    /*------------------------------------------------------------------------------------------------*/

    // Jose
    void Microenvironment::resize_space(double x_start, double x_end, double y_start, double y_end, double z_start, double z_end, double dx_new, double dy_new, double dz_new, int *dims, int *coords)
    {
        mesh.resize(x_start, x_end, y_start, y_end, z_start, z_end, dx_new, dy_new, dz_new, dims, coords);

        // temporary_density_vectors1.assign(mesh.voxels.size(), zero);
        // temporary_density_vectors2.assign(mesh.voxels.size(), zero);
        /*
        gradient_vectors.resize(mesh.voxels.size());
        for (int k = 0; k < mesh.voxels.size(); k++)
        {
            gradient_vectors[k].resize(number_of_densities());
            for (int i = 0; i < number_of_densities(); i++)
            {
                (gradient_vectors[k])[i].resize(3, 0.0);
            }
        }
        gradient_vector_computed.resize(mesh.voxels.size(), false);
        */
        int box_elements = mesh.x_size * mesh.y_size * mesh.z_size * number_of_densities();
        dirichlet_value_vectors.assign(box_elements, 100.0);
        p_density_vectors.resize(box_elements);
        return;
    }

    /*-------------------------------------------------------------------------------------------*/
    /* This is the main function which performs the Domain Partitioning in the x-direction 			 */
    /* It also assings the x,y and z-coordinates to each voxel and makes a list of neighbouring  */
    /* voxels in the x,y and z-directions. There are 2 types of indices : Local and Global 			 */
    /* The voxels at the boundary of a sub-domain only have the global index of the neighbouring */
    /* voxel in the adjacent process (assuming its not the last process)												 */
    /*-------------------------------------------------------------------------------------------*/



    void Cartesian_Mesh::resize(double x_start, double x_end, double y_start, double y_end, double z_start, double z_end, double dx_new, double dy_new, double dz_new, int *dims, int *coords)
    {
        /*------------------------------------------------------------------------------------------------*/
        /*		local_x/y/z_start give local starting coordinates at each process	                      */
        /*------------------------------------------------------------------------------------------------*/
        /*
        double local_x_start;
        double local_y_start;
        double local_z_start;*/

        /*------------------------------------------------------------------------------------------------*/
        /*		To find global_mesh_index, we declare following	                                          */
        /*------------------------------------------------------------------------------------------------*/

        int x_index;
        int y_index;
        int z_index;
        int local_start_of_global_index;

        dx = dx_new;
        dy = dy_new;
        dz = dz_new;

        double eps = 1e-16;

        /*--------------------------------------------*/
        /*		Global Nodes	                      */
        /*--------------------------------------------*/

        int x_nodes = (int)ceil(eps + (x_end - x_start) / dx);
        int y_nodes = (int)ceil(eps + (y_end - y_start) / dy);
        int z_nodes = (int)ceil(eps + (z_end - z_start) / dz);

        /*--------------------------------------------*/
        /*		Local Nodes on MPI Processes	      */
        /*--------------------------------------------*/

        int local_x_nodes = x_nodes / dims[1];
        int local_y_nodes = y_nodes / dims[0];
        int local_z_nodes = z_nodes / dims[2];

        // Assign the size of each dimension
        x_size = local_x_nodes;
        y_size = local_y_nodes;
        z_size = local_z_nodes;

        x_coordinates.assign(local_x_nodes, 0.0);
        y_coordinates.assign(local_y_nodes, 0.0);
        z_coordinates.assign(local_z_nodes, 0.0);

        uniform_mesh = true;
        regular_mesh = true;
        double tol = 1e-16;
        if (fabs(dx - dy) > tol || fabs(dy - dz) > tol || fabs(dx - dz) > tol)
        {
            uniform_mesh = false;
        }

        local_x_start = x_start + (coords[1] * local_x_nodes * dx);
        for (int i = 0; i < x_coordinates.size(); i++)
        {
            x_coordinates[i] = local_x_start + (i + 0.5) * dx;
        }

        local_y_start = y_start + ((dims[0] - coords[0] - 1) * local_y_nodes * dy);
        for (int i = 0; i < y_coordinates.size(); i++)
        {
            y_coordinates[i] = local_y_start + (i + 0.5) * dy;
        }

        local_z_start = z_start + (coords[2] * local_z_nodes * dz);
        for (int i = 0; i < z_coordinates.size(); i++)
        {
            z_coordinates[i] = local_z_start + (i + 0.5) * dz;
        }

        /*--------------------------------------------*/
        /*		Global bounding Box	                  */
        /*--------------------------------------------*/

        bounding_box[0] = x_start;
        bounding_box[3] = x_end;
        bounding_box[1] = y_start;
        bounding_box[4] = y_end;
        bounding_box[2] = z_start;
        bounding_box[5] = z_end;

        dV = dx * dy * dz;
        dS = dx * dy;

        dS_xy = dx * dy;
        dS_yz = dy * dz;
        dS_xz = dx * dz;

        /*--------------------------------------------------------------------------------------------------------------------*/
        /*		Creates a Voxel defined in BioFVM_mesh.cpp to a default Voxel having index 0, center (0,0,0), volume = 1000	  */
        /*      The template Voxel now has a new volume i.e. template_voxel.volume = dV; see below                            */
        /*--------------------------------------------------------------------------------------------------------------------*/

        Voxel template_voxel;
        template_voxel.volume = dV;

        /*-----------------------------------------------------*/
        /*		voxels is std::vector<Voxel> voxels;           	 */
        /*      total voxels = sum of voxels on MPI processes  */
        /*      Size of x/y/z_coordinates is local_x/y/z_nodes */
        /*-----------------------------------------------------*/

        voxels.assign(x_coordinates.size() * y_coordinates.size() * z_coordinates.size(), template_voxel);

        //This depend on the indeaxation from BioFVM-B
        local_start_of_global_index = (coords[1] * z_nodes * y_nodes * local_x_nodes) +       //Imagine 3rd plate 'beginning' point (leftmost bottom point)
                                      (dims[0]-coords[0]-1) * z_nodes * local_y_nodes +       //Imagine going up in 3rd plate
                                      (coords[2] * local_z_nodes) ;
        
        int n = 0;
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < x_coordinates.size(); i++)
        {
            for (int j = 0; j < y_coordinates.size(); j++)
            {
                for (int k = 0; k < z_coordinates.size(); k++)
                {
                    int z_index = k;
				    int y_index = j * z_nodes;  
				    int x_index = i * y_nodes * z_nodes;
                    BioFVM::Voxel aux;
                    aux = template_voxel;
                    aux.center[0] = x_coordinates[i];
                    aux.center[1] = y_coordinates[j];
                    aux.center[2] = z_coordinates[k];
                    aux.mesh_index = n;
                    aux.global_mesh_index = local_start_of_global_index + z_index + y_index + x_index; 
                    aux.volume = dV;               
                    voxels[x_index + y_index + z_index] = aux;
                }
            }
        }

        /*--------------------------*/
	    /* Make Connections next ...*/
  	    /*--------------------------*/

        connected_voxel_indices.resize(voxels.size()); //Connected voxels at local level
        connected_voxel_global_indices.resize(voxels.size()); //Connected voxels at the global level
        voxel_faces.clear();

        /*--------------------------------------------------------------------------------*/
        /*		We need to break it in two parts                                          */
        /*      (i) Inner Compute (IC) which will have two x, two y and two z neighbours  */
        /*      (ii) border part for X, Y and Z directions                                */
        /*      For example for left border i.e. lower X axis part, we can check if       */
        /*      local_index = k * local_x_nodes * local_y_nodes + j * local_y_nodes + i   */
        /*      if voxels[local_index].center[0] - dx/2 > bounding_box[0]                 */
        /*      If true then this voxel has a valid left neighbours                       */
        /*      Let there be two connected voxel lists: local and global                  */
        /*      If left neighbour is across process then local = -1 and global = +ve val  */
        /*      If left neighbour is not valid then do not add anything to the list.      */
        /*      Then the size of the connected_voxels_indices gives number of neighbours  */
        /*--------------------------------------------------------------------------------*/
        
        //Clear for safety
        for (int i = 0; i < connected_voxel_indices.size(); i++)
        {
            connected_voxel_indices[i].clear();
            connected_voxel_global_indices[i].clear();
        }

        int i_jump = local_y_nodes * local_z_nodes; //Local x-jump
	    int j_jump = local_z_nodes; //Local y-jump
	    int k_jump = 1; //Local z-jump
	
	    int i_global_jump = z_nodes*y_nodes; //Global x-jump
  	    int j_global_jump = z_nodes; //Global y-jump
  	    int k_global_jump = k_jump; //Global z-jump
        
        /*----------------------------------------------------------------------------------------------------------------------------------------------------*/
        /* x-aligned connections, first tackle non-boundary voxels in each process, then tackle left boundary then right boundary                             */
        /* We first go from 1st voxel to 2nd last voxel, the 2nd last voxel will connect to last voxel and the last voxel will connect to the 2nd last voxel  */
        /* The problem with the last Voxel is that its right neighbour will be on the next process (or doesn't exist if it is the last process)               */
        /* Whenever we call functions: connect_voxels_indices_only() and connect_voxels_global_indices_only(), the jump will be a local jump only.            */
        /*----------------------------------------------------------------------------------------------------------------------------------------------------*/
        
        for (int k = 0; k < z_coordinates.size(); k++)
        {
            for (int j = 0; j < y_coordinates.size(); j++)
            {
                for (int i = 1; i < x_coordinates.size() - 1; i++)
                {
                    int n = voxel_index(i, j, k);                             // Returns local index of voxel
                    connect_voxels_indices_only(n, n + i_jump, dS_yz);        // Guranteed that adjacent local index will be present
                    connect_voxels_global_indices_only(n, n + i_jump, dS_yz); // Guranteed that global adjacent index will be present
                }
            }
        }

        // Tackling left boundary of each process

        for (int k = 0; k < z_coordinates.size(); k++)
        {
            for (int j = 0; j < y_coordinates.size(); j++)
            {
                int n = voxel_index(0, j, k);               // Returns local index of voxel
                if (voxels[n].center[0] - dx / 2 > x_start) // i.e. it is not a process aligned with left physical boundary
                {
                    // First connect this to right neighbour then right neighbour to this.
                    connect_voxels_indices_only(n, n + i_jump, dS_yz);        // Guranteed that adjacent local index will be present
                    connect_voxels_global_indices_only(n, n + i_jump, dS_yz); // Guranteed that global adjacent index will be present

                    connected_voxel_indices[n].push_back(-1);                                                 // There is no local left neighbour
                    connected_voxel_global_indices[n].push_back(voxels[n].global_mesh_index - i_global_jump); // But there is a neighbour on previous process, so use global index
                }
                else // It is the process that is aligned with left physical boundary
                {
                    connect_voxels_indices_only(n, n + i_jump, dS_yz);        // Guranteed that adjacent local index will be present
                    connect_voxels_global_indices_only(n, n + i_jump, dS_yz); // Guranteed that global adjacent index will be present
                                                                              // No left local OR global voxel is present
                }
            }
        }

        // Tacking right boundary of each process
        // Because of loops above, the rightmost voxel is already connected to its left neighbour
        // but its not connected to its right neighbour (if it exists !)

        for (int k = 0; k < z_coordinates.size(); k++)
        {
            for (int j = 0; j < y_coordinates.size(); j++)
            {
                int n = voxel_index(x_coordinates.size() - 1, j, k); // Returns local index of voxel
                if (voxels[n].center[0] + dx / 2 < x_end)            // i.e. it is not a process aligned with right physical boundary
                {
                    connected_voxel_indices[n].push_back(-1);                                                 // There is no local right neighbour
                    connected_voxel_global_indices[n].push_back(voxels[n].global_mesh_index + i_global_jump); // But there is a neighbour on next process, so use global index
                }
                else // It is the process that is aligned with right physical boundary
                {
                    // There is no right local or neighbour across this process. Do nothing.
                }
            }
        }


        /*--------------------------------------------------------------------------*/
        /*                  Y-aligned connections                                   */
        /*                Again broken into three parts                             */
        /*--------------------------------------------------------------------------*/

        // Non-boundary parts of each sub-domain
        
        for (int k = 0; k < z_coordinates.size(); k++)
        {
            for (int i = 0; i < x_coordinates.size(); i++)
            {
                for (int j = 1; j < y_coordinates.size() - 1; j++)
                {
                    int n = voxel_index(i, j, k);
                    connect_voxels_indices_only(n, n + j_jump, dS_xz);
                    connect_voxels_global_indices_only(n, n + j_jump, dS_xz);
                }
            }
        }

        // Lower boundary of each sub-domain

        for (int k = 0; k < z_coordinates.size(); k++)
        {
            for (int i = 0; i < x_coordinates.size(); i++)
            {
                int n = voxel_index(i, 0, k);               // Returns local index of voxel
                if (voxels[n].center[1] - dy / 2 > y_start) // i.e. it is not a process aligned with bottom physical boundary
                {
                    // First connect this to right neighbour then right neighbour to this.
                    connect_voxels_indices_only(n, n + j_jump, dS_xz);        // Guranteed that adjacent local index will be present
                    connect_voxels_global_indices_only(n, n + j_jump, dS_xz); // Guranteed that global adjacent index will be present

                    connected_voxel_indices[n].push_back(-1);                                                 // There is no local left neighbour
                    connected_voxel_global_indices[n].push_back(voxels[n].global_mesh_index - j_global_jump); // But there is a neighbour on previous process, so use global index
                }
                else // It is the process that is aligned with left physical boundary
                {
                    connect_voxels_indices_only(n, n + j_jump, dS_xz);        // Guranteed that adjacent local index will be present
                    connect_voxels_global_indices_only(n, n + j_jump, dS_xz); // Guranteed that global adjacent index will be present
                                                                              // No downward local OR global voxel is present
                }
            }
        }

        // Upper boundary of each sub-domain

        for (int k = 0; k < z_coordinates.size(); k++)
        {
            for (int i = 0; i < x_coordinates.size(); i++)
            {
                int n = voxel_index(i, y_coordinates.size() - 1, k); // Returns local index of voxel
                if (voxels[n].center[1] + dy / 2 < y_end)            // i.e. it is not a process aligned with right physical boundary
                {
                    connected_voxel_indices[n].push_back(-1);                                                 // There is no local right neighbour
                    connected_voxel_global_indices[n].push_back(voxels[n].global_mesh_index + j_global_jump); // But there is a neighbour on next process, so use global index
                }
                else // It is the process that is aligned with right physical boundary
                {
                    // There is no right local or neighbour across this process. Do nothing.
                }
            }
        }
        
        /*--------------------------------------------------------------------------*/
        /*                  Z-aligned connections                                   */
        /*                Again broken into three parts                             */
        /*--------------------------------------------------------------------------*/
        
        for (int j = 0; j < y_coordinates.size(); j++)
        {
            for (int i = 0; i < x_coordinates.size(); i++)
            {
                for (int k = 1; k < z_coordinates.size() - 1; k++)
                {
                    int n = voxel_index(i, j, k);
                    connect_voxels_indices_only(n, n + k_jump, dS_xy);
                    connect_voxels_global_indices_only(n, n + k_jump, dS_xy);
                }
            }
        }

        // Front boundary of each sub-domain

        for (int j = 0; j < y_coordinates.size(); j++)
        {
            for (int i = 0; i < x_coordinates.size(); i++)
            {
                int n = voxel_index(i, j, 0);               // Returns local index of voxel
                if (voxels[n].center[2] - dz / 2 > z_start) // i.e. it is not a process aligned with bottom physical boundary
                {
                    // First connect this to right neighbour then right neighbour to this.
                    connect_voxels_indices_only(n, n + k_jump, dS_xy);        // Guranteed that adjacent local index will be present
                    connect_voxels_global_indices_only(n, n + k_jump, dS_xy); // Guranteed that global adjacent index will be present

                    connected_voxel_indices[n].push_back(-1);                                                 // There is no local left neighbour
                    connected_voxel_global_indices[n].push_back(voxels[n].global_mesh_index - k_global_jump); // But there is a neighbour on previous process, so use global index
                }
                else // It is the process that is aligned with left physical boundary
                {
                    connect_voxels_indices_only(n, n + k_jump, dS_xy);        // Guranteed that adjacent local index will be present
                    connect_voxels_global_indices_only(n, n + k_jump, dS_xy); // Guranteed that global adjacent index will be present
                                                                              // No downward local OR global voxel is present
                }
            }
        }

        // Back boundary of each sub-domain

        for (int j = 0; j < y_coordinates.size(); j++)
        {
            for (int i = 0; i < x_coordinates.size(); i++)
            {
                int n = voxel_index(i, j, z_coordinates.size() - 1); // Returns local index of voxel
                if (voxels[n].center[2] + dz / 2 < z_end)            // i.e. it is not a process aligned with right physical boundary
                {
                    connected_voxel_indices[n].push_back(-1);                                                 // There is no local right neighbour
                    connected_voxel_global_indices[n].push_back(voxels[n].global_mesh_index + k_global_jump); // But there is a neighbour on next process, so use global index
                }
                else // It is the process that is aligned with right physical boundary
                {
                    // There is no right local or neighbour across this process. Do nothing.
                }
            }
        }

        /*--------------------------------------------------------------------------------------------------------------------------- */
        /* In the example that I am following, use_voxel_faces is false, hence no need to parallelize this yet.                       */
        /* This is very similar to finding neighbours of voxels but most importantly, the connected_voxels_indices[] vector           */
        /* is again initialized over here.                                                                                            */
        /*--------------------------------------------------------------------------------------------------------------------------- */
        
        if (use_voxel_faces)
        {
            create_voxel_faces();
        }
        
        /*--------------------------------------------------------------------------------------------------------------------------- */
        /* Moore neighbourhood is possibly not used anywhere, hence leave parallelization for later
        /*--------------------------------------------------------------------------------------------------------------------------- */

        create_moore_neighborhood();
        return;
    }

    void General_Mesh::connect_voxels_indices_only(int i, int j, double SA) // done
    {
        // Create local index adjacency list
        connected_voxel_indices[i].push_back(j);
        connected_voxel_indices[j].push_back(i);

        return;
    }

    void General_Mesh::connect_voxels_global_indices_only(int i, int j, double SA) // done
    {

        // Create an adjacency list of global indexes

        connected_voxel_global_indices[i].push_back(voxels[j].global_mesh_index);
        connected_voxel_global_indices[j].push_back(voxels[i].global_mesh_index);

        return;
    }

    int Cartesian_Mesh::nearest_voxel_index(std::vector<double> &position)
    {
        /*----------------------------------------------------*/
        /* Routine should return the global index of the voxel*/
        /*----------------------------------------------------*/

        int i = (int)floor((position[0] - bounding_box[0]) / dx);
        int j = (int)floor((position[1] - bounding_box[1]) / dy);
        int k = (int)floor((position[2] - bounding_box[2]) / dz);

        int global_num_x_voxels = (bounding_box[3] - bounding_box[0]) / dx;
        int global_num_y_voxels = (bounding_box[4] - bounding_box[1]) / dy;
        int global_num_z_voxels = (bounding_box[5] - bounding_box[2]) / dz;

        //  add some bounds checking -- truncate to inside the computational domain

        if (i >= global_num_x_voxels)
        {
            i = global_num_x_voxels - 1;
        }
        if (i < 0)
        {
            i = 0;
        }

        if (j >= global_num_y_voxels)
        {
            j = global_num_y_voxels - 1;
        }
        if (j < 0)
        {
            j = 0;
        }

        if (k >= global_num_z_voxels)
        {
            k = global_num_z_voxels - 1;
        }
        if (k < 0)
        {
            k = 0;
        }

        return (i * global_num_y_voxels * global_num_z_voxels + j * global_num_z_voxels + k);
    }

    /*--------------------------------------------------------------------------------*/
    /* This function returns the local voxel index in which a basic_agent resides and */
    /* NOT the global voxel index 																										 */
    /*--------------------------------------------------------------------------------*/

    int Cartesian_Mesh::nearest_voxel_local_index(std::vector<double> &position, int mpi_Rank, int *mpi_Dims)
    {
        /*----------------------------------------------------*/
        /* Routine should return the local index of the voxel */
        /* of the process having rank mpi_Rank that contains  */
        /* the Basic_Agent. The local index is needed because */
        /* voxels[global_index] is not a valid position       */
        /* voxels[local_index] = some_global_index is ok      */
        /*----------------------------------------------------*/

        /*----------------------------------------------------*/
        /* Coordinates of Voxel containing Basic_Agent        */
        /*----------------------------------------------------*/

        int x_vox = (int)floor((position[0] - bounding_box[0]) / dx);
        int y_vox = (int)floor((position[1] - bounding_box[1]) / dy);
        int z_vox = (int)floor((position[2] - bounding_box[2]) / dz);

        /*----------------------------------------------------*/
        /* Global Voxels in each directions                   */
        /*----------------------------------------------------*/

        int global_num_x_voxels = (bounding_box[3] - bounding_box[0]) / dx;
        int global_num_y_voxels = (bounding_box[4] - bounding_box[1]) / dy;
        int global_num_z_voxels = (bounding_box[5] - bounding_box[2]) / dz;

        /*----------------------------------------------------*/
        /* Local Voxels in each directions                    */
        /*----------------------------------------------------*/

        int local_num_x_voxels = (bounding_box[3] - bounding_box[0]) / (mpi_Dims[1] * dx);
        int local_num_y_voxels = (bounding_box[4] - bounding_box[1]) / (mpi_Dims[0] * dy);
        int local_num_z_voxels = (bounding_box[5] - bounding_box[2]) / (mpi_Dims[2] * dz);

        /*---------------------------------------------------------------*/
        /* bounds checking - truncate to inside the computational domain */
        /*---------------------------------------------------------------*/

        if (x_vox >= global_num_x_voxels)
        {
            x_vox = global_num_x_voxels - 1;
        }
        if (x_vox < 0)
        {
            x_vox = 0;
        }

        if (y_vox >= global_num_y_voxels)
        {
            y_vox = global_num_y_voxels - 1;
        }
        if (y_vox < 0)
        {
            y_vox = 0;
        }

        if (z_vox >= global_num_z_voxels)
        {
            z_vox = global_num_z_voxels - 1;
        }
        if (z_vox < 0)
        {
            z_vox = 0;
        }

        /*---------------------------------------------------------------*/
        /* Find process coordinates using mpi_Rank and mpi_Dims          */
        /*---------------------------------------------------------------*/

        int prod12 = mpi_Dims[1] * mpi_Dims[2];
        int proc_x_coord = floor(mpi_Rank / prod12);
        int proc_y_coord = floor((mpi_Rank - proc_x_coord * prod12) / mpi_Dims[2]);
        int proc_z_coord = mpi_Rank - proc_x_coord * prod12 - proc_y_coord * mpi_Dims[2];

        /*---------------------------------------------------------------*/
        /* Calculate the X/Y/Z coordinate of the first voxel             */
        /* of the process (given its process coordinates as above)       */
        /* Remember X/Y Mesh direction and MPI are different             */
        /*---------------------------------------------------------------*/

        int proc_start_vox_x_coord = proc_y_coord * local_num_x_voxels;
        int proc_start_vox_y_coord = (mpi_Dims[0] - 1 - proc_x_coord) * local_num_y_voxels;
        int proc_start_vox_z_coord = proc_z_coord * local_num_z_voxels;

        /*---------------------------------------------------------------*/
        /* Calculate the difference between x/y/z coord of the Voxel     */
        /* that contains the Basic_Agent and the first starting Voxel    */
        /* of that process. Clearly, this diff >= 0.                     */
        /*---------------------------------------------------------------*/

        int diff_x_coord = x_vox - proc_start_vox_x_coord;
        int diff_y_coord = y_vox - proc_start_vox_y_coord;
        int diff_z_coord = z_vox - proc_start_vox_z_coord;

        /*---------------------------------------------------------------*/
        /* Now calculate how many voxels are between the starting voxel  */
        /* and the voxel that contains the Basic_Agent.                  */
        /*---------------------------------------------------------------*/

        int process_local_index_of_voxel_containing_basic_agent = diff_x_coord * local_num_y_voxels * local_num_z_voxels +
                                                                  diff_y_coord * local_num_z_voxels +
                                                                  diff_z_coord;

        return (process_local_index_of_voxel_containing_basic_agent);
    }

    /*------------------------------------------------------------------------------------------------------------------*/
    /* A matlab header is needed by BioFVM for plots and this function uses the MPI process rank 0 to write this header */
    /*------------------------------------------------------------------------------------------------------------------*/

    void write_matlab4_header(int nrows, int ncols, std::string filename, std::string variable_name, int rank, int size, MPI_Comm mpi_Cart_comm)
    {

        MPI_File fh; // Equivalent FILE* fp;
        char char_filename[filename.size() + 1];

        /*--------------------------------------------------------------------------------------------------------------------*/
        /*             C++ string doesn't work within MPI_File_open()                                                         */
        /*             so it needs to be converted to a const char * like below                                               */
        /*             One extra space is for the NULL character in C++                                                       */
        /*--------------------------------------------------------------------------------------------------------------------*/

        strcpy(char_filename, filename.c_str());

        MPI_File_open(mpi_Cart_comm, char_filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh); // Equivalent fp = fopen( filename.c_str() , "wb" );

        unsigned int temp;
        unsigned int type_numeric_format = 0; // little-endian assumed for now!
        unsigned int type_reserved = 0;
        unsigned int type_data_format = 0; // doubles for all entries
        unsigned int type_matrix_type = 0; // full matrix, not sparse

        temp = 1000 * type_numeric_format + 100 * type_reserved + 10 * type_data_format + type_matrix_type;

        if (rank == 0)
            MPI_File_write(fh, &temp, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE); // fwrite( (char*) &temp , UINTs , 1 , fp );

        // UINT rows = (UINT) number_of_data_entries; // storing data as rows
        unsigned int rows = (unsigned int)nrows; // size_of_each_datum; // storing data as cols

        if (rank == 0)
            MPI_File_write(fh, &rows, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE); // fwrite( (char*) &rows , UINTs , 1, fp );

        /*--------------------------------------------------------------------------------------------------------------------*/
        /*             The number of columns is equal to number of voxels                                                     */
        /*             Row0 = center[0], Row1=center[1]..., Row[5]=densities		                                               */
        /*--------------------------------------------------------------------------------------------------------------------*/

        // UINT cols = (UINT) size_of_each_datum; // storing data as rows
        unsigned int cols = (unsigned int)(ncols); // size*ncols: change for cohesion

        if (rank == 0)
            MPI_File_write(fh, &cols, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE); // fwrite( (char*) &cols, UINTs , 1 , fp );

        unsigned int imag = 0; // no complex matrices!

        if (rank == 0)
            MPI_File_write(fh, &imag, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE); // fwrite( (char*) &imag, UINTs, 1 , fp );

        unsigned int name_length = variable_name.size(); // strlen( variable_name );

        if (rank == 0)
            MPI_File_write(fh, &name_length, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE); // fwrite( (char*) &name_length, UINTs, 1 , fp );

        // this is the end of the 20-byte header

        // write the name

        if (rank == 0)
            MPI_File_write(fh, variable_name.c_str(), name_length, MPI_CHARACTER, MPI_STATUS_IGNORE); // fwrite( variable_name.c_str() , name_length , 1 , fp );

        MPI_File_close(&fh);
        return;
    }

    /*----------------------------------------------------------------------------------------------------------*/
    /* This is a dummy function which calls write_matlab4_header(...) function to write header for MATLAB files */
    /*----------------------------------------------------------------------------------------------------------*/

    void write_matlab_header(int rows, int cols, std::string filename, std::string variable_name, int rank, int size, MPI_Comm mpi_Cart_comm)
    {
        write_matlab4_header(rows, cols, filename, variable_name, rank, size, mpi_Cart_comm);
        return;
    }

    /*--------------------------------------------------------------------------------*/
    /* This function uses ALL the MPI processes to write the MATLAB file in parallel. */
    /* This is called MPI-IO and each process sees only a specific part of the file 	*/
    /* which it needs to write - this is called a "view" 															*/
    /*--------------------------------------------------------------------------------*/

    void Microenvironment::write_to_matlab(std::string filename, int rank, int size, MPI_Comm mpi_Cart_comm)
    {

        MPI_File fh;
        MPI_Offset file_size, offset;
        MPI_Datatype etype, filetype;
        double *buffer; // Will contain center[0],center[1],center[2],volume and densities in a contiguous buffer
        char char_filename[filename.size() + 1];
        int elements_to_write;

        /*----------------------------------------------------------------------------------------*/
        /* Now total data entries is now the sum of all entries on all processes                  */
        /* size of datum remains the same                                                         */
        /*----------------------------------------------------------------------------------------*/

        int number_of_data_entries = mesh.voxels.size();
        int size_of_each_datum = 3 + 1 + number_of_densities(); 

        // Possibly we do not need to return anything over here, we can write a separate file at Master
        // All processes call this function - because William Groppe says MPI_File_open is collective operation

        write_matlab_header(size_of_each_datum, number_of_data_entries, filename, "multiscale_microenvironment", rank, size, mpi_Cart_comm);

        MPI_Barrier(mpi_Cart_comm);

        // storing data as cols
        buffer = new double[number_of_data_entries * size_of_each_datum];

        // std::cout<<"CX	"<<"CY	"<<"CZ	"<<"Vol	"<<"Density	\n";

        int n = 0;
        int density_index = 0;

        for (int i = 0; i < number_of_data_entries; i++)
        {

            buffer[n++] = mesh.voxels[i].center[0];
            buffer[n++] = mesh.voxels[i].center[1];
            buffer[n++] = mesh.voxels[i].center[2];
            buffer[n++] = mesh.voxels[i].volume;

            // fwrite( (char*) &( mesh.voxels[i].center[0] ) , sizeof(double) , 1 , fp );
            // fwrite( (char*) &( mesh.voxels[i].center[1] ) , sizeof(double) , 1 , fp );
            // fwrite( (char*) &( mesh.voxels[i].center[2] ) , sizeof(double) , 1 , fp );
            // fwrite( (char*) &( mesh.voxels[i].volume ) , sizeof(double) , 1 , fp );

            for (int j = 0; j < number_of_densities(); j++)
            {
                buffer[n++] = p_density_vectors[density_index];
			    ++density_index;      
            }
        }
        
        strcpy(char_filename, filename.c_str());

        MPI_File_open(mpi_Cart_comm, char_filename, MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); // This file is already created while writing Matlab header
        MPI_File_get_size(fh, &file_size);

        offset = file_size + rank * sizeof(double) * number_of_data_entries * size_of_each_datum;
        etype = MPI_DOUBLE;
        filetype = MPI_DOUBLE;
        elements_to_write = number_of_data_entries * size_of_each_datum;

        MPI_File_set_view(fh, offset, etype, filetype, "native", MPI_INFO_NULL);
        MPI_File_write(fh, buffer, elements_to_write, MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_close(&fh);
        delete[] buffer;

        return;
    }

    /*------------------------------Implementing the X-decomposition-------------------------------------*/

    /*------------------------------------------------------------------------------------------*/
    /* This is the actual Thomas solver which solves a tridigonal system of linear equations. 	*/
    /* Its working can be understood from any book on Numerical Methods 												*/
    /* The Forward pass eliminates all lower triangular coefficients and the Backward pass 			*/
    /* uses simple substitution to find the unknowns. This is the best known algorithm to 			*/
    /* solve such a system of equations in serial. This CANNOT be parallelized. 								*/
    /* Thus, although x-direction is divided among processes, the solver is still serial 				*/
    /* In the future, we plan to replace this solver by Modified Thomas algorithm which can be 	*/
    /* parallelized 																																						*/
    /*------------------------------------------------------------------------------------------*/
    // #define DENSITY(X,Y,Z) (*(M.p_density_vectors[(X)]))[(Y)*M.mesh.z_size+(Z)]
    void diffusion_decay_solver__constant_coefficients_LOD_3D(Microenvironment &M, double dt, int size, int rank, int granurality, int *coords, int *dims, MPI_Comm mpi_Cart_comm)
    {
        
        std::ofstream file(M.timing_csv, std::ios::app);
      
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
            M.mesh.x_size = M.mesh.x_coordinates.size();
            M.mesh.y_size = M.mesh.y_coordinates.size();
            M.mesh.z_size = M.mesh.z_coordinates.size();
            M.mesh.n_substrates = M.number_of_densities();

            int y_size = M.mesh.y_coordinates.size();
            int z_size = M.mesh.z_coordinates.size();
            int p_size = M.number_of_densities(); 

            int step_size = (z_size * y_size) / granurality;

            M.snd_data_size = step_size * p_size; // Number of data elements to be sent
            M.rcv_data_size = step_size * p_size; // All p_density_vectors elements have same size, use anyone

            M.snd_data_size_last = ((z_size * y_size) % granurality) * p_size; // Number of data elements to be sent
            M.rcv_data_size_last = ((z_size * y_size) % granurality) * p_size;
            M.last_iteration = ((z_size * y_size) % granurality) > 0;

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

            M.thomas_i_jump = M.number_of_densities() * M.mesh.z_coordinates.size() * M.mesh.y_coordinates.size();
            M.thomas_j_jump = M.number_of_densities() * M.mesh.z_coordinates.size();
            M.thomas_k_jump = M.number_of_densities(); // M.thomas_j_jump * M.mesh.y_coordinates.size();

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
            // cout << "Rank " << rank << endl;
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
                        MPI_Request send_req;
                        MPI_Isend(&(M.thomas_cx[M.mesh.x_coordinates.size() - 1][0]), M.thomas_cx[M.mesh.x_coordinates.size() - 1].size(), MPI_DOUBLE, ser_ctr + 1, 1111, mpi_Cart_comm, &send_req);
                    }
                }

                if (rank == (ser_ctr + 1) && (ser_ctr + 1) <= (size - 1))
                {

                    std::vector<double> temp_cx(M.thomas_cx[0].size());
                    MPI_Request recv_req;
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

            
            if (rank == 0) file << "X-diffusion,Y-diffusion,Z-diffusion,Apply Dirichlet" << std::endl;
        }

        int n_req = granurality;
        if (M.last_iteration > 0) n_req+=1;
        MPI_Request send_req[n_req];
        MPI_Request recv_req[n_req];
        std::vector<double> block3d(M.thomas_i_jump); //The message to send is of the size Y_voxels * Z_voxels * Substrates

        auto start_time = std::chrono::high_resolution_clock::now();
        M.apply_dirichlet_conditions(rank, size);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto apply_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
       

        // cout << "Rank " << rank << " apply dirichlet condition done" << endl;
        /*-----------------------------------------------------------------------------------*/
        /*                        FORWARD ELIMINATION - x DIRECTION/DECOMPOSITION            */
        /*-----------------------------------------------------------------------------------*/

        /* For data packing...                                                                                 */
        /* My direction of traversing is go up up up i.e. y direction points then go in i.e. Z-direction       */
        /* Remember to visualize 3-D as 2-D plates kept after one another. Hence Z-direction data is farther   */
        /* apart than X/Y direction                                                                            */
        // cout << "Rank " << rank << " snd_data_size: " << snd_data_size << " rcv_data_size: " << rcv_data_size << endl;

        /* So row is along Z axis, column of each row is along Y-axis and each element has p_density_vector*/

        

        // t_strt_x = MPI_Wtime();
        // cout << "Rank " << rank << " starting forward" << endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        if (rank == 0)
        {
            for (int step = 0; step < granurality; ++step)
            {
                int initial_index = step * M.snd_data_size;
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + M.snd_data_size; index += M.mesh.n_substrates)
                {
                    int index_dec = index; 
                    for (int d = 0; d < M.thomas_denomx[0].size(); d++)
                    {
                        M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                    }

                    for (int i = 1; i < M.mesh.x_size; i++)
                    {
                        
                        int index_inc = index_dec + M.thomas_i_jump;
                        // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                        }

                        //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                        }
                        index_dec = index_inc;
                    }
                }

                if (size > 1) {
                    int x_end = M.mesh.x_size - 1;
                    int offset = step * M.snd_data_size;
                    MPI_Status status;
                    MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + offset]), M.snd_data_size, MPI_DOUBLE, rank + 1, step, mpi_Cart_comm, &send_req[step]);
                }
            }
            //Last iteration
            if (M.last_iteration) {
                int initial_index = granurality * M.snd_data_size;
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + M.snd_data_size_last; index += M.mesh.n_substrates)
                {
                    int index_dec = index; 
                    for (int d = 0; d < M.mesh.n_substrates; d++)
                    {
                        M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                    }

                    for (int i = 1; i < M.mesh.x_size; i++)
                    {
                        int index_inc = index_dec + M.thomas_i_jump;
                        // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                        }

                        //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                        for (int d = 0; d < M.mesh.n_substrates; d++)
                        {
                            M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                        }
                        index_dec = index_inc;
                    }
                }

                if (size > 1) {
                    int x_end = M.mesh.x_size - 1;
                    int offset = granurality * M.snd_data_size;
                    MPI_Status status;
                    MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + offset]), M.snd_data_size_last, MPI_DOUBLE, rank + 1, granurality, mpi_Cart_comm, &send_req[granurality]);
                    
                }
            }
        }
        else
        {
            if (rank >= 1 && rank <= (size - 1))
            {
                for (int step = 0; step < granurality; ++step)
                {
                    int initial_index = step * M.snd_data_size;
                    MPI_Irecv(&(block3d[initial_index]), M.rcv_data_size, MPI_DOUBLE, rank-1, step, mpi_Cart_comm, &(recv_req[step]));
                }
                if (M.last_iteration)
                    MPI_Irecv(&(block3d[granurality*M.snd_data_size]), M.rcv_data_size_last, MPI_DOUBLE, rank-1, granurality, mpi_Cart_comm, &(recv_req[granurality]));
                for (int step = 0; step < granurality; ++step)
                {
                    int initial_index = step * M.snd_data_size;
                    MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                    #pragma omp parallel for
                    for (int index = initial_index; index < initial_index + M.snd_data_size; index += M.mesh.n_substrates)
                    {
                        // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, block3d[k][j]);
                        int index_dec = index;
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index + d] += M.thomas_constant1[d] * block3d[index + d];
                        }

                        //(*M.p_density_vectors)[n] /= M.thomas_denomx[0];
                        for (int d = 0; d < M.mesh.n_substrates; d++)
                        {
                            M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                        }

                        for (int i = 1; i < M.mesh.x_size; i++)
                        {

                            int index_inc = index_dec + M.thomas_i_jump;
                            // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                            for (int d = 0; d < M.thomas_k_jump; d++)
                            {
                                M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                            }
                            //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                            for (int d = 0; d < M.mesh.n_substrates; d++)
                            {
                                M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                            }

                            index_dec = index_inc;
                        }
                        
                    }
                    if (rank < (size - 1))
                    {
                        int x_end = M.mesh.x_size - 1;
                        MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + initial_index]), M.snd_data_size, MPI_DOUBLE, rank + 1, step, mpi_Cart_comm, &send_req[step]);
                    }
                }
                if (M.last_iteration)
                {
                    int initial_index = granurality * M.snd_data_size;
                    MPI_Wait(&recv_req[granurality], MPI_STATUS_IGNORE); //Need to change
                    #pragma omp parallel for
                    for (int index = initial_index; index < initial_index + M.snd_data_size_last; index += M.mesh.n_substrates)
                    {
                        // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, block3d[k][j]);
                        int index_dec = index;
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index + d] += M.thomas_constant1[d] * block3d[index + d];
                        }

                        //(*M.p_density_vectors)[n] /= M.thomas_denomx[0];
                        for (int d = 0; d < M.mesh.n_substrates; d++)
                        {
                            M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                        }

                        for (int i = 1; i < M.mesh.x_size; i++)
                        {

                            int index_inc = index_dec + M.thomas_i_jump;
                            // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                            for (int d = 0; d < M.thomas_k_jump; d++)
                            {
                                M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                            }
                            //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                            for (int d = 0; d < M.mesh.n_substrates; d++)
                            {
                                M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                            }

                            index_dec = index_inc;
                        }
                        
                    }
                    // End of computation region
                    if (rank < (size - 1))
                    {
                        int x_end = M.mesh.x_size - 1;
                        MPI_Request aux;
                        MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + initial_index]), M.snd_data_size_last, MPI_DOUBLE, rank + 1, granurality, mpi_Cart_comm, &send_req[granurality]);
                      
                    }
                }
            }
        }
        
        /*-----------------------------------------------------------------------------------*/
        /*                         CODE FOR BACK SUBSITUTION                                 */
        /*-----------------------------------------------------------------------------------*/
        
        //cout << "Rank " << rank << " starting backward substitution" << endl;

        if (rank == (size - 1))
        {
            for (int step = 0; step < granurality; ++step)
            {
                int initial_index = ((M.mesh.x_size - 1)*M.thomas_i_jump) + (step * M.snd_data_size);
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + M.snd_data_size; index += M.mesh.n_substrates)
                {
                    int index_aux = index;
                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {

                        int index_dec = index_aux - M.thomas_i_jump;
                        // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[i], (*M.p_density_vectors)[n + M.thomas_i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_dec + d] -= M.thomas_cx[i][d] * M.p_density_vectors[index_aux + d];
                        }
                        index_aux = index_dec;
                    }
                }
                if (size > 1) {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[step * M.snd_data_size]), M.snd_data_size, MPI_DOUBLE, rank - 1, step, mpi_Cart_comm, &send_req[step]);
                }
            }

            //Last iteration
            if (M.last_iteration) {
                int initial_index = ((M.mesh.x_size - 1)*M.thomas_i_jump) + (granurality * M.snd_data_size);
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + M.snd_data_size_last; index += M.mesh.n_substrates)
                {
                    int index_aux = index;
                    for (int i = M.mesh.x_coordinates.size() - 2; i >= 0; i--)
                    {

                        int index_dec = index_aux - M.thomas_i_jump;
                        // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[i], (*M.p_density_vectors)[n + M.thomas_i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_dec + d] -= M.thomas_cx[i][d] * M.p_density_vectors[index_aux + d];
                        }
                        index_aux = index_dec;
                    }
                }
                if (size > 1) {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[granurality * M.snd_data_size]), M.snd_data_size_last, MPI_DOUBLE, rank - 1, granurality, mpi_Cart_comm, &send_req[granurality]);
                    //cout << "Rank " << rank << " has send" << endl;
                }
            
            }
        }
        else
        {
            for (int step = 0; step < granurality; ++step)
            {
                MPI_Irecv(&(block3d[step*M.snd_data_size]), M.rcv_data_size, MPI_DOUBLE, rank+1, step, mpi_Cart_comm, &recv_req[step]);
            }
            if (M.last_iteration)
                MPI_Irecv(&(block3d[granurality*M.snd_data_size]), M.rcv_data_size_last, MPI_DOUBLE, rank+1, granurality, mpi_Cart_comm, &recv_req[granurality]);

            
            for (int step = 0; step < granurality; ++step)
            {
                int initial_index = ((M.mesh.x_size - 1)*M.thomas_i_jump) + (step * M.snd_data_size);
                int index_3d_initial = (step * M.snd_data_size);
                MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (int offset = 0; offset < M.snd_data_size; offset += M.mesh.n_substrates)
                {
                    int index_aux = initial_index + offset;
                    int index_3d = index_3d_initial + offset;
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_aux + d] -= M.thomas_cx[M.mesh.x_size - 1][d] * block3d[index_3d + d];
                    }

                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {

                        int index_dec = index_aux - M.thomas_i_jump;
                        // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[i], (*M.p_density_vectors)[n + M.thomas_i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_dec + d] -= M.thomas_cx[i][d] * M.p_density_vectors[index_aux + d];
                        }
                        index_aux = index_dec;
                        
                    }
                }
                if (rank > 0)
                {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[step * M.snd_data_size]), M.snd_data_size, MPI_DOUBLE, rank - 1, step, mpi_Cart_comm, &send_req[step]);
                    // cout << "Rank " << rank << " has send" << endl;
                }
            }
            if (M.last_iteration)
            {
                int initial_index = ((M.mesh.x_coordinates.size() - 1)*M.thomas_i_jump) + (granurality * M.snd_data_size);
                int index_3d_initial = (granurality * M.snd_data_size);
                MPI_Wait(&recv_req[granurality], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (int offset = 0; offset < M.snd_data_size_last; offset += M.mesh.n_substrates)
                {
                    int index_aux = initial_index + offset;
                    //int index = j * M.thomas_j_jump + k * M.thomas_k_jump + (M.mesh.x_coordinates.size() - 1) * M.thomas_i_jump;
                    int index_3d = index_3d_initial + offset;
                    // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[M.mesh.x_coordinates.size() - 1], block3d[k][j]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_aux + d] -= M.thomas_cx[M.mesh.x_size - 1][d] * block3d[index_3d + d];
                    }

                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {

                        int index_dec = index_aux - M.thomas_i_jump;
                        // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[i], (*M.p_density_vectors)[n + M.thomas_i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_dec + d] -= M.thomas_cx[i][d] * M.p_density_vectors[index_aux + d];
                        }
                        index_aux = index_dec;
                        
                    }
                }
                if (rank > 0)
                {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[granurality * M.snd_data_size]), M.snd_data_size_last, MPI_DOUBLE, rank - 1, granurality, mpi_Cart_comm, &send_req[granurality]);
                }
            }
        }
        MPI_Barrier(mpi_Cart_comm);

        end_time = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (rank == 0)
            //std::cout << "   X diffusion: " << duration_ms << "ms" << std::endl;
            file << duration_us << ",";

       
        start_time = std::chrono::high_resolution_clock::now();
        M.apply_dirichlet_conditions(rank, size);
        end_time = std::chrono::high_resolution_clock::now();
        apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        start_time = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < M.mesh.x_coordinates.size(); i++)
        {
            for (int k = 0; k < M.mesh.z_coordinates.size(); k++)
            {

                int index = i * M.thomas_i_jump + k * M.thomas_k_jump;
                //(*M.p_density_vectors)[n] /= M.thomas_denomy[0];
                for (int d = 0; d < M.mesh.n_substrates; d++)
                {
                    M.p_density_vectors[index + d] /= M.thomas_denomy[0][d];
                }

                for (int j = 1; j < M.mesh.y_size; j++)
                {

                    int index_inc = index + M.thomas_j_jump;
                    // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_j_jump]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index + d];
                    }
                    //(*M.p_density_vectors)[n] /= M.thomas_denomy[j];
                    for (int d = 0; d < M.mesh.n_substrates; d++)
                    {
                        M.p_density_vectors[index_inc + d] /= M.thomas_denomy[j][d];
                    }
                    index = index_inc;
                }

                // back substitution

                index = i * M.thomas_i_jump + k * M.thomas_k_jump + (M.thomas_j_jump * (M.mesh.y_size - 1));
                for (int j = M.mesh.y_size - 2; j >= 0; j--)
                {

                    int index_dec = index - M.thomas_j_jump;
                    // naxpy(&(*M.p_density_vectors)[n], M.thomas_cy[j], (*M.p_density_vectors)[n + M.thomas_j_jump]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_dec + d] -= M.thomas_cy[j][d] * M.p_density_vectors[index + d];
                    }
                    index = index_dec;
                }
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (rank == 0)
            //std::cout << "   Y diffussion: " << duration_ms << "ms" << std::endl;
            file << duration_us << ",";
         
        start_time = std::chrono::high_resolution_clock::now();
        M.apply_dirichlet_conditions(rank, size);
        end_time = std::chrono::high_resolution_clock::now();
        apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (rank == 0)
        //std::cout << "   Apply dirichlet: " << duration_ms << "ms" << std::endl;

        // cout << "Rank " << rank << " Z diffusion" << endl;
        
        start_time = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M.mesh.x_size; i++)
        {
           for (int j = 0; j < M.mesh.y_size; j++)
            {

                int index = i * M.thomas_i_jump + j * M.thomas_j_jump;
                //(*M.p_density_vectors)[n] /= M.thomas_denomz[0];
                for (int d = 0; d < M.mesh.n_substrates; d++)
                {
                    M.p_density_vectors[index + d] /= M.thomas_denomz[0][d];
                }

                // should be an empty loop if mesh.z_coordinates.size() < 2
                for (int k = 1; k < M.mesh.z_size; k++)
                {

                    int index_inc = index + M.thomas_k_jump;
                    // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_k_jump]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index + d];
                    }
                    //(*M.p_density_vectors)[n] /= M.thomas_denomz[k];
                    for (int d = 0; d < M.mesh.n_substrates; d++)
                    {
                        M.p_density_vectors[index_inc + d] /= M.thomas_denomz[k][d];
                    }

                    index = index_inc;
                }

                // for parallelization need to break forward elimination and back substitution into
                // should be an empty loop if mesh.z_coordinates.size() < 2

                index = i * M.thomas_i_jump + j * M.thomas_j_jump + (M.thomas_k_jump * (M.mesh.z_size - 1));
                for (int k = M.mesh.z_coordinates.size() - 2; k >= 0; k--)
                {

                    int index_dec = index - M.thomas_k_jump;
                    // naxpy(&(*M.p_density_vectors)[n], M.thomas_cz[k], (*M.p_density_vectors)[n + M.thomas_k_jump]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_dec + d] -= M.thomas_cz[k][d] * M.p_density_vectors[index + d];
                    }
                    index = index_dec;
                }
            }
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (rank == 0)
            //std::cout << "   Z diffussion: " << duration_ms << "ms" << std::endl;
            file << duration_us << ",";

        // t_end_z = MPI_Wtime();
        // std::cout<<"Z solve time = "<<(t_end_z-t_strt_z)<<std::endl;
        // cout << "Rank " << rank << " apply dirichlet" << endl;
        start_time = std::chrono::high_resolution_clock::now();
        M.apply_dirichlet_conditions(rank, size);
        end_time = std::chrono::high_resolution_clock::now();
        apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (rank == 0)
            //std::cout << "   Apply dirichlet: " << duration_ms << "ms" << std::endl;
            file << apply_us/4.0 << std::endl;
        // reset gradient vectors
        //	M.reset_all_gradient_vectors();
        // cout << "Rank " << rank << " have finished the diffusion" << endl;
        
        return;
    }

    int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    // Function to compute the least common multiple (LCM)
    int lcm(int a, int b) {
        return (a * b) / gcd(a, b);
    }

    void diffusion_decay_solver__constant_coefficients_LOD_3D_vectorized(Microenvironment &M, double dt, int size, int rank, int granurality, int *coords, int *dims, MPI_Comm mpi_Cart_comm){
        MPI_Request send_req[granurality], recv_req[granurality];
        
        /*
        string path  = "./timing/voxels_" + std::to_string((int)cube_side/2.0) + "/substrates_" + std::to_string(M.mesh.n_substrates) 
        + "/factor_" + std::to_string(factor) +  "/" + std::to_string(mpi_size) + "_node.csv";*/
        std::ofstream file(M.timing_csv, std::ios::app);
        int vl = 4;

        if (M.diffusion_solver_setup_done == false) {
            
            MPI_Request send_req[size];
            MPI_Request recv_req[size];
            
            vector<double> zero(M.mesh.n_substrates, 0.0);
            vector<double> one(M.mesh.n_substrates, 1.0);
            double dt = 0.01;

            int step_size = (M.mesh.z_size * M.mesh.y_size) / granurality;

            M.snd_data_size = step_size * M.mesh.n_substrates; // Number of data elements to be sent
            M.rcv_data_size = step_size * M.mesh.n_substrates; // All p_density_vectors elements have same size, use anyone

            M.snd_data_size_last = ((M.mesh.z_size * M.mesh.y_size) % granurality) * M.mesh.n_substrates; // Number of data elements to be sent
            M.rcv_data_size_last = ((M.mesh.z_size * M.mesh.y_size) % granurality) * M.mesh.n_substrates;

            //Thomas initialization
            M.thomas_denomx.resize(M.mesh.x_size, zero); // sizeof(x_coordinates) = local_x_nodes, denomx is the main diagonal elements
            M.thomas_cx.resize(M.mesh.x_size, zero);     // Both b and c of tridiagonal matrix are equal, hence just one array needed

            /*-------------------------------------------------------------*/
            /* y_coordinates are of size of local_y_nodes.                 */
            /* Each line of Voxels going                                   */
            /* from bottom to top forms a tridiagonal system of Equations  */
            /*-------------------------------------------------------------*/

            M.thomas_denomy.resize(M.mesh.y_size, zero);
            M.thomas_cy.resize(M.mesh.y_size, zero);

            /*-------------------------------------------------------------*/
            /* z_coordinates are of size of local_z_nodes.                 */
            /* Each line of Voxels going                                   */
            /* from front to back forms a tridiagonal system of Equations  */
            /*-------------------------------------------------------------*/

            M.thomas_denomz.resize(M.mesh.z_size, zero);
            M.thomas_cz.resize(M.mesh.z_size, zero);

            /*-------------------------------------------------------------*/
            /* For X-decomposition thomas_i_jump - 1 can be in the previous*/
            /* process and thomas_i_jump+1 can be in the next processs     */
            /* hence we can use thomas_j_jump and thomas_k_jump safely     */
            /* but we CANNOT use thomas_i_jump safely                      */
            /*-------------------------------------------------------------*/

            int i_jump = M.mesh.y_size*M.mesh.z_size*M.mesh.n_substrates;
            int j_jump = M.mesh.z_size*M.mesh.n_substrates;
            int k_jump = M.mesh.n_substrates; // M.thomas_j_jump * M.mesh.y_coordinates.size();

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
            M.thomas_constant1 /= M.mesh.dx; //dx
            M.thomas_constant1 /= M.mesh.dx; //dx

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

            M.thomas_cx.assign(M.mesh.x_size, M.thomas_constant1a);    // Fill b and c elements with -D * dt/dx^2
            M.thomas_denomx.assign(M.mesh.x_size, M.thomas_constant3); // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dx^2)

            if (rank == 0)
                M.thomas_denomx[0] = M.thomas_constant3a; // First diagonal element is   (1 + 1/3 * lambda * dt + 1*D*dt/dx^2)

            if (rank == (size - 1))
                M.thomas_denomx[M.mesh.x_size - 1] = M.thomas_constant3a; // Last diagonal element  is   (1 + 1/3 * lambda * dt + 1*D*dt/dx^2)

            if (rank == 0)
                if (M.mesh.x_size == 1) // This is an extreme case, won't exist, still if it does
                {                                     // then this must be at rank 0
                    M.thomas_denomx[0] = M.one;
                    M.thomas_denomx[0] += M.thomas_constant2;
                }
            if (rank == 0)
                M.thomas_cx[0] /= M.thomas_denomx[0]; // The first c element of tridiagonal matrix is div by first diagonal el.

            // axpy(1st, 2nd, 3rd) => 1st = 1st + 2nd * 3rd
            // the value at  size-1 is not actually used
            // Since value of size-1 is not used, it means it is the value after the last Diagonal element
            // cout << "Rank " << rank << endl;
            for (int ser_ctr = 0; ser_ctr <= size - 1; ser_ctr++)
            {
                if (rank == ser_ctr)
                {
                    if (rank == 0 && rank <= size - 1) // If size=1, then this process does not send data
                    {

                        for (int i = 1; i <= M.mesh.x_size - 1; i++)
                        {
                            axpy(&M.thomas_denomx[i], M.thomas_constant1, M.thomas_cx[i - 1]);
                            M.thomas_cx[i] /= M.thomas_denomx[i]; // the value at  size-1 is not actually used
                        }
                    }
                    else
                    {
                        for (int i = 1; i <= M.mesh.x_size - 1; i++)
                        {
                            axpy(&M.thomas_denomx[i], M.thomas_constant1, M.thomas_cx[i - 1]);
                            M.thomas_cx[i] /= M.thomas_denomx[i]; // the value at  size-1 is not actually used
                        }
                    }

                    if (rank < (size - 1))
                    {
                        MPI_Isend(&(M.thomas_cx[M.mesh.x_size - 1][0]), M.thomas_cx[M.mesh.x_size - 1].size(), MPI_DOUBLE, ser_ctr + 1, 1111, mpi_Cart_comm, &send_req[0]);
                    }
                }

                if (rank == (ser_ctr + 1) && (ser_ctr + 1) <= (size - 1))
                {

                    std::vector<double> temp_cx(M.thomas_cx[0].size());

                    MPI_Irecv(&temp_cx[0], temp_cx.size(), MPI_DOUBLE, ser_ctr, 1111, mpi_Cart_comm, &recv_req[0]);
                    MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

                    axpy(&M.thomas_denomx[0], M.thomas_constant1, temp_cx); // CHECK IF &temp_cz[0] is OK, axpy() in BioFVM_vector.cpp
                    M.thomas_cx[0] /= M.thomas_denomx[0];                   // the value at  size-1 is not actually used
                }

                MPI_Barrier(mpi_Cart_comm);
            }

            /*--------------------------------------------------------------------*/
            /* In 1-D X decomposition, z and y-lines are contiguous adn typically */
            /* the assignments below for z,y should not be changed                */
            /* Both the first voxel i.e. index 0 and last voxel i.e. index=       */
            /* y_coordinates.size()-1 are on the same process                     */
            /*--------------------------------------------------------------------*/

            M.thomas_cy.assign(M.mesh.y_size, M.thomas_constant1a);
            M.thomas_denomy.assign(M.mesh.y_size, M.thomas_constant3);
            M.thomas_denomy[0] = M.thomas_constant3a;
            M.thomas_denomy[M.mesh.y_size - 1] = M.thomas_constant3a;
            if (M.mesh.y_size == 1)
            {
                M.thomas_denomy[0] = M.one;
                M.thomas_denomy[0] += M.thomas_constant2;
            }
            M.thomas_cy[0] /= M.thomas_denomy[0];
            for (int i = 1; i <= M.mesh.y_size - 1; i++)
            {
                axpy(&M.thomas_denomy[i], M.thomas_constant1, M.thomas_cy[i - 1]);
                M.thomas_cy[i] /= M.thomas_denomy[i]; // the value at  size-1 is not actually used
            }

            M.thomas_cz.assign(M.mesh.z_size, M.thomas_constant1a);
            M.thomas_denomz.assign(M.mesh.z_size, M.thomas_constant3);
            M.thomas_denomz[0] = M.thomas_constant3a;
            M.thomas_denomz[M.mesh.z_size - 1] = M.thomas_constant3a;
            if (M.mesh.z_size == 1)
            {
                M.thomas_denomz[0] = M.one;
                M.thomas_denomz[0] += M.thomas_constant2;
            }
            M.thomas_cz[0] /= M.thomas_denomz[0];
            for (int i = 1; i <= M.mesh.z_size - 1; i++)
            {
                axpy(&M.thomas_denomz[i], M.thomas_constant1, M.thomas_cz[i - 1]);
                M.thomas_cz[i] /= M.thomas_denomz[i]; // the value at  size-1 is not actually used
            }
            M.diffusion_solver_vectorized_setup_done = true;
            //if (rank == 0) file << "X-diffusion,Y-diffusion,Z-diffusion,Apply Dirichlet" << std::endl;
        }
        if (M.diffusion_solver_vectorized_setup_done == false) {
            //Vectorization initialization
            M.gvec_size = lcm(M.mesh.n_substrates, vl);

            //X-diffusion
            M.gthomas_constant1.resize(M.gvec_size, 0.0);
            auto dest_iter =  M.gthomas_constant1.begin();
            for (int j = 0; j < M.gvec_size; j+=M.mesh.n_substrates){
                copy(M.thomas_constant1.begin(), M.thomas_constant1.end(), dest_iter);
                dest_iter+=M.mesh.n_substrates;
            }

            M.gthomas_denomx.resize(M.mesh.x_size);
            M.gthomas_cx.resize(M.mesh.x_size);
            for (int i = 0; i < M.mesh.x_size; ++i){
                M.gthomas_denomx[i].resize(M.gvec_size, 0.0);
                M.gthomas_cx[i].resize(M.gvec_size, 0.0);
                auto dest_denomx = M.gthomas_denomx[i].begin();
                auto dest_cx = M.gthomas_cx[i].begin();
                for (int d = 0; d < M.gvec_size; d+=M.mesh.n_substrates){
                    copy(M.thomas_denomx[i].begin(), M.thomas_denomx[i].end(), dest_denomx);
                    copy(M.thomas_cx[i].begin(), M.thomas_cx[i].end(), dest_cx);
                    dest_denomx+=M.mesh.n_substrates;
                    dest_cx+=M.mesh.n_substrates;
                }
            }
            //Y-diffusion

            M.gthomas_denomy.resize(M.mesh.y_size);
            M.gthomas_cy.resize(M.mesh.y_size);
            for (int j = 0; j < M.mesh.y_size; ++j){
                M.gthomas_denomy[j].resize(M.gvec_size, 0.0);
                M.gthomas_cy[j].resize(M.gvec_size, 0.0);
                auto dest_denomy = M.gthomas_denomy[j].begin();
                auto dest_cy = M.gthomas_cy[j].begin();
                for (int d = 0; d < M.gvec_size; d+=M.mesh.n_substrates){
                    copy(M.thomas_denomy[j].begin(), M.thomas_denomy[j].end(), dest_denomy);
                    copy(M.thomas_cy[j].begin(), M.thomas_cy[j].end(), dest_cy);
                    dest_denomy+=M.mesh.n_substrates;
                    dest_cy+=M.mesh.n_substrates;
                }
            }
            //Z - diffusion

            M.gthomas_denomz.resize(M.mesh.z_size);
            M.gthomas_cz.resize(M.mesh.z_size);
            for (int k = 0; k < M.mesh.z_size; ++k){
                M.gthomas_denomz[k].resize(M.gvec_size, 0.0);
                M.gthomas_cz[k].resize(M.gvec_size, 0.0);
                auto dest_denomz = M.gthomas_denomz[k].begin();
                auto dest_cz = M.gthomas_cz[k].begin();
                for (int d = 0; d < M.gvec_size; d+=M.mesh.n_substrates){
                    copy(M.thomas_denomz[k].begin(), M.thomas_denomz[k].end(), dest_denomz);
                    copy(M.thomas_cz[k].begin(), M.thomas_cz[k].end(), dest_cz);
                    dest_denomz+=M.mesh.n_substrates;
                    dest_cz+=M.mesh.n_substrates;
                }
            }
            /*
            string path  = "./timing/voxels_" + std::to_string((int)cube_side/2.0) + "/substrates_" + std::to_string(number_of_densities) 
            + "/factor_" + std::to_string(factor) +  "/" + std::to_string(mpi_size) + "_node.csv";*/
            std::ofstream file(M.timing_csv, std::ios::app);
            if (rank == 0) {
                file << "X-diffusion,Y-diffusion,Z-diffusion,Apply Dirichlet" << std::endl;
            }
            M.diffusion_solver_vectorized_setup_done = true;
            } 
    
        double block3d[M.thomas_i_jump]; //Aux structure of the size: Y*Z*Substrates

        auto start_time = std::chrono::high_resolution_clock::now();
        M.apply_dirichlet_conditions_v2(rank, size);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto apply_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        //cout << "Peto aqui!" << endl;
        start_time = std::chrono::high_resolution_clock::now();
        //X-diffusion vector_x_v2
        //start_time = std::chrono::high_resolution_clock::now();
        if (rank == 0)
        {
            for (int step = 0; step < granurality; ++step)
            {
                int initial_index = step * M.snd_data_size;
                int limit = (initial_index + M.snd_data_size);
                int limit_vec = limit -(M.snd_data_size%vl);
                #pragma omp parallel for
                for (int index = initial_index; index < limit_vec; index += vl)
                {
                    int index_dec = index;
                    int gd = index%M.gvec_size;

                    __m256d denomx1 = _mm256_loadu_pd(&M.gthomas_denomx[0][gd]);
                    __m256d density1 = _mm256_loadu_pd(&M.p_density_vectors[index]);
                    __m256d aux1 = _mm256_div_pd(density1, denomx1);

                    _mm256_storeu_pd(&M.p_density_vectors[index], aux1);

                    for (int i = 1; i < M.mesh.x_size; i++)
                    {
                        int index_inc = index_dec + M.thomas_i_jump;
                        __m256d constant1 = _mm256_loadu_pd(&M.gthomas_constant1[gd]);
                        __m256d density_curr1 = _mm256_loadu_pd(&M.p_density_vectors[index_dec ]);
                        __m256d density_inc1 = _mm256_loadu_pd(&M.p_density_vectors[index_inc]);
                        __m256d denomy1 = _mm256_loadu_pd(&M.gthomas_denomx[i][gd]);
                    
                        density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
                
                        density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                        _mm256_storeu_pd(&M.p_density_vectors[index_inc], density_curr1);
                        
                        index_dec = index_inc;
                    }
                }

                
                //Epilogo vectorization
                //int ep =limit - limit_vec;
                //int ep_it = ep / number_of_M.p_density_vectors + (ep > 0);
                for (int index = limit_vec; index < limit; ++index)
                {
                    int index_dec = index;
                    int d = index % M.mesh.n_substrates; 

                    M.p_density_vectors[index] /= M.thomas_denomx[0][d];

                    for (int i = 1; i < M.mesh.x_size; i++)
                    {
                        int index_inc = index_dec + M.thomas_i_jump;
                        // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
                        M.p_density_vectors[index_inc] += M.thomas_constant1[d] * M.p_density_vectors[index_dec];
                        
                        //(*M.microenvironment)[n] /= M.thomas_denomx[i];
                        
                        M.p_density_vectors[index_inc] /= M.thomas_denomx[i][d];
                        
                        index_dec = index_inc;
                    }
                }

                if (size > 1) {
                    int x_end = M.mesh.x_size - 1;
                    int offset = step * M.snd_data_size;
                    MPI_Status status;
                    MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + offset]), M.snd_data_size, MPI_DOUBLE, rank + 1, step, MPI_COMM_WORLD, &send_req[step]);
                }
            }
            //Last iteration
            if (M.snd_data_size_last != 0) {
                int initial_index = granurality * M.snd_data_size;
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + M.snd_data_size_last; index += M.mesh.n_substrates)
                {
                    int index_dec = index; 
                    for (int d = 0; d < M.mesh.n_substrates; d++)
                    {
                        M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                    }

                    for (int i = 1; i < M.mesh.x_size; i++)
                    {
                        int index_inc = index_dec + M.thomas_i_jump;
                        // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
                        for (int d = 0; d < M.mesh.n_substrates; d++)
                        {
                            M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                        }

                        //(*M.microenvironment)[n] /= M.thomas_denomx[i];
                        for (int d = 0; d < M.mesh.n_substrates; d++)
                        {
                            M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                        }
                        index_dec = index_inc;
                    }
                }

                if (size > 1) {
                    int x_end = M.mesh.x_size - 1;
                    int offset = granurality * M.snd_data_size;
                    MPI_Status status;
                    MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + offset]), M.snd_data_size_last, MPI_DOUBLE, rank + 1, granurality, MPI_COMM_WORLD, &send_req[granurality]);
                    
                }
            }
        }
        else
        {
            if (rank >= 1 && rank <= (size - 1))
            {
                for (int step = 0; step < granurality; ++step)
                {
                    int initial_index = step * M.snd_data_size;
                    MPI_Irecv(&(block3d[initial_index]), M.rcv_data_size, MPI_DOUBLE, rank-1, step, MPI_COMM_WORLD, &(recv_req[step]));
                }
                if (M.snd_data_size_last != 0)
                    MPI_Irecv(&(block3d[granurality*M.snd_data_size]), M.rcv_data_size_last, MPI_DOUBLE, rank-1, granurality, MPI_COMM_WORLD, &(recv_req[granurality]));
                for (int step = 0; step < granurality; ++step)
                {
                    int initial_index = step * M.snd_data_size;
                    int limit = (initial_index + M.snd_data_size);
                    int limit_vec = limit - (M.snd_data_size%vl);
                    MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                    #pragma omp parallel for
                    for (int index = initial_index; index < limit; index += vl)
                    {
                        // axpy(&(*M.microenvironment)[n], M.thomas_constant1, block3d[k][j]);
                        int index_dec = index;
                        int gd = index%M.gvec_size;
                        __m256d constant1 = _mm256_loadu_pd(&M.gthomas_constant1[gd]);
                        __m256d density_curr1 = _mm256_loadu_pd(&M.p_density_vectors[index]);
                        __m256d density_inc1 = _mm256_loadu_pd(&block3d[index]);
                        __m256d denomy1 = _mm256_loadu_pd(&M.gthomas_denomx[0][gd]);
                

                        density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
            
                        //_mm256_storeu_pd(&microenvironment[index_inc + zd], density_curr);
            
                        //(*M.p_density_vectors)[n] /= M.thomas_denomy[j];
                        //Fer unrolling aqui
                
                        //__m256d density = _mm256_loadu_pd(&microenvironment[index_inc + zd]);
                        density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                        _mm256_storeu_pd(&M.p_density_vectors[index], density_curr1);

                        for (int i = 1; i < M.mesh.x_size; i++)
                        {

                            int index_inc = index_dec + M.thomas_i_jump;
                            __m256d constant1 = _mm256_loadu_pd(&M.gthomas_constant1[gd]);
                            __m256d density_curr1 = _mm256_loadu_pd(&M.p_density_vectors[index_dec ]);
                            __m256d density_inc1 = _mm256_loadu_pd(&M.p_density_vectors[index_inc]);
                            __m256d denomy1 = _mm256_loadu_pd(&M.gthomas_denomx[i][gd]);
                    

                            density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
                
                            //_mm256_storeu_pd(&microenvironment[index_inc + zd], density_curr);
                
                            //(*M.p_density_vectors)[n] /= M.thomas_denomy[j];
                            //Fer unrolling aqui
                   
                            //__m256d density = _mm256_loadu_pd(&microenvironment[index_inc + zd]);
                            density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                            _mm256_storeu_pd(&M.p_density_vectors[index_inc], density_curr1);

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
                        int d = index % M.mesh.n_substrates; 
                        
                        M.p_density_vectors[index] += M.thomas_constant1[d] * block3d[index];
                        M.p_density_vectors[index] /= M.thomas_denomx[0][d];
                        
                        for (int i = 1; i < M.mesh.x_size; i++)
                        {
                            int index_inc = index_dec + M.thomas_i_jump;
                            // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
                        
                            M.p_density_vectors[index_inc] += M.thomas_constant1[d] * M.p_density_vectors[index_dec];
                        
                            //(*M.microenvironment)[n] /= M.thomas_denomx[i];
                            M.p_density_vectors[index_inc] /= M.thomas_denomx[i][d];
                            index_dec = index_inc;
                        }
                    }


                    if (rank < (size - 1))
                    {
                        int x_end = M.mesh.x_size - 1;
                        MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + initial_index]), M.snd_data_size, MPI_DOUBLE, rank + 1, step, MPI_COMM_WORLD, &send_req[step]);
                    }
                }
                if (M.snd_data_size_last != 0)
                {
                    int initial_index = granurality * M.snd_data_size;
                    MPI_Wait(&recv_req[granurality], MPI_STATUS_IGNORE); //Need to change
                    #pragma omp parallel for
                    for (int index = initial_index; index < initial_index + M.snd_data_size_last; index += M.mesh.n_substrates)
                    {
                        // axpy(&(*M.microenvironment)[n], M.thomas_constant1, block3d[k][j]);
                        int index_dec = index;
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index + d] += M.thomas_constant1[d] * block3d[index + d];
                        }

                        //(*M.microenvironment)[n] /= M.thomas_denomx[0];
                        for (int d = 0; d < M.mesh.n_substrates; d++)
                        {
                            M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                        }

                        for (int i = 1; i < M.mesh.x_size; i++)
                        {

                            int index_inc = index_dec + M.thomas_i_jump;
                            // axpy(&(*M.microenvironment)[n], M.thomas_constant1, (*M.microenvironment)[n - M.i_jump]);
                            for (int d = 0; d < M.mesh.n_substrates; d++)
                            {
                                M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                            }
                            //(*M.microenvironment)[n] /= M.thomas_denomx[i];
                            for (int d = 0; d < M.mesh.n_substrates; d++)
                            {
                                M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                            }

                            index_dec = index_inc;
                        }
                        
                    }
                    // End of computation region
                    if (rank < (size - 1))
                    {
                        int x_end = M.mesh.x_size - 1;
                        MPI_Request aux;
                        MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + initial_index]), M.snd_data_size_last, MPI_DOUBLE, rank + 1, granurality, MPI_COMM_WORLD, &send_req[granurality]);
                      
                    }
                }
            }
        }
        
        /*-----------------------------------------------------------------------------------*/
        /*                         CODE FOR BACK SUBSITUTION                                 */
        /*-----------------------------------------------------------------------------------*/
        
        //cout << "Rank " << rank << " starting backward substitution" << endl;

        if (rank == (size - 1))
        {
            for (int step = 0; step < granurality; ++step)
            {
                int last_xplane = ((M.mesh.x_size - 1)*M.thomas_i_jump);
                int initial_index = last_xplane + (step * M.snd_data_size);
                int limit = initial_index + M.snd_data_size;
                int limit_vec = limit - (M.snd_data_size%vl);
                #pragma omp parallel for 
                for (int index = initial_index; index < limit_vec; index += vl)
                {
                    int index_aux = index;
                    int gd = (index - last_xplane)%M.gvec_size;
                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {

                        int index_dec = index_aux - M.thomas_i_jump;
                        __m256d cy1 = _mm256_loadu_pd(&M.gthomas_cx[i][gd]);
                        __m256d density_curr1 = _mm256_loadu_pd(&M.p_density_vectors[index_aux]);
                        __m256d density_dec1 = _mm256_loadu_pd(&M.p_density_vectors[index_dec]);

                        density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                        _mm256_storeu_pd(&M.p_density_vectors[index_dec], density_curr1);
                        index_aux = index_dec;
                    }
                }

                //Epilogo Vectorizacion Back Last rank
                
                for (int index = limit_vec; index < limit; ++index){
                    int index_aux = index;
                    int d = (index - last_xplane) % M.mesh.n_substrates;
                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {
                        int index_dec = index_aux - M.thomas_i_jump;
                        // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
                        M.p_density_vectors[index_dec] -= M.thomas_cx[i][d] * M.p_density_vectors[index_aux];
                        
                        index_aux = index_dec;
                    }
                }

                if (size > 1) {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[step * M.snd_data_size]), M.snd_data_size, MPI_DOUBLE, rank - 1, step, MPI_COMM_WORLD, &send_req[step]);
                }
            }

            //Last iteration
            if (M.snd_data_size_last != 0) {
                int initial_index = ((M.mesh.x_size - 1)*M.thomas_i_jump) + (granurality * M.snd_data_size);
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + M.snd_data_size_last; index += M.mesh.n_substrates)
                {
                    int index_aux = index;
                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {

                        int index_dec = index_aux - M.thomas_i_jump;
                        // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_dec + d] -= M.thomas_cx[i][d] * M.p_density_vectors[index_aux + d];
                        }
                        index_aux = index_dec;
                    }
                }
                if (size > 1) {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[granurality * M.snd_data_size]), M.snd_data_size_last, MPI_DOUBLE, rank - 1, granurality, MPI_COMM_WORLD, &send_req[granurality]);
                    //cout << "Rank " << rank << " has send" << endl;
                }
            
            }
        }
        else
        {
            for (int step = 0; step < granurality; ++step)
            {
                MPI_Irecv(&(block3d[step*M.snd_data_size]), M.rcv_data_size, MPI_DOUBLE, rank+1, step, MPI_COMM_WORLD, &recv_req[step]);
            }
            if (M.snd_data_size_last != 0)
                MPI_Irecv(&(block3d[granurality*M.snd_data_size]), M.rcv_data_size_last, MPI_DOUBLE, rank+1, granurality, MPI_COMM_WORLD, &recv_req[granurality]);

            
            for (int step = 0; step < granurality; ++step)
            {
                int last_xplane = ((M.mesh.x_size - 1)*M.thomas_i_jump);
                int initial_index = last_xplane + (step * M.snd_data_size);
                int limit = initial_index + M.snd_data_size;
                int limit_vec = limit - (M.snd_data_size%vl);
                MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (int index = initial_index; index < limit_vec; index += vl)
                {
                    int index_aux = index;
                    int index_3d = index - last_xplane;
                    int gd = index_3d%M.gvec_size;
                    __m256d cy1 = _mm256_loadu_pd(&M.gthomas_cx[M.mesh.x_size-1][gd]);
                    __m256d density_curr1 = _mm256_loadu_pd(&M.p_density_vectors[index_aux]);
                    __m256d density_dec1 = _mm256_loadu_pd(&block3d[index_3d]);

                    density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                    _mm256_storeu_pd(&M.p_density_vectors[index_aux], density_curr1);

                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {

                        int index_dec = index_aux - M.thomas_i_jump;
                        __m256d cy1 = _mm256_loadu_pd(&M.gthomas_cx[i][gd]);
                        __m256d density_curr1 = _mm256_loadu_pd(&M.p_density_vectors[index_aux]);
                        __m256d density_dec1 = _mm256_loadu_pd(&M.p_density_vectors[index_dec]);

                        density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                        _mm256_storeu_pd(&M.p_density_vectors[index_dec], density_curr1);
                        index_aux = index_dec;
                    }
                }
                
                //Epilogo Vectorizacion
                for (int index = limit_vec; index < limit; ++index){
                    int index_aux = index;
                    int index_3d = index - last_xplane;
                    int d = (index - last_xplane) % M.mesh.n_substrates;

                    M.p_density_vectors[index_aux] -= M.thomas_cx[M.mesh.x_size - 1][d] * block3d[index_3d];


                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {
                        int index_dec = index_aux - M.thomas_i_jump;
                        // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
                        M.p_density_vectors[index_dec] -= M.thomas_cx[i][d] * M.p_density_vectors[index_aux];
                        
                        index_aux = index_dec;
                    }
                }

                if (rank > 0)
                {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[step * M.snd_data_size]), M.snd_data_size, MPI_DOUBLE, rank - 1, step, MPI_COMM_WORLD, &send_req[step]);
                    // cout << "Rank " << rank << " has send" << endl;
                }
            }
            if (M.snd_data_size_last != 0)
            {
                int initial_index = ((M.mesh.x_size - 1)*M.thomas_i_jump) + (granurality * M.snd_data_size);
                int index_3d_initial = (granurality * M.snd_data_size);
                MPI_Wait(&recv_req[granurality], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (int offset = 0; offset < M.snd_data_size_last; offset += M.mesh.n_substrates)
                {
                    int index_aux = initial_index + offset;
                    //int index = j * M.j_jump + k * M.k_jump + (M.mesh.x_coordinates.size() - 1) * M.i_jump;
                    int index_3d = index_3d_initial + offset;
                    // naxpy(&(*M.microenvironment)[n], M.thomas_cx[M.mesh.x_coordinates.size() - 1], block3d[k][j]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_aux + d] -= M.thomas_cx[M.mesh.x_size - 1][d] * block3d[index_3d + d];
                    }

                    for (int i = M.mesh.x_size - 2; i >= 0; i--)
                    {

                        int index_dec = index_aux - M.thomas_i_jump;
                        // naxpy(&(*M.microenvironment)[n], M.thomas_cx[i], (*M.microenvironment)[n + M.i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_dec + d] -= M.thomas_cx[i][d] * M.p_density_vectors[index_aux + d];
                        }
                        index_aux = index_dec;
                        
                    }
                }
                if (rank > 0)
                {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[granurality * M.snd_data_size]), M.snd_data_size_last, MPI_DOUBLE, rank - 1, granurality, MPI_COMM_WORLD, &send_req[granurality]);
                }
            }
        }
    end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    //cout << "Peto aqui2!" << endl;

    if (rank == 0)
        file << duration_us << ",";
    start_time = std::chrono::high_resolution_clock::now();
    M.apply_dirichlet_conditions(rank, size);
    end_time = std::chrono::high_resolution_clock::now();
    apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    //Y-diffusion
    start_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < M.mesh.x_size; i++)
    {
        //Forward Elimination
        //J = 0
        int gd = 0; //Density pointer
        int index = i * M.thomas_i_jump;
        int zd;
        for (zd = 0; zd + vl < M.thomas_j_jump; zd+=vl)
        {
            __m256d denomy1 = _mm256_loadu_pd(&M.gthomas_denomy[0][gd]);
            __m256d density1 = _mm256_loadu_pd(&M.p_density_vectors[index + zd]);
            gd+=vl;
            if (gd == M.gvec_size) gd = 0;
            __m256d aux1 = _mm256_div_pd(density1, denomy1);
            _mm256_storeu_pd(&M.p_density_vectors[index + zd], aux1);
        }
        //Epilogo
        int ep = M.thomas_j_jump - zd;
        int z_ini = zd / M.mesh.n_substrates;
        int d_ini = zd % M.mesh.n_substrates;
        index = index + z_ini * M.thomas_k_jump;
        for (int k = z_ini; k < M.mesh.z_size; ++k)
        {
            for (int d = d_ini; d < M.mesh.n_substrates; ++d){
                d_ini = 0;
                M.p_density_vectors[index + d] /= M.thomas_denomy[0][d];
            }
            index+=M.thomas_k_jump;
        }
        //J = 1..(y_size-1)
        for (int j = 1; j < M.mesh.y_size; j++)
        {
            int index_base = i * M.thomas_i_jump +  (j-1)*M.thomas_j_jump;
            int index_inc =  index_base + M.thomas_j_jump;
            int zd;
            gd = 0;
            for (zd = 0; zd + vl < M.thomas_j_jump; zd+=vl)
            {
                __m256d constant1 = _mm256_loadu_pd(&M.gthomas_constant1[gd]);
                __m256d density_curr1 = _mm256_loadu_pd(&M.p_density_vectors[index_base + zd]);
                __m256d density_inc1 = _mm256_loadu_pd(&M.p_density_vectors[index_inc + zd]);
                __m256d denomy1 = _mm256_loadu_pd(&M.gthomas_denomy[j][gd]);
                gd+=vl;
                if (gd == M.gvec_size) gd = 0;
                density_curr1 = _mm256_fmadd_pd(constant1, density_curr1, density_inc1);
                density_curr1 = _mm256_div_pd(density_curr1, denomy1);
                _mm256_storeu_pd(&M.p_density_vectors[index_inc + zd], density_curr1);
            }
            //Epilogo
            int ep = M.thomas_j_jump - zd;
            int z_ini = zd / M.mesh.n_substrates;
            int d_ini = zd % M.mesh.n_substrates;
            index_base = index_base + z_ini * M.thomas_k_jump;
            index_inc = index_inc + z_ini * M.thomas_k_jump;
            for (int k = z_ini; k < M.mesh.z_size; ++k)
            {
                for (int d = d_ini; d < M.mesh.n_substrates; ++d){
                    d_ini = 0;
                    M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_base + d];
                    M.p_density_vectors[index_inc + d] /= M.thomas_denomy[j][d];
                }
                index_base+=M.thomas_k_jump;
                index_inc+=M.thomas_k_jump;
            }
        }
        // Back substitution
        for (int j = M.mesh.y_size - 2; j >= 0; j--)
        {
            int index_base = i * M.thomas_i_jump + (j+1) * M.thomas_j_jump;
            int index_dec = index_base - M.thomas_j_jump;
            int zd;
            gd = 0;
            for ( zd = 0; zd + vl < M.thomas_j_jump; zd+=vl)
            {
                __m256d cy1 = _mm256_loadu_pd(&M.gthomas_cy[j][gd]);
                __m256d density_curr1 = _mm256_loadu_pd(&M.p_density_vectors[index_base + zd]);
                __m256d density_dec1 = _mm256_loadu_pd(&M.p_density_vectors[index_dec+ zd]);
                gd+=vl;
                if (gd == M.gvec_size) gd = 0;

                density_curr1 = _mm256_fnmadd_pd(cy1, density_curr1, density_dec1);

                _mm256_storeu_pd(&M.p_density_vectors[index_dec + zd], density_curr1);
                
            }

            //Epilogo
            int ep = M.thomas_j_jump - zd;
            int z_ini = zd / M.mesh.n_substrates;
            int d_ini = zd % M.mesh.n_substrates;
            index_base = index_base + z_ini * M.thomas_k_jump;
            index_dec = index_dec + z_ini * M.thomas_k_jump;
            for (int k = z_ini; k < M.mesh.z_size; ++k)
            {
                for (int d = d_ini; d < M.mesh.n_substrates; ++d){
                    d_ini = 0;
                    M.p_density_vectors[index_dec + d] -= M.thomas_cy[j][d] * M.p_density_vectors[index_base + d];
                }
                index_base+=M.thomas_k_jump;
                index_dec+=M.thomas_k_jump;
            }

        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    if (rank == 0)
        file << duration_us << ",";
    M.apply_dirichlet_conditions_v2(rank, size);
    end_time = std::chrono::high_resolution_clock::now();
    apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    //Z-diffusion
    //cout << "Peto aqui 3!" << endl;

    start_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M.mesh.x_size; i++)
        {
        for (int j = 0; j < M.mesh.y_size; j++)
            {
            int index = i * M.thomas_i_jump + j * M.thomas_j_jump;
            //(*M.p_density_vectors)[n] /= M.thomas_denomz[0];
            for (int d = 0; d < M.mesh.n_substrates; d++)
            {
                M.p_density_vectors[index + d] /= M.thomas_denomz[0][d];
            }

            // should be an empty loop if mesh.z_coordinates.size() < 2
            for (int k = 1; k < M.mesh.z_size; k++)
            {

                int index_inc = index + M.thomas_k_jump;
                // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_k_jump]);
                for (int d = 0; d < M.mesh.n_substrates; d++)
                {
                    M.p_density_vectors[index_inc + d] += M.p_density_vectors[d] * M.p_density_vectors[index + d];
                }
                //(*M.p_density_vectors)[n] /= M.thomas_denomz[k];
                for (int d = 0; d < M.mesh.n_substrates; d++)
                {
                    M.p_density_vectors[index_inc + d] /= M.thomas_denomz[k][d];
                }

                index = index_inc;
            }

            index = i * M.thomas_i_jump + j * M.thomas_j_jump + (M.thomas_k_jump * (M.mesh.z_size - 1));
            for (int k = M.mesh.z_size - 2; k >= 0; k--)
            {

                int index_dec = index - M.thomas_k_jump;
                // naxpy(&(*M.p_density_vectors)[n], M.thomas_cz[k], (*M.p_density_vectors)[n + M.thomas_k_jump]);
                for (int d = 0; d < M.mesh.n_substrates; d++)
                {
                    M.p_density_vectors[index_dec + d] -= M.thomas_cz[k][d] * M.p_density_vectors[index + d];
                }
                index = index_dec;
            }
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    if (rank == 0)
        file << duration_us << ",";
    
    start_time = std::chrono::high_resolution_clock::now();
    M.apply_dirichlet_conditions_v2(rank, size);
    end_time = std::chrono::high_resolution_clock::now();
    apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
     if (rank == 0)
        file << apply_us/4.0 << std::endl;
    }   

    //-->Had to create a function like this else compiler complains of undefined reference to this function due to call in initialize_microenvironment() in microenvironment.cpp
    //-->Remember I have commented out LOD_2D and LOD_3D in solvers.cpp
    void diffusion_decay_solver__constant_coefficients_LOD_2D(Microenvironment &M, double dt, int mpi_Size, int mpi_Rank, int *mpi_Coords, int *mpi_Dims, MPI_Comm mpi_Cart_comm)
    {
    }

    void print_voxels_densities(Microenvironment &M, double dt, int size, int rank, int *coords, std::string *file_name, int *dims, MPI_Comm mpi_Cart_comm)
    {
        // std::string filename = std::to_string(rank) + "_" + *file_name;
        //std::cout << "Rank " << rank << " esta imprimiendo densidades" << std::endl;
        std::string filename = *file_name;
        std::ofstream outputFile(filename, std::ios::app);
        int index = 0;
        for (int i = 0; i < M.mesh.x_size; i++)
        {
            for (int j = 0; j < M.mesh.y_size; j++)
            {
                for (int k = 0; k < M.mesh.z_size; k++)
                {
                    std::cout << (M.mesh.x_coordinates.size()*rank) + i << " " << j << " " << k << " : ";
                    for (int d = 0; d < M.number_of_densities(); ++d)
                    {
                        outputFile << M.p_density_vectors[index] << " ";
                        std::cout << M.p_density_vectors[index] << " ";
                        ++index;
                    }
                    std::cout << endl;
                    outputFile << std::endl;
                }
            }
        }
        std::chrono::seconds dura(15);
        std::this_thread::sleep_for( dura );

        //flush write buffer
        std::cout << std::flush;
        outputFile << std::flush;
    }

};
