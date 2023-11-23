#include "BioFVM_microenvironment.h"
#include "BioFVM_mesh.h"
#include "BioFVM_basic_agent.h"
#include <cmath>
#include <chrono>
#include <thread>

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
        cout << "Microenvironment resize space in progress" << endl;
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
        // dirichlet_value_vectors.assign(mesh.voxels.size(), one); // need tor redo this
        // Jose: commented because it won't be used
        dirichlet_value_vectors.assign(mesh.x_size * mesh.y_size * mesh.z_size * number_of_densities(), 100.0);
        p_density_vectors.resize(mesh.x_size * mesh.y_size * mesh.z_size * number_of_densities());
        cout << "Microenvironment resize space completed" << endl;
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

        cout << "x_start = " << x_start << " | x_end = " << x_end << endl;
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

        cout << "Local nodes " << endl;
        cout << "   X: " << local_x_nodes << endl
             << "   Y: " << local_y_nodes << endl
             << "   Z: " << local_z_nodes << endl;

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

        cout << "Rank will assign the mesh size" << endl;
        voxels.resize(x_coordinates.size());
        cout << "Size of X voxels = " << x_coordinates.size() << endl;

        // voxels.assign(x_coordinates.size() * y_coordinates.size() * z_coordinates.size(), template_voxel);

        cout << "Rank have asign the mesh size" << endl;
        cout << "size of voxels vector " << voxels.size() << endl;
        local_start_of_global_index = (dims[0] - coords[0] - 1) * x_nodes * local_y_nodes +
                                      (coords[1] * local_x_nodes) +
                                      (coords[2] * x_nodes * y_nodes * local_z_nodes);
        int n = 0;

#pragma omp parallel for
        for (int i = 0; i < x_coordinates.size(); i++)
        {
            // cout << "Generating plane x = " << i << endl;
            // voxels[i] = new BioFVM::Voxel[y_coordinates.size() * z_coordinates.size()];
            voxels[i] = new std::vector<Voxel>(y_coordinates.size() * z_coordinates.size(), template_voxel);
            for (int j = 0; j < y_coordinates.size(); j++)
            {
                y_index = j * x_nodes;
                int j_index = j * y_coordinates.size();
                for (int k = 0; k < z_coordinates.size(); k++)
                {
                    z_index = k * x_nodes * y_nodes;
                    // cout << "K " << k << " J " << j << " I " << i << " Address to consult " << (voxels_jose[i]) + (j * y_coordinates.size()) + k << endl;
                    BioFVM::Voxel aux; // = *((voxels_jose[i]) + (j * y_coordinates.size()) + k);
                    aux = template_voxel;
                    aux.center[0] = x_coordinates[i];
                    aux.center[1] = y_coordinates[j];
                    aux.center[2] = z_coordinates[k];
                    aux.mesh_index = n;                                                          // This now becomes the local index (Jose: will not be necessary)
                    aux.global_mesh_index = local_start_of_global_index + z_index + y_index + i; // This is now the global index of the Voxel in the global mesh.
                    aux.volume = dV;
                    //*((voxels[i]) + (j * y_coordinates.size()) + k) = aux;
                    (*(voxels[i]))[j_index + k] = aux;
                }
            }
        }
        cout << "Voxels are generated" << endl;

        // make connections
        /*

        connected_voxel_indices.resize(voxels.size());
        connected_voxel_global_indices.resize(voxels.size()); //--->This field in class Voxels has been added by Gaurav Saxena
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
        /*
        for (int i = 0; i < connected_voxel_indices.size(); i++)
        {
            connected_voxel_indices[i].clear();
        }

        int i_jump = 1;
        int j_jump = local_x_nodes;
        int k_jump = local_x_nodes * local_y_nodes;

        int i_global_jump = i_jump;
        int j_global_jump = x_nodes;
        int k_global_jump = x_nodes * y_nodes;
        */
        /*----------------------------------------------------------------------------------------------------------------------------------------------------*/
        /* x-aligned connections, first tackle non-boundary voxels in each process, then tackle left boundary then right boundary                             */
        /* We first go from 1st voxel to 2nd last voxel, the 2nd last voxel will connect to last voxel and the last voxel will connect to the 2nd last voxel  */
        /* The problem with the last Voxel is that its right neighbour will be on the next process (or doesn't exist if it is the last process)               */
        /* Whenever we call functions: connect_voxels_indices_only() and connect_voxels_global_indices_only(), the jump will be a local jump only.            */
        /*----------------------------------------------------------------------------------------------------------------------------------------------------*/
        /*
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
        */

        /*--------------------------------------------------------------------------*/
        /*                  Y-aligned connections                                   */
        /*                Again broken into three parts                             */
        /*--------------------------------------------------------------------------*/

        // Non-boundary parts of each sub-domain
        /*
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
        */
        /*--------------------------------------------------------------------------*/
        /*                  Z-aligned connections                                   */
        /*                Again broken into three parts                             */
        /*--------------------------------------------------------------------------*/
        /*
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
        */
        /*--------------------------------------------------------------------------------------------------------------------------- */
        /* In the example that I am following, use_voxel_faces is false, hence no need to parallelize this yet.                       */
        /* This is very similar to finding neighbours of voxels but most importantly, the connected_voxels_indices[] vector           */
        /* is again initialized over here.                                                                                            */
        /*--------------------------------------------------------------------------------------------------------------------------- */
        /*
        if (use_voxel_faces)
        {
            create_voxel_faces();
        }
        */
        /*--------------------------------------------------------------------------------------------------------------------------- */
        /* Moore neighbourhood is possibly not used anywhere, hence leave parallelization for later
        /*--------------------------------------------------------------------------------------------------------------------------- */

        // create_moore_neighborhood();*/
        return;
    }

    void General_Mesh::connect_voxels_indices_only(int i, int j, double SA) // done
    {

        // Create local index adjacency list

        // connected_voxel_indices[i].push_back(j);
        // connected_voxel_indices[j].push_back(i);

        return;
    }

    void General_Mesh::connect_voxels_global_indices_only(int i, int j, double SA) // done
    {

        // Create an adjacency list of global indexes

        // connected_voxel_global_indices[i].push_back(voxels[j].global_mesh_index);
        // connected_voxel_global_indices[j].push_back(voxels[i].global_mesh_index);

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

        return (k * global_num_x_voxels * global_num_y_voxels + j * global_num_x_voxels + i);
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

        int process_local_index_of_voxel_containing_basic_agent = diff_z_coord * local_num_x_voxels * local_num_y_voxels +
                                                                  diff_y_coord * local_num_x_voxels +
                                                                  diff_x_coord;

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
        unsigned int cols = (unsigned int)(size * ncols); // number_of_data_entries; // storing data as cols

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

        std::cout << "Write to matlab is desactivated " << std::endl;

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
        // int size_of_each_datum = 3 + 1 + (*p_density_vectors)[0].size(); // Jose : to be done

        // Possibly we do not need to return anything over here, we can write a separate file at Master
        // All processes call this function - because William Groppe says MPI_File_open is collective operation

        // write_matlab_header(size_of_each_datum, number_of_data_entries, filename, "multiscale_microenvironment", rank, size, mpi_Cart_comm);

        MPI_Barrier(mpi_Cart_comm);

        // storing data as cols
        // buffer = new double[number_of_data_entries * size_of_each_datum];

        // std::cout<<"CX	"<<"CY	"<<"CZ	"<<"Vol	"<<"Density	\n";

        int n = 0;
        /*

        for (int i = 0; i < number_of_data_entries; i++)
        {

            buffer[n++] = mesh.voxels[i].center[0];
            // std::cout<<mesh.voxels[i].center[0]<<"	";
            buffer[n++] = mesh.voxels[i].center[1];
            // std::cout<<mesh.voxels[i].center[1]<<"	";
            buffer[n++] = mesh.voxels[i].center[2];
            // std::cout<<mesh.voxels[i].center[2]<<"	";
            buffer[n++] = mesh.voxels[i].volume;
            // std::cout<<mesh.voxels[i].volume<<"	";

            // fwrite( (char*) &( mesh.voxels[i].center[0] ) , sizeof(double) , 1 , fp );
            // fwrite( (char*) &( mesh.voxels[i].center[1] ) , sizeof(double) , 1 , fp );
            // fwrite( (char*) &( mesh.voxels[i].center[2] ) , sizeof(double) , 1 , fp );
            // fwrite( (char*) &( mesh.voxels[i].volume ) , sizeof(double) , 1 , fp );

            // densities
            // Jose: to be done
            for (int j = 0; j < (*p_density_vectors)[i].size(); j++)
            {
                buffer[n++] = ((*p_density_vectors)[i])[j];
                // std::cout<<((*p_density_vectors)[i])[j]<<"	";
                // fwrite( (char*) &( ((*p_density_vectors)[i])[j] ) , sizeof(double) , 1 , fp );
            }
            // std::cout<<"\n";
        }
        */
        strcpy(char_filename, filename.c_str());

        MPI_File_open(mpi_Cart_comm, char_filename, MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); // This file is already created while writing Matlab header
        MPI_File_get_size(fh, &file_size);

        // offset = file_size + rank * sizeof(double) * number_of_data_entries * size_of_each_datum;
        etype = MPI_DOUBLE;
        filetype = MPI_DOUBLE;
        // fvoxelselements_to_write = number_of_data_entries * size_of_each_datum;

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
        //int granurality = 64;
        //std::vector<MPI_Request *> send_req(granurality + 1);
        //std::vector<MPI_Request *> recv_req(granurality + 1);
        MPI_Request send_req[granurality+1];
        MPI_Request recv_req[granurality+1];
        std::ofstream file(M.timing_csv, std::ios::app);
        /*
        for (int i  = 0; i <= granurality; ++i) 
        {
            
            send_req[i] = new MPI_Request;
            recv_req[i] = new MPI_Request;
        }*/
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
                        MPI_Isend(&(M.thomas_cx[M.mesh.x_coordinates.size() - 1][0]), M.thomas_cx[M.mesh.x_coordinates.size() - 1].size(), MPI_DOUBLE, ser_ctr + 1, 1111, mpi_Cart_comm, &send_req[0]);
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


        // x-diffusion
        // cout << "Rank " << rank << " starting x-diffusion" << endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        M.apply_dirichlet_conditions(rank, size);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto apply_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (rank == 0){
            //std::cout << "   Apply dirichlet: " << duration_ms << "ms" << std::endl;
        }

        // cout << "Rank " << rank << " apply dirichlet condition done" << endl;
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

        int step_size = (z_size * y_size) / granurality;

        int snd_data_size = step_size * p_size; // Number of data elements to be sent
        int rcv_data_size = step_size * p_size; // All p_density_vectors elements have same size, use anyone

        int snd_data_size_last = ((z_size * y_size) % granurality) * p_size; // Number of data elements to be sent
        int rcv_data_size_last = ((z_size * y_size) % granurality) * p_size;
        bool last_iteration = ((z_size * y_size) % granurality) > 0;
        // cout << "Rank " << rank << " snd_data_size: " << snd_data_size << " rcv_data_size: " << rcv_data_size << endl;

        /* So row is along Z axis, column of each row is along Y-axis and each element has p_density_vector*/

        std::vector<double> block3d(z_size * y_size * p_size);

        // t_strt_x = MPI_Wtime();
        // cout << "Rank " << rank << " starting forward" << endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        if (rank == 0)
        {
            // cout << "Rank " << rank << "is computing" << endl;
            for (int step = 0; step < granurality; ++step)
            {
                int initial_index = step * snd_data_size;
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + snd_data_size; index += p_size)
                {
                    int index_dec = index; 
                    for (int d = 0; d < M.thomas_denomx[0].size(); d++)
                    {
                        M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                    }

                    for (int i = 1; i < M.mesh.x_coordinates.size(); i++)
                    {
                        
                        int index_inc = index_dec + M.thomas_i_jump;
                        // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                        }

                        //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                        for (int d = 0; d < M.thomas_denomx[i].size(); d++)
                        {
                            M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                        }
                        index_dec = index_inc;
                    }
                }

                if (size > 1) {
                    int x_end = M.mesh.x_coordinates.size() - 1;
                    int offset = step * snd_data_size;
                    MPI_Status status;
                    MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + offset]), snd_data_size, MPI_DOUBLE, rank + 1, step, mpi_Cart_comm, &send_req[step]);
                }
            }
            //Last iteration
            if (last_iteration) {
                int initial_index = granurality * snd_data_size;
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + snd_data_size_last; index += p_size)
                {
                    int index_dec = index; 
                    for (int d = 0; d < M.thomas_denomx[0].size(); d++)
                    {
                        M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                    }

                    for (int i = 1; i < M.mesh.x_coordinates.size(); i++)
                    {
                        int index_inc = index_dec + M.thomas_i_jump;
                        // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                        }

                        //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                        for (int d = 0; d < M.thomas_denomx[i].size(); d++)
                        {
                            M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                        }
                        index_dec = index_inc;
                    }
                }

                if (size > 1) {
                    int x_end = M.mesh.x_coordinates.size() - 1;
                    int offset = granurality * snd_data_size;
                    MPI_Status status;
                    MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + offset]), snd_data_size_last, MPI_DOUBLE, rank + 1, granurality, mpi_Cart_comm, &send_req[granurality]);
                    
                }
            }
        }
        else
        {
            if (rank >= 1 && rank <= (size - 1))
            {
                for (int step = 0; step < granurality; ++step)
                {
                    int initial_index = step * snd_data_size;
                    MPI_Irecv(&(block3d[initial_index]), rcv_data_size, MPI_DOUBLE, rank-1, step, mpi_Cart_comm, &(recv_req[step]));
                }
                if (last_iteration)
                    MPI_Irecv(&(block3d[granurality*snd_data_size]), rcv_data_size_last, MPI_DOUBLE, rank-1, granurality, mpi_Cart_comm, &(recv_req[granurality]));
                for (int step = 0; step < granurality; ++step)
                {
                    int initial_index = step * snd_data_size;
                    MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                    #pragma omp parallel for
                    for (int index = initial_index; index < initial_index + snd_data_size; index += p_size)
                    {
                        // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, block3d[k][j]);
                        int index_dec = index;
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index + d] += M.thomas_constant1[d] * block3d[index + d];
                        }

                        //(*M.p_density_vectors)[n] /= M.thomas_denomx[0];
                        for (int d = 0; d < M.thomas_denomx[0].size(); d++)
                        {
                            M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                        }

                        for (int i = 1; i < M.mesh.x_coordinates.size(); i++)
                        {

                            int index_inc = index_dec + M.thomas_i_jump;
                            // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                            for (int d = 0; d < M.thomas_k_jump; d++)
                            {
                                M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                            }
                            //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                            for (int d = 0; d < M.thomas_denomx[i].size(); d++)
                            {
                                M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                            }

                            index_dec = index_inc;
                        }
                        
                    }
                    if (rank < (size - 1))
                    {
                        int x_end = M.mesh.x_coordinates.size() - 1;
                        MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + initial_index]), snd_data_size, MPI_DOUBLE, rank + 1, step, mpi_Cart_comm, &send_req[step]);
                    }
                }
                if (last_iteration)
                {
                    int initial_index = granurality * snd_data_size;
                    MPI_Wait(&recv_req[granurality], MPI_STATUS_IGNORE); //Need to change
                    #pragma omp parallel for
                    for (int index = initial_index; index < initial_index + snd_data_size_last; index += p_size)
                    {
                        // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, block3d[k][j]);
                        int index_dec = index;
                        for (int d = 0; d < M.thomas_k_jump; d++)
                        {
                            M.p_density_vectors[index + d] += M.thomas_constant1[d] * block3d[index + d];
                        }

                        //(*M.p_density_vectors)[n] /= M.thomas_denomx[0];
                        for (int d = 0; d < M.thomas_denomx[0].size(); d++)
                        {
                            M.p_density_vectors[index + d] /= M.thomas_denomx[0][d];
                        }

                        for (int i = 1; i < M.mesh.x_coordinates.size(); i++)
                        {

                            int index_inc = index_dec + M.thomas_i_jump;
                            // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_i_jump]);
                            for (int d = 0; d < M.thomas_k_jump; d++)
                            {
                                M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index_dec + d];
                            }
                            //(*M.p_density_vectors)[n] /= M.thomas_denomx[i];
                            for (int d = 0; d < M.thomas_denomx[i].size(); d++)
                            {
                                M.p_density_vectors[index_inc + d] /= M.thomas_denomx[i][d];
                            }

                            index_dec = index_inc;
                        }
                        
                    }
                    // End of computation region
                    if (rank < (size - 1))
                    {
                        int x_end = M.mesh.x_coordinates.size() - 1;
                        MPI_Request aux;
                        MPI_Isend(&(M.p_density_vectors[x_end * M.thomas_i_jump + initial_index]), snd_data_size_last, MPI_DOUBLE, rank + 1, granurality, mpi_Cart_comm, &send_req[granurality]);
                      
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
                int initial_index = ((M.mesh.x_coordinates.size() - 1)*M.thomas_i_jump) + (step * snd_data_size);
                #pragma omp parallel for

                for (int index = initial_index; index < initial_index + snd_data_size; index += p_size)
                {
                    int index_aux = index;
                    //int index = j * M.thomas_j_jump + k * M.thomas_k_jump + (M.mesh.x_coordinates.size() - 1) * M.thomas_i_jump;
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
                    MPI_Isend(&(M.p_density_vectors[step * snd_data_size]), snd_data_size, MPI_DOUBLE, rank - 1, step, mpi_Cart_comm, &send_req[step]);
                }
            }

            //Last iteration
            if (last_iteration) {
                int initial_index = ((M.mesh.x_coordinates.size() - 1)*M.thomas_i_jump) + (granurality * snd_data_size);
                #pragma omp parallel for
                for (int index = initial_index; index < initial_index + snd_data_size_last; index += p_size)
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
                    MPI_Isend(&(M.p_density_vectors[granurality * snd_data_size]), snd_data_size_last, MPI_DOUBLE, rank - 1, granurality, mpi_Cart_comm, &send_req[granurality]);
                    //cout << "Rank " << rank << " has send" << endl;
                }
            
            }
        }
        else
        {
            MPI_Request status[granurality]; 
            MPI_Request status_last;
            for (int step = 0; step < granurality; ++step)
            {
                MPI_Irecv(&(block3d[step*snd_data_size]), rcv_data_size, MPI_DOUBLE, rank+1, step, mpi_Cart_comm, &recv_req[step]);
            }
            if (last_iteration)
                MPI_Irecv(&(block3d[granurality*snd_data_size]), rcv_data_size_last, MPI_DOUBLE, rank+1, granurality, mpi_Cart_comm, &recv_req[granurality]);

            
            for (int step = 0; step < granurality; ++step)
            {
                int initial_index = ((M.mesh.x_coordinates.size() - 1)*M.thomas_i_jump) + (step * snd_data_size);
                int index_3d_initial = (step * snd_data_size);
                MPI_Wait(&recv_req[step], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (int offset = 0; offset < snd_data_size; offset += p_size)
                {
                    int index_aux = initial_index + offset;
                    int index_3d = index_3d_initial + offset;
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_aux + d] -= M.thomas_cx[M.mesh.x_coordinates.size() - 1][d] * block3d[index_3d + d];
                    }

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
                if (rank > 0)
                {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[step * snd_data_size]), snd_data_size, MPI_DOUBLE, rank - 1, step, mpi_Cart_comm, &send_req[step]);
                    // cout << "Rank " << rank << " has send" << endl;
                }
            }
            if (last_iteration)
            {
                int initial_index = ((M.mesh.x_coordinates.size() - 1)*M.thomas_i_jump) + (granurality * snd_data_size);
                int index_3d_initial = (granurality * snd_data_size);
                MPI_Wait(&recv_req[granurality], MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for (int offset = 0; offset < snd_data_size_last; offset += p_size)
                {
                    int index_aux = initial_index + offset;
                    //int index = j * M.thomas_j_jump + k * M.thomas_k_jump + (M.mesh.x_coordinates.size() - 1) * M.thomas_i_jump;
                    int index_3d = index_3d_initial + offset;
                    // naxpy(&(*M.p_density_vectors)[n], M.thomas_cx[M.mesh.x_coordinates.size() - 1], block3d[k][j]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_aux + d] -= M.thomas_cx[M.mesh.x_coordinates.size() - 1][d] * block3d[index_3d + d];
                    }

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
                if (rank > 0)
                {
                    MPI_Request aux;
                    MPI_Isend(&(M.p_density_vectors[granurality * snd_data_size]), snd_data_size_last, MPI_DOUBLE, rank - 1, granurality, mpi_Cart_comm, &send_req[granurality]);
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
        if (rank == 0)
            //std::cout << "   Apply dirichlet: " << duration_ms << "ms" << std::endl;
            //std::ofstream << duration_us << ",";

        // cout << "Rank " << rank << " Y diffusion" << endl;
        // t_strt_y = MPI_Wtime();
        start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static,8)
        for (int k = 0; k < M.mesh.z_coordinates.size(); k++)
        {
            for (int i = 0; i < M.mesh.x_coordinates.size(); i++)
            {

                int index = i * M.thomas_i_jump + k * M.thomas_k_jump;
                //(*M.p_density_vectors)[n] /= M.thomas_denomy[0];
                for (int d = 0; d < M.thomas_denomy[0].size(); d++)
                {
                    M.p_density_vectors[index + d] /= M.thomas_denomy[0][d];
                }

                for (int j = 1; j < M.mesh.y_coordinates.size(); j++)
                {

                    int index_inc = index + M.thomas_j_jump;
                    // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_j_jump]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index + d];
                    }
                    //(*M.p_density_vectors)[n] /= M.thomas_denomy[j];
                    for (int d = 0; d < M.thomas_denomy[j].size(); d++)
                    {
                        M.p_density_vectors[index_inc + d] /= M.thomas_denomy[j][d];
                    }
                    index = index_inc;
                }

                // back substitution

                index = i * M.thomas_i_jump + k * M.thomas_k_jump + (M.thomas_j_jump * (M.mesh.y_coordinates.size() - 1));
                for (int j = M.mesh.y_coordinates.size() - 2; j >= 0; j--)
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
        // t_end_y = MPI_Wtime();
        // std::cout<<"Y solve time = "<<(t_end_y-t_strt_y)<<std::endl;


        // cout << "Rank " << rank << " apply dirichlet" << endl;
         



        
        start_time = std::chrono::high_resolution_clock::now();
        M.apply_dirichlet_conditions(rank, size);
        end_time = std::chrono::high_resolution_clock::now();
        apply_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (rank == 0)
        //std::cout << "   Apply dirichlet: " << duration_ms << "ms" << std::endl;

        // cout << "Rank " << rank << " Z diffusion" << endl;
        
        start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
        for (int j = 0; j < M.mesh.y_coordinates.size(); j++)
        {

            for (int i = 0; i < M.mesh.x_coordinates.size(); i++)
            {

                int index = i * M.thomas_i_jump + j * M.thomas_j_jump;
                //(*M.p_density_vectors)[n] /= M.thomas_denomz[0];
                for (int d = 0; d < M.thomas_denomz[0].size(); d++)
                {
                    M.p_density_vectors[index + d] /= M.thomas_denomz[0][d];
                }

                // should be an empty loop if mesh.z_coordinates.size() < 2
                for (int k = 1; k < M.mesh.z_coordinates.size(); k++)
                {

                    int index_inc = index + M.thomas_k_jump;
                    // axpy(&(*M.p_density_vectors)[n], M.thomas_constant1, (*M.p_density_vectors)[n - M.thomas_k_jump]);
                    for (int d = 0; d < M.thomas_k_jump; d++)
                    {
                        M.p_density_vectors[index_inc + d] += M.thomas_constant1[d] * M.p_density_vectors[index + d];
                    }
                    //(*M.p_density_vectors)[n] /= M.thomas_denomz[k];
                    for (int d = 0; d < M.thomas_denomz[k].size(); d++)
                    {
                        M.p_density_vectors[index_inc + d] /= M.thomas_denomz[k][d];
                    }

                    index = index_inc;
                }

                // for parallelization need to break forward elimination and back substitution into
                // should be an empty loop if mesh.z_coordinates.size() < 2

                index = i * M.thomas_i_jump + j * M.thomas_j_jump + (M.thomas_k_jump * (M.mesh.z_coordinates.size() - 1));
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
