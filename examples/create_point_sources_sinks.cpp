/*
#############################################################################
# This file has been written by Gaurav Saxena                               #
# for parallelizing the example of ./examples/tutorial1_BioFVM.cpp          #
# It contains two major functions (1) Creating point sources in parallel    #
# Second one creates point sinks in parallel                                #
# These functions are general and can be used by other examples to          #
# to create point sources and sinks.                                        #
#############################################################################
*/

#include<vector>
#include "../BioFVM_microenvironment.h"
#include "../BioFVM_basic_agent.h"

using namespace BioFVM; 


double pi= 3.1415926535897932384626433832795;

double UniformRandom()
{	
	return ((double) rand() / (RAND_MAX));
}


/*---------------------------------------------------------------------------------------------*/
/* The Microenvironment object should be passed by reference otherwise it creates a new object */
/* The other alternative is to make this function a member function of Microenvironment class  */
/* then manipulate using the 'this' pointer                                                    */
/*---------------------------------------------------------------------------------------------*/


void create_point_sources(double cell_radius, double dt, int num_sources, Microenvironment &microenvironment, int *mpi_Dims, int mpi_Rank, int mpi_Size, MPI_Comm mpi_Cart_comm)
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
        std::vector<std::vector<double>> vector_IDs(mpi_Size);    //vector_IDs[r] contains list of IDs to be sent to rank 'r', IDs of Basic_Agents are unique in domain
        std::vector<int> sources_per_proc_at_root(mpi_Size); 
        std::vector<MPI_Request> send_req(mpi_Size);
        std::vector<double> list_of_coords;
        std::vector<int> list_of_IDs;
        
        MPI_Request recv_req;
        int sources_per_proc; 
        
        
        if(mpi_Rank == 0)
        {
            
            for(int i=0; i<num_sources; i++)
            {
                
                /*------------------------------------------------------------------------------------------------*/
                /* Generate 3 random numbers between [0,1] and multiply by Range to get random position in domain */
                /* i.e. Multiply by x_max - x_min OR y_max - y_min OR z_max-z_min => in general the Range         */
                /*------------------------------------------------------------------------------------------------*/
                
                for(int j=0; j < 3 ; j++ )
                    tempPoint[j] = UniformRandom()*1000;   
                
                /*------------------------------------------------------------------------------------------------*/
                /* Find the MPI process coordinate using local_x/y/z_voxels                                       */
                /* Remember BioFVM X direction is left to right                                                   */
                /* MPI Process X direction is top to bottom                                                       */
                /*------------------------------------------------------------------------------------------------*/
                
                proc_y_coord = tempPoint[0]/(local_x_voxels * dx);
                proc_x_coord = (mpi_Dims[0] - 1 - tempPoint[1]/(local_y_voxels * dy));
                proc_z_coord = tempPoint[2]/(local_z_voxels * dz); 
                
                proc_index_rank = proc_x_coord * mpi_Dims[1] * mpi_Dims[2] + proc_y_coord * mpi_Dims[2] + proc_z_coord; 
                
                vector_coords[proc_index_rank].push_back(tempPoint[0]);          //Create a list of coordinates for a particular rank 
                vector_coords[proc_index_rank].push_back(tempPoint[1]);          //i.e. for some rank 'r', vector_coords[r] = tempPoint[0]-->temPoint[1]-->temPoint[2]
                vector_coords[proc_index_rank].push_back(tempPoint[2]);
                vector_IDs[proc_index_rank].push_back(unique_ID);               //Similarly list of unique IDs for that particular rank 
                
                unique_ID++; 
            }
            
            for(int k=0 ; k<vector_IDs.size(); k++)                                   //Need to know if there is a rank which has got no points
                sources_per_proc_at_root[k] = vector_IDs[k].size();                   //sources_per_proc[k] contains no. of points going to kth rank
            
            
            for(int k=1 ; k<mpi_Size; k++)                                            
            {
                MPI_Isend(&sources_per_proc_at_root[k], 1, MPI_INT, k, 0, mpi_Cart_comm, &send_req[k]); //Every process needs to know how many points it will receive
                MPI_Wait(&send_req[k], MPI_STATUS_IGNORE);
                
                if(sources_per_proc_at_root[k] != 0)                                                    //Send to only those processes which have got at least one point
                {
                   MPI_Isend(&vector_coords[k], 3 * sources_per_proc_at_root[k], MPI_DOUBLE, k, 0, mpi_Cart_comm, &send_req[k]); 
                   MPI_Wait(&send_req[k], MPI_STATUS_IGNORE);
                   
                   MPI_Isend(&vector_IDs[k], sources_per_proc_at_root[k], MPI_INT, k, 0, mpi_Cart_comm, &send_req[k]); 
                   MPI_Wait(&send_req[k], MPI_STATUS_IGNORE);
                   
                }
            }
           
                /*------------------------------------------------------------------------------------------------*/
                /* Since the send above was not for the root, we need to copy parameters at root                  */
                /*------------------------------------------------------------------------------------------------*/
                
            if(sources_per_proc_at_root[0] != 0)                                            
            {
               sources_per_proc = sources_per_proc_at_root[0]; 
               
               list_of_coords.resize(3 * sources_per_proc);
               list_of_IDs.resize(sources_per_proc);
               
               for(int k=0; k<vector_coords[0].size(); k++)
                   list_of_coords[k] = vector_coords[0][k]; 
               
               for(int k=0; k<sources_per_proc; k++)
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
                
          MPI_Irecv(&sources_per_proc, 1, MPI_INT, 0, 0, mpi_Cart_comm, &recv_req); 
          MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
          
          if(sources_per_proc != 0)
          {
              list_of_coords.resize(3 * sources_per_proc);
              list_of_IDs.resize(sources_per_proc);
              
              MPI_Irecv((void*)&list_of_coords[0], 3 * sources_per_proc, MPI_DOUBLE, 0, 0, mpi_Cart_comm, &recv_req);
              MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
              
              MPI_Irecv((void*)&list_of_IDs[0], sources_per_proc, MPI_INT, 0, 0, mpi_Cart_comm, &recv_req);
              MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
              
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
            temp_point_source->assign_position(tempPoint);
            temp_point_source->ID = list_of_IDs[k];                                         //Added this statement to overwrite Constructor generated ID. 
            temp_point_source->set_total_volume( (4.0/3.0)*pi*pow(cell_radius,3.0) );
            (*temp_point_source->secretion_rates)[0]=10;
            (*temp_point_source->saturation_densities)[0]=1;
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
        std::vector<std::vector<double>> vector_IDs(mpi_Size);    //vector_IDs[r] contains list of IDs to be sent to rank 'r', IDs of Basic_Agents are unique in domain
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
                    tempPoint[j] = UniformRandom()*1000;   
                
                /*------------------------------------------------------------------------------------------------*/
                /* Find the MPI process coordinate using local_x/y/z_voxels                                       */
                /* Remember BioFVM X direction is left to right                                                   */
                /* MPI Process X direction is top to bottom                                                       */
                /*------------------------------------------------------------------------------------------------*/
                
                proc_y_coord = tempPoint[0]/(local_x_voxels * dx);
                proc_x_coord = (mpi_Dims[0] - 1 - tempPoint[1]/(local_y_voxels * dy));
                proc_z_coord = tempPoint[2]/(local_z_voxels * dz); 
                
                proc_index_rank = proc_x_coord * mpi_Dims[1] * mpi_Dims[2] + proc_y_coord * mpi_Dims[2] + proc_z_coord; 
                
                vector_coords[proc_index_rank].push_back(tempPoint[0]);          //Create a list of coordinates for a particular rank 
                vector_coords[proc_index_rank].push_back(tempPoint[1]);          //i.e. for some rank 'r', vector_coords[r] = tempPoint[0]-->temPoint[1]-->temPoint[2]
                vector_coords[proc_index_rank].push_back(tempPoint[2]);
                vector_IDs[proc_index_rank].push_back(unique_ID);               //Similarly list of unique IDs for that particular rank 
                
                unique_ID++; 
            }
            
            for(int k=0 ; k<vector_IDs.size(); k++)                                   //Need to know if there is a rank which has got no points
                sinks_per_proc_at_root[k] = vector_IDs[k].size();                   //sinks_per_proc[k] contains no. of points going to kth rank
            
            
            for(int k=1 ; k<mpi_Size; k++)                                            
            {
                MPI_Isend(&sinks_per_proc_at_root[k], 1, MPI_INT, k, 0, mpi_Cart_comm, &send_req[k]); //Every process needs to know how many points it will receive
                MPI_Wait(&send_req[k], MPI_STATUS_IGNORE);
                
                if(sinks_per_proc_at_root[k] != 0)                                                    //Send to only those processes which have got at least one point
                {
                   MPI_Isend(&vector_coords[k], 3 * sinks_per_proc_at_root[k], MPI_DOUBLE, k, 0, mpi_Cart_comm, &send_req[k]); 
                   MPI_Wait(&send_req[k], MPI_STATUS_IGNORE);
                   
                   MPI_Isend(&vector_IDs[k], sinks_per_proc_at_root[k], MPI_INT, k, 0, mpi_Cart_comm, &send_req[k]); 
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
              
              MPI_Irecv((void*)&list_of_coords[0], 3 * sinks_per_proc, MPI_DOUBLE, 0, 0, mpi_Cart_comm, &recv_req);
              MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
              
              MPI_Irecv((void*)&list_of_IDs[0], sinks_per_proc, MPI_INT, 0, 0, mpi_Cart_comm, &recv_req);
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
            temp_point_sink->assign_position(tempPoint);
            temp_point_sink->ID = list_of_IDs[k];                                         //Added this statement to overwrite Constructor generated ID. 
            temp_point_sink->set_total_volume( (4.0/3.0)*pi*pow(cell_radius,3.0) );
            (*temp_point_sink->uptake_rates)[0]=0.8;
            temp_point_sink->set_internal_uptake_constants(dt); 
        }
            
}
