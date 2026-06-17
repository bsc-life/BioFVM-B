/*
 * BioFVM-B diffusion example with deterministic global agents.
 *
 * This keeps the same initial density field as density_diffusion_demo.cpp, but
 * creates source/sink agents from one global random sequence and lets each MPI
 * rank instantiate only the agents owned by its local subdomain.
 */

#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cerrno>
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../BioFVM.h"

using namespace BioFVM;

namespace
{
const int oxygen = 0;
const double pi = 3.1415926535897932384626433832795;
int center_voxel_index = 0;

struct AgentSpec
{
	int id;
	std::vector<double> position;
	double secretion_rate;
	double saturation_density;
	double uptake_rate;
};

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

double squared_distance(const std::vector<double>& a, const std::vector<double>& b)
{
	double out = 0.0;
	for (int i = 0; i < 3; ++i)
	{
		const double d = a[i] - b[i];
		out += d * d;
	}
	return out;
}

std::vector<double> domain_center(const Microenvironment* microenvironment)
{
	std::vector<double> center(3, 0.0);
	center[0] = 0.5 * (microenvironment->mesh.bounding_box[0] + microenvironment->mesh.bounding_box[3]);
	center[1] = 0.5 * (microenvironment->mesh.bounding_box[1] + microenvironment->mesh.bounding_box[4]);
	center[2] = 0.5 * (microenvironment->mesh.bounding_box[2] + microenvironment->mesh.bounding_box[5]);
	return center;
}

double parse_double(const char* input, double fallback)
{
	std::stringstream stream(input);
	double value = fallback;
	stream >> value;
	return value > 0.0 ? value : fallback;
}

std::vector<double> random_position(const Microenvironment& microenvironment, std::mt19937& rng)
{
	std::vector<double> position(3, 0.0);
	const std::vector<double>* coordinates[3] = {
		&microenvironment.mesh.x_coordinates,
		&microenvironment.mesh.y_coordinates,
		&microenvironment.mesh.z_coordinates
	};

	for (int i = 0; i < 3; ++i)
	{
		const std::vector<double>& axis = *(coordinates[i]);
		std::uniform_real_distribution<double> distribution(axis.front(), axis.back());
		position[i] = distribution(rng);
	}

	return position;
}

std::vector<AgentSpec> make_global_agents(
	int number_of_agents,
	const std::vector<double>& bounding_box,
	double mesh_resolution)
{
	std::vector<AgentSpec> agents;
	agents.reserve(number_of_agents);

	std::mt19937 rng(13);
	std::uniform_real_distribution<double> x_distribution(
		bounding_box[0] + 0.5 * mesh_resolution,
		bounding_box[3] - 0.5 * mesh_resolution);
	std::uniform_real_distribution<double> y_distribution(
		bounding_box[1] + 0.5 * mesh_resolution,
		bounding_box[4] - 0.5 * mesh_resolution);
	std::uniform_real_distribution<double> z_distribution(
		bounding_box[2] + 0.5 * mesh_resolution,
		bounding_box[5] - 0.5 * mesh_resolution);

	const int number_of_sources = number_of_agents / 2;
	for (int i = 0; i < number_of_agents; ++i)
	{
		AgentSpec spec;
		spec.id = i;
		spec.position.assign(3, 0.0);
		spec.position[0] = x_distribution(rng);
		spec.position[1] = y_distribution(rng);
		spec.position[2] = z_distribution(rng);

		if (i < number_of_sources)
		{
			spec.secretion_rate = 10.0;
			spec.saturation_density = 5.0;
			spec.uptake_rate = 0.0;
		}
		else
		{
			spec.secretion_rate = 0.0;
			spec.saturation_density = 0.0;
			spec.uptake_rate = 0.8;
		}

		agents.push_back(spec);
	}

	return agents;
}

bool coordinate_in_local_axis(
	double value,
	const std::vector<double>& coordinates,
	double spacing,
	double global_max)
{
	const double local_min = coordinates.front() - 0.5 * spacing;
	const double local_max = coordinates.back() + 0.5 * spacing;
	return value >= local_min && (value < local_max || (local_max >= global_max && value <= local_max));
}

bool rank_owns_position(const Microenvironment& microenvironment, const std::vector<double>& position)
{
	return coordinate_in_local_axis(
			position[0],
			microenvironment.mesh.x_coordinates,
			microenvironment.mesh.dx,
			microenvironment.mesh.bounding_box[3]) &&
		coordinate_in_local_axis(
			position[1],
			microenvironment.mesh.y_coordinates,
			microenvironment.mesh.dy,
			microenvironment.mesh.bounding_box[4]) &&
		coordinate_in_local_axis(
			position[2],
			microenvironment.mesh.z_coordinates,
			microenvironment.mesh.dz,
			microenvironment.mesh.bounding_box[5]);
}

bool ensure_directory(const std::string& directory)
{
	if (directory.empty() || directory == ".")
	{
		return true;
	}

	std::string current;
	for (std::size_t i = 0; i < directory.size(); ++i)
	{
		current.push_back(directory[i]);
		if (directory[i] != '/' && i + 1 != directory.size())
		{
			continue;
		}

		if (current.empty() || current == "/")
		{
			continue;
		}

		if (mkdir(current.c_str(), 0775) != 0 && errno != EEXIST)
		{
			return false;
		}
	}

	return true;
}

int split_count_for_rank(int total_count, int mpi_Rank, int mpi_Size)
{
	const int base_count = total_count / mpi_Size;
	const int remainder = total_count % mpi_Size;
	return base_count + (mpi_Rank < remainder ? 1 : 0);
}

void create_agent_from_spec(
	Microenvironment& microenvironment,
	const AgentSpec& spec,
	double cell_radius,
	double dt,
	int mpi_Rank,
	int* mpi_Dims)
{
	Basic_Agent* agent = create_basic_agent();
	agent->ID = spec.id;
	agent->register_microenvironment(&microenvironment);
	agent->assign_position(spec.position, mpi_Rank, mpi_Dims);
	agent->set_total_volume((4.0 / 3.0) * pi * std::pow(cell_radius, 3.0));
	(*agent->secretion_rates)[oxygen] = spec.secretion_rate;
	(*agent->saturation_densities)[oxygen] = spec.saturation_density;
	(*agent->uptake_rates)[oxygen] = spec.uptake_rate;
	agent->set_internal_uptake_constants(dt);
}

int create_deterministic_sources_and_sinks(
	Microenvironment& microenvironment,
	double cell_radius,
	double dt,
	int number_of_agents,
	int mpi_Rank,
	int* mpi_Dims)
{
	const std::vector<AgentSpec> global_agents = make_global_agents(
		number_of_agents,
		microenvironment.mesh.bounding_box,
		microenvironment.mesh.dx);

	int local_agents_created = 0;
	for (std::size_t i = 0; i < global_agents.size(); ++i)
	{
		if (!rank_owns_position(microenvironment, global_agents[i].position))
		{
			continue;
		}

		create_agent_from_spec(microenvironment, global_agents[i], cell_radius, dt, mpi_Rank, mpi_Dims);
		++local_agents_created;
	}

	return local_agents_created;
}

std::string join_path(const std::string& directory, const std::string& filename)
{
	if (directory.empty() || directory == ".")
	{
		return filename;
	}

	if (directory[directory.size() - 1] == '/')
	{
		return directory + filename;
	}

	return directory + "/" + filename;
}

int parse_int(const char* input, int fallback)
{
	std::stringstream stream(input);
	int value = fallback;
	stream >> value;
	return value > 0 ? value : fallback;
}
}

void dirichlet_initial_conditions(Microenvironment* microenvironment, int mpi_Rank, int mpi_Size) {
	int num_substrates = microenvironment->number_of_densities();
	vector<double> subs(1, 1.0);
	for (int i = 0; i < microenvironment->mesh.x_size; i++)
	{
		for (int j = 0; j < microenvironment->mesh.y_size; j++)
		{
			for (int k = 0; k < microenvironment->mesh.z_size; k++)
			{
				int index =  i*microenvironment->mesh.y_size*microenvironment->mesh.z_size*num_substrates + 
							 j*microenvironment->mesh.z_size*num_substrates+
							 k*num_substrates;
				copy(subs.begin(), subs.end(), (*microenvironment->p_density_vectors).begin() + index);
			}
		}
	}

	microenvironment->set_substrate_dirichlet_value(oxygen, 100.0);

	microenvironment->apply_dirichlet_conditions(mpi_Rank, mpi_Size);


}

void centered_initial_conditions(
	Microenvironment* microenvironment,
	double dt,
	int mpi_Rank,
	int mpi_Size,
	int* mpi_Dims,
	int number_of_agents,
	double center_concentration)
{
	int num_substrates = microenvironment->number_of_densities();
	vector<double> subs(1, 0.0);
	for (int i = 0; i < microenvironment->mesh.x_size; i++)
	{
		for (int j = 0; j < microenvironment->mesh.y_size; j++)
		{
			for (int k = 0; k < microenvironment->mesh.z_size; k++)
			{
				int index =  i*microenvironment->mesh.y_size*microenvironment->mesh.z_size*num_substrates + 
							 j*microenvironment->mesh.z_size*num_substrates+
							 k*num_substrates;
				copy(subs.begin(), subs.end(), (*microenvironment->p_density_vectors).begin() + index);
			}
		}
	}

	std::vector<double> center = domain_center(microenvironment);
	const double center_radius = 50.0;
	const double center_radius_squared = center_radius * center_radius;

	center_voxel_index = microenvironment->mesh.nearest_voxel_local_index(center, mpi_Rank, mpi_Dims);
	const bool rank_owns_center =
		center_voxel_index >= 0 &&
		center_voxel_index < microenvironment->number_of_voxels() &&
		squared_distance(microenvironment->voxels(center_voxel_index).center, center) <= center_radius_squared;
	if (!rank_owns_center)
	{
		center_voxel_index = -1;
	}

	for (int i = 0; i < microenvironment->number_of_voxels(); ++i)
	{
		const double r2 = squared_distance(microenvironment->voxels(i).center, center);
		if (r2 <= center_radius_squared)
		{
			microenvironment->density_vector(i)[oxygen] = center_concentration * std::exp(-r2 / center_radius_squared);
		}
	}

	microenvironment->set_substrate_dirichlet_value(oxygen, 0.0);

	const double cell_radius = 5.0;

	create_deterministic_sources_and_sinks(
		*microenvironment,
		cell_radius,
		dt,
		number_of_agents,
		mpi_Rank,
		mpi_Dims);
}

int main(int argc, char* argv[])
{
	unsigned int mpi_Error;
	int mpi_Requested_level = MPI_THREAD_FUNNELED;
	int mpi_Provided_level;
	MPI_Comm mpi_Comm = MPI_COMM_WORLD;
	MPI_Comm mpi_Cart_comm;
	int mpi_Size, mpi_Rank;
	int mpi_Dims[3], mpi_Is_periodic[3], mpi_Coords[3];
	int mpi_Reorder;

	mpi_Error = MPI_Init_thread(&argc, &argv, mpi_Requested_level, &mpi_Provided_level);
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

	const std::string output_dir = argc > 1 ? argv[1] : "output";
	const int iterations = argc > 2 ? parse_int(argv[2], 500) : 500;
	const double dt = argc > 3 ? parse_double(argv[3], 0.01) : 0.01;
	if (!ensure_directory(output_dir))
	{
		if (mpi_Rank == 0)
		{
			std::cerr << "Error: could not create output directory " << output_dir << std::endl;
		}
		MPI_Finalize();
		return 1;
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

	MPI_Cart_create(mpi_Comm, 3, mpi_Dims, mpi_Is_periodic, mpi_Reorder, &mpi_Cart_comm);
	MPI_Comm_rank(mpi_Cart_comm, &mpi_Rank);
	MPI_Cart_coords(mpi_Cart_comm, mpi_Rank, 3, mpi_Coords);

	omp_set_num_threads(1);

	Microenvironment microenvironment;
	microenvironment.name = "density diffusion demo";
	microenvironment.spatial_units = "microns";
	microenvironment.mesh.units = "microns";
	microenvironment.time_units = "minutes";

	microenvironment.set_density(oxygen, "oxygen", "dimensionless");

	microenvironment.diffusion_coefficients[oxygen] = 1000.0;
	microenvironment.decay_rates[oxygen] = 0.01;

	const double min = 0.0;
	const double max = 1000.0;
	const double mesh_resolution = 10.0;
	microenvironment.resize_space_uniform(min, max, min, max, min, max, mesh_resolution, mpi_Dims, mpi_Coords);

	// register the diffusion solver
	microenvironment.diffusion_decay_solver = diffusion_decay_solver__constant_coefficients_LOD_3D_AVX256D;
	microenvironment.print_voxels_densities = print_voxels_densities;

	microenvironment.bulk_supply_rate_function = supply_function;
	microenvironment.bulk_supply_target_densities_function = supply_target_function;

	if (mpi_Rank == 0)
		microenvironment.display_information(std::cout);

	// Timing measurements will start, hence a barrier

	MPI_Barrier(mpi_Cart_comm);

	BioFVM::RUNTIME_TIC();
	BioFVM::TIC();

	int output_index = 0;

	const int number_of_agents = 1000;
	const double center_concentration = 100.0;
	centered_initial_conditions(
		&microenvironment,
		dt,
		mpi_Rank,
		mpi_Size,
		mpi_Dims,
		number_of_agents,
		center_concentration);

	microenvironment.write_to_matlab(join_path(output_dir, "density_diffusion_initial.mat"), mpi_Rank, mpi_Size, mpi_Cart_comm);

	const int number_of_sources = number_of_agents / 2;
	const int number_of_sinks = number_of_agents - number_of_sources;

	for (int n = 0; n < iterations; ++n)
	{
		microenvironment.simulate_cell_sources_and_sinks(dt);
		microenvironment.simulate_diffusion_decay(dt, mpi_Size, mpi_Rank, mpi_Coords,mpi_Dims, mpi_Cart_comm);
	}

	microenvironment.write_to_matlab(join_path(output_dir, "density_diffusion_final.mat"), mpi_Rank, mpi_Size, mpi_Cart_comm);

	if (mpi_Rank == 0)
	{
		std::cout << "Wrote " << join_path(output_dir, "density_diffusion_initial.mat") << std::endl;
		std::cout << "Wrote " << join_path(output_dir, "density_diffusion_final.mat") << std::endl;
		std::cout << "Iterations: " << iterations << ", dt: " << dt << std::endl;
		std::cout << "Domain: 1000^3 microns, mesh dx: " << mesh_resolution << " microns" << std::endl;
		std::cout << "Basic Agents: " << number_of_agents << " (" << number_of_sources
			<< " sources, " << number_of_sinks << " sinks)" << std::endl;
	}

	MPI_Comm_free(&mpi_Cart_comm);
	MPI_Finalize();
	return 0;
}
