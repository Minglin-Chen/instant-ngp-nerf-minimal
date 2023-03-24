/** @author	Minglin Chen
 *  @date 	2023/3/23
 *  @Email 	chenmlin8@mail2.sysu.edu.cn
 */

#pragma once

#include <json/json.hpp>
#include <tiny-cuda-nn/gpu_memory.h>

#include "internal/utils/common.h"
#include "internal/utils/render_buffer.h"
#include "internal/utils/random_val.cuh"
#include "internal/sampler/bounding_box.h"


class OccupancySampler {
    using json = nlohmann::json;

public:
    OccupancySampler(json& config);

	void init(
        const uint32_t n_images,
        const Eigen::Vector2i& image_resolution,
        const Eigen::Vector2f& focal_length,
        const Eigen::Vector2f& principal_point,
        const tcnn::GPUMemory<Eigen::Matrix<float, 3, 4>>& camera_matrices,
        const tcnn::GPUMemory<__half>& images,
        default_rng_t& rng,
        cudaStream_t stream
	);

	/* *********************************************************************************
     * Sample positions from grid & update grid and its bitfield
     * ********************************************************************************/
    void sample_positions_from_grid(
		const uint32_t n_uniform_positions,
		const uint32_t n_nonuniform_positions,
        tcnn::GPUMemory<Position>& grid_positions,
        tcnn::GPUMemory<uint32_t>& grid_indices,
        cudaStream_t stream
    );

	void update_grid(
		const uint32_t n_total_positions,
		const uint32_t padded_output_width,
        tcnn::GPUMemory<precision_t>& grid_densities,
        tcnn::GPUMemory<uint32_t>& grid_indices,
        EActivation density_activation,
		cudaStream_t stream
    );

	void update_grid_bitfield(cudaStream_t stream);

	/* *********************************************************************************
     * Sample coordinates (posisiton & view direction) & rays
     * ********************************************************************************/
	void sample_coords_from_dataset(
        const uint32_t n_rays,
        const uint32_t n_rays_shift,
        const uint32_t n_coords,
		const NeRFSynthetic& dataset,
		const bool train_envmap,
        tcnn::GPUMemory<Ray>& rays, 
	    tcnn::GPUMemory<Eigen::Array4f>& rays_rgba,
        uint32_t* __restrict__ n_rays_counter,
        tcnn::GPUMemory<uint32_t>& ray_coord_indices,
        tcnn::GPUMemory<Coordinate>& coords,
        uint32_t* __restrict__ n_coords_counter,
        default_rng_t rng,
        cudaStream_t stream
    );

	void generate_rays_from_camera_matrix(
		const Eigen::Vector2i& resolution, 
		const Eigen::Vector2f& focal_length, 
		const Eigen::Vector2f& principal_point, 
		const Eigen::Matrix<float, 3, 4>& camera_matrix, 
		const precision_t* __restrict__ envmap_data,
		const Eigen::Vector2i& envmap_resolution,
		tcnn::GPUMemory<RayPayload>& payload,
		CudaRenderBuffer& render_buffer, 
		cudaStream_t stream
	);

	void generate_coords_from_rays(
        const uint32_t n_rays,
        const uint32_t n_steps,
        tcnn::GPUMemory<RayPayload> payloads,
        tcnn::GPUMemory<Coordinate> coords,
        cudaStream_t stream
    );

public:
    // AABB parameters
    uint32_t    aabb_scale;
    BoundingBox aabb;

    // Grid parameters
    uint32_t    n_grid_size;            // size of the density/occupancy grid.
    uint32_t    n_grid_elements;
	uint32_t 	n_cascades;
	uint32_t	n_total_elements;
    
    // EMA parameters
    uint32_t    i_step;
    float       ema_decay;

    // Sampling paramters
    uint32_t    n_max_steps;            // finest number of steps per unit length
	float		cone_angle_constant;
    float       min_cone_stepsize;      // Minimum step size is the width of the coarsest gridsize cell.
    float       max_cone_stepsize;      // Maximum step size is the width of the coarsest gridsize cell.
	float		near_distance;
	float		far_distance;
    // Prior nerf papers don't typically do multi-sample anti aliasing.
	// So snap all pixels to the pixel centers by default.
    bool        snap_to_pixel_centers_in_training;
	bool		snap_to_pixel_centers_in_rendering;
	float       min_optical_thickness;  // Any alpha below this is considered "invisible" and is thus culled away.

    // N^3 grid of densities from the network
    tcnn::GPUMemory<float>      density_grid_current;
    // N^3 grid of EMA smoothed densities from the network
    tcnn::GPUMemory<float>      density_grid;
    // Density mean
    tcnn::GPUMemory<float>      density_grid_mean;
    // Bitfield
    tcnn::GPUMemory<uint8_t>    density_grid_bitfield;

    // Random number generator
    default_rng_t 		        rng;
};