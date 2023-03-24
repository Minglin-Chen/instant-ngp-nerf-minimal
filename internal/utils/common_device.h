/** @author	Minglin Chen
 *  @date 	2023/3/23
 *  @Email 	chenmlin8@mail2.sysu.edu.cn
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "internal/utils/common.h"
#include "internal/utils/random_val.cuh"

#include "internal/sampler/bounding_box.h"


/* *********************************************************************
 * Common
 * *********************************************************************/
inline HOST_DEVICE float4 to_float4(const Eigen::Array4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline HOST_DEVICE float4 to_float4(const Eigen::Vector4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline HOST_DEVICE float3 to_float3(const Eigen::Array3f& x) {
	return {x.x(), x.y(), x.z()};
}

inline HOST_DEVICE float3 to_float3(const Eigen::Vector3f& x) {
	return {x.x(), x.y(), x.z()};
}

inline HOST_DEVICE float2 to_float2(const Eigen::Array2f& x) {
	return {x.x(), x.y()};
}

inline HOST_DEVICE float2 to_float2(const Eigen::Vector2f& x) {
	return {x.x(), x.y()};
}

inline HOST_DEVICE Eigen::Array4f to_array4(const float4& x) {
	return {x.x, x.y, x.z, x.w};
}

inline HOST_DEVICE Eigen::Vector4f to_vec4(const float4& x) {
	return {x.x, x.y, x.z, x.w};
}

inline HOST_DEVICE Eigen::Array3f to_array3(const float3& x) {
	return {x.x, x.y, x.z};
}

inline HOST_DEVICE Eigen::Vector3f to_vec3(const float3& x) {
	return {x.x, x.y, x.z};
}

inline HOST_DEVICE Eigen::Array2f to_array2(const float2& x) {
	return {x.x, x.y};
}

inline HOST_DEVICE Eigen::Vector2f to_vec2(const float2& x) {
	return {x.x, x.y};
}

/* *********************************************************************
 * Color space (i.e., sRGB or Linear)
 * *********************************************************************/
inline HOST_DEVICE float srgb_to_linear(float srgb) {
	return srgb <= 0.04045f ? srgb / 12.92f : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
}

inline HOST_DEVICE Eigen::Array3f srgb_to_linear(const Eigen::Array3f& x) {
	return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}

inline HOST_DEVICE float srgb_to_linear_derivative(float srgb) {
	return srgb <= 0.04045f ? 1.0f / 12.92f : 2.4f / 1.055f * std::pow((srgb + 0.055f) / 1.055f, 1.4f);
}

inline HOST_DEVICE Eigen::Array3f srgb_to_linear_derivative(const Eigen::Array3f& x) {
	return {srgb_to_linear_derivative(x.x()), srgb_to_linear_derivative(x.y()), (srgb_to_linear_derivative(x.z()))};
}

inline HOST_DEVICE float linear_to_srgb(float linear) {
	return linear < 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 0.41666f) - 0.055f;
}

inline HOST_DEVICE Eigen::Array3f linear_to_srgb(const Eigen::Array3f& x) {
	return {linear_to_srgb(x.x()), linear_to_srgb(x.y()), (linear_to_srgb(x.z()))};
}

inline HOST_DEVICE float linear_to_srgb_derivative(float linear) {
	return linear < 0.0031308f ? 12.92f : 1.055f * 0.41666f * std::pow(linear, 0.41666f - 1.0f);
}

inline HOST_DEVICE Eigen::Array3f linear_to_srgb_derivative(const Eigen::Array3f& x) {
	return {linear_to_srgb_derivative(x.x()), linear_to_srgb_derivative(x.y()), (linear_to_srgb_derivative(x.z()))};
}

/* *********************************************************************
 * Camera
 * *********************************************************************/
inline HOST_DEVICE float fov_to_focal_length(int resolution, float rad) {
	return 0.5f * (float)resolution / tanf(0.5f * rad);
}

inline HOST_DEVICE Eigen::Vector2f fov_to_focal_length(const Eigen::Vector2i& resolution, const Eigen::Vector2f& rads) {
	return 0.5f * resolution.cast<float>().cwiseQuotient((0.5f * rads).array().tan().matrix());
}

inline HOST_DEVICE float focal_length_to_fov(int resolution, float focal_length) {
	return 2.f * 180.f / PI() * atanf(float(resolution)/(focal_length*2.f));
}

inline HOST_DEVICE Eigen::Vector2f focal_length_to_fov(const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length) {
	return 2.f * 180.f / PI() * resolution.cast<float>().cwiseQuotient(focal_length*2).array().atan().matrix();
}

Eigen::Matrix<float, 3, 4> spec_opengl_to_opencv(
	const Eigen::Matrix<float, 3, 4> ogl_matrix,
	const float scale,
	const Eigen::Vector3f offset
);

Eigen::Matrix<float, 3, 4> log_space_lerp(const Eigen::Matrix<float, 3, 4>& begin, const Eigen::Matrix<float, 3, 4>& end, float t);

inline HOST_DEVICE Ray pixel_to_ray(
	const Eigen::Vector2i& 				pixel,
	const Eigen::Vector2i& 				resolution,
	const Eigen::Vector2f& 				focal_length,
	const Eigen::Vector2f& 				principal_point,
	const Eigen::Matrix<float, 3, 4>& 	camera_matrix,
	const uint32_t 						spp,
	const bool 							snap_to_pixel_centers
) {
	const Eigen::Vector2f offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : spp, pixel.x(), pixel.y());
	auto xy = (pixel.cast<float>() + offset).cwiseQuotient(resolution.cast<float>());

	Eigen::Vector3f origin = camera_matrix.col(3);
	Eigen::Vector3f dir = {
		(xy.x() - principal_point.x()) * (float)resolution.x() / focal_length.x(),
		(xy.y() - principal_point.y()) * (float)resolution.y() / focal_length.y(),
		1.0f
	};
	dir = camera_matrix.block<3, 3>(0, 0) * dir;
	dir = dir.normalized();

	return {origin, dir};
}

/* *********************************************************************
 * Image IO
 * *********************************************************************/
tcnn::GPUMemory<float> load_stbi(const std::string& filename, int& width, int& height);

template <typename T>
__global__ void from_rgba32(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, T* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	uint8_t rgba[4];
	*((uint32_t*)&rgba[0]) = *((uint32_t*)&pixels[i*4]);

	tcnn::vector_t<T, 4> rgba_out;
	float alpha = rgba[3] * (1.0f/255.0f);
	rgba_out[0] = (T)(srgb_to_linear(rgba[0] * (1.0f/255.0f)) * alpha);
	rgba_out[1] = (T)(srgb_to_linear(rgba[1] * (1.0f/255.0f)) * alpha);
	rgba_out[2] = (T)(srgb_to_linear(rgba[2] * (1.0f/255.0f)) * alpha);
	rgba_out[3] = (T)alpha;

	*((tcnn::vector_t<T, 4>*)&out[i*4]) = rgba_out;
}

/* *********************************************************************
 * Image manipulation
 * *********************************************************************/
inline __device__ Eigen::Vector2i image_pos(const Eigen::Vector2f& pos, const Eigen::Vector2i& resolution) {
	return pos.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMin(resolution - Eigen::Vector2i::Constant(1)).cwiseMax(0);
}

inline __device__ uint64_t pixel_idx(const Eigen::Vector2i& pos, const Eigen::Vector2i& resolution, uint32_t img) {
	return pos.x() + pos.y() * resolution.x() + img * (uint64_t)resolution.x() * resolution.y();
}

inline __device__ uint64_t pixel_idx(const Eigen::Vector2f& xy, const Eigen::Vector2i& resolution, uint32_t img) {
	return pixel_idx(image_pos(xy, resolution), resolution, img);
}

inline __device__ Eigen::Array4f read_rgba(Eigen::Vector2f pos, const Eigen::Vector2i& resolution, uint32_t img, const __half* training_images) {
	Eigen::Vector2i idx = image_pos(pos, resolution);

	auto read_val = [&](const Eigen::Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Eigen::Array4f{val[0], val[1], val[2], val[3]};
	};

	return read_val(idx);
}

template <uint32_t N_DIMS, typename T>
HOST_DEVICE Eigen::Matrix<float, N_DIMS, 1> read_image(const T* __restrict__ data, const Eigen::Vector2i& resolution, const Eigen::Vector2f& pos) {
	auto pos_float = Eigen::Vector2f{pos.x() * (float)(resolution.x()-1), pos.y() * (float)(resolution.y()-1)};
	Eigen::Vector2i texel = pos_float.cast<int>();

	auto weight = pos_float - texel.cast<float>();

	auto read_val = [&](Eigen::Vector2i pos) {
		pos.x() = std::max(std::min(pos.x(), resolution.x()-1), 0);
		pos.y() = std::max(std::min(pos.y(), resolution.y()-1), 0);

		Eigen::Matrix<float, N_DIMS, 1> result;
		if (std::is_same<T, float>::value) {
			result = *(Eigen::Matrix<T, N_DIMS, 1>*)&data[(pos.x() + pos.y() * resolution.x()) * N_DIMS];
		} else {
			auto val = *(tcnn::vector_t<T, N_DIMS>*)&data[(pos.x() + pos.y() * resolution.x()) * N_DIMS];

			PRAGMA_UNROLL
			for (uint32_t i = 0; i < N_DIMS; ++i) {
				result[i] = (float)val[i];
			}
		}
		return result;
	};

	auto result = (
		(1 - weight.x()) * (1 - weight.y()) * read_val({texel.x(), texel.y()}) +
		(weight.x()) * (1 - weight.y()) * read_val({texel.x()+1, texel.y()}) +
		(1 - weight.x()) * (weight.y()) * read_val({texel.x(), texel.y()+1}) +
		(weight.x()) * (weight.y()) * read_val({texel.x()+1, texel.y()+1})
	);

	return result;
}

inline __device__ Eigen::Array3f composit(
	Eigen::Vector2f pos, 
	const Eigen::Vector2i& resolution, 
	uint32_t img, 
	const __half* training_images, 
	const Eigen::Array3f& background_color, 
	const Eigen::Array3f& exposure_scale = Eigen::Array3f::Ones()
) {
	Eigen::Vector2i idx = image_pos(pos, resolution);

	auto read_val = [&](const Eigen::Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Eigen::Array3f{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
	};

	return read_val(idx);
}

inline __device__ Eigen::Array3f composit_and_lerp(
	Eigen::Vector2f pos, 
	const Eigen::Vector2i& resolution, 
	uint32_t img, 
	const __half* training_images, 
	const Eigen::Array3f& background_color, 
	const Eigen::Array3f& exposure_scale = Eigen::Array3f::Ones()
) {
	pos = (pos.cwiseProduct(resolution.cast<float>()) - Eigen::Vector2f::Constant(0.5f)).cwiseMax(0.0f).cwiseMin(resolution.cast<float>() - Eigen::Vector2f::Constant(1.0f + 1e-4f));

	Eigen::Vector2i pos_int = pos.cast<int>();
	auto weight = pos - pos_int.cast<float>();

	Eigen::Vector2i idx = pos_int.cwiseMin(resolution - Eigen::Vector2i::Constant(2)).cwiseMax(0);

	auto read_val = [&](const Eigen::Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Eigen::Array3f{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
	};

	Eigen::Array3f result = (
		(1 - weight.x()) * (1 - weight.y()) * read_val({idx.x(), idx.y()}) +
		(weight.x()) * (1 - weight.y()) * read_val({idx.x()+1, idx.y()}) +
		(1 - weight.x()) * (weight.y()) * read_val({idx.x(), idx.y()+1}) +
		(weight.x()) * (weight.y()) * read_val({idx.x()+1, idx.y()+1})
	);

	return result;
}

/* *********************************************************************
 * Network input & output
 * *********************************************************************/
inline __device__ float network_to_rgb(float val, EActivation activation) {
	switch (activation) {
		case EActivation::None: return val;
		case EActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case EActivation::Logistic: return tcnn::logistic(val);
		case EActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

inline __device__ float network_to_rgb_derivative(float val, EActivation activation) {
	switch (activation) {
		case EActivation::None: return 1.0f;
		case EActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case EActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case EActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

inline __device__ float network_to_density(float val, EActivation activation) {
	switch (activation) {
		case EActivation::None: return val;
		case EActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case EActivation::Logistic: return tcnn::logistic(val);
		case EActivation::Exponential: return __expf(val);
		default: assert(false);
	}
	return 0.0f;
}

inline __device__ float network_to_density_derivative(float val, EActivation activation) {
	switch (activation) {
		case EActivation::None: return 1.0f;
		case EActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case EActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case EActivation::Exponential: return __expf(tcnn::clamp(val, -15.0f, 15.0f));
		default: assert(false);
	}
	return 0.0f;
}

inline __device__ Eigen::Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, EActivation activation) {
	return {
		network_to_rgb(float(local_network_output[0]), activation),
		network_to_rgb(float(local_network_output[1]), activation),
		network_to_rgb(float(local_network_output[2]), activation)
	};
}

/* *********************************************************************
 * Coordinate transformation
 * *********************************************************************/
inline __device__ Eigen::Vector3f warp_position(const Eigen::Vector3f& pos, const BoundingBox& aabb) {
	// return {tcnn::logistic(pos.x() - 0.5f), tcnn::logistic(pos.y() - 0.5f), tcnn::logistic(pos.z() - 0.5f)};
	// return pos;

	return aabb.relative_pos(pos);
}

inline __device__ Eigen::Vector3f unwarp_position(const Eigen::Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.min + pos.cwiseProduct(aabb.diag());
}


inline __device__ Eigen::Vector3f unwarp_position_derivative(const Eigen::Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.diag();
}


inline __device__ Eigen::Vector3f warp_position_derivative(const Eigen::Vector3f& pos, const BoundingBox& aabb) {
	return unwarp_position_derivative(pos, aabb).cwiseInverse();
}


inline __device__ Eigen::Vector3f warp_direction(const Eigen::Vector3f& dir) {
	return (dir + Eigen::Vector3f::Ones()) * 0.5f;
}


inline __device__ Eigen::Vector3f unwarp_direction(const Eigen::Vector3f& dir) {
	return dir * 2.0f - Eigen::Vector3f::Ones();
}


inline __device__ Eigen::Vector3f warp_direction_derivative(const Eigen::Vector3f& dir) {
	return Eigen::Vector3f::Constant(0.5f);
}


inline __device__ Eigen::Vector3f unwarp_direction_derivative(const Eigen::Vector3f& dir) {
	return Eigen::Vector3f::Constant(2.0f);
}


inline __device__ float warp_dt(float dt, const float min_cone_stepsize, const uint32_t n_cascades) {
	float max_stepsize = min_cone_stepsize * (1<<(n_cascades-1));
	return (dt - min_cone_stepsize) / (max_stepsize - min_cone_stepsize);
}


inline __device__ float unwarp_dt(float dt, const float min_cone_stepsize, const uint32_t n_cascades) {
	float max_stepsize = min_cone_stepsize * (1<<(n_cascades-1));
	return dt * (max_stepsize - min_cone_stepsize) + min_cone_stepsize;
}