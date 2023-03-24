/** @brief	An implementation of a trainable N-channel buffer within the tcnn API.
 * 	@author	Minglin Chen
 *  @date 	2023/3/23
 *  @Email 	chenmlin8@mail2.sysu.edu.cn
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>

template <uint32_t N_DIMS, uint32_t RANK, typename T>
class TrainableBuffer : public tcnn::DifferentiableObject<float, T, T> {
	using ResVector = Eigen::Matrix<int, RANK, 1>;

public:
	TrainableBuffer(const ResVector& resolution) : m_resolution{resolution} {}

	virtual ~TrainableBuffer() { }

	void inference(cudaStream_t stream, const tcnn::GPUMatrix<float>& input, tcnn::GPUMatrix<float>& output) override {
		throw std::runtime_error{"The trainable buffer does not support inference(). Its content is meant to be used externally."};
	}

	void forward(cudaStream_t stream, const tcnn::GPUMatrix<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_matrices = false, bool prepare_input_gradients = false) override {
		throw std::runtime_error{"The trainable buffer does not support forward(). Its content is meant to be used externally."};
	}

	void backward(
		cudaStream_t stream,
		const tcnn::GPUMatrix<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrix<float>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override {
		throw std::runtime_error{"The trainable buffer does not support backward(). Its content is meant to be used externally."};
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		m_params_full_precision = params_full_precision;
		m_params = params;
		m_params_inference = inference_params;
		m_params_gradient = gradients;

		// Initialize the buffer to zero from the GPU
		CUDA_CHECK_THROW(cudaMemset(m_params_full_precision, 0, n_params()*sizeof(float)));

		m_params_gradient_weight.resize(n_params());
	}

	size_t n_params() const override {
		return m_resolution.prod() * N_DIMS;
	}

	uint32_t padded_output_width() const override {
		return N_DIMS;
	}

	uint32_t output_width() const override {
		return N_DIMS;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return {};
	}

	T* gradients() const {
		return m_params_gradient;
	}

	T* gradient_weights() const {
		return m_params_gradient_weight.data();
	}

	T* params() const {
		return m_params;
	}

	T* params_inference() const {
		return m_params_inference;
	}

	float* params_full_precision() const {
		return m_params_full_precision;
	}

private:
	ResVector m_resolution;

	float* m_params_full_precision = nullptr;
	T* m_params = nullptr;
	T* m_params_inference = nullptr;
	T* m_params_gradient = nullptr;
	tcnn::GPUMemory<T> m_params_gradient_weight;
};