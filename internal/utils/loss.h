/** @author	Minglin Chen
 *  @date 	2023/3/23
 *  @Email 	chenmlin8@mail2.sysu.edu.cn
 */

#pragma once

#include "internal/utils/common.h"

struct LossAndGradient {
	Eigen::Array3f loss;
	Eigen::Array3f gradient;
};

enum class ELossType : int {
	L2,
	L1,
	Mape,
	Smape,
	SmoothL1,
	LogL1,
	RelativeL2,
};

ELossType string_to_loss_type(const std::string& str);
__device__ LossAndGradient loss_and_gradient(const Eigen::Vector3f& target, const Eigen::Vector3f& prediction, ELossType loss_type);
