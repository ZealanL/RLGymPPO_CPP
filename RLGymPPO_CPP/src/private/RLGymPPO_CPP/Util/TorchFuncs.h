#pragma once
#include <RLGymPPO_CPP/Lists.h>
#include "../FrameworkTorch.h"
#include <torch/optim/adam.h>

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/util/torch_functions.py
	namespace TorchFuncs {
		void ComputeGAE(
			const FList& rews, const FList& dones, const FList& truncated, const FList& values,
			torch::Tensor& outAdvantages, torch::Tensor& outValues, FList& outReturns,
			float gamma = 0.99f, float lambda = 0.95f, float returnStd = 0
		);

		// torch::cat({a, b}, 0) but returns b.clone() if a is undefined
		torch::Tensor ConcatSafe(torch::Tensor a, torch::Tensor b);

		constexpr uint32_t OPTIMIZER_SERIALIZE_PREFIX = 0xB73AC038;
		void SerializeOptimizer(torch::optim::Adam* optim, DataStreamOut& out);
		void DeserializeOptimizer(torch::optim::Adam* optim, DataStreamIn& in);
	}
}