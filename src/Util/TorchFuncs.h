#pragma once
#include "../Lists.h"

#include <torch/nn/Module.h>

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/util/torch_functions.py
	namespace TorchFuncs {
		void ComputeGAE(
			const FList& rews, const FList& dones, const FList& truncated, const FList& values,
			torch::Tensor& outAdvantages, torch::Tensor& outValues, FList& outReturns,
			float gamma = 0.99f, float lambda = 0.95f, float returnStd = 1
		);

		// torch::cat({a, b}, 0) but returns b.clone() if a is undefined
		torch::Tensor ConcatSafe(torch::Tensor a, torch::Tensor b);

		void LoadStateDict(torch::nn::Module* mod, std::filesystem::path path);
	}
}