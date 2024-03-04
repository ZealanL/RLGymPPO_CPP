#pragma once
#include "../Lists.h"

#include <torch/nn/modules/container/sequential.h>

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/discrete_policy.py
	class DiscretePolicy : public torch::nn::Module {
	public:
		torch::Device device;
		torch::nn::Sequential seq;
		int inputAmount;
		int actionAmount;
		bool isHalf = false;

		// Min probability that an action will be taken
		constexpr static float ACTION_MIN_PROB = 1e-11;

		DiscretePolicy(int inputAmount, int actionAmount, const IList& layerSizes, torch::Device device);

		torch::Tensor GetOutput(torch::Tensor input) {
			return seq->forward(input);
		}

		struct ActionResult {
			torch::Tensor action, logProb;
		};
		// NOTE: For deterministic, use GetDeterministicAction()
		ActionResult GetAction(torch::Tensor obs);

		int GetDeterministicActionIdx(torch::Tensor obs);
		
		struct BackpropResult {
			torch::Tensor actionLogProbs;
			torch::Tensor entropy;
		};
		BackpropResult GetBackpropData(torch::Tensor obs, torch::Tensor acts);
	};
}