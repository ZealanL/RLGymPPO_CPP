#pragma once
#include <RLGymPPO_CPP/Lists.h>

#include <torch/nn/modules/container/sequential.h>

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/discrete_policy.py
	class DiscretePolicy : public torch::nn::Module {
	public:
		torch::Device device;
		torch::nn::Sequential seq;
		int inputAmount;
		int actionAmount;
		IList layerSizes;

		// Min probability that an action will be taken
		constexpr static float ACTION_MIN_PROB = 1e-11;

		DiscretePolicy(int inputAmount, int actionAmount, const IList& layerSizes, torch::Device device);

		RG_NO_COPY(DiscretePolicy);

		void CopyTo(DiscretePolicy& to);

		torch::Tensor GetOutput(torch::Tensor input) {
			return seq->forward(input);
		}

		torch::Tensor GetActionProbs(torch::Tensor obs);

		struct ActionResult {
			torch::Tensor action, logProb;
		};
		ActionResult GetAction(torch::Tensor obs, bool deterministic);
		
		struct BackpropResult {
			torch::Tensor actionLogProbs;
			torch::Tensor entropy;
		};
		BackpropResult GetBackpropData(torch::Tensor obs, torch::Tensor acts);

		~DiscretePolicy() = default;
	};
}