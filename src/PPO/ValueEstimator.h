#pragma once
#include "../Lists.h"

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/value_estimator.py
	class ValueEstimator : public torch::nn::Module {
	public:
		torch::Device device;
		torch::nn::Sequential seq;

		ValueEstimator(int inputAmount, const IList& layerSizes, torch::Device device);

		torch::Tensor Forward(torch::Tensor input) {
			return seq->forward(input).to(device);
		}
	};
}