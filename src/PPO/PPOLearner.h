#pragma once
#include "DiscretePolicy.h";
#include "ValueEstimator.h";
#include "ExperienceBuffer.h";
#include "../Util/Report.h"
#include "../Util/Timer.h"

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {
		IList policyLayerSizes = { 256, 256, 256 };
		IList criticLayerSizes = { 256, 256, 256 };
		int batchSize = 50 * 1000;
		int epochs = 10;
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate
		float entCoef = 0.005f; // Entropy coefficient
		float clipRange = 0.2f;
		int miniBatchSize = 0; // Set to 0 to just use batchSize
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	class PPOLearner {
	public:
		DiscretePolicy* policy;
		ValueEstimator* valueNet;
		torch::optim::Adam *policyOptimizer, *valueOptimizer;
		torch::nn::MSELoss valueLossFn;

		PPOLearnerConfig config;
		torch::Device device;

		int cumulativeModelUpdates = 0;

		PPOLearner(
			int obsSpaceSize, int actSpaceSize,
			PPOLearnerConfig config, torch::Device device
		);
		

		void Learn(ExperienceBuffer* expBuffer, Report& report);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath, bool isFromPython);
	};
}