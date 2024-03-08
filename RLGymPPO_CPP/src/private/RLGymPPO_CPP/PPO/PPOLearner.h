#pragma once
#include "DiscretePolicy.h";
#include "ValueEstimator.h";
#include "ExperienceBuffer.h";
#include <RLGymPPO_CPP/Util/Report.h>
#include <RLGymPPO_CPP/Util/Timer.h>
#include <RLGymPPO_CPP/PPO/PPOLearnerConfig.h>

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>

namespace RLGPC {
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
		void LoadFrom(std::filesystem::path folderPath);

		void UpdateLearningRates(float policyLR, float criticLR);
	};
}