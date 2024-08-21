#pragma once
#include "../Lists.h"

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {
		IList policyLayerSizes = { 256, 256, 256 };
		IList criticLayerSizes = { 256, 256, 256 };
		int64_t batchSize = 50 * 1000;
		int epochs = 10;
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate
		float entCoef = 0.005f; // Entropy coefficient
		float clipRange = 0.2f;
		int64_t miniBatchSize = 0; // Set to 0 to just use batchSize

		// Experimental, improves PPO learn speed
		// If this causes your learning to collapse, please let me know
		bool autocastLearn = false;

		// Very experimental, uses half-precision versions of models where beneficial
		bool halfPrecModels = false;

		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		// https://openai.com/index/how-ai-training-scales/
		// Measures the noise of both policy and critic gradients every epoch
		bool measureGradientNoise = false;
		int gradientNoiseUpdateInterval = 10;
		float gradientNoiseAvgDecay = 0.9925f;
	};
}