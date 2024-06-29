#pragma once
#include "../FrameworkTorch.h"
#include <torch/nn/modules/container/sequential.h>

namespace RLGPC {
	// Based on https://github.com/shreyansh26/An-Empirical-Model-of-Large-Batch-Training/blob/master/noise_scale.py
	struct GradNoiseTracker {
		int stepCount = 0;

		int batchSize;
		int updateInterval;
		float averageDecay;

		float 
			batchBig, 
			batchSmall;
		
		float 
			movingAvgScale = 0,
			movingAvgNoise = 0;

		std::vector<torch::Tensor> batchesGrad;
		float lastNoiseScale = 0;

		GradNoiseTracker(int batchSize, int updateInterval, float averageDecay = 0.99f);

		void Update(torch::nn::Sequential& seq);
	};
}