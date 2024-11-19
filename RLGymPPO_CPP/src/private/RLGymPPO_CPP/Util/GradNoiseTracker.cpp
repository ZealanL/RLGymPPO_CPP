#include "GradNoiseTracker.h"

// Based on https://github.com/shreyansh26/An-Empirical-Model-of-Large-Batch-Training/blob/master/utils.py

torch::Tensor GetAllGrad(torch::nn::Sequential seq) {
	std::vector<torch::Tensor> grads = {};

	auto params = seq->parameters();
	for (auto& param : params) {
		auto& grad = param.grad();
		if (!grad.defined())
			continue;

		grads.push_back(grad.flatten().view({ -1, 1 }));
	}

	return torch::concat(grads);
}

float UpdateExpMovingAvg(float& avg, float x, float decay, int step) {
	avg = (avg * decay) + (x * (1 - decay));
	return avg / (1 - powf(decay, step + 1));
}

// Based on https://github.com/shreyansh26/An-Empirical-Model-of-Large-Batch-Training/blob/master/noise_scale.py

RLGPC::GradNoiseTracker::GradNoiseTracker(int batchSize, int updateInterval, float beta)
	: batchSize(batchSize), updateInterval(updateInterval), averageDecay(averageDecay) {
	batchSmall = batchSize;
	batchBig = batchSize * updateInterval;
}

void RLGPC::GradNoiseTracker::Update(torch::nn::Sequential& seq) {
	auto grad = GetAllGrad(seq);
	batchesGrad.push_back(grad);

	if ((stepCount % updateInterval) == (updateInterval - 1)) {
		auto batchesGradAll = torch::concat(batchesGrad);
		batchesGrad.clear();

		auto batches_grad_mean = batchesGradAll.mean(1);
		float g_big = batches_grad_mean.square().mean().cpu().item<float>();
		float g_small = grad.square().mean().cpu().item<float>();

		auto curNoise = (batchBig * g_big - batchSmall * g_small) / (batchBig - batchSmall);
		auto curScale = abs((g_small - g_big) / ((1 / batchSmall) - (1 / batchBig))); // TODO: Not sure why this needs an abs()

		float scale = UpdateExpMovingAvg(movingAvgScale, curScale, averageDecay, stepCount);
		float noise = UpdateExpMovingAvg(movingAvgNoise, curNoise, averageDecay, stepCount);

		lastNoiseScale = scale / noise;
	}

	stepCount++;
}
