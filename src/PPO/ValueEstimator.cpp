#include "ValueEstimator.h"

RLGPC::ValueEstimator::ValueEstimator(int inputAmount, const IList& layerSizes, torch::Device device) : device(device) {
	using namespace torch;

	seq = {};

	seq->push_back(nn::Linear(inputAmount, layerSizes[0]));
	seq->push_back(nn::ReLU());

	int prevLayerSize = layerSizes[0];
	for (int i = 1; i < layerSizes.size(); i++) {
		int layerSize = layerSizes[i];
		seq->push_back(nn::Linear(prevLayerSize, layerSize));
		seq->push_back(nn::ReLU());
		prevLayerSize = layerSize;
	}

	// Output layer, just gives 1 output for value estimate
	seq->push_back(nn::Linear(layerSizes.back(), 1));

	register_module("seq", seq);

	this->to(device);
}