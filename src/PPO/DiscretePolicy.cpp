#include "DiscretePolicy.h"

RLGPC::DiscretePolicy::DiscretePolicy(int inputAmount, int actionAmount, const IList& layerSizes, torch::Device device) : 
	device(device), inputAmount(inputAmount), actionAmount(actionAmount) {
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

	// Output layer, for each action
	seq->push_back(nn::Linear(layerSizes.back(), actionAmount));
	seq->push_back(nn::Softmax(nn::SoftmaxOptions(-1)));

	register_module("seq", seq);

	this->to(device);
}

RLGPC::DiscretePolicy::ActionResult RLGPC::DiscretePolicy::GetAction(torch::Tensor obs) {
	// Get probability of each action
	auto probs = GetOutput(obs);
	probs = probs.view({ -1, actionAmount });
	probs = torch::clamp(probs, ACTION_MIN_PROB, 1); // Prevent actions from being impossible, and also log(0)

	auto action = torch::multinomial(probs, 1, true);
	auto logProb = torch::log(probs).gather(-1, action);
	return ActionResult{ action.cpu().flatten(), logProb.cpu().flatten() };
}

int RLGPC::DiscretePolicy::GetDeterministicActionIdx(torch::Tensor obs) {
	// Get probability of each action
	auto probs = GetOutput(obs);
	probs = probs.view({ -1, actionAmount });
	probs = torch::clamp(probs, ACTION_MIN_PROB, 1); // Prevent actions from being impossible, and also log(0)

	auto cpuProbs = probs.cpu();
	float* probsData = probs.data_ptr<float>();

	// Find the index with the greated probability
	// Probably some stdlib stuff that could do this for me
	int bestIdx = 0;
	float largestVal = probsData[0];
	for (int i = 1; i < cpuProbs.size(0); i++) {
		float val = probsData[i];
		if (val > largestVal) {
			largestVal = val;
			bestIdx = i;
		}
	}

	return bestIdx;
}

RLGPC::DiscretePolicy::BackpropResult RLGPC::DiscretePolicy::GetBackpropData(torch::Tensor obs, torch::Tensor acts) {
	// Get probability of each action
	acts = acts.to(torch::kInt64);
	auto probs = GetOutput(obs);
	probs = probs.view({ -1, actionAmount });
	probs = torch::clamp(probs, ACTION_MIN_PROB, 1); // Prevent actions from being impossible, and also log(0)
	// TODO: 3x repeated code block

	// Man I give up on commenting this I got no clue
	auto logProbs = torch::log(probs);
	auto actionLogProbs = logProbs.gather(-1, acts);
	auto entropy = -(logProbs * probs).sum(-1);

	return BackpropResult{ actionLogProbs.to(device), entropy.to(device).mean() };
}
