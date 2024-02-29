#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <ATen/autocast_mode.h>

using namespace torch;

Tensor _CopyParams(nn::Module* mod) {
	return torch::nn::utils::parameters_to_vector(mod->parameters()).cpu();
}

RLGPC::PPOLearner::PPOLearner(int obsSpaceSize, int actSpaceSize, PPOLearnerConfig config, Device device) 
	: config(config), device(device) {

	if (config.miniBatchSize == 0)
		this->config.miniBatchSize = config.batchSize;

	policy = new DiscretePolicy(obsSpaceSize, actSpaceSize, config.policyLayerSizes, device);
	valueNet = new ValueEstimator(obsSpaceSize, config.criticLayerSizes, device);
	policyOptimizer = new optim::Adam(policy->parameters(), optim::AdamOptions(config.policyLR));
	valueOptimizer = new optim::Adam(valueNet->parameters(), optim::AdamOptions(config.criticLR));
	valueLossFn = nn::MSELoss();
}

void RLGPC::PPOLearner::Learn(ExperienceBuffer* expBuffer, Report& report) {
	bool autocast = config.autocastLearn;
	if (autocast) RG_AUTOCAST_ON();

	int
		numIterations = 0,
		numMinibatchIterations = 0;
	float
		meanEntropy = 0,
		meanDivergence = 0,
		meanValLoss = 0;
	FList clipFractions = {};

	// Save parameters first
	auto policyBefore = _CopyParams(policy);
	auto criticBefore = _CopyParams(valueNet);

	float batchSizeRatio = config.miniBatchSize / config.batchSize;

	Timer totalTimer = {};
	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = expBuffer->GetAllBatchesShuffled(config.batchSize);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchTargetValues = batch.values;
			auto batchAdvantages = batch.advantages;

			batchActs = batchActs.view({ config.batchSize, -1 });
			policyOptimizer->zero_grad();
			valueOptimizer->zero_grad();

			for (int mbs = 0; mbs < config.batchSize; mbs += config.miniBatchSize) {
				Timer timer = {};

				int start = mbs;
				int stop = start + config.miniBatchSize;

				// Send everything to the device and enforce correct shapes
				auto acts = batchActs.slice(0, start, stop).to(device, true);
				auto obs = batchObs.slice(0, start, stop).to(device, true);
				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true);
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true);

				timer.Reset();
				// Compute value estimates
				auto vals = valueNet->Forward(obs);
				report.Accum("PPO Value Estimate Time", timer.Elapsed());

				timer.Reset();
				// Get policy log probs & entropy
				DiscretePolicy::BackpropResult bpResult = policy->GetBackpropData(obs, acts);
				auto logProbs = bpResult.actionLogProbs;
				auto entropy = bpResult.entropy;

				logProbs = logProbs.view_as(oldProbs);
				report.Accum("PPO Backprop Data Time", timer.Elapsed());

				// Compute PPO loss
				auto ratio = exp(logProbs - oldProbs);
				auto clipped = clamp(
					ratio, 1 - config.clipRange, 1 + config.clipRange
				);

				vals = vals.view_as(targetValues);

				// Compute policy loss
				auto policyLoss = -min(
					ratio * advantages, clipped * advantages
				).mean();
				auto valueLoss = valueLossFn(vals, targetValues);
				auto ppoLoss = (policyLoss - entropy * config.entCoef) * batchSizeRatio;

				// Compute KL divergence & clip fraction using SB3 method for reporting
				float kl;
				float clipFraction;
				{
					RG_NOGRAD;

					auto logRatio = logProbs - oldProbs;
					auto klTensor = (exp(logRatio) - 1) - logRatio;
					kl = klTensor.mean().detach().cpu().item<float>();

					clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat)).cpu().item<float>();
					clipFractions.push_back(clipFraction);
				}
				

				timer.Reset();
				// NOTE: These gradient calls are a substantial portion of learn time
				//	From my testing, they are around 61% of learn time
				//	Results will probably vary heavily depending on model size and GPU strength
				ppoLoss.backward();
				valueLoss.backward();
				report.Accum("PPO Gradient Time", timer.Elapsed());

				meanValLoss += valueLoss.cpu().detach().item<float>();
				meanDivergence += kl;
				meanEntropy += entropy.cpu().detach().item<float>();
				numMinibatchIterations += 1;
			}

			nn::utils::clip_grad_norm_(valueNet->parameters(), 0.5f);
			nn::utils::clip_grad_norm_(policy->parameters(), 0.5f);

			policyOptimizer->step();
			valueOptimizer->step();

			numIterations += 1;
		}
	}

	numIterations = RS_MAX(numIterations, 1);
	numMinibatchIterations = RS_MAX(numMinibatchIterations, 1);

	// Compute averages for the metrics that will be reported
	meanEntropy /= numMinibatchIterations;
	meanDivergence /= numMinibatchIterations;
	meanValLoss /= numMinibatchIterations;

	float meanClip = 0;
	if (!clipFractions.empty()) {
		for (float f : clipFractions)
			meanClip += f;
		meanClip /= clipFractions.size();
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = _CopyParams(policy);
	auto criticAfter = _CopyParams(valueNet);

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	float totalTime = totalTimer.Elapsed();

	// Assemble and return report
	cumulativeModelUpdates += numIterations;
	report["PPO Batch Consumption Time"] = totalTime / numIterations;
	report["Cumulative Model Updates"] = cumulativeModelUpdates;
	report["Policy Entropy"] = meanEntropy;
	report["Mean KL Divergence"] = meanDivergence;
	report["Value Function Loss"] = meanValLoss;
	report["SB3 Clip Fraction"] = meanClip;
	report["Policy Update Magnitude"] = policyUpdateMagnitude;
	report["Value Function Update Magnitude"] = criticUpdateMagnitude;
	report["PPO Learn Time"] = totalTimer.Elapsed();

	policyOptimizer->zero_grad();
	valueOptimizer->zero_grad();
	
	if (autocast) RG_AUTOCAST_OFF();
}

// Code in here is by alireza_dizaji
// Found at https://discuss.pytorch.org/t/how-would-i-do-load-state-dict-in-c/76720/5
// Modified heavily to be one function, use RG_ERR_CLOSE(), etc.
void load_state_dict(nn::Module* model, std::filesystem::path filename) {
	std::string errorPrefix = RS_STR("Failed to load model state dict at path ", filename, ", ");

	std::ifstream input(filename, std::ios::binary);
	std::vector<char> bytes(
		(std::istreambuf_iterator<char>(input)),
		(std::istreambuf_iterator<char>()));

	input.close();
	c10::Dict<IValue, IValue> weights = pickle_load(bytes).toGenericDict();

	const OrderedDict<std::string, at::Tensor>& model_params = model->named_parameters();
	std::vector<std::string> param_names;
	for (auto const& w : model_params) {
		param_names.push_back(w.key());
	}

	NoGradGuard no_grad;
	for (auto const& w : weights) {
		std::string name = w.key().toStringRef();
		at::Tensor src = w.value().toTensor();

		if (std::find(param_names.begin(), param_names.end(), name) != param_names.end()) {
			auto dst = model_params.find(name);
			auto dstSizes = dst->sizes();
			auto srcSizes = src.sizes();

			if (dstSizes != srcSizes) {
				RG_ERR_CLOSE(
					errorPrefix << "source and destination tensor sizes do not match (" << dstSizes << ", " << srcSizes << ")"
				);
			}

			dst->copy_(src);
		} else {
			RG_ERR_CLOSE(
				errorPrefix << name << " does not exist in model parameters"
			);
		};
	}
}

template<typename T>
void TorchLoadSave(T* obj, std::filesystem::path path, bool load) {
	if (load) {
		auto streamIn = std::ifstream(path, std::ios::binary);
		streamIn >> std::noskipws;
		torch::load(*obj, streamIn);
	} else {
		auto streamOut = std::ofstream(path, std::ios::binary);
		torch::save(*obj, streamOut);
	}
}

void TorchLoadSaveAll(RLGPC::PPOLearner* learner, std::filesystem::path folderPath, bool load) {
	TorchLoadSave(&learner->policy->seq, folderPath / "PPO_POLICY.pt", load);
	TorchLoadSave(&learner->valueNet->seq, folderPath / "PPO_VALUE_NET.pt", load);
	TorchLoadSave(&(learner->policyOptimizer->parameters()), folderPath / "PPO_POLICY_OPTIMIZER.pt", load);
	TorchLoadSave(&(learner->valueOptimizer->parameters()), folderPath / "PPO_VALUE_NET_OPTIMIZER.pt", load);
}

void RLGPC::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	RG_LOG("PPOLearner(): Saving models to: " << folderPath);
	if (!std::filesystem::is_directory(folderPath))
		std::filesystem::create_directories(folderPath);
	TorchLoadSaveAll(this, folderPath, false);
}

void RLGPC::PPOLearner::LoadFrom(std::filesystem::path folderPath, bool isFromPython)  {
	RG_LOG("PPOLearner(): Loading models from: " << folderPath);
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	if (isFromPython) {
		load_state_dict(policy, folderPath / "PPO_POLICY.pt");
		load_state_dict(valueNet, folderPath / "PPO_VALUE_NET.pt");

		// TODO: Load optimizer
	} else {
		TorchLoadSaveAll(this, folderPath, true);
	}
}