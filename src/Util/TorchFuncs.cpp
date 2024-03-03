#include "TorchFuncs.h"

#include <torch/csrc/api/include/torch/serialize.h>

void RLGPC::TorchFuncs::ComputeGAE(
	const FList& rews, const FList& dones, const FList& truncated, const FList& values, 
	torch::Tensor& outAdvantages, torch::Tensor& outValues, FList& outReturns, 
	float gamma, float lambda, float returnStd
) {
	auto next_values = FList(values.begin() + 1, values.end());
	auto& terminal = dones;

	float returnScale = 1 / returnStd;
	if (isnan(returnScale))
		returnScale = 0;

	float lastGAE_LAM = 0;
	int nReturns = rews.size();
	FList adv = FList(nReturns);
	FList returns = FList(nReturns);
	float lastReturn = 0;

	for (int step = nReturns - 1; step >= 0; step--) {
		float done, trunc;
		if (step == nReturns - 1) {
			done = 1 - terminal[terminal.size() - 1];
			trunc = 1 - truncated[truncated.size() - 1];
		} else {
			done = 1 - terminal[step + 1];
			trunc = 1 - truncated[step + 1];
		}

		float norm_rew;
		if (returnStd != 0) {
			norm_rew = RS_CLAMP(rews[step] * returnScale, -10, 10);
		} else {
			norm_rew = rews[step];
		}

		float pred_ret = norm_rew + gamma * next_values[step] * done;
		float delta = pred_ret - values[step];
		float ret = rews[step] + lastReturn * gamma * done * trunc;
		returns[step] = ret;
		lastReturn = ret;
		lastGAE_LAM = delta + gamma * lambda * done * lastGAE_LAM;
		adv[step] = lastGAE_LAM;
	}

	outAdvantages = torch::tensor(adv);
	auto outValuesList = FList(values.size() - 1);
	for (int i = 0; i < values.size() - 1; i++)
		outValuesList[i] = values[i] + adv[i];

	outValues = torch::tensor(outValuesList);
	outReturns = returns;
}

torch::Tensor RLGPC::TorchFuncs::ConcatSafe(torch::Tensor a, torch::Tensor b) {
	if (a.defined()) {
		return torch::cat({ a,b });
	} else {
		return b.clone();
	}
}

void RLGPC::TorchFuncs::LoadStateDict(torch::nn::Module* mod, std::filesystem::path path) {
	constexpr const char* ERROR_PREFIX = "TorchFuncs::LoadStateDict(): ";

	// From: https://github.com/pytorch/pytorch/issues/36577#issuecomment-1279666295

	std::ifstream input(path, std::ios::binary);
	input >> std::noskipws;
	if (!input.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Cannot open input file at " << path);

	std::vector<char> bytes(
		(std::istreambuf_iterator<char>(input)),
		(std::istreambuf_iterator<char>())
	);

	try {
		c10::IValue depickled = torch::pickle_load(bytes);
		c10::Dict<c10::IValue, c10::IValue> weights = depickled.toGenericDict();

		const auto& modelParams = mod->named_parameters();
		std::vector<std::string> paramNames;
		for (auto const& w : modelParams) {
			paramNames.push_back(w.key());
		}

		RG_NOGRAD;
		for (auto const& w : weights) {
			std::string name = w.key().toStringRef();
			at::Tensor param = w.value().toTensor();

			if (std::find(paramNames.begin(), paramNames.end(), name) != paramNames.end()) {
				modelParams.find(name)->copy_(param);
			} else {
				RS_ERR_CLOSE(ERROR_PREFIX << "Cannot find model param name \"" << name << "\"");
			}
		}
	} catch (std::exception& e) {
		RG_ERR_CLOSE(ERROR_PREFIX << "Exception while loading " << path << ":\n" << e.what());
	}
}