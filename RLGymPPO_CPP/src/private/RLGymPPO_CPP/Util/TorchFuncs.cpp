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
		float done = 1 - terminal[step];
		float trunc = 1 - truncated[step];

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
		lastGAE_LAM = delta + gamma * lambda * done * trunc * lastGAE_LAM;
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