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

void RLGPC::TorchFuncs::SerializeOptimizer(torch::optim::Adam* optim, DataStreamOut& out) {
	out.Write<uint32_t>(OPTIMIZER_SERIALIZE_PREFIX);
	auto& groups = optim->param_groups();
	out.Write<uint32_t>(groups.size());
	for (auto& group : groups) {
		auto& params = group.params();
		out.Write<uint32_t>(params.size());
		for (auto param : params) {
			auto flatParam = param.cpu().flatten();
			out.Write<uint64_t>(flatParam.numel());

			if (param.scalar_type() != torch::ScalarType::Float)
				RG_ERR_CLOSE("TorchFuncs::SerializeOptimizer(): Failed to serialize param of non-float type");

			out.WriteBytes(flatParam.data_ptr<float>(), flatParam.numel() * sizeof(float));
		}
	}
}

void RLGPC::TorchFuncs::DeserializeOptimizer(torch::optim::Adam* optim, DataStreamIn& in) {
	constexpr const char* ERR_PREFIX = "TorchFuncs::DeserializeOptimizer(): ";
	RG_NOGRAD;

	uint32_t key = in.Read<uint32_t>();
	if (key != OPTIMIZER_SERIALIZE_PREFIX)
		RS_ERR_CLOSE(ERR_PREFIX << "File is not a valid serialized optimizer (bad prefix)");

	auto& groups = optim->param_groups();
	uint32_t targetGroupsSize = in.Read<uint32_t>();
	if (targetGroupsSize != groups.size())
		RG_ERR_CLOSE(ERR_PREFIX << "Mismatched groups size (" << targetGroupsSize << "/" << groups.size() << ")");

	for (auto& group : groups) {
		auto& params = group.params();
		uint32_t targetParamsSize = in.Read<uint32_t>();
		if (targetParamsSize != params.size())
			RG_ERR_CLOSE(ERR_PREFIX << "Mismatched params size (" << targetParamsSize << "/" << params.size() << ")");
		for (auto& param : params) {
			uint64_t targetSize = in.Read<uint64_t>();
			if (targetSize != param.numel())
				RG_ERR_CLOSE(ERR_PREFIX << "Mismatched tensor size (" << targetSize << "/" << params.size() << ")");

			if (param.scalar_type() != torch::ScalarType::Float)
				RG_ERR_CLOSE(ERR_PREFIX << "Failed to deserialize param of non-float type");

			FList data = FList(targetSize);
			in.ReadBytes(data.data(), targetSize * sizeof(float));

			auto newParam = torch::tensor(data).reshape_as(param).to(param.device());
			
			// Must be in place, param is referenced elsewhere
			param.copy_(newParam);
		}
	}
}