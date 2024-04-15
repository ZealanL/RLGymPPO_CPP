#include "PolicyInferUnit.h"

#include <RLGymPPO_CPP/PPO/DiscretePolicy.h>

using namespace RLGSC;
using namespace RLGPC;

RLGPC::PolicyInferUnit::PolicyInferUnit(
	OBSBuilder* obsBuilder, ActionParser* actionParser, 
	std::filesystem::path policyPath, int obsSize, const IList& policyLayerSizes, bool gpu)
	: obsBuilder(obsBuilder), actionParser(actionParser) {

	RG_LOG("PolicyInferUnit():");

	RG_LOG(" > Creating policy...");
	torch::Device device = gpu ? torch::kCUDA : torch::kCPU;
	policy = new DiscretePolicy(obsSize, actionParser->GetActionAmount(), policyLayerSizes, device);
	
	RG_LOG(" > Loading policy...");
	try {
		auto streamIn = std::ifstream(path, std::ios::binary);
		torch::load(seq, streamIn, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"Failed to load model, checkpoint may be corrupt or of different model arch.\n" <<
			"Exception: " << e.what()
		);
	}

	RG_LOG(" > Done!");
}

ActionSet RLGPC::PolicyInferUnit::InferPolicy(const GameState& state, const ActionSet& prevActions, bool deterministic) {
	FList2 obsSet = {};
	for (auto& player : state.players)
		obsSet.push_back(obsBuilder->BuildOBS(player, state));
	
	RG_NOGRAD;
	torch::Tensor inputTen = FLIST2_TO_TENSOR(obsSet);
	auto actionResult = policy->GetAction(inputTen, deterministic);
	return TENSOR_TO_ILIST(actionResult.action);
}