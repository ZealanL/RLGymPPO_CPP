#include "PolicyInferUnit.h"

#include <RLGymPPO_CPP/PPO/DiscretePolicy.h>
#include <RLGymPPO_CPP/FrameworkTorch.h>
#include <torch/csrc/api/include/torch/serialize.h>

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
		auto streamIn = std::ifstream(policyPath, std::ios::binary);
		torch::load(policy->seq, streamIn, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"Failed to load model, checkpoint may be corrupt or of different model arch.\n" <<
			"Exception: " << e.what()
		);
	}

	RG_LOG(" > Done!");
}

ActionSet RLGPC::PolicyInferUnit::InferPolicyAll(const GameState& state, const ActionSet& prevActions, bool deterministic) {
	FList2 obsSet = {};
	for (int i = 0; i < state.players.size(); i++)
		obsSet.push_back(obsBuilder->BuildOBS(state.players[i], state, prevActions[i]));
	
	RG_NOGRAD;
	torch::Tensor inputTen = FLIST2_TO_TENSOR(obsSet).to(policy->device);
	auto actionResult = policy->GetAction(inputTen, deterministic);
	auto actionParserInput = TENSOR_TO_ILIST(actionResult.action);

	return actionParser->ParseActions(actionParserInput, state);
}

Action RLGPC::PolicyInferUnit::InferPolicySingle(const PlayerData& player, const GameState& state, const Action& prevAction, bool deterministic) {
	FList obs = obsBuilder->BuildOBS(player, state, prevAction);

	int playerIndex = 0;
	for (int i = 1; i < state.players.size(); i++) {
		if (state.players[i].carId == player.carId) {
			playerIndex = i;
			break;
		}
	}

	RG_NOGRAD;
	torch::Tensor inputTen = torch::tensor(obs).to(policy->device);
	auto actionResult = policy->GetAction(inputTen, deterministic);
	IList actionParserInput = IList(state.players.size());
	actionParserInput[playerIndex] = actionResult.action.item<int>();

	return actionParser->ParseActions(actionParserInput, state)[playerIndex];
}