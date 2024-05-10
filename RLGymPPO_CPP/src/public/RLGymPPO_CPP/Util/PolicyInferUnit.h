#pragma once
#include "../Lists.h"
#include "../Threading/GameInst.h"
#include "../LearnerConfig.h"

namespace RLGPC {
	class RG_IMEXPORT PolicyInferUnit {
	public:

		RLGSC::OBSBuilder* obsBuilder;
		RLGSC::ActionParser* actionParser;
		class DiscretePolicy* policy;

		PolicyInferUnit(
			RLGSC::OBSBuilder* obsBuilder, RLGSC::ActionParser* actionParser, 
			std::filesystem::path policyPath, int obsSize, const RLGPC::IList& policyLayerSizes, bool gpu);

		RLGSC::ActionSet InferPolicyAll(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions, bool deterministic);
		RLGSC::Action InferPolicySingle(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction, bool deterministic);
	};
}