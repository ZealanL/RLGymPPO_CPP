#pragma once
#include "../Lists.h"
#include "../Threading/GameInst.h"
#include "../LearnerConfig.h"

namespace RLGPC {
	class RG_IMEXPORT InferUnit {
	public:

		RLGSC::OBSBuilder* obsBuilder;
		RLGSC::ActionParser* actionParser;
		class DiscretePolicy* policy;
		class ValueEstimator* critic;

		InferUnit(
			RLGSC::OBSBuilder* obsBuilder, RLGSC::ActionParser* actionParser, 
			std::filesystem::path modelPath, bool isPolicy, int obsSize, const RLGPC::IList& layerSizes, bool gpu = false);

		RLGSC::FList GetObs(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction);
		RLGSC::FList2 GetObs(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions);

		RLGSC::ActionSet InferPolicyAll(
			const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions, 
			bool deterministic, float temperature = 1.0f
		);
		RLGSC::Action InferPolicySingle(
			const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction, 
			bool deterministic, float temperature = 1.0f
		);
		RLGSC::FList InferPolicySingleDistrib(
			const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction, 
			float temperature = 1.0f
		);

		RLGSC::FList InferCriticAll(
			const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions
		);
		float InferCriticSingle(
			const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction
		);
	};
}