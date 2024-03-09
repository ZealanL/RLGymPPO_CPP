#pragma once
#include "RewardFunction.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/combined_reward.py
namespace RLGSC {
	class CombinedReward : public RewardFunction {
	public:
		std::vector<RewardFunction*> rewardFuncs;
		std::vector<float> rewardWeights;
		bool ownsFuncs;

		CombinedReward(std::vector<RewardFunction*> rewardFuncs, std::vector<float> rewardWeights, bool ownsFuncs = false) :
			rewardFuncs(rewardFuncs), rewardWeights(rewardWeights), ownsFuncs(ownsFuncs) {
			assert(rewardFuncs.size() == rewardWeights.size());
		}

		CombinedReward(std::vector<std::pair<RewardFunction*, float>> funcsWithWeights, bool ownsFuncs = false) :
			ownsFuncs(ownsFuncs) {
			for (auto& pair : funcsWithWeights) {
				rewardFuncs.push_back(pair.first);
				rewardWeights.push_back(pair.second);
			}
		}

		virtual void Reset(const GameState& initialState) {
			for (auto func : rewardFuncs)
				func->Reset(initialState);
		}

		virtual void PreStep(const GameState& state) {
			for (auto func : rewardFuncs)
				func->PreStep(state);
		}

		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			float totalReward = 0;
			for (int i = 0; i < rewardFuncs.size(); i++)
				totalReward += rewardFuncs[i]->GetReward(player, state, prevAction) * rewardWeights[i];
			return totalReward;
		}

		virtual float GetFinalReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			float totalReward = 0;
			for (int i = 0; i < rewardFuncs.size(); i++)
				totalReward += rewardFuncs[i]->GetFinalReward(player, state, prevAction) * rewardWeights[i];
			return totalReward;
		}

		virtual ~CombinedReward() {
			if (ownsFuncs)
				for (auto func : rewardFuncs)
					delete func;
		}
	};
}