#pragma once
#include "RewardFunction.h"

namespace RLGSC {
	// This is a wrapper class that makes another reward function zero-sum and team-distributed
	// Per-player reward is calculated using: ownReward*(1-teamSpirit) + avgTeamReward*teamSpirit - avgOpponentReward
	class ZeroSumReward : public RewardFunction {
	public:

		RewardFunction* childFunc;
		bool ownsFunc;

		float teamSpirit;

		// Team spirit is the fraction of reward shared between teammates
		ZeroSumReward(RewardFunction* childFunc, float teamSpirit, bool ownsFunc = true)
			: childFunc(childFunc), teamSpirit(teamSpirit), ownsFunc(ownsFunc) {

		}

		~ZeroSumReward() {
			if (ownsFunc)
				delete childFunc;
		}

	protected:
		virtual void Reset(const GameState& initialState) {
			childFunc->Reset(initialState);
		}

		virtual void PreStep(const GameState& state) {
			childFunc->PreStep(state);
		}

		// Get all rewards for all players
		virtual std::vector<float> GetAllRewards(const GameState& state, const ActionSet& prevActions, bool final);
	};
}