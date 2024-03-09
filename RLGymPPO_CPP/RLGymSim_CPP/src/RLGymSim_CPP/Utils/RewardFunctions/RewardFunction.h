#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/reward_function.py
namespace RLGSC {
	class RewardFunction {
	public:
		virtual void Reset(const GameState& initialState) {}

		virtual void PreStep(const GameState& state) {}

		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) = 0;

		virtual float GetFinalReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			return GetReward(player, state, prevAction);
		}

		virtual ~RewardFunction() {};
	};
}